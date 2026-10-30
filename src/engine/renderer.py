"""Master FFmpeg pipeline (no MoviePy).

1. Each scene becomes one or two shots (second image cut on the middle word), rendered in parallel:
   - eased Ken Burns move; impact scenes get a punch-in + decaying shake (plus a white flash in the .ass)
   - 2.5D scenes: blurred parallax background + rembg sprite with sway, bob, drop shadow and pop-in entrance
2. Clips are concatenated losslessly (concat demuxer, stream copy).
3. Final pass: dust overlay (screen blend), .ass karaoke + callouts burn-in, narration + side-chain-ducked BGM
   + optional wind/sea bed + SFX hits.
"""
from __future__ import annotations

import logging
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from src.config_loader import ChannelConfig, get_settings
from src.engine.audio_tts import TAIL_SECONDS
from src.engine.media_fetcher import SceneAsset
from src.engine.schemas import CameraMove, Scene, TransitionSfx, VideoScript
from src.engine.subtitles_ass import Word, add_overlays

log = logging.getLogger(__name__)

W, H, FPS = 1080, 1920, 30
AUDIO_FMT = "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"


class RenderError(Exception):
    pass


def run_ffmpeg(args: list[str], cwd: Path | None = None, timeout: int = 1800) -> None:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args]
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RenderError(f"ffmpeg exited {proc.returncode}: {proc.stderr[-3000:]}")


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def _escape_filter_path(path: Path) -> str:
    """Escape a path for use inside a filter option (handles Windows drive colons)."""
    return "'" + str(path).replace("\\", "/").replace(":", "\\:").replace("'", "\\'") + "'"


# --------------------------------------------------------------------------- timing


def plan_scene_frames(total: float, scene_word_counts: list[int], words: list[Word], max_shift: float = 0.2,
                      min_scene: float = 3.4) -> list[int]:
    """Cuts land on the spoken word that starts each scene, clamped to +-max_shift around the equal split
    so pacing stays ~3.8-4.2 s per scene."""
    n = len(scene_word_counts)
    equal = total / n
    total_script_words = max(1, sum(scene_word_counts))
    boundaries = [0.0]
    cumulative = 0
    for k in range(n - 1):
        cumulative += scene_word_counts[k]
        idx = min(len(words) - 1, round(cumulative / total_script_words * len(words)))
        aligned = words[idx].start - 0.06  # cut slightly before the word: feels snappier
        ideal = equal * (k + 1)
        b = min(max(aligned, ideal - max_shift), ideal + max_shift)
        boundaries.append(max(b, boundaries[-1] + min_scene))
    boundaries.append(total)
    frames = [round(boundaries[i + 1] * FPS) - round(boundaries[i] * FPS) for i in range(n)]
    return frames


def split_frame(scene_idx: int, start: float, frames: int, scene_word_counts: list[int], words: list[Word],
                min_shot: float = 1.2) -> int:
    """Frames of the first shot when a scene has two: cut just before its middle word."""
    total_script_words = max(1, sum(scene_word_counts))
    middle = sum(scene_word_counts[:scene_idx]) + scene_word_counts[scene_idx] / 2
    idx = min(len(words) - 1, round(middle / total_script_words * len(words)))
    dur = frames / FPS
    cut = min(max(words[idx].start - 0.06 - start, min_shot), dur - min_shot)
    return max(1, min(frames - 1, round(cut * FPS)))


# --------------------------------------------------------------------------- camera math


# Second shot of a scene moves the opposite way so the cut reads as a new angle
COUNTER_MOVE = {
    CameraMove.PUSH_IN: CameraMove.PULL_OUT,
    CameraMove.PULL_OUT: CameraMove.PUSH_IN,
    CameraMove.PAN_LEFT: CameraMove.PAN_RIGHT,
    CameraMove.PAN_RIGHT: CameraMove.PAN_LEFT,
    CameraMove.SHAKE: CameraMove.PULL_OUT,
}


PARALLAX_MOVES = {CameraMove.PUSH_IN, CameraMove.PULL_OUT, CameraMove.SHAKE}  # pans would misalign the layers
BG_DEPTH, FG_DEPTH = 0.45, 1.45  # zoom travel of each 2.5D layer relative to a flat shot


def _zoompan(move: CameraMove, frames: int, k: float, seed: int, d: int, punch: bool = False,
             depth: float = 1.0) -> str:
    """zoompan expression set. `k` = supersampling factor of the input (amplitudes are in input pixels).

    Smooth eased moves on every shot; only impact beats (`punch`) get a hard punch-in and a decaying shake.
    `depth` scales the zoom travel: 2.5D layers share the start zoom and centre but travel different amounts,
    so the foreground always stays larger than its copy in the background.
    """
    n = max(frames - 1, 1)
    e = f"((1-cos(PI*on/{n}))/2)"  # ease in-out 0 -> 1
    travel, shake = 0.13 * depth, 0.06 * depth
    base = {
        CameraMove.PUSH_IN: f"1.04+{travel:.3f}*{e}",
        CameraMove.PULL_OUT: f"{1.04 + travel:.3f}-{travel:.3f}*{e}",
        CameraMove.PAN_LEFT: "1.16",
        CameraMove.PAN_RIGHT: "1.16",
        CameraMove.SHAKE: f"1.10+{shake:.3f}*{e}",
    }[move]
    x = {
        CameraMove.PAN_LEFT: f"(iw-iw/zoom)*(1-{e})",
        CameraMove.PAN_RIGHT: f"(iw-iw/zoom)*{e}",
    }.get(move, "(iw-iw/zoom)/2")
    y = "(ih-ih/zoom)/2"
    z = base
    if punch or move == CameraMove.SHAKE:
        amp = 20 * k
        z = f"{base}+0.22*pow(max(0,1-on/8),2)"  # +22% snapping back in ~0.27 s
        x = f"{x}+{amp:.1f}*exp(-on/6)*sin(on*2.7+{seed % 6})"
        y = f"{y}+{amp:.1f}*exp(-on/6)*cos(on*3.1+{seed % 6})"
    return (f"zoompan=z='{z}':x='clip({x},0,iw-iw/zoom)':y='clip({y},0,ih-ih/zoom)'"
            f":d={d}:s={W}x{H}:fps={FPS}")


def _sprite_size(sprite: Path) -> tuple[int, int]:
    with Image.open(sprite) as img:
        sw, sh = img.size
    scale = min(W * 0.85 / sw, H * 0.58 / sh)
    return max(2, int(sw * scale) // 2 * 2), max(2, int(sh * scale) // 2 * 2)


def _render_scene(idx: int, asset: SceneAsset, scene: Scene, frames: int, channel: ChannelConfig,
                  out: Path, mirror: bool, path: Path | None = None, move: CameraMove | None = None,
                  punch: bool = False, fg: Path | None = None) -> Path:
    """Render one shot. `path`/`move` override the asset's main image and the scene's camera (second shots);
    `fg` is that image's foreground layer for 2.5D parallax."""
    dur = frames / FPS
    seed = idx * 7 + 3
    flip = ",hflip" if mirror else ""
    move = move or scene.camera
    if asset.kind == "video":
        k = 1.5
        inputs = ["-stream_loop", "-1", "-i", str(asset.path)]
        src = f"[0:v]fps={FPS},scale={int(W * k)}:{int(H * k)}:force_original_aspect_ratio=increase," \
              f"crop={int(W * k)}:{int(H * k)},setsar=1{flip},{_zoompan(move, frames, k, seed, 1, punch)}"
    else:
        k = 2.0
        inputs = ["-i", str(path or asset.path)]
        src = f"[0:v]scale={int(W * k)}:{int(H * k)}:force_original_aspect_ratio=increase," \
              f"crop={int(W * k)}:{int(H * k)},setsar=1{flip},{_zoompan(move, frames, k, seed, frames, punch)}"

    grade = f",{channel.grade}" if channel.grade else ""
    if fg is not None and asset.kind == "image" and asset.sprite is None and move in PARALLAX_MOVES:
        # 2.5D: background travels little and goes slightly soft, the lifted subject travels a lot
        frame = f"scale={int(W * k)}:{int(H * k)}:force_original_aspect_ratio=increase,crop={int(W * k)}:{int(H * k)},setsar=1{flip}"
        inputs += ["-loop", "1", "-framerate", str(FPS), "-t", f"{dur:.3f}", "-i", str(fg)]
        graph = (
            f"[0:v]{frame},{_zoompan(move, frames, k, seed, frames, punch, BG_DEPTH)},gblur=sigma=1.5[bg];"
            f"[1:v]format=rgba,{frame},{_zoompan(move, frames, k, seed, frames, punch, FG_DEPTH)},format=rgba[fg];"
            f"[bg][fg]overlay=0:0:format=auto{grade},format=yuv420p[v]"
        )
    elif asset.sprite is not None:
        sw, sh = _sprite_size(asset.sprite)
        side = -1 if idx % 2 else 1
        inputs += ["-loop", "1", "-framerate", str(FPS), "-t", f"{dur:.3f}", "-i", str(asset.sprite)]
        x = f"(W-w)/2{side * 110:+d}+26*sin(2*PI*t/3.4)"
        y = "H-h-150+14*sin(2*PI*t/1.3)+760*pow(max(0,1-t/0.32),3)"  # cubic ease-out pop-in from below
        graph = (
            f"{src},gblur=sigma=2.5[bg];"  # depth of field: background slightly out of focus
            # Library sprites repeat across scenes: mirror them on the left side so they read as new shots
            f"[1:v]format=rgba,scale={sw}:{sh}{',hflip' if side < 0 else ''},"
            f"rotate='0.035*sin(2*PI*t/2.6)':c=none:ow='rotw(0.06)':oh='roth(0.06)',split[spr][sh];"
            f"[sh]colorchannelmixer=rr=0:gg=0:bb=0:aa=0.5,gblur=sigma=16[shadow];"
            f"[bg][shadow]overlay=x='{x}+22':y='{y}+34'[b1];"
            f"[b1][spr]overlay=x='{x}':y='{y}'{grade},format=yuv420p[v]"
        )
    else:
        graph = f"{src}{grade},format=yuv420p[v]"

    run_ffmpeg([
        *inputs, "-filter_complex", graph, "-map", "[v]", "-frames:v", str(frames), "-r", str(FPS),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-an", str(out),
    ])
    return out


# --------------------------------------------------------------------------- audio events


def _sfx_events(scenes: list[Scene], starts: list[float], assets: list[SceneAsset]) -> list[tuple[str, float, float]]:
    """(sfx name, time, volume) for every cut."""
    events: list[tuple[str, float, float]] = [("impact", 0.0, 0.75)]  # hook hit
    for i in range(1, len(scenes)):
        t = starts[i]
        kind = scenes[i].sfx
        if kind == TransitionSfx.IMPACT:
            # Build-up into the twist: heartbeat, then a riser that ends exactly on the impact
            events += [("heartbeat", max(0.0, t - 3.6), 0.6), ("riser", max(0.0, t - 2.2), 0.5), ("impact", t, 0.9)]
        elif kind == TransitionSfx.POP:
            events += [("whoosh", max(0.0, t - 0.12), 0.25), ("pop", t, 0.5)]
        elif kind == TransitionSfx.WHOOSH:
            events += [("whoosh", max(0.0, t - 0.12), 0.55)]
        else:
            events += [("whoosh", max(0.0, t - 0.12), 0.2)]
        if assets[i].sprite is not None and kind != TransitionSfx.POP:
            events.append(("pop", t + 0.1, 0.4))  # character entrance
    return events


MUSIC_DROP_S = 0.45  # the music cuts out right before each impact so the hit lands in near-silence


def _final_pass(base: Path, narration: Path, subs: Path, channel: ChannelConfig, events, total: float,
                out: Path, workdir: Path, drops: list[float] = ()) -> None:
    media = get_settings().media_dir
    args = ["-i", str(base), "-i", str(narration)]
    next_input = 2
    graph: list[str] = []

    bgm = media / channel.bgm.file
    bgm_idx = None
    windows = "+".join(f"between(t,{max(0.0, t - MUSIC_DROP_S):.2f},{t:.2f})" for t in drops)
    music_drops = f",volume='if({windows},0.06,1)':eval=frame" if windows else ""
    if bgm.exists():
        args += ["-stream_loop", "-1", "-i", str(bgm)]
        bgm_idx, next_input = next_input, next_input + 1

    overlay = media / channel.overlay.file
    if overlay.exists():
        args += ["-stream_loop", "-1", "-i", str(overlay)]
        graph.append(
            f"[{next_input}:v]scale={W}:{H},fps={FPS},format=gbrp[dust];[0:v]format=gbrp[base];"
            f"[base][dust]blend=all_mode=screen:all_opacity={channel.overlay.opacity}:shortest=1,format=yuv420p[vb]"
        )
        next_input += 1
    else:
        graph.append("[0:v]null[vb]")

    fonts = media / "fonts"
    fontsdir = f":fontsdir={_escape_filter_path(fonts)}" if fonts.is_dir() else ""
    graph.append(f"[vb]ass={subs.name}{fontsdir}[vout]")

    mix = ["[narr]"]
    graph.append(f"[1:a]{AUDIO_FMT},asplit=2[narr][key]")
    if bgm_idx is not None:
        graph.append(
            f"[{bgm_idx}:a]{AUDIO_FMT},atrim=0:{total:.3f},volume={channel.bgm.volume},afade=t=in:d=1.2{music_drops}[bgraw];"
            f"[bgraw][key]sidechaincompress=threshold=0.025:ratio=10:attack=15:release=350[bgm]"
        )
        mix.append("[bgm]")
    else:
        graph.append("[key]anullsink")

    if channel.ambience == "wind":  # brown noise, band-limited and slowly breathing: distant wind and sea
        args += ["-f", "lavfi", "-t", f"{total:.3f}", "-i", "anoisesrc=color=brown:amplitude=0.6:r=48000"]
        graph.append(f"[{next_input}:a]{AUDIO_FMT},lowpass=f=420,highpass=f=60,tremolo=f=0.13:d=0.6,volume=0.10[amb]")
        mix.append("[amb]")
        next_input += 1

    for n, (name, t, vol) in enumerate(events):
        sfx = media / "sfx" / f"{name}.wav"
        if not sfx.exists():
            continue
        args += ["-i", str(sfx)]
        ms = int(t * 1000)
        graph.append(f"[{next_input}:a]{AUDIO_FMT},volume={vol},adelay={ms}|{ms}[s{n}]")
        mix.append(f"[s{n}]")
        next_input += 1

    graph.append(
        f"{''.join(mix)}amix=inputs={len(mix)}:duration=longest:dropout_transition=0:normalize=0,"
        # No long fade-out: the last frame must cut straight back to 0.0 s (loop). 40 ms only avoids a click.
        f"apad=whole_dur={total:.3f},atrim=0:{total:.3f},alimiter=limit=0.95,"
        f"afade=t=out:st={total - 0.04:.3f}:d=0.04[aout]"
    )
    run_ffmpeg([
        *args, "-filter_complex", ";".join(graph), "-map", "[vout]", "-map", "[aout]",
        # maxrate caps texture-heavy shots (fog, dust, rain): uncapped CRF 20 once produced 450 MB per minute
        "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-maxrate", "8M", "-bufsize", "16M",
        "-profile:v", "high", "-pix_fmt", "yuv420p",
        "-r", str(FPS), "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-t", f"{total:.3f}",
        "-movflags", "+faststart", str(out),
    ], cwd=workdir)


# --------------------------------------------------------------------------- entry point


def render_video(script: VideoScript, assets: list[SceneAsset], narration: Path, narration_duration: float,
                 words: list[Word], subs: Path, channel: ChannelConfig, workdir: Path) -> tuple[Path, float]:
    total = narration_duration + TAIL_SECONDS
    frames = plan_scene_frames(total, [len(s.narration.split()) for s in script.scenes], words)
    starts = [sum(frames[:i]) / FPS for i in range(len(frames))]
    log.info("[%s] %d scenes, %.2fs total, scene lengths %.2f-%.2fs", channel.id, len(frames), total,
             min(frames) / FPS, max(frames) / FPS)

    counts = [len(s.narration.split()) for s in script.scenes]
    # Loop: the very last shot reuses the opening image, so the end flows straight back into frame 0
    last = len(assets) - 1
    if assets[0].kind == "image" and assets[last].second is not None:
        assets[last].second, assets[last].second_fg = assets[0].path, assets[0].fg

    clips_dir = workdir / "clips"
    clips_dir.mkdir(exist_ok=True)
    jobs = []
    with ThreadPoolExecutor(max_workers=get_settings().render_threads) as pool:
        for i, (asset, scene, n) in enumerate(zip(assets, script.scenes, frames)):
            mirror = i > 0 and asset.path == assets[i - 1].path  # reused asset: flip so it reads as a new shot
            punch = i == 0 or scene.sfx == TransitionSfx.IMPACT  # the opening frame hits too
            if asset.second is None:
                jobs.append(pool.submit(_render_scene, i, asset, scene, n, channel,
                                        clips_dir / f"scene_{i:02d}.mp4", mirror, punch=punch, fg=asset.fg))
                continue
            first = split_frame(i, starts[i], n, counts, words)
            jobs.append(pool.submit(_render_scene, i, asset, scene, first, channel,
                                    clips_dir / f"scene_{i:02d}a.mp4", mirror, punch=punch, fg=asset.fg))
            jobs.append(pool.submit(_render_scene, i + 100, asset, scene, n - first, channel,
                                    clips_dir / f"scene_{i:02d}b.mp4", False, asset.second,
                                    COUNTER_MOVE[scene.camera], fg=asset.second_fg))
        clips = [job.result() for job in jobs]

    # Callouts slam in at their scene's cut; impact beats get a white flash
    callouts = [(0.0 if i == 0 else starts[i] + 0.1, min(starts[i] + 2.6, starts[i] + frames[i] / FPS - 0.1),
                 s.callout, i == 0 or s.sfx == TransitionSfx.IMPACT) for i, s in enumerate(script.scenes) if s.callout]
    flashes = [starts[i] for i, s in enumerate(script.scenes) if i > 0 and s.sfx == TransitionSfx.IMPACT]
    add_overlays(subs, channel.subtitle, callouts, flashes)
    events = _sfx_events(script.scenes, starts, assets) + [("pop", c[0], 0.35) for c in callouts if c[0] > 0]

    concat_list = workdir / "concat.txt"
    concat_list.write_text("".join(f"file '{c.as_posix()}'\n" for c in clips), encoding="utf-8")
    base = workdir / "base.mp4"
    run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy", str(base)])

    final = workdir / "final.mp4"
    _final_pass(base, narration, subs, channel, events, total, final, workdir, drops=flashes)
    duration = probe_duration(final)
    if not channel.min_duration - 0.1 <= duration <= channel.max_duration + 0.1:
        raise RenderError(f"Final duration {duration:.2f}s outside [{channel.min_duration}, {channel.max_duration}]")
    if math.isclose(duration, 0):
        raise RenderError("Empty output")
    log.info("[%s] rendered %s (%.2fs)", channel.id, final, duration)
    return final, duration
