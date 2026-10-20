"""edge-tts voice-over + FFmpeg pitch/tempo shaping that lands the narration inside the monetization window."""
from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from pathlib import Path

import edge_tts

from src.config_loader import ChannelConfig

log = logging.getLogger(__name__)

TAIL_SECONDS = 0.6  # silence kept after the last word (video = narration + tail)
MIN_TEMPO, MAX_TEMPO = 0.82, 1.30  # beyond this the voice sounds broken -> regenerate the script


class AudioLengthError(Exception):
    """Narration can't be stretched into the duration window; the script must be regenerated."""

    def __init__(self, raw_duration: float, needed_tempo: float):
        self.raw_duration = raw_duration
        self.needed_tempo = needed_tempo
        super().__init__(f"raw narration {raw_duration:.1f}s would need tempo {needed_tempo:.2f}")


def probe_audio(path: Path) -> tuple[float, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate:format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(out.stdout)
    return float(data["format"]["duration"]), int(data["streams"][0]["sample_rate"])


def _atempo_chain(factor: float) -> list[str]:
    """atempo accepts 0.5..2.0 per instance; chain for anything outside."""
    filters = []
    while factor > 2.0:
        filters.append("atempo=2.0")
        factor /= 2.0
    while factor < 0.5:
        filters.append("atempo=0.5")
        factor /= 0.5
    filters.append(f"atempo={factor:.5f}")
    return filters


async def _edge_tts(text: str, voice: str, rate: str, out_path: Path, retries: int = 4) -> None:
    for attempt in range(retries):
        try:
            await edge_tts.Communicate(text, voice, rate=rate).save(str(out_path))
            if out_path.exists() and out_path.stat().st_size > 1000:
                return
        except Exception as exc:  # edge-tts raises a variety of aiohttp/websocket errors
            log.warning("edge-tts attempt %d failed: %s", attempt + 1, exc)
        await asyncio.sleep(3 * (attempt + 1))
    raise RuntimeError("edge-tts failed after retries")


def synthesize(text: str, channel: ChannelConfig, workdir: Path) -> tuple[Path, float]:
    """Return (narration.wav, duration). Duration + TAIL_SECONDS is guaranteed inside [min, max]."""
    voice = channel.voice
    raw_mp3 = workdir / "narration_raw.mp3"
    out_wav = workdir / "narration.wav"
    asyncio.run(_edge_tts(text, voice.voice, voice.rate, raw_mp3))
    raw_duration, sample_rate = probe_audio(raw_mp3)

    lo = channel.min_duration - TAIL_SECONDS + 0.3
    hi = channel.max_duration - TAIL_SECONDS - 0.3
    tempo = voice.tempo
    if not lo <= raw_duration / tempo <= hi:
        tempo = raw_duration / (channel.target_duration - TAIL_SECONDS)
    if not MIN_TEMPO <= tempo <= MAX_TEMPO:
        raise AudioLengthError(raw_duration, tempo)

    # asetrate shifts pitch AND speed by p; atempo then compensates so the net speed == tempo.
    p = 2 ** (voice.pitch_semitones / 12.0)
    filters = [
        "silenceremove=start_periods=1:start_threshold=-50dB:start_silence=0.05",
        f"asetrate={round(sample_rate * p)}",
        f"aresample={sample_rate}",
        *_atempo_chain(tempo / p),
        "highpass=f=70",
        "acompressor=threshold=-18dB:ratio=3:attack=5:release=120:makeup=2",
        "loudnorm=I=-16:TP=-1.5:LRA=11",
        "aresample=48000",
    ]
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw_mp3), "-af", ",".join(filters),
         "-ac", "2", "-c:a", "pcm_s16le", str(out_wav)],
        check=True, capture_output=True, text=True,
    )
    duration, _ = probe_audio(out_wav)
    log.info("[%s] TTS raw=%.1fs tempo=%.3f pitch=%+.1fst -> %.1fs", channel.id, raw_duration, tempo,
             voice.pitch_semitones, duration)
    if not channel.min_duration <= duration + TAIL_SECONDS <= channel.max_duration:
        raise AudioLengthError(raw_duration, tempo)
    return out_wav, duration
