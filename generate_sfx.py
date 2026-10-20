"""Bootstrap: synthesize SFX, background music and the dust overlay with FFmpeg lavfi (zero cost, no licensing).

Existing files are never overwritten unless --force is given, so you can drop real assets in
media_assets/ and they will be used instead.

    python generate_sfx.py [--force]
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

MEDIA = Path(os.getenv("MEDIA_DIR", Path(__file__).resolve().parent / "media_assets"))

SR = 44100

# name -> (relative path, ffmpeg args after the inputs are defined)
SOUNDS: dict[str, tuple[str, list[str]]] = {
    # Filtered pink-noise swoosh with a bandpass "sweep" via fades
    "whoosh": ("sfx/whoosh.wav", [
        "-f", "lavfi", "-i", f"anoisesrc=d=0.7:c=pink:r={SR}:a=0.9",
        "-af", "highpass=f=300,lowpass=f=6000,bandpass=f=1400:width_type=h:w=1800,"
               "afade=t=in:st=0:d=0.32:curve=exp,afade=t=out:st=0.32:d=0.38:curve=exp,"
               "aecho=0.6:0.4:40:0.25,volume=2.2,aformat=channel_layouts=stereo",
    ]),
    # Sub boom with pitch drop + noisy transient
    "impact": ("sfx/impact.wav", [
        "-f", "lavfi", "-i",
        f"aevalsrc='0.95*sin(2*PI*(38+70*exp(-9*t))*t)*exp(-4.5*t)+0.35*(random(0)*2-1)*exp(-30*t)':d=1.4:s={SR}",
        "-af", "lowpass=f=4000,aecho=0.7:0.5:60|120:0.3|0.2,alimiter=limit=0.95,aformat=channel_layouts=stereo",
    ]),
    # Short bubbly pop
    "pop": ("sfx/pop.wav", [
        "-f", "lavfi", "-i", f"aevalsrc='0.8*sin(2*PI*(1300-900*t/0.12)*t)*exp(-28*t)':d=0.22:s={SR}",
        "-af", "highpass=f=200,aformat=channel_layouts=stereo",
    ]),
    # 96 s dark drone: detuned low sines, slow tremolo, heartbeat pulse every 1.2 s, filtered noise bed
    "dark_tension": ("bgm/dark_tension.mp3", [
        "-f", "lavfi", "-i",
        "aevalsrc='0.22*sin(2*PI*55*t)+0.16*sin(2*PI*55.4*t)"
        "+0.12*sin(2*PI*82.4*t)*(0.6+0.4*sin(2*PI*0.08*t))"
        "+0.07*sin(2*PI*130.8*t+2*sin(2*PI*0.2*t))*(0.5+0.5*sin(2*PI*0.05*t))"
        "+0.45*sin(2*PI*48*t)*exp(-14*mod(t,1.2))+0.3*sin(2*PI*44*t)*exp(-16*mod(t-0.25,1.2))*gte(mod(t,1.2),0.25)"
        f"+0.04*(random(0)*2-1)':d=96:s={SR}",
        "-af", "lowpass=f=1800,aecho=0.8:0.7:500|900:0.3|0.2,afade=t=in:d=2,afade=t=out:st=93:d=3,"
               "loudnorm=I=-20,aformat=channel_layouts=stereo",
        "-c:a", "libmp3lame", "-b:a", "192k",
    ]),
    # 96 s quirky pizzicato loop: C-E-G-A arpeggio at 4 notes/s + kick on the beat + hat offbeats
    "quirky_upbeat": ("bgm/quirky_upbeat.mp3", [
        "-f", "lavfi", "-i",
        "aevalsrc='0.35*sin(2*PI*if(eq(mod(floor(t*4),8),0),523.3,if(eq(mod(floor(t*4),8),1),659.3,"
        "if(eq(mod(floor(t*4),8),2),784,if(eq(mod(floor(t*4),8),3),880,if(eq(mod(floor(t*4),8),4),784,"
        "if(eq(mod(floor(t*4),8),5),659.3,if(eq(mod(floor(t*4),8),6),587.3,493.9)))))))*t)*exp(-14*mod(t,0.25))"
        "+0.18*sin(2*PI*if(lt(mod(t,4),2),130.8,174.6)*t)*exp(-6*mod(t,0.5))"
        "+0.6*sin(2*PI*(55+90*exp(-40*mod(t,0.5)))*t)*exp(-18*mod(t,0.5))"
        f"+0.08*(random(0)*2-1)*exp(-60*mod(t+0.25,0.5))':d=96:s={SR}",
        "-af", "aecho=0.6:0.3:180:0.2,afade=t=in:d=1,afade=t=out:st=93:d=3,loudnorm=I=-20,"
               "aformat=channel_layouts=stereo",
        "-c:a", "libmp3lame", "-b:a", "192k",
    ]),
}


def ffmpeg(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def make_dust(dest: Path) -> None:
    """Seamless 40 s loop of drifting dust specks on black (for screen blending)."""
    with tempfile.TemporaryDirectory() as tmp:
        dots = Path(tmp) / "dots.png"
        ffmpeg([
            "-f", "lavfi", "-i", "color=c=black:s=1080x1920:d=1",
            "-vf", "format=gray,geq=lum='if(gt(random(1),0.99965),170+random(2)*85,0)',"
                   "gblur=sigma=1.6,eq=brightness=0.02:contrast=1.6",
            "-frames:v", "1", str(dots),
        ])
        # Stack the tile twice and scroll exactly one tile height in 40 s -> perfect loop
        ffmpeg([
            "-loop", "1", "-i", str(dots),
            "-filter_complex", "[0]split[a][b];[a][b]vstack,crop=1080:1920:0:'1920-mod(t*48,1920)',"
                               "fps=30,format=yuv420p",
            "-t", "40", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-an", str(dest),
        ])


def main(force: bool = False) -> None:
    for name, (rel, args) in SOUNDS.items():
        dest = MEDIA / rel
        if dest.exists() and not force:
            print(f"[skip] {rel}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg([*args, str(dest)])
        print(f"[ok]   {rel}")

    dust = MEDIA / "overlays" / "dust_particles.mp4"
    if dust.exists() and not force:
        print("[skip] overlays/dust_particles.mp4")
    else:
        dust.parent.mkdir(parents=True, exist_ok=True)
        make_dust(dust)
        print("[ok]   overlays/dust_particles.mp4")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
