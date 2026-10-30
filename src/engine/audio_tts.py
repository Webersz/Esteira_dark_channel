"""Voice-over (Gemini TTS, edge-tts fallback) + FFmpeg pitch/tempo shaping that lands the narration inside
the monetization window."""
from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import wave
from pathlib import Path

import edge_tts
from google.genai import errors as genai_errors
from google.genai import types

from src.config_loader import ChannelConfig, get_settings

log = logging.getLogger(__name__)

TAIL_SECONDS = 0.15  # tiny gap after the last word: scene 16 must flow straight back into scene 1 (loop)
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


GEMINI_TTS_MODELS = ["gemini-3.8-flash-tts", "gemini-3.1-flash-tts-preview", "gemini-2.5-flash-preview-tts"]


def _gemini_tts(text: str, voice: str, direction: str, out_wav: Path) -> None:
    """Gemini TTS over models x keys. The direction goes in a structured prompt: these models reject
    system instructions, and a plain "Read like...: <text>" prefix gets spoken aloud."""
    from src.engine.script_generator import _get_client  # shared per-key client cache

    prompt = f"# AUDIO PROFILE\n{direction}\n\n# TRANSCRIPT (read only this)\n{text}"
    config = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice))
        ),
    )
    failures = []
    for model in GEMINI_TTS_MODELS:
        for idx, key in enumerate(get_settings().gemini_api_keys, start=1):
            try:
                response = _get_client(key).models.generate_content(model=model, contents=prompt, config=config)
                pcm = response.candidates[0].content.parts[0].inline_data.data
            except genai_errors.APIError as exc:
                failures.append(f"{model} key#{idx}: HTTP {exc.code}")
                if exc.code in (400, 403, 404):
                    break  # model unusable: skip its other keys
                continue
            except (AttributeError, IndexError, TypeError):
                failures.append(f"{model} key#{idx}: no audio")
                continue
            with wave.open(str(out_wav), "wb") as wav:  # 24 kHz mono 16-bit PCM
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(24000)
                wav.writeframes(pcm)
            log.info("Narration via %s (key #%d, voice %s)", model, idx, voice)
            return
    raise RuntimeError(f"Gemini TTS failed: {'; '.join(failures[-6:])}")


def _voice_over(text: str, channel: ChannelConfig, workdir: Path) -> Path:
    voice = channel.voice
    if voice.engine == "gemini" and get_settings().gemini_api_keys:
        raw_wav = workdir / "narration_raw.wav"
        try:
            _gemini_tts(text, voice.gemini_voice, voice.direction, raw_wav)
            return raw_wav
        except RuntimeError as exc:
            log.warning("[%s] %s -> falling back to edge-tts", channel.id, exc)
    raw_mp3 = workdir / "narration_raw.mp3"
    asyncio.run(_edge_tts(text, voice.voice, voice.rate, raw_mp3))
    return raw_mp3


def synthesize(text: str, channel: ChannelConfig, workdir: Path) -> tuple[Path, float]:
    """Return (narration.wav, duration). Duration + TAIL_SECONDS is guaranteed inside [min, max]."""
    voice = channel.voice
    out_wav = workdir / "narration.wav"
    raw_audio = _voice_over(text, channel, workdir)
    raw_duration, sample_rate = probe_audio(raw_audio)

    # Margins absorb the leading/trailing silence trim applied below (measured on the raw file here)
    lo = channel.min_duration - TAIL_SECONDS + 0.8
    hi = channel.max_duration - TAIL_SECONDS - 0.8
    tempo = voice.tempo
    if not lo <= raw_duration / tempo <= hi:
        tempo = raw_duration / (channel.target_duration - TAIL_SECONDS)
    if not MIN_TEMPO <= tempo <= MAX_TEMPO:
        raise AudioLengthError(raw_duration, tempo)

    # asetrate shifts pitch AND speed by p; atempo then compensates so the net speed == tempo.
    p = 2 ** (voice.pitch_semitones / 12.0)
    filters = [
        "silenceremove=start_periods=1:start_threshold=-50dB:start_silence=0.05",
        # Trailing-silence trim for the seamless loop. Done as reverse -> trim start -> reverse because
        # `silenceremove=stop_periods=-1` would also delete every natural pause in the middle of the speech.
        "areverse,silenceremove=start_periods=1:start_duration=0.1:start_threshold=-45dB,areverse",
        f"asetrate={round(sample_rate * p)}",
        f"aresample={sample_rate}",
        *_atempo_chain(tempo / p),
        "highpass=f=70",
        "acompressor=threshold=-18dB:ratio=3:attack=5:release=120:makeup=2",
        "loudnorm=I=-16:TP=-1.5:LRA=11",
        "aresample=48000",
    ]
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw_audio), "-af", ",".join(filters),
         "-ac", "2", "-c:a", "pcm_s16le", str(out_wav)],
        check=True, capture_output=True, text=True,
    )
    duration, _ = probe_audio(out_wav)
    log.info("[%s] TTS raw=%.1fs tempo=%.3f pitch=%+.1fst -> %.1fs", channel.id, raw_duration, tempo,
             voice.pitch_semitones, duration)
    if not channel.min_duration <= duration + TAIL_SECONDS <= channel.max_duration:
        raise AudioLengthError(raw_duration, tempo)
    return out_wav, duration
