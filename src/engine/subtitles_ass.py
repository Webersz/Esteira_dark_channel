"""faster-whisper word timestamps -> SubStation Alpha (.ass) word-by-word karaoke highlight."""
from __future__ import annotations

import difflib
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from src.config_loader import SubtitleConfig, get_settings

log = logging.getLogger(__name__)

PLAY_RES_X, PLAY_RES_Y = 1080, 1920

_model = None


@dataclass
class Word:
    start: float
    end: float
    text: str


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel  # heavy import, keep lazy

        settings = get_settings()
        _model = WhisperModel(settings.whisper_model, device=settings.whisper_device, compute_type="int8")
    return _model


def transcribe_words(audio_path: Path, language: str) -> list[Word]:
    segments, _info = _get_model().transcribe(
        str(audio_path),
        language=language.split("-")[0],
        word_timestamps=True,
        beam_size=5,
        vad_filter=False,
        condition_on_previous_text=False,
        # No initial_prompt: passing the script made Whisper skip the first ~27 s of speech, and
        # align_to_script already restores the script's spelling.
    )
    words: list[Word] = []
    for segment in segments:
        for w in segment.words or []:
            text = w.word.strip()
            if text:
                words.append(Word(start=float(w.start), end=float(w.end), text=text))
    if not words:
        raise RuntimeError("Whisper returned no words")
    # Enforce monotonic, non-overlapping timings
    for prev, cur in zip(words, words[1:]):
        if cur.start < prev.end:
            cur.start = prev.end
        if cur.end <= cur.start:
            cur.end = cur.start + 0.05
    return words


def _norm(token: str) -> str:
    token = unicodedata.normalize("NFKD", token).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", token)


def align_to_script(words: list[Word], script_text: str) -> list[Word]:
    """Keep Whisper's timings but the script's spelling (Whisper mishears names, accents, 'Olhe' -> 'Oi').

    Tokens are matched with difflib; mismatched stretches get the script words spread evenly over
    the time span Whisper heard there.
    """
    tokens = script_text.split()
    if not tokens:
        return words
    matcher = difflib.SequenceMatcher(None, [_norm(w.text) for w in words], [_norm(t) for t in tokens], autojunk=False)
    aligned: list[Word] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            aligned += [Word(words[i1 + k].start, words[i1 + k].end, tokens[j1 + k]) for k in range(i2 - i1)]
            continue
        if j1 == j2:  # Whisper hallucinated extra words: drop them
            continue
        if i1 < i2:
            start, end = words[i1].start, words[i2 - 1].end
        else:  # Whisper missed words: squeeze them into the gap between neighbours
            start = words[i1 - 1].end if i1 > 0 else 0.0
            end = words[i1].start if i1 < len(words) else start + 0.3 * (j2 - j1)
            end = max(end, start + 0.12 * (j2 - j1))
        step = (end - start) / (j2 - j1)
        aligned += [Word(start + k * step, start + (k + 1) * step, tokens[j1 + k]) for k in range(j2 - j1)]
    for prev, cur in zip(aligned, aligned[1:]):  # keep timings monotonic
        if cur.start < prev.end:
            cur.start = prev.end
        if cur.end <= cur.start:
            cur.end = cur.start + 0.05
    matched = sum(b.size for b in matcher.get_matching_blocks())
    log.info("subtitle alignment: %d/%d script words matched Whisper exactly", matched, len(tokens))
    return aligned


# --------------------------------------------------------------------------- ASS writer


def _ass_color(hex_color: str, alpha: int = 0) -> str:
    h = hex_color.lstrip("#")
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H{alpha:02X}{b}{g}{r}".upper()


def _ts(seconds: float) -> str:
    cs = max(0, int(round(seconds * 100)))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _clean(text: str, uppercase: bool) -> str:
    text = re.sub(r"[.,;:\"“”«»]+", "", text)  # keep ? and ! for emotion
    text = text.replace("\\", "").replace("{", "").replace("}", "")
    return text.upper() if uppercase else text


def _chunk(words: list[Word], max_words: int, max_chars: int = 18, max_gap: float = 0.55) -> list[list[Word]]:
    chunks: list[list[Word]] = []
    current: list[Word] = []
    for word in words:
        chars = sum(len(w.text) + 1 for w in current) + len(word.text)
        gap = word.start - current[-1].end if current else 0.0
        if current and (len(current) >= max_words or chars > max_chars or gap > max_gap):
            chunks.append(current)
            current = []
        current.append(word)
        if re.search(r"[.!?]$", word.text):  # sentence end closes the line
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)
    return chunks


def build_ass(words: list[Word], style: SubtitleConfig, out_path: Path, video_duration: float) -> Path:
    primary = _ass_color(style.primary_color)
    highlight = _ass_color(style.highlight_color)
    outline = _ass_color(style.outline_color)
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {PLAY_RES_X}
PlayResY: {PLAY_RES_Y}
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Karaoke,{style.font},{style.font_size},{primary},{highlight},{outline},&H96000000,-1,0,0,0,100,100,1,0,1,{style.outline},{style.shadow},2,60,60,{style.margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines: list[str] = []
    chunks = _chunk(words, style.words_per_line)
    for ci, chunk in enumerate(chunks):
        next_start = chunks[ci + 1][0].start if ci + 1 < len(chunks) else video_duration
        chunk_end = min(chunk[-1].end + 0.35, next_start)
        tokens = [_clean(w.text, style.uppercase) for w in chunk]
        for wi, word in enumerate(chunk):
            start = word.start if wi else chunk[0].start
            end = chunk[wi + 1].start if wi + 1 < len(chunk) else chunk_end
            if end - start < 0.02:
                continue
            parts = []
            for ti, token in enumerate(tokens):
                if ti == wi:
                    # Active word: highlight colour + pop (125% -> 110% in 90ms)
                    parts.append(
                        f"{{\\c{highlight}\\fscx125\\fscy125\\t(0,90,\\fscx110\\fscy110)}}{token}{{\\r}}"
                    )
                else:
                    parts.append(token)
            intro = "{\\fad(40,0)}" if wi == 0 else ""
            lines.append(f"Dialogue: 0,{_ts(start)},{_ts(end)},Karaoke,,0,0,0,,{intro}{' '.join(parts)}")
    out_path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    log.info("ASS written: %d chunks, %d events", len(chunks), len(lines))
    return out_path


CALLOUT_MAX_SIZE = 112
CALLOUT_MAX_WIDTH = 900  # px of the 1080 frame
CALLOUT_CHAR_EM = 0.70  # average uppercase advance of heavy sans fonts, in em
CALLOUT_BOX = "&H33000000"  # ~80% opaque black band behind the text
CALLOUT_ALERT_BOX = "&H1A0E0EC8"  # ~90% opaque deep red for impact beats


def add_overlays(ass_path: Path, style: SubtitleConfig, callouts: list[tuple[float, float, str, bool]],
                 flashes: list[float]) -> None:
    """Append top-of-frame headline callouts and full-frame white flashes to an existing .ass file.

    Callouts sit on an opaque band (BorderStyle 3) so they stay legible on bright skies, shrink to fit
    the frame width, and the band turns red on impact beats (`alert`).
    """
    styles = (
        f"Style: Callout,{style.font},{CALLOUT_MAX_SIZE},&H00FFFFFF,&H00FFFFFF,{CALLOUT_BOX},&H00000000,"
        "-1,0,0,0,100,100,1,0,3,12,0,8,60,60,300,1\n"
        "Style: Flash,Arial,10,&H00FFFFFF,&H00FFFFFF,&H00FFFFFF,&H00FFFFFF,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1\n"
    )
    events = []
    for start, end, text, alert in callouts:
        if end - start < 0.3:
            continue
        lines = [_clean(text, True)]
        if len(lines[0]) > 12 and " " in lines[0]:  # two big lines beat one tiny line: break near the middle
            words = lines[0].split()
            cut = min(range(1, len(words)), key=lambda k: abs(len(" ".join(words[:k])) - len(" ".join(words[k:]))))
            lines = [" ".join(words[:cut]), " ".join(words[cut:])]
        longest = max(len(line) for line in lines)
        size = min(CALLOUT_MAX_SIZE, int(CALLOUT_MAX_WIDTH / (CALLOUT_CHAR_EM * max(longest, 1))))
        label = "\\N".join(lines)
        box = f"\\3c&H{CALLOUT_ALERT_BOX[4:]}&\\3a&H{CALLOUT_ALERT_BOX[2:4]}&" if alert else ""
        pop = f"{{\\fs{size}{box}\\fad(80,200)\\fscx135\\fscy135\\t(0,160,\\fscx100\\fscy100)}}"  # slams in, settles
        events.append(f"Dialogue: 2,{_ts(start)},{_ts(end)},Callout,,0,0,0,,{pop}{label}")
    for t in flashes:
        events.append(f"Dialogue: 3,{_ts(t)},{_ts(t + 0.35)},Flash,,0,0,0,,"
                      "{\\an7\\pos(0,0)\\fad(0,300)\\p1}m 0 0 l 1080 0 1080 1920 0 1920{\\p0}")
    text = ass_path.read_text(encoding="utf-8").replace("\n[Events]", styles + "\n[Events]", 1)
    ass_path.write_text(text + "\n".join(events) + "\n", encoding="utf-8")
