"""Fixed mascot library per channel: media_assets/characters/<channel_id>/<pose>.png

The library is generated once (`python main_cli.py --init-mascots`, or lazily on the first render) and
reused by every video, so the mascot looks the same across the whole channel. Replace or add PNGs freely:
- any `*.png` works (names starting with `_` are ignored); name it after a pose key to be picked for it
- opaque images get their background removed with rembg; everything is cropped to the character
Prepared copies are cached in `_cache/` and rebuilt when the source file changes.
"""
from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from src.config_loader import ChannelConfig, get_settings

log = logging.getLogger(__name__)

POSES: dict[str, str] = {
    "neutral": "standing relaxed, friendly smile, looking at the camera",
    "pointing": "pointing up with the index finger, excited, mouth open",
    "shocked": "shocked expression, both hands on cheeks, wide eyes",
    "thinking": "thinking pose, hand on chin, curious look",
    "explaining": "explaining with both hands open, confident smile",
    "laughing": "laughing hard, eyes closed, holding belly",
}

# Free-text fallback when the script's pose isn't an exact library key (EN + PT-BR stems)
_KEYWORDS: dict[str, tuple[str, ...]] = {
    "pointing": ("point", "apont", "indic"),
    "shocked": ("shock", "surpr", "choc", "scared", "assust", "gasp", "wide eye", "espant"),
    "thinking": ("think", "pens", "chin", "queixo", "curious", "curios", "wonder", "duvid"),
    "explaining": ("explain", "expli", "hands open", "present", "gestur", "mostr"),
    "laughing": ("laugh", "rindo", "risad", "giggl", "funny", "engra"),
    "neutral": ("neutral", "smil", "sorri", "calm", "relax"),
}


def library_dir(channel: ChannelConfig) -> Path:
    return get_settings().media_dir / "characters" / channel.id


def source_pngs(channel: ChannelConfig) -> list[Path]:
    folder = library_dir(channel)
    return sorted(p for p in folder.glob("*.png") if not p.name.startswith("_")) if folder.is_dir() else []


def pose_names(channel: ChannelConfig) -> list[str]:
    """Pose keys the script writer may use: the library's file names, or the default set if not built yet."""
    return [p.stem for p in source_pngs(channel)] or list(POSES)


def _prepare(src: Path, dest: Path) -> bool:
    from src.engine.media_fetcher import _cutout_sync, save_clean_cutout  # lazy: media_fetcher imports us

    with Image.open(src) as img:
        rgba = img.convert("RGBA")
    if rgba.getchannel("A").getextrema()[0] == 255:  # fully opaque -> needs background removal
        return _cutout_sync(src, dest)
    return save_clean_cutout(rgba, dest)


def prepared_sprites(channel: ChannelConfig) -> dict[str, Path]:
    """{pose: ready-to-composite PNG} for the channel (blocking; may run rembg on new user files)."""
    cache = library_dir(channel) / "_cache"
    sprites: dict[str, Path] = {}
    for src in source_pngs(channel):
        dest = cache / src.name
        if not dest.exists() or dest.stat().st_mtime < src.stat().st_mtime:
            cache.mkdir(parents=True, exist_ok=True)
            if not _prepare(src, dest):
                log.warning("[%s] mascot file %s unusable (no character found), skipped", channel.id, src.name)
                continue
        sprites[src.stem] = dest
    return sprites


def pick_sprite(sprites: dict[str, Path], pose: str, index: int) -> Path:
    """Exact pose key -> key mentioned in the text -> keyword match -> rotate through the library."""
    text = pose.strip().lower()
    if text in sprites:
        return sprites[text]
    for key, path in sprites.items():
        if key.lower() in text:
            return path
    for key, stems in _KEYWORDS.items():
        if key in sprites and any(stem in text for stem in stems):
            return sprites[key]
    ordered = list(sprites.values())
    return ordered[index % len(ordered)]


def save_preview(channel: ChannelConfig, sprites: dict[str, Path]) -> Path | None:
    """Contact sheet `_preview.png` to review the generated library at a glance."""
    if not sprites:
        return None
    thumbs = []
    for path in sprites.values():
        with Image.open(path) as img:
            img = img.convert("RGBA")
            img.thumbnail((300, 450))
            thumbs.append(img.copy())
    sheet = Image.new("RGBA", (310 * len(thumbs), 460), (235, 235, 235, 255))
    for i, thumb in enumerate(thumbs):
        sheet.alpha_composite(thumb, (310 * i + (310 - thumb.width) // 2, 460 - thumb.height - 5))
    out = library_dir(channel) / "_preview.png"
    sheet.convert("RGB").save(out)
    return out
