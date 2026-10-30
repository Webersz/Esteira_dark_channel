"""Async parallel asset fetching: Pollinations Flux images, optional Pexels B-roll, rembg 2.5D sprite cutouts."""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import json
import logging
import random
import subprocess
import types
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import aiohttp
from PIL import Image, ImageChops, ImageFilter

from src.config_loader import ChannelConfig, get_settings
from src.engine import mascot
from src.engine.schemas import VideoScript

log = logging.getLogger(__name__)

POLLINATIONS_URL = "https://image.pollinations.ai/prompt/{prompt}?width={w}&height={h}&model={model}&seed={seed}&nologo=true"
# Pollen keys (sk_...) from enter.pollinations.ai only lift the 45 s limit on the gen endpoint;
# image.pollinations.ai ignores them. That endpoint has no "turbo": tier 2 uses Z-Image Turbo.
POLLINATIONS_KEY_URL = "https://gen.pollinations.ai/image/{prompt}?width={w}&height={h}&model={model}&seed={seed}&nologo=true"
PEXELS_SEARCH = "https://api.pexels.com/videos/search"
NEGATIVE_HINT = "no text, no watermark, no letters, no logo"

_rembg_session = None
_rembg_pool = ThreadPoolExecutor(max_workers=1)  # rembg is CPU-bound; serialize to keep RAM stable


@dataclass
class SceneAsset:
    index: int
    kind: str  # "image" | "video"
    path: Path
    sprite: Path | None = None  # transparent PNG for 2.5D compositing
    procedural: bool = False  # tier-3 local fallback was used
    second: Path | None = None  # image for the second half of the scene (two shots per scene)


# --------------------------------------------------------------------------- download helpers


async def _fetch_bytes(session: aiohttp.ClientSession, url: str, sem: asyncio.Semaphore, *, headers=None,
                       retries: int = 4, timeout: int = 180) -> bytes | None:
    for attempt in range(retries):
        async with sem:
            try:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                    if resp.status == 200:
                        return await resp.read()
                    log.warning("GET %s -> HTTP %s", url[:90], resp.status)
                    if resp.status in (400, 401, 403, 404):
                        return None
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                log.warning("GET %s failed (%s)", url[:90], type(exc).__name__)
        if attempt < retries - 1:
            await asyncio.sleep(4 * (2**attempt) + random.random() * 2)
    return None


def _valid_image(data: bytes | None, min_side: int = 512) -> Image.Image | None:
    if not data:
        return None
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img.convert("RGB") if min(img.size) >= min_side else None
    except Exception:
        return None


TIER1_BACKOFF = (2, 5, 10)  # seconds after Flux attempts 1, 2, 3
TIER2_ATTEMPTS = 2  # Turbo attempts 4 and 5
# Anonymous Pollinations tier: 1 image / 45 s per IP (HTTP 402 FREE_LIMIT otherwise). A pollen key removes it.
ANON_INTERVAL_S = 46.0

_throttle_lock: asyncio.Lock | None = None
_last_request = 0.0


@contextlib.asynccontextmanager
async def _anon_slot():
    """Serialize anonymous Pollinations requests and space them ANON_INTERVAL_S after the previous one
    *finished* (the free-tier window is counted from generation end). No-op with POLLINATIONS_TOKEN."""
    global _throttle_lock, _last_request
    slot = types.SimpleNamespace(penalize=False)  # caller sets True on failure -> one extra window of rest
    if get_settings().pollinations_token:
        yield slot
        return
    if _throttle_lock is None:
        _throttle_lock = asyncio.Lock()
    async with _throttle_lock:
        loop = asyncio.get_running_loop()
        wait = _last_request + ANON_INTERVAL_S - loop.time()
        if wait > 0:
            await asyncio.sleep(wait)
        try:
            yield slot
        finally:
            _last_request = loop.time() + (ANON_INTERVAL_S if slot.penalize else 0.0)


async def _pollinations_image(session, sem, prompt: str, seed: int, dest: Path, w: int = 1080, h: int = 1920) -> bool:
    """Tier 1: Flux x3 (backoff 2/5/10 s). Tier 2: Turbo/Z-Image x2 with a mutated seed. False if both tiers fail."""
    settings = get_settings()
    headers = {"Authorization": f"Bearer {settings.pollinations_token}"} if settings.pollinations_token else None
    encoded = quote(prompt[:900], safe="")
    url_template, tier2 = (POLLINATIONS_KEY_URL, "zimage") if settings.pollinations_token else (POLLINATIONS_URL, "turbo")
    plan = [("flux", seed, wait) for wait in TIER1_BACKOFF]
    plan += [(tier2, seed + random.randint(1, 9999), 0) for _ in range(TIER2_ATTEMPTS)]
    for attempt, (model, current_seed, wait) in enumerate(plan, start=1):
        url = url_template.format(prompt=encoded, w=w, h=h, seed=current_seed, model=model)
        async with _anon_slot() as slot:
            img = _valid_image(await _fetch_bytes(session, url, sem, headers=headers, retries=1))
            # After a timeout the server keeps generating and counts it against the free window
            slot.penalize = img is None
        if img is not None:
            img.save(dest, "JPEG", quality=94)
            if attempt > 1:
                log.info("image %s OK on attempt %d (%s)", dest.name, attempt, model)
            return True
        if wait:
            await asyncio.sleep(wait)
    return False


def _procedural_image_sync(dest: Path, seed: int) -> None:
    """Tier 3: dark textured 1080x1920 frame made locally with FFmpeg lavfi (never fails, no network)."""
    rng = random.Random(seed)
    cx, cy = rng.randint(250, 830), rng.randint(500, 1300)
    r, g, b = rng.choice([(70, 20, 30), (20, 40, 80), (60, 40, 10), (25, 60, 50), (50, 25, 70)])
    glow = f"exp(-((X-{cx})*(X-{cx})+(Y-{cy})*(Y-{cy}))/260000)"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=0x0d1117:s=1080x1920:d=1",
         "-vf", f"format=rgb24,geq=r='13+{r}*{glow}':g='17+{g}*{glow}':b='23+{b}*{glow}',"
                "noise=alls=20:allf=t+u,vignette=PI/4",
         "-frames:v", "1", "-q:v", "2", str(dest)],
        check=True, capture_output=True,
    )


async def _pexels_clip(session, sem, query: str, dest: Path, used_ids: set[int], rng: random.Random) -> bool:
    settings = get_settings()
    if not settings.pexels_api_key or not query:
        return False
    params = f"?query={quote(query)}&orientation=portrait&size=medium&per_page=15"
    data = await _fetch_bytes(session, PEXELS_SEARCH + params, sem, headers={"Authorization": settings.pexels_api_key},
                              retries=2, timeout=30)
    if not data:
        return False
    videos = [v for v in json.loads(data).get("videos", []) if v.get("duration", 0) >= 4 and v["id"] not in used_ids]
    rng.shuffle(videos)
    for video in videos[:4]:
        files = [f for f in video.get("video_files", [])
                 if f.get("height") and f.get("width") and f["height"] > f["width"] and f["height"] >= 1280]
        if not files:
            continue
        best = min(files, key=lambda f: f["height"])  # smallest file that is still >= 1280p
        clip = await _fetch_bytes(session, best["link"], sem, retries=2, timeout=180)
        if clip and len(clip) > 100_000:
            dest.write_bytes(clip)
            used_ids.add(video["id"])
            return True
    return False


# --------------------------------------------------------------------------- rembg


def _cutout_sync(src: Path, dest: Path) -> bool:
    global _rembg_session
    from rembg import new_session, remove  # heavy import, keep lazy

    if _rembg_session is None:
        _rembg_session = new_session("isnet-general-use")
    img = Image.open(src).convert("RGB")
    cut = remove(img, session=_rembg_session, post_process_mask=True)
    coverage = sum(1 for a in cut.getchannel("A").getdata() if a > 20) / (img.width * img.height)
    if not 0.04 <= coverage <= 0.85:  # empty or failed segmentation
        return False
    return save_clean_cutout(cut, dest)


def save_clean_cutout(rgba: Image.Image, dest: Path, pad: int = 8) -> bool:
    """Drop stray specks left by segmentation, then crop tight to the character and save as PNG."""
    alpha = rgba.getchannel("A")
    mask = alpha.point(lambda a: 255 if a > 20 else 0)
    # Opening erases islands < ~11 px; the generous dilation then keeps thin parts attached to the body
    body = mask.filter(ImageFilter.MinFilter(11)).filter(ImageFilter.MaxFilter(11)).filter(ImageFilter.MaxFilter(31))
    alpha = ImageChops.multiply(alpha, body)
    rgba = rgba.copy()
    rgba.putalpha(alpha)
    bbox = alpha.point(lambda a: 255 if a > 20 else 0).getbbox()
    if not bbox:
        return False
    bbox = (max(0, bbox[0] - pad), max(0, bbox[1] - pad), min(rgba.width, bbox[2] + pad), min(rgba.height, bbox[3] + pad))
    rgba.crop(bbox).save(dest, "PNG")
    return True


async def _cutout(src: Path, dest: Path) -> bool:
    try:
        return await asyncio.get_running_loop().run_in_executor(_rembg_pool, _cutout_sync, src, dest)
    except Exception as exc:
        log.warning("rembg failed for %s: %s", src.name, exc)
        return False


# --------------------------------------------------------------------------- orchestration


async def _fetch_scene(i, scene, channel: ChannelConfig, assets_dir: Path, session, sem, base_seed: int,
                       used_ids: set[int], rng: random.Random, sprites: dict[str, Path]) -> SceneAsset:
    use_sprite = bool(sprites) and bool(scene.character_pose)

    # 1) Optional B-roll for hybrid channels
    if channel.use_broll and not use_sprite and i % max(1, channel.broll_every) == channel.broll_every - 1:
        clip = assets_dir / f"scene_{i:02d}.mp4"
        if await _pexels_clip(session, sem, scene.broll_query, clip, used_ids, rng):
            return SceneAsset(i, "video", clip)

    # 2) Background / main image
    bg_prompt = scene.image_prompt
    if use_sprite:
        bg_prompt += ", empty environment background plate, no characters, no people"
    bg_prompt = f"{bg_prompt}, {channel.visual_style}, vertical 9:16 composition, {NEGATIVE_HINT}"
    bg = assets_dir / f"scene_{i:02d}.jpg"

    procedural = not await _pollinations_image(session, sem, bg_prompt, base_seed + i, bg)
    if procedural:
        log.warning("[%s] scene %d: Pollinations tiers 1+2 failed -> tier 3 procedural frame", channel.id, i)
        await asyncio.to_thread(_procedural_image_sync, bg, base_seed + i)

    # 3) Second shot for the other half of the line. Skipped on the anonymous tier (it would double the
    #    46 s/image pacing) and for mascot scenes, whose sprite already carries the motion.
    second = None
    if scene.image_prompt_b and not use_sprite and get_settings().pollinations_token:
        b_prompt = f"{scene.image_prompt_b}, {channel.visual_style}, vertical 9:16 composition, {NEGATIVE_HINT}"
        b_path = assets_dir / f"scene_{i:02d}_b.jpg"
        if await _pollinations_image(session, sem, b_prompt, base_seed + 500 + i, b_path):
            second = b_path

    # 4) Mascot from the channel's fixed library (same character in every video)
    sprite = mascot.pick_sprite(sprites, scene.character_pose, i) if use_sprite else None
    return SceneAsset(i, "image", bg, sprite, procedural, second)


# --------------------------------------------------------------------------- mascot library


async def ensure_mascot_library(channel: ChannelConfig, session, sem, force: bool = False) -> None:
    """Generate media_assets/characters/<channel>/<pose>.png once (skipped if any PNG already exists)."""
    folder = mascot.library_dir(channel)
    if mascot.source_pngs(channel) and not force:
        return
    raw_dir = folder / "_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    log.info("[%s] building mascot library (%d poses) in %s", channel.id, len(mascot.POSES), folder)
    for pose, description in mascot.POSES.items():
        prompt = (
            f"{channel.character_description}, {description}, full body, centered, single character, "
            f"isolated on plain pure white background, soft studio lighting, 3D render, {NEGATIVE_HINT}"
        )
        raw = raw_dir / f"{pose}.jpg"
        # Same seed + same description for every pose maximizes resemblance between poses
        if await _pollinations_image(session, sem, prompt, channel.character_seed, raw, 1024, 1536) \
                and await _cutout(raw, folder / f"{pose}.png"):
            log.info("[%s] mascot pose '%s' ready", channel.id, pose)
        else:
            log.warning("[%s] mascot pose '%s' failed, skipped", channel.id, pose)
    sprites = await asyncio.to_thread(mascot.prepared_sprites, channel)
    preview = mascot.save_preview(channel, sprites)
    log.info("[%s] mascot library: %d poses, preview %s", channel.id, len(sprites), preview)


async def build_mascot_library(channel: ChannelConfig, force: bool = False) -> None:
    """Standalone entry point (CLI)."""
    global _throttle_lock
    _throttle_lock = None
    async with aiohttp.ClientSession(headers={"User-Agent": "dark-factory-os/1.0"}) as session:
        await ensure_mascot_library(channel, session, asyncio.Semaphore(1), force=force)


async def fetch_scene_assets(script: VideoScript, channel: ChannelConfig, workdir: Path) -> list[SceneAsset]:
    assets_dir = workdir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    global _throttle_lock
    _throttle_lock = None  # asyncio.Lock is bound to the event loop; each asyncio.run() gets a fresh one
    base_seed = int(hashlib.md5(script.topic.encode()).hexdigest()[:7], 16)
    if not get_settings().pollinations_token:
        n = len(script.scenes)
        log.info("[%s] no POLLINATIONS_TOKEN: anonymous tier paced at 1 image/%.0fs (~%.0f min for %d images)",
                 channel.id, ANON_INTERVAL_S, n * ANON_INTERVAL_S / 60, n)
    rng = random.Random(base_seed)
    sem = asyncio.Semaphore(get_settings().image_concurrency)
    used_ids: set[int] = set()
    connector = aiohttp.TCPConnector(limit=8)
    async with aiohttp.ClientSession(connector=connector, headers={"User-Agent": "dark-factory-os/1.0"}) as session:
        sprites: dict[str, Path] = {}
        if channel.is_character_channel:
            await ensure_mascot_library(channel, session, sem)  # one-time, ~5 min on the free tier
            sprites = await asyncio.to_thread(mascot.prepared_sprites, channel)
            if not sprites:
                log.error("[%s] mascot library is empty: rendering without the character", channel.id)
        assets = await asyncio.gather(*(
            _fetch_scene(i, scene, channel, assets_dir, session, sem, base_seed, used_ids, rng, sprites)
            for i, scene in enumerate(script.scenes)
        ))

    procedural = sum(a.procedural for a in assets)
    log.info("[%s] assets ready: %d images, %d clips, %d sprites, %d procedural", channel.id,
             sum(a.kind == "image" for a in assets) + sum(a.second is not None for a in assets),
             sum(a.kind == "video" for a in assets),
             sum(a.sprite is not None for a in assets), procedural)
    return list(assets)
