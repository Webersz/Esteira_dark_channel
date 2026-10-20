"""Async parallel asset fetching: Pollinations Flux images, optional Pexels B-roll, rembg 2.5D sprite cutouts."""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import aiohttp
from PIL import Image

from src.config_loader import ChannelConfig, get_settings
from src.engine.schemas import VideoScript

log = logging.getLogger(__name__)

POLLINATIONS_URL = "https://image.pollinations.ai/prompt/{prompt}?width={w}&height={h}&model=flux&seed={seed}&nologo=true"
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


async def _pollinations_image(session, sem, prompt: str, seed: int, dest: Path, w: int = 1080, h: int = 1920) -> bool:
    settings = get_settings()
    headers = {"Authorization": f"Bearer {settings.pollinations_token}"} if settings.pollinations_token else None
    for variant, current_prompt in enumerate((prompt, prompt.split(",")[0] + f", {NEGATIVE_HINT}")):
        url = POLLINATIONS_URL.format(prompt=quote(current_prompt[:900], safe=""), w=w, h=h, seed=seed + variant * 991)
        img = _valid_image(await _fetch_bytes(session, url, sem, headers=headers))
        if img is not None:
            img.save(dest, "JPEG", quality=94)
            return True
    return False


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
    alpha = cut.getchannel("A")
    bbox = alpha.point(lambda a: 255 if a > 20 else 0).getbbox()
    if not bbox:
        return False
    coverage = sum(1 for a in alpha.getdata() if a > 20) / (img.width * img.height)
    if not 0.04 <= coverage <= 0.85:  # empty or failed segmentation
        return False
    pad = 8
    bbox = (max(0, bbox[0] - pad), max(0, bbox[1] - pad), min(img.width, bbox[2] + pad), min(img.height, bbox[3] + pad))
    cut.crop(bbox).save(dest, "PNG")
    return True


async def _cutout(src: Path, dest: Path) -> bool:
    try:
        return await asyncio.get_running_loop().run_in_executor(_rembg_pool, _cutout_sync, src, dest)
    except Exception as exc:
        log.warning("rembg failed for %s: %s", src.name, exc)
        return False


# --------------------------------------------------------------------------- orchestration


async def _fetch_scene(i, scene, channel: ChannelConfig, assets_dir: Path, session, sem, base_seed: int,
                       used_ids: set[int], rng: random.Random) -> SceneAsset | None:
    use_sprite = channel.is_character_channel and bool(scene.character_pose)

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

    tasks = [_pollinations_image(session, sem, bg_prompt, base_seed + i, bg)]
    sprite_raw = assets_dir / f"sprite_{i:02d}_raw.jpg"
    if use_sprite:
        sprite_prompt = (
            f"{channel.character_description}, {scene.character_pose}, full body, centered, single character, "
            f"isolated on plain pure white background, soft studio lighting, 3D render, {NEGATIVE_HINT}"
        )
        # Same seed for every sprite of the channel keeps the mascot visually consistent
        tasks.append(_pollinations_image(session, sem, sprite_prompt, channel.character_seed, sprite_raw, 1024, 1536))
    results = await asyncio.gather(*tasks)
    if not results[0]:
        return None

    sprite = None
    if use_sprite and results[1]:
        sprite_png = assets_dir / f"sprite_{i:02d}.png"
        if await _cutout(sprite_raw, sprite_png):
            sprite = sprite_png
    return SceneAsset(i, "image", bg, sprite)


async def fetch_scene_assets(script: VideoScript, channel: ChannelConfig, workdir: Path) -> list[SceneAsset]:
    assets_dir = workdir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    base_seed = int(hashlib.md5(script.topic.encode()).hexdigest()[:7], 16)
    rng = random.Random(base_seed)
    sem = asyncio.Semaphore(get_settings().image_concurrency)
    used_ids: set[int] = set()
    connector = aiohttp.TCPConnector(limit=8)
    async with aiohttp.ClientSession(connector=connector, headers={"User-Agent": "dark-factory-os/1.0"}) as session:
        results = await asyncio.gather(*(
            _fetch_scene(i, scene, channel, assets_dir, session, sem, base_seed, used_ids, rng)
            for i, scene in enumerate(script.scenes)
        ))

    failed = [i for i, r in enumerate(results) if r is None]
    if len(failed) > len(results) // 4:
        raise RuntimeError(f"Too many scene assets failed: {failed}")
    # Fill gaps with the nearest successful asset (the renderer varies the camera, so it still feels new)
    ok = [r for r in results if r is not None]
    final: list[SceneAsset] = []
    for i, asset in enumerate(results):
        if asset is None:
            nearest = min(ok, key=lambda a: abs(a.index - i))
            asset = SceneAsset(i, nearest.kind, nearest.path, nearest.sprite)
            log.warning("Scene %d reused asset from scene %d", i, nearest.index)
        final.append(asset)
    log.info("[%s] assets ready: %d images, %d clips, %d sprites, %d reused", channel.id,
             sum(a.kind == "image" for a in final), sum(a.kind == "video" for a in final),
             sum(a.sprite is not None for a in final), len(failed))
    return final
