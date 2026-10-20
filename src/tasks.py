"""Celery workers + Beat schedule.

Queues:  render  -> produce_video (CPU heavy, concurrency 1)
         publish -> publish_youtube / publish_tiktok (concurrency 1: browser profiles must not collide)

Manual run (no Celery needed, great for testing a channel end to end):

    python -m src.tasks run nicho_1 [--publish] [--force]
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from celery import Celery
from celery.schedules import crontab

from src import database as db
from src.alerts import send_alert
from src.config_loader import get_channel, get_settings, load_channels
from src.engine.audio_tts import AudioLengthError, synthesize
from src.engine.media_fetcher import fetch_scene_assets
from src.engine.renderer import render_video
from src.engine.script_generator import generate_script
from src.engine.subtitles_ass import build_ass, transcribe_words

log = logging.getLogger(__name__)
settings = get_settings()

celery_app = Celery("dark_factory", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.update(
    timezone=settings.timezone,
    enable_utc=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=20,  # release whisper/rembg memory periodically
    broker_connection_retry_on_startup=True,
    result_expires=7 * 24 * 3600,
    task_routes={
        "src.tasks.produce_video": {"queue": "render"},
        "src.tasks.cleanup_outputs": {"queue": "render"},
        "src.tasks.publish_youtube": {"queue": "publish"},
        "src.tasks.publish_tiktok": {"queue": "publish"},
    },
)


def _beat_schedule() -> dict:
    schedule = {
        "cleanup-outputs": {"task": "src.tasks.cleanup_outputs", "schedule": crontab(hour=4, minute=0)},
    }
    for channel in load_channels().values():
        if not channel.enabled:
            continue
        hour, minute = channel.post_hour_minute
        # Start production ~75 min before the publish slot (render takes 10-40 min on CPU)
        start = (hour * 60 + minute - 75) % (24 * 60)
        schedule[f"produce-{channel.id}"] = {
            "task": "src.tasks.produce_video",
            "schedule": crontab(hour=start // 60, minute=start % 60),
            "args": (channel.id,),
        }
    return schedule


celery_app.conf.beat_schedule = _beat_schedule()


# --------------------------------------------------------------------------- pipeline


def run_pipeline(channel_id: str, force: bool = False) -> int | None:
    """Script -> TTS -> Whisper/ASS -> assets -> render. Returns the DB video id (None if skipped)."""
    channel = get_channel(channel_id)
    if not channel.enabled:
        log.info("[%s] disabled, skipping", channel_id)
        return None
    db.init_db()
    if not force and db.produced_since(channel_id, hours=20):
        log.info("[%s] already produced in the last 20h, skipping", channel_id)
        return None

    workdir = settings.work_dir / channel_id / datetime.now().strftime("%Y%m%d_%H%M%S")
    workdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    recent = db.get_recent_topics(channel_id)
    feedback: list[str] = []
    for attempt in range(3):
        script = generate_script(channel, recent, feedback)
        if db.topic_exists(channel_id, script.topic):
            feedback = [f"Topic '{script.topic}' already exists. Choose a completely different subject."]
            recent = [script.topic, *recent]
            continue
        try:
            narration, narration_dur = synthesize(script.narration, channel, workdir)
            break
        except AudioLengthError as exc:
            direction = "shorter" if exc.needed_tempo > 1 else "longer"
            feedback = [f"The narration read in {exc.raw_duration:.0f}s; make it {direction} so it lasts "
                        f"~{channel.target_duration:.0f}s at a natural pace."]
            log.warning("[%s] %s -> regenerating script", channel_id, exc)
    else:
        raise RuntimeError("Could not obtain a script with valid narration length")

    (workdir / "script.json").write_text(script.model_dump_json(indent=2), encoding="utf-8")
    video_id = db.create_video(channel_id, script.topic, script.title, script.model_dump(mode="json"), str(workdir))
    try:
        words = transcribe_words(narration, channel.language, script.narration)
        total = narration_dur + 0.6
        subs = build_ass(words, channel.subtitle, workdir / "subs.ass", total)
        assets = asyncio.run(fetch_scene_assets(script, channel, workdir))
        final, duration = render_video(script, assets, narration, narration_dur, words, subs, channel, workdir)
    except Exception as exc:
        db.update_video(video_id, status=db.VideoStatus.FAILED, error=str(exc)[:4000])
        raise
    db.update_video(video_id, status=db.VideoStatus.RENDERED, video_path=str(final), duration=duration)
    send_alert(f"[{channel.name}] rendered '{script.title}' ({duration:.1f}s) in {(time.time() - t0) / 60:.1f} min",
               "success")
    return video_id


def dispatch_publish(video_id: int) -> None:
    video = db.get_video(video_id)
    channel = get_channel(video.channel_id)
    if channel.youtube.enabled:
        publish_youtube.delay(video_id)
    if channel.tiktok.enabled:
        publish_tiktok.delay(video_id)


# --------------------------------------------------------------------------- tasks


@celery_app.task(bind=True, max_retries=2, default_retry_delay=900, soft_time_limit=3300, time_limit=3600)
def produce_video(self, channel_id: str, force: bool = False):
    try:
        video_id = run_pipeline(channel_id, force=force)
    except Exception as exc:
        send_alert(f"[{channel_id}] production failed (try {self.request.retries + 1}/3): {exc}", "error")
        raise self.retry(exc=exc)
    if video_id and not settings.dry_run:
        dispatch_publish(video_id)
    return video_id


@celery_app.task(bind=True, max_retries=3, default_retry_delay=1200)
def publish_youtube(self, video_id: int):
    from src.publishers import youtube_api

    video = db.get_video(video_id)
    channel = get_channel(video.channel_id)
    if video.youtube_status == "published":
        return video.youtube_id
    try:
        yt_id = youtube_api.publish(channel, Path(video.video_path), video.script)
    except youtube_api.YouTubeAuthError as exc:
        db.update_video(video_id, youtube_status="auth_error", error=str(exc))
        send_alert(f"[{channel.name}] YouTube auth needed: {exc}", "error")
        return None
    except Exception as exc:
        db.update_video(video_id, youtube_status="error", error=str(exc)[:4000])
        if self.request.retries >= self.max_retries:
            send_alert(f"[{channel.name}] YouTube upload failed for good: {exc}", "error")
        raise self.retry(exc=exc)
    db.update_video(video_id, youtube_status="published", youtube_id=yt_id)
    db.refresh_publish_status(video_id, channel.youtube.enabled, channel.tiktok.enabled)
    send_alert(f"[{channel.name}] YouTube: https://youtube.com/shorts/{yt_id}", "success")
    return yt_id


@celery_app.task(bind=True, max_retries=2, default_retry_delay=1800)
def publish_tiktok(self, video_id: int):
    from src.publishers import tiktok_playwright

    video = db.get_video(video_id)
    channel = get_channel(video.channel_id)
    if video.tiktok_status == "published":
        return "published"
    try:
        status = tiktok_playwright.upload(channel, Path(video.video_path), tiktok_playwright.build_caption(video.script),
                                          debug_dir=Path(video.work_dir))
    except tiktok_playwright.TikTokAuthError as exc:
        db.update_video(video_id, tiktok_status="auth_error", error=str(exc))
        send_alert(f"[{channel.name}] TikTok login needed: {exc}", "error")
        return None
    except Exception as exc:
        db.update_video(video_id, tiktok_status="error", error=str(exc)[:4000])
        if self.request.retries >= self.max_retries:
            send_alert(f"[{channel.name}] TikTok upload failed for good: {exc}", "error")
        raise self.retry(exc=exc)
    db.update_video(video_id, tiktok_status=status)
    db.refresh_publish_status(video_id, channel.youtube.enabled, channel.tiktok.enabled)
    send_alert(f"[{channel.name}] TikTok posted", "success")
    return status


@celery_app.task
def cleanup_outputs():
    """Delete work dirs older than KEEP_DAYS (published or not)."""
    cutoff = time.time() - settings.keep_days * 86400
    removed = 0
    for run_dir in settings.work_dir.glob("*/*"):
        if run_dir.is_dir() and run_dir.stat().st_mtime < cutoff:
            shutil.rmtree(run_dir, ignore_errors=True)
            removed += 1
    log.info("cleanup removed %d run dirs", removed)
    return removed


# --------------------------------------------------------------------------- CLI

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if len(sys.argv) >= 3 and sys.argv[1] == "run":
        vid = run_pipeline(sys.argv[2], force="--force" in sys.argv)
        print(f"video_id={vid}")
        if vid and "--publish" in sys.argv:
            ch = get_channel(sys.argv[2])
            if ch.youtube.enabled:
                publish_youtube.apply(args=(vid,))
            if ch.tiktok.enabled:
                publish_tiktok.apply(args=(vid,))
    else:
        print("usage: python -m src.tasks run <channel_id> [--publish] [--force]")
