"""PostgreSQL state: video lifecycle tracking + topic anti-duplication."""
from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Iterator

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, UniqueConstraint, create_engine, func, select
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from src.config_loader import get_settings

log = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class VideoStatus:
    SCRIPTED = "scripted"
    RENDERED = "rendered"
    PUBLISHED = "published"  # all enabled platforms OK
    PARTIAL = "partial"  # at least one platform OK
    FAILED = "failed"


class Video(Base):
    __tablename__ = "videos_history"
    __table_args__ = (UniqueConstraint("channel_id", "topic_key", name="uq_channel_topic"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    channel_id: Mapped[str] = mapped_column(String(64), index=True)
    topic: Mapped[str] = mapped_column(String(400))
    topic_key: Mapped[str] = mapped_column(String(64), index=True)
    title: Mapped[str | None] = mapped_column(String(200))
    script: Mapped[dict | None] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(32), default=VideoStatus.SCRIPTED, index=True)
    work_dir: Mapped[str | None] = mapped_column(String(500))
    video_path: Mapped[str | None] = mapped_column(String(500))
    duration: Mapped[float | None] = mapped_column(Float)
    youtube_id: Mapped[str | None] = mapped_column(String(32))
    youtube_status: Mapped[str | None] = mapped_column(String(32))
    tiktok_status: Mapped[str | None] = mapped_column(String(32))
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class DuplicateTopicError(Exception):
    pass


_engine = None
_SessionLocal: sessionmaker | None = None


def get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        _engine = create_engine(get_settings().database_url, pool_pre_ping=True, pool_size=5, future=True,
                                connect_args={"connect_timeout": 5})
        _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def init_db() -> None:
    Base.metadata.create_all(get_engine())


def db_available() -> bool:
    """True if PostgreSQL is reachable (tables are created on the way). Used by local test renders."""
    try:
        init_db()
        return True
    except OperationalError as exc:
        log.warning("PostgreSQL unreachable (%s) - running without history/de-duplication", str(exc).splitlines()[0])
        return False


@contextmanager
def session_scope() -> Iterator[Session]:
    get_engine()
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# --------------------------------------------------------------------------- topic helpers


def normalize_topic(topic: str) -> str:
    text = unicodedata.normalize("NFKD", topic).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def topic_key(topic: str) -> str:
    return hashlib.sha1(normalize_topic(topic).encode()).hexdigest()


def get_recent_topics(channel_id: str, limit: int = 120) -> list[str]:
    with session_scope() as s:
        rows = s.execute(
            select(Video.topic).where(Video.channel_id == channel_id).order_by(Video.created_at.desc()).limit(limit)
        ).scalars()
        return list(rows)


def topic_exists(channel_id: str, topic: str) -> bool:
    with session_scope() as s:
        return (
            s.execute(
                select(Video.id).where(Video.channel_id == channel_id, Video.topic_key == topic_key(topic))
            ).first()
            is not None
        )


def produced_since(channel_id: str, hours: int = 20) -> bool:
    """Idempotency guard: was a (non-failed) video already produced recently for this channel?"""
    since = datetime.now() - timedelta(hours=hours)
    with session_scope() as s:
        return (
            s.execute(
                select(Video.id).where(
                    Video.channel_id == channel_id,
                    Video.created_at >= since,
                    Video.status != VideoStatus.FAILED,
                )
            ).first()
            is not None
        )


# --------------------------------------------------------------------------- CRUD


def create_video(channel_id: str, topic: str, title: str, script: dict, work_dir: str) -> int:
    try:
        with session_scope() as s:
            video = Video(
                channel_id=channel_id,
                topic=topic[:400],
                topic_key=topic_key(topic),
                title=title[:200],
                script=script,
                work_dir=work_dir,
                status=VideoStatus.SCRIPTED,
            )
            s.add(video)
            s.flush()
            return video.id
    except IntegrityError as exc:
        raise DuplicateTopicError(topic) from exc


def update_video(video_id: int, **fields: Any) -> None:
    with session_scope() as s:
        video = s.get(Video, video_id)
        if video is None:
            raise KeyError(video_id)
        for key, value in fields.items():
            setattr(video, key, value)


def get_video(video_id: int) -> Video:
    with session_scope() as s:
        video = s.get(Video, video_id)
        if video is None:
            raise KeyError(video_id)
        return video


def refresh_publish_status(video_id: int, youtube_enabled: bool, tiktok_enabled: bool) -> str:
    """Derive the aggregated status from per-platform results."""
    with session_scope() as s:
        video = s.get(Video, video_id)
        results = []
        if youtube_enabled:
            results.append(video.youtube_status)
        if tiktok_enabled:
            results.append(video.tiktok_status)
        if results and all(r == "published" for r in results):
            video.status = VideoStatus.PUBLISHED
        elif any(r == "published" for r in results):
            video.status = VideoStatus.PARTIAL
        return video.status
