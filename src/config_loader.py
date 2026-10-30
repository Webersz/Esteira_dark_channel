"""Loads `.env` settings and the channel matrix (`config/channels_config.json`)."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import quote, unquote

from dotenv import load_dotenv
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

CHANNELS_FILE = BASE_DIR / "config" / "channels_config.json"
REPORTS_DIR = BASE_DIR / "reports"
OUTPUT_TEST_DIR = BASE_DIR / "output_test"


def _bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value) if value else default


def _list(name: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, "").split(",") if item.strip()]


# Tried in order; a model that 404s for a key is skipped instantly. Check what your key sees with:
#   python -c "from google import genai; c = genai.Client(api_key='KEY'); print([m.name for m in c.models.list()])"
DEFAULT_GEMINI_MODELS = ["gemini-3.5-flash", "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash",
                         "gemini-3-flash-preview", "gemini-3.5-flash-lite"]


def _gemini_keys() -> list[str]:
    """GEMINI_API_KEYS (comma-separated, ideally one per Google Cloud project) + legacy GEMINI_API_KEY."""
    keys = _list("GEMINI_API_KEYS")
    single = os.getenv("GEMINI_API_KEY", "").strip()
    if single and single not in keys:
        keys.append(single)
    return keys


def _gemini_models() -> list[str]:
    models = _list("GEMINI_MODELS")
    legacy = os.getenv("GEMINI_MODEL", "").strip()
    if not models:
        models = list(DEFAULT_GEMINI_MODELS)
        if legacy:  # an explicit GEMINI_MODEL goes first, the defaults stay as cascade
            models = [legacy] + [m for m in models if m != legacy]
    return models


class FallbackLLM(BaseModel):
    """OpenAI-compatible chat-completions endpoint used when every Gemini key/model fails."""

    name: str
    url: str
    api_key: str
    model: str


def _fallback_llms() -> list[FallbackLLM]:
    candidates = [
        ("groq", "https://api.groq.com/openai/v1/chat/completions",
         os.getenv("GROQ_API_KEY"), os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")),
        ("github", "https://models.github.ai/inference/chat/completions",
         os.getenv("GITHUB_MODELS_TOKEN"), os.getenv("GITHUB_MODELS_MODEL", "openai/gpt-4o-mini")),
        ("openrouter", "https://openrouter.ai/api/v1/chat/completions",
         os.getenv("OPENROUTER_API_KEY"), os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")),
        ("custom", os.getenv("FALLBACK_LLM_URL", ""),
         os.getenv("FALLBACK_LLM_API_KEY"), os.getenv("FALLBACK_LLM_MODEL", "")),
    ]
    return [FallbackLLM(name=name, url=url, api_key=key.strip(), model=model)
            for name, url, key, model in candidates if key and key.strip() and url and model]


def _sanitize_database_url(url: str) -> str:
    """URL-encode the password so '@', '#', ':', '/', '%', '?' and '&' don't break SQLAlchemy's parser."""
    scheme, sep, rest = url.partition("://")
    if not sep or "@" not in rest:
        return url
    userinfo, _, hostpart = rest.rpartition("@")  # host/db never contain '@'; the password may
    user, colon, password = userinfo.partition(":")
    if not colon or quote(unquote(password), safe="") == password:  # no password, or already encoded
        return url
    return f"{scheme}://{user}:{quote(password, safe='')}@{hostpart}"


def _database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        user, password = os.getenv("POSTGRES_USER"), os.getenv("POSTGRES_PASSWORD")
        if user and password:
            host = os.getenv("POSTGRES_HOST", "localhost")
            port = os.getenv("POSTGRES_PORT", "5432")
            db = os.getenv("POSTGRES_DB", "dark_factory")
            return f"postgresql+psycopg2://{quote(user, safe='')}:{quote(password, safe='')}@{host}:{port}/{db}"
        url = "postgresql+psycopg2://factory:change_me@localhost:5432/dark_factory"
    return _sanitize_database_url(url)


class Settings(BaseModel):
    gemini_api_keys: list[str]
    gemini_models: list[str] = Field(default_factory=lambda: list(DEFAULT_GEMINI_MODELS))
    fallback_llms: list[FallbackLLM] = Field(default_factory=list)
    pexels_api_key: str | None = None
    pollinations_token: str | None = None
    image_concurrency: int = 3
    render_threads: int = 2

    whisper_model: str = "small"
    whisper_device: str = "cpu"

    database_url: str
    redis_url: str
    timezone: str = "America/Sao_Paulo"

    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None

    youtube_privacy: Literal["public", "unlisted", "private"] = "public"
    tiktok_headless: bool = True
    tiktok_proxy_server: str | None = None  # global fallback when a channel has no "proxy"
    tiktok_proxy_user: str | None = None
    tiktok_proxy_pass: str | None = None
    dry_run: bool = False
    keep_days: int = 3

    work_dir: Path
    media_dir: Path
    credentials_dir: Path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        gemini_api_keys=_gemini_keys(),
        gemini_models=_gemini_models(),
        fallback_llms=_fallback_llms(),
        pexels_api_key=os.getenv("PEXELS_API_KEY") or None,
        pollinations_token=os.getenv("POLLINATIONS_TOKEN") or None,
        image_concurrency=int(os.getenv("IMAGE_CONCURRENCY", "3")),
        render_threads=int(os.getenv("RENDER_THREADS", "2")),
        whisper_model=os.getenv("WHISPER_MODEL", "small"),
        whisper_device=os.getenv("WHISPER_DEVICE", "cpu"),
        database_url=_database_url(),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        timezone=os.getenv("TZ", "America/Sao_Paulo"),
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN") or None,
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID") or None,
        youtube_privacy=os.getenv("YOUTUBE_PRIVACY", "public"),
        tiktok_headless=_bool("TIKTOK_HEADLESS", True),
        tiktok_proxy_server=os.getenv("TIKTOK_PROXY_SERVER") or None,
        tiktok_proxy_user=os.getenv("TIKTOK_PROXY_USER") or None,
        tiktok_proxy_pass=os.getenv("TIKTOK_PROXY_PASS") or None,
        dry_run=_bool("DRY_RUN", False),
        keep_days=int(os.getenv("KEEP_DAYS", "3")),
        work_dir=_path("WORK_DIR", BASE_DIR / "output"),
        media_dir=_path("MEDIA_DIR", BASE_DIR / "media_assets"),
        credentials_dir=_path("CREDENTIALS_DIR", BASE_DIR / "credentials"),
    )


# --------------------------------------------------------------------------- channels


class VoiceConfig(BaseModel):
    voice: str = "pt-BR-AntonioNeural"
    rate: str = "+0%"  # edge-tts native rate
    pitch_semitones: float = 0.0  # applied with FFmpeg asetrate/aresample
    tempo: float = 1.0  # applied with FFmpeg atempo (auto-adjusted to hit the duration window)


class SubtitleConfig(BaseModel):
    font: str = "Montserrat ExtraBold"
    font_size: int = 92
    primary_color: str = "#FFFFFF"
    highlight_color: str = "#FFE600"
    outline_color: str = "#000000"
    outline: int = 6
    shadow: int = 3
    margin_v: int = 620
    words_per_line: int = 3
    uppercase: bool = True


class AudioBed(BaseModel):
    file: str
    volume: float = 0.15


class OverlayConfig(BaseModel):
    file: str = "overlays/dust_particles.mp4"
    opacity: float = 0.35


class YouTubeConfig(BaseModel):
    enabled: bool = True
    token_file: str
    category_id: str = "24"


class TikTokConfig(BaseModel):
    enabled: bool = True
    profile_dir: str
    proxy: str | None = None  # e.g. "http://user:pass@host:port"


class ChannelConfig(BaseModel):
    id: str
    name: str
    enabled: bool = True
    language: str = "pt-BR"
    niche: str
    style: Literal["cinematic", "3d_character"] = "cinematic"
    visual_style: str
    character_description: str = ""
    character_seed: int = 1234
    use_rembg: bool = False
    use_broll: bool = False
    broll_every: int = 4
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    bgm: AudioBed
    post_time: str = "12:00"
    prompt_guidelines: str = ""
    hashtags: list[str] = Field(default_factory=list)
    youtube: YouTubeConfig
    tiktok: TikTokConfig

    words_per_second: float = 2.65
    target_duration: float = 65.0
    min_duration: float = 62.0
    max_duration: float = 68.0
    scenes: int = 16
    subtitle: SubtitleConfig = Field(default_factory=SubtitleConfig)
    overlay: OverlayConfig = Field(default_factory=OverlayConfig)
    grade: str = ""
    ambience: Literal["none", "wind"] = "none"  # low synthetic wind/sea bed under the music

    @property
    def is_character_channel(self) -> bool:
        return self.style == "3d_character" and self.use_rembg

    @property
    def target_words(self) -> int:
        return round(self.target_duration * self.words_per_second)

    @property
    def post_hour_minute(self) -> tuple[int, int]:
        hour, minute = self.post_time.split(":")
        return int(hour), int(minute)


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@lru_cache(maxsize=1)
def load_channels() -> dict[str, ChannelConfig]:
    raw = json.loads(CHANNELS_FILE.read_text(encoding="utf-8"))
    defaults = raw.get("defaults", {})
    channels = [ChannelConfig(**_deep_merge(defaults, ch)) for ch in raw["channels"]]
    return {ch.id: ch for ch in channels}


def get_channel(channel_id: str) -> ChannelConfig:
    channels = load_channels()
    if channel_id not in channels:
        raise KeyError(f"Unknown channel '{channel_id}'. Known: {', '.join(channels)}")
    return channels[channel_id]
