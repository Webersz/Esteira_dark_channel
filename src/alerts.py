"""Telegram notifications. Never raises: alerting must not break the pipeline."""
from __future__ import annotations

import html
import logging

import requests

from src.config_loader import get_settings

log = logging.getLogger(__name__)

_ICONS = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "🚨"}


def send_alert(message: str, level: str = "info") -> None:
    settings = get_settings()
    text = f"{_ICONS.get(level, '')} <b>dark-factory</b>\n{html.escape(message)[:3900]}"
    log.log(logging.ERROR if level == "error" else logging.INFO, "ALERT[%s] %s", level, message)
    if not (settings.telegram_bot_token and settings.telegram_chat_id):
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage",
            json={
                "chat_id": settings.telegram_chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=10,
        )
    except requests.RequestException as exc:
        log.warning("Telegram alert failed: %s", exc)
