"""Telegram notifications. Never raises: alerting must not break the pipeline."""
from __future__ import annotations

import html
import logging

import requests

from src.config_loader import get_settings

log = logging.getLogger(__name__)

_ICONS = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "🚨"}


def _post(text: str) -> None:
    settings = get_settings()
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


def send_alert(message: str, level: str = "info") -> None:
    log.log(logging.ERROR if level == "error" else logging.INFO, "ALERT[%s] %s", level, message)
    _post(f"{_ICONS.get(level, '')} <b>dark-factory</b>\n{html.escape(message)[:3900]}")


def send_report(title: str, markdown: str) -> None:
    """Send a Markdown table as monospace (Telegram has no table rendering), split into 4 kB messages."""
    lines, chunk, chunks = markdown.splitlines(), "", []
    for line in lines:
        if len(chunk) + len(line) > 3500:
            chunks.append(chunk)
            chunk = ""
        chunk += line + "\n"
    chunks.append(chunk)
    for i, part in enumerate(chunks):
        header = f"📊 <b>{html.escape(title)}</b>\n" if i == 0 else ""
        _post(f"{header}<pre>{html.escape(part)}</pre>")
