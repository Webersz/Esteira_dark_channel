"""Weekly YouTube Analytics v2 report across all channels with automatic retention diagnosis.

    python main_cli.py --report-analytics

Note: YouTube Analytics data lags ~48-72 h, so the most recent days are usually partial.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from src.alerts import send_report
from src.config_loader import REPORTS_DIR, ChannelConfig, load_channels
from src.publishers.youtube_api import YouTubeAuthError, load_credentials

log = logging.getLogger(__name__)

METRICS = "views,averageViewPercentage,averageViewDuration,likes,subscribersGained"
HEADER = "| Canal | Views (7d) | Retenção Média (%) | Tempo Médio (s) | Likes | Inscritos (+) | Diagnóstico IA |"
SEPARATOR = "|---|---:|---:|---:|---:|---:|---|"


@dataclass
class ChannelStats:
    channel: ChannelConfig
    views: int = 0
    avg_view_pct: float = 0.0
    avg_view_duration: float = 0.0
    likes: int = 0
    subscribers: int = 0
    error: str | None = None


def _thousands(value: int) -> str:
    return f"{value:,}".replace(",", ".")  # pt-BR thousands separator


def diagnose(avg_view_pct: float) -> str:
    if avg_view_pct < 50:
        return "⚠️ ALERTA: Gancho fraco (Ajustar Cena 1)"
    if avg_view_pct < 70:
        return "🟡 ESTÁVEL: Otimizar twist da Cena 7"
    return "🔥 EXCELENTE: Algoritmo escalando"


def fetch_channel_stats(channel: ChannelConfig, start: date, end: date) -> ChannelStats:
    try:
        creds = load_credentials(channel)
        analytics = build("youtubeAnalytics", "v2", credentials=creds, cache_discovery=False)
        response = analytics.reports().query(
            ids="channel==MINE", startDate=start.isoformat(), endDate=end.isoformat(), metrics=METRICS
        ).execute()
    except YouTubeAuthError as exc:
        return ChannelStats(channel, error=f"sem token ({exc.__class__.__name__})")
    except HttpError as exc:
        if exc.resp.status == 403:
            return ChannelStats(channel, error=f"sem permissão: python auth_youtube.py --channel {channel.id} --force")
        return ChannelStats(channel, error=f"HTTP {exc.resp.status}")
    rows = response.get("rows") or [[0, 0, 0, 0, 0]]
    views, pct, duration, likes, subs = rows[0]
    return ChannelStats(channel, int(views), float(pct), float(duration), int(likes), int(subs))


def build_report(today: date | None = None) -> tuple[str, str]:
    """Return (title, markdown)."""
    end = today or date.today()
    start = end - timedelta(days=7)
    lines = [f"# Relatório semanal — {start:%d/%m} a {end:%d/%m/%Y}", "", HEADER, SEPARATOR]
    for channel in load_channels().values():
        if not (channel.enabled and channel.youtube.enabled):
            continue
        s = fetch_channel_stats(channel, start, end)
        if s.error:
            lines.append(f"| {channel.name} | – | – | – | – | – | ❌ {s.error} |")
        else:
            lines.append(
                f"| {channel.name} | {_thousands(s.views)} | {s.avg_view_pct:.1f} | {s.avg_view_duration:.1f} | "
                f"{_thousands(s.likes)} | {s.subscribers:+d} | {diagnose(s.avg_view_pct)} |"
            )
    lines += ["", "_Dados do YouTube Analytics têm atraso de 48–72 h._"]
    return f"Analytics {start:%d/%m}–{end:%d/%m}", "\n".join(lines)


def run_report(send: bool = True) -> tuple[str, Path]:
    title, markdown = build_report()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"analytics_{date.today():%Y_%m_%d}.md"
    path.write_text(markdown + "\n", encoding="utf-8")
    if send:
        send_report(title, markdown)
    log.info("analytics report saved to %s", path)
    return markdown, path
