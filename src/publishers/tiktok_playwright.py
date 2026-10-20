"""TikTok upload through TikTok Studio using a persistent Chromium profile per channel (optional proxy).

One-time login per channel: `python auth_tiktok.py` (headed browser; you log in by hand).

Besides the Chromium profile, the login exports `storage_state.json` (cookies + localStorage).
Chromium profiles created on Windows/macOS are encrypted with OS keychains and won't decrypt inside
the Linux container, so the container re-imports cookies from that portable JSON when needed.

TikTok's web UI changes often: every selector below has text fallbacks (EN + PT-BR) and failures
save a screenshot + HTML dump into the video's work dir for quick fixing.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import BrowserContext, Page, TimeoutError as PWTimeout, sync_playwright

from src.config_loader import ChannelConfig, get_settings

log = logging.getLogger(__name__)

UPLOAD_URL = "https://www.tiktok.com/tiktokstudio/upload?from=upload"
UPLOAD_TIMEOUT_S = 600


class TikTokAuthError(Exception):
    """Session expired: run the login command again (not retryable)."""


class TikTokUploadError(Exception):
    pass


def _profile_dir(channel: ChannelConfig) -> Path:
    path = get_settings().credentials_dir / channel.tiktok.profile_dir
    path.mkdir(parents=True, exist_ok=True)
    return path


def _proxy(channel: ChannelConfig) -> dict | None:
    """Per-channel `proxy` URL wins; otherwise TIKTOK_PROXY_SERVER/USER/PASS from .env; otherwise direct."""
    if channel.tiktok.proxy:
        parsed = urlparse(channel.tiktok.proxy)
        proxy = {"server": f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"}
        if parsed.username:
            proxy.update(username=parsed.username, password=parsed.password or "")
        return proxy
    settings = get_settings()
    if settings.tiktok_proxy_server:
        proxy = {"server": settings.tiktok_proxy_server}
        if settings.tiktok_proxy_user:
            proxy.update(username=settings.tiktok_proxy_user, password=settings.tiktok_proxy_pass or "")
        return proxy
    return None


def storage_state_path(channel: ChannelConfig) -> Path:
    return _profile_dir(channel) / "storage_state.json"


def launch(p, channel: ChannelConfig, headless: bool) -> BrowserContext:
    return p.chromium.launch_persistent_context(
        user_data_dir=str(_profile_dir(channel)),
        headless=headless,
        proxy=_proxy(channel),
        viewport={"width": 1366, "height": 900},
        locale=channel.language,
        args=["--disable-dev-shm-usage", "--password-store=basic"],
    )


def _ensure_cookies(context: BrowserContext, channel: ChannelConfig) -> None:
    state_file = storage_state_path(channel)
    has_session = any(c["name"] == "sessionid" for c in context.cookies("https://www.tiktok.com"))
    if not has_session and state_file.exists():
        cookies = json.loads(state_file.read_text(encoding="utf-8")).get("cookies", [])
        context.add_cookies(cookies)
        log.info("[%s] imported %d cookies from storage_state.json", channel.id, len(cookies))


def _first_visible(page: Page, selectors: list[str], timeout_ms: int = 5000):
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        for selector in selectors:
            loc = page.locator(selector).first
            try:
                if loc.count() and loc.is_visible():
                    return loc
            except Exception:
                pass
        page.wait_for_timeout(400)
    return None


def _dump_debug(page: Page, debug_dir: Path | None, tag: str) -> None:
    if not debug_dir:
        return
    try:
        page.screenshot(path=str(debug_dir / f"tiktok_{tag}.png"), full_page=True)
        (debug_dir / f"tiktok_{tag}.html").write_text(page.content(), encoding="utf-8")
    except Exception:
        pass


def _type_caption(page: Page, caption: str) -> None:
    editor = _first_visible(page, ['div[contenteditable="true"]', '.public-DraftEditor-content'], 120_000)
    if editor is None:
        raise TikTokUploadError("caption editor not found")
    editor.click()
    page.keyboard.press("Control+A")
    page.keyboard.press("Backspace")
    for token in caption.split(" "):
        if token.startswith("#"):
            page.keyboard.type(token, delay=60)
            page.wait_for_timeout(900)  # hashtag suggestion popup
            page.keyboard.press("Escape")
            page.keyboard.type(" ")
        else:
            page.keyboard.insert_text(token + " ")


def _toggle_ai_label(page: Page) -> None:
    """Best effort: enable the 'AI-generated content' disclosure (TikTok policy for synthetic media)."""
    try:
        more = _first_visible(page, ['text="Show more"', 'text="Mostrar mais"', 'text="Ver mais"'], 3000)
        if more:
            more.click()
        label = _first_visible(page, ['text=/AI-generated content/i', 'text=/gerado por IA/i'], 4000)
        if label is None:
            return
        switch = label.locator("xpath=ancestor::*[.//*[@role='switch'] or .//input[@type='checkbox']][1]")
        target = switch.locator("[role='switch'], input[type='checkbox']").first
        if target.get_attribute("aria-checked") != "true" and not target.is_checked():
            target.click(force=True)
            confirm = _first_visible(page, ['button:has-text("Turn on")', 'button:has-text("Ativar")'], 2500)
            if confirm:
                confirm.click()
    except Exception as exc:
        log.info("AI label toggle skipped: %s", exc)


def _wait_upload_finished(page: Page) -> None:
    deadline = time.time() + UPLOAD_TIMEOUT_S
    post = 'button[data-e2e="post_video_button"]'
    while time.time() < deadline:
        btn = page.locator(post).first
        if btn.count() and btn.is_enabled() and btn.get_attribute("aria-disabled") not in ("true",):
            uploading = page.locator("text=/Uploading|Enviando|Carregando/i").first
            if not (uploading.count() and uploading.is_visible()):
                return
        page.wait_for_timeout(2000)
    raise TikTokUploadError("upload did not finish in time")


def upload(channel: ChannelConfig, video_path: Path, caption: str, debug_dir: Path | None = None) -> str:
    settings = get_settings()
    with sync_playwright() as p:
        context = launch(p, channel, settings.tiktok_headless)
        page = context.pages[0] if context.pages else context.new_page()
        try:
            _ensure_cookies(context, channel)
            page.goto(UPLOAD_URL, wait_until="domcontentloaded", timeout=90_000)
            page.wait_for_timeout(4000)
            if "login" in page.url:
                raise TikTokAuthError(f"Session expired. Run: python auth_tiktok.py --channel {channel.id}")

            file_input = page.locator('input[type="file"]').first
            file_input.wait_for(state="attached", timeout=60_000)
            file_input.set_input_files(str(video_path))

            _type_caption(page, caption[:2100])
            _toggle_ai_label(page)
            _wait_upload_finished(page)

            page.locator('button[data-e2e="post_video_button"]').first.click()
            confirm = _first_visible(page, ['button:has-text("Post now")', 'button:has-text("Publicar agora")'], 6000)
            if confirm:
                confirm.click()

            try:
                page.wait_for_url("**/tiktokstudio/content**", timeout=120_000)
            except PWTimeout:
                ok = _first_visible(page, ["text=/posted|publicado|Your video is being uploaded|sendo carregado/i"], 15_000)
                if ok is None:
                    raise TikTokUploadError("no confirmation after clicking Post")
            log.info("[%s] TikTok post submitted", channel.id)
            return "published"
        except Exception:
            _dump_debug(page, debug_dir, "error")
            raise
        finally:
            context.close()


def build_caption(script: dict) -> str:
    hashtags = " ".join(script.get("hashtags", []))
    return f"{script['title']} {script.get('pinned_comment', '')} {hashtags}".strip()

