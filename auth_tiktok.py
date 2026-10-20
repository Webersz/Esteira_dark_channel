"""Interactive TikTok session initializer: opens a headed Chromium per channel so you can log in once.

    python auth_tiktok.py                    # every enabled channel without a saved session
    python auth_tiktok.py --channel nicho_1  # just one
    python auth_tiktok.py --force            # redo all (session expired)

Saves the persistent profile in credentials/tiktok_profiles/<channel_id>/ plus a portable
storage_state.json (cookies) that the Linux container re-imports. Uses the same proxy as uploads
(per-channel "proxy" or TIKTOK_PROXY_* from .env), so TikTok sees a consistent network per account.
"""
from __future__ import annotations

import argparse

from playwright.sync_api import sync_playwright

from src.config_loader import ChannelConfig, load_channels
from src.publishers.tiktok_playwright import launch, storage_state_path

LOGIN_URL = "https://www.tiktok.com/login"


def login(channel: ChannelConfig) -> None:
    print(f"\n=== [{channel.id}] {channel.name} ===")
    with sync_playwright() as p:
        context = launch(p, channel, headless=False)
        page = context.pages[0] if context.pages else context.new_page()
        page.goto(LOGIN_URL, wait_until="domcontentloaded")
        input("Log in to TikTok in the opened window (this channel's account), then press ENTER here... ")
        has_session = any(c["name"] == "sessionid" for c in context.cookies("https://www.tiktok.com"))
        path = storage_state_path(channel)
        context.storage_state(path=str(path))
        context.close()
    print(f"Saved {path}" + ("" if has_session else "  (WARNING: no 'sessionid' cookie found - login may have failed)"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Log in to TikTok once per channel")
    parser.add_argument("--channel", help="only this channel id")
    parser.add_argument("--force", action="store_true", help="redo even if a session exists")
    args = parser.parse_args(argv)

    channels = [c for c in load_channels().values() if c.tiktok.enabled and (not args.channel or c.id == args.channel)]
    if args.channel and not channels:
        raise SystemExit(f"Unknown or TikTok-disabled channel: {args.channel}")
    pending = [c for c in channels if args.force or not storage_state_path(c).exists()]
    if not pending:
        print("All TikTok sessions already exist (use --force to redo).")
    for channel in pending:
        login(channel)


if __name__ == "__main__":
    main()
