"""Interactive YouTube OAuth2 token generator (run on a machine with a browser, NOT inside Docker).

Needs `credentials/client_secret.json` (Google Cloud Console -> OAuth client of type "Desktop app",
with YouTube Data API v3 and YouTube Analytics API enabled).

    python auth_youtube.py                    # every enabled channel that has no token yet
    python auth_youtube.py --channel nicho_1  # just one
    python auth_youtube.py --force            # re-authorize all (e.g. to add the analytics scope)

For each channel, sign in with the Google account that owns it and pick the right channel/brand account.
Tokens land in credentials/yt_tokens/yt_<channel_id>.json and are refreshed automatically afterwards.
"""
from __future__ import annotations

import argparse
import json

from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from src.config_loader import ChannelConfig, get_settings, load_channels
from src.publishers.youtube_api import SCOPES, token_path


def authorize(channel: ChannelConfig) -> None:
    secrets = get_settings().credentials_dir / "client_secret.json"
    if not secrets.exists():
        raise SystemExit(f"Missing {secrets}. Download the OAuth 'Desktop app' client JSON from Google Cloud Console.")
    print(f"\n=== [{channel.id}] {channel.name}: a browser window will open; sign in to THIS channel's account ===")
    flow = InstalledAppFlow.from_client_secrets_file(str(secrets), SCOPES)
    creds = flow.run_local_server(port=0, access_type="offline", prompt="consent")
    path = token_path(channel)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(creds.to_json(), encoding="utf-8")
    me = build("youtube", "v3", credentials=creds, cache_discovery=False).channels().list(part="snippet", mine=True).execute()
    names = [item["snippet"]["title"] for item in me.get("items", [])]
    print(f"Saved {path}\nAuthorized YouTube channel(s): {json.dumps(names, ensure_ascii=False)}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Create YouTube OAuth tokens per channel")
    parser.add_argument("--channel", help="only this channel id")
    parser.add_argument("--force", action="store_true", help="re-authorize even if a token exists")
    args = parser.parse_args(argv)

    channels = [c for c in load_channels().values() if c.youtube.enabled and (not args.channel or c.id == args.channel)]
    if args.channel and not channels:
        raise SystemExit(f"Unknown or YouTube-disabled channel: {args.channel}")
    pending = [c for c in channels if args.force or not token_path(c).exists()]
    if not pending:
        print("All YouTube tokens already exist (use --force to redo).")
    for channel in pending:
        authorize(channel)


if __name__ == "__main__":
    main()
