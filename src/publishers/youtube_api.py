"""YouTube Data API v3: resumable upload, automatic token refresh, first engagement comment.

One-time authorization per channel (run on a machine with a browser, NOT inside Docker):

    python -m src.publishers.youtube_api auth nicho_1

Needs `credentials/client_secret.json` (OAuth client of type "Desktop app" from Google Cloud Console).
Sign in with the Google account that owns the channel; when it has several channels, pick the right one.
"""
from __future__ import annotations

import json
import logging
import random
import sys
import time
from pathlib import Path

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

from src.config_loader import ChannelConfig, get_channel, get_settings

log = logging.getLogger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.force-ssl",  # comments
]
RETRYABLE_STATUS = {500, 502, 503, 504}


class YouTubeAuthError(Exception):
    """Token missing/revoked: needs a manual re-auth (not retryable)."""


def _token_path(channel: ChannelConfig) -> Path:
    return get_settings().credentials_dir / channel.youtube.token_file


def _load_credentials(channel: ChannelConfig) -> Credentials:
    path = _token_path(channel)
    if not path.exists():
        raise YouTubeAuthError(f"Missing token {path}. Run: python -m src.publishers.youtube_api auth {channel.id}")
    creds = Credentials.from_authorized_user_file(str(path), SCOPES)
    if not creds.valid:
        if not creds.refresh_token:
            raise YouTubeAuthError(f"Token for {channel.id} has no refresh_token; re-run auth")
        try:
            creds.refresh(Request())
        except RefreshError as exc:
            raise YouTubeAuthError(f"Refresh failed for {channel.id} (revoked/expired consent): {exc}") from exc
        path.write_text(creds.to_json(), encoding="utf-8")  # persist the rotated access token
    return creds


def _build_title(title: str) -> str:
    title = title.strip()
    return f"{title} #shorts" if len(title) <= 92 else title[:100]


def upload_video(channel: ChannelConfig, video_path: Path, title: str, description: str, tags: list[str],
                 hashtags: list[str]) -> str:
    youtube = build("youtube", "v3", credentials=_load_credentials(channel), cache_discovery=False)
    body = {
        "snippet": {
            "title": _build_title(title),
            "description": f"{description}\n\n{' '.join(hashtags)}"[:4900],
            "tags": tags[:15],
            "categoryId": channel.youtube.category_id,
            "defaultLanguage": channel.language,
            "defaultAudioLanguage": channel.language,
        },
        "status": {
            "privacyStatus": get_settings().youtube_privacy,
            "selfDeclaredMadeForKids": False,
            "containsSyntheticMedia": True,  # YouTube altered/synthetic content disclosure
        },
    }
    media = MediaFileUpload(str(video_path), mimetype="video/mp4", chunksize=8 * 1024 * 1024, resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    response, retries = None, 0
    while response is None:
        try:
            status, response = request.next_chunk()
            if status:
                log.info("[%s] YouTube upload %d%%", channel.id, int(status.progress() * 100))
        except HttpError as exc:
            if exc.resp.status not in RETRYABLE_STATUS or retries >= 6:
                raise
            retries += 1
            time.sleep(min(64, 2**retries) + random.random())
    video_id = response["id"]
    log.info("[%s] YouTube upload OK: https://youtube.com/shorts/%s", channel.id, video_id)
    return video_id


def post_comment(channel: ChannelConfig, video_id: str, text: str) -> str | None:
    """Post the engagement comment. NOTE: the Data API cannot pin comments; pin manually if desired."""
    youtube = build("youtube", "v3", credentials=_load_credentials(channel), cache_discovery=False)
    body = {"snippet": {"videoId": video_id, "topLevelComment": {"snippet": {"textOriginal": text[:9000]}}}}
    for attempt in range(4):
        try:
            result = youtube.commentThreads().insert(part="snippet", body=body).execute()
            return result["id"]
        except HttpError as exc:
            # Freshly uploaded videos often reject comments for a minute while processing
            log.warning("[%s] comment attempt %d failed: %s", channel.id, attempt + 1, exc)
            time.sleep(45 * (attempt + 1))
    return None


def publish(channel: ChannelConfig, video_path: Path, script: dict) -> str:
    video_id = upload_video(channel, video_path, script["title"], script["description"], script.get("tags", []),
                            script.get("hashtags", []))
    if script.get("pinned_comment"):
        post_comment(channel, video_id, script["pinned_comment"])
    return video_id


# --------------------------------------------------------------------------- CLI


def _auth(channel_id: str) -> None:
    from google_auth_oauthlib.flow import InstalledAppFlow

    channel = get_channel(channel_id)
    secrets = get_settings().credentials_dir / "client_secret.json"
    flow = InstalledAppFlow.from_client_secrets_file(str(secrets), SCOPES)
    creds = flow.run_local_server(port=0, access_type="offline", prompt="consent")
    path = _token_path(channel)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(creds.to_json(), encoding="utf-8")
    me = build("youtube", "v3", credentials=creds, cache_discovery=False).channels().list(part="snippet", mine=True).execute()
    names = [item["snippet"]["title"] for item in me.get("items", [])]
    print(f"Saved {path}\nAuthorized channel(s): {json.dumps(names, ensure_ascii=False)}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "auth":
        _auth(sys.argv[2])
    else:
        print("usage: python -m src.publishers.youtube_api auth <channel_id>")
