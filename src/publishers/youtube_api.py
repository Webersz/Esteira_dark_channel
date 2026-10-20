"""YouTube Data API v3: resumable upload, automatic token refresh, first engagement comment.

Tokens are created once per channel with `python auth_youtube.py` (see that file).
"""
from __future__ import annotations

import logging
import random
import time
from pathlib import Path

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

from src.config_loader import ChannelConfig, get_settings

log = logging.getLogger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.force-ssl",  # comments
    "https://www.googleapis.com/auth/yt-analytics.readonly",  # weekly report
]
RETRYABLE_STATUS = {500, 502, 503, 504}


class YouTubeAuthError(Exception):
    """Token missing/revoked: needs a manual re-auth (not retryable)."""


def token_path(channel: ChannelConfig) -> Path:
    return get_settings().credentials_dir / channel.youtube.token_file


def load_credentials(channel: ChannelConfig) -> Credentials:
    path = token_path(channel)
    if not path.exists():
        raise YouTubeAuthError(f"Missing token {path}. Run: python auth_youtube.py --channel {channel.id}")
    # No `scopes` override: keep whatever the token was granted (older tokens may lack analytics)
    creds = Credentials.from_authorized_user_file(str(path))
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except RefreshError as exc:
            raise YouTubeAuthError(f"Refresh failed for {channel.id} (revoked/expired consent): {exc}") from exc
        path.write_text(creds.to_json(), encoding="utf-8")  # persist the rotated access token immediately
    elif not creds.valid:
        raise YouTubeAuthError(f"Token for {channel.id} is invalid and has no refresh_token; re-run auth")
    return creds


def _build_title(title: str) -> str:
    title = title.strip()
    return f"{title} #shorts" if len(title) <= 92 else title[:100]


def upload_video(channel: ChannelConfig, video_path: Path, title: str, description: str, tags: list[str],
                 hashtags: list[str]) -> str:
    youtube = build("youtube", "v3", credentials=load_credentials(channel), cache_discovery=False)
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
    youtube = build("youtube", "v3", credentials=load_credentials(channel), cache_discovery=False)
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

