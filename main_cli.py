"""Operational CLI for dark-factory-os.

    python main_cli.py --init-assets              # synth SFX/BGM/overlay + create PostgreSQL tables
    python main_cli.py --auth-youtube             # OAuth for channels missing a YouTube token
    python main_cli.py --auth-tiktok              # log in to TikTok profiles missing a session
    python main_cli.py --test-render nicho_1      # full 16-scene video into ./output_test/nicho_1/, no upload
    python main_cli.py --run-channel nicho_1      # full pipeline incl. YouTube + TikTok + cleanup + alerts
    python main_cli.py --report-analytics         # 7-day table to stdout + Telegram
    python main_cli.py --init-mascots [--channel nicho_2] [--force]   # fixed mascot pose library
"""
from __future__ import annotations

import argparse
import logging
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="main_cli.py", description="dark-factory-os operations")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--init-assets", action="store_true", help="generate SFX/BGM/overlay and init DB tables")
    group.add_argument("--auth-youtube", action="store_true", help="interactive YouTube OAuth for missing tokens")
    group.add_argument("--auth-tiktok", action="store_true", help="interactive TikTok login for missing sessions")
    group.add_argument("--test-render", metavar="CHANNEL_ID", help="render one video locally, no upload")
    group.add_argument("--run-channel", metavar="CHANNEL_ID", help="full production + publishing pipeline")
    group.add_argument("--report-analytics", action="store_true", help="7-day analytics report (stdout + Telegram)")
    group.add_argument("--init-mascots", action="store_true",
                       help="generate the fixed mascot pose library for 3D character channels")
    parser.add_argument("--channel", help="--init-mascots: only this channel id")
    parser.add_argument("--force", action="store_true",
                        help="auth: redo existing; run: ignore the 20h guard; mascots: regenerate the library")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.init_assets:
        import generate_sfx
        from src.database import init_db

        generate_sfx.main(force=args.force)
        init_db()
        print("Assets ready and table 'videos_history' initialized.")

    elif args.auth_youtube:
        import auth_youtube

        auth_youtube.main(["--force"] if args.force else [])

    elif args.auth_tiktok:
        import auth_tiktok

        auth_tiktok.main(["--force"] if args.force else [])

    elif args.test_render:
        from src import database as db
        from src.config_loader import OUTPUT_TEST_DIR
        from src.tasks import run_pipeline

        result = run_pipeline(args.test_render, force=True, work_root=OUTPUT_TEST_DIR / args.test_render,
                              use_db=db.db_available())
        print(f"\nVideo: {result.final}\nDuration: {result.duration:.2f}s\nTitle: {result.script.title}"
              f"\nHook: {result.script.hook_archetype}")

    elif args.run_channel:
        from src.tasks import run_channel_now

        result = run_channel_now(args.run_channel, force=args.force)
        if result is None:
            print("Skipped (already produced in the last 20h or channel disabled). Use --force to override.")
            return 1
        print(f"\nDone: {result.final} ({result.duration:.2f}s)")

    elif args.report_analytics:
        from src.analytics_reporter import run_report

        markdown, path = run_report(send=True)
        print(markdown)
        print(f"\nSaved to {path}")

    elif args.init_mascots:
        import asyncio

        from src.config_loader import load_channels
        from src.engine import mascot
        from src.engine.media_fetcher import build_mascot_library

        channels = [c for c in load_channels().values()
                    if c.is_character_channel and (not args.channel or c.id == args.channel)]
        if not channels:
            print("No 3D character channel matches (needs style=3d_character and use_rembg=true).")
            return 1
        for channel in channels:
            asyncio.run(build_mascot_library(channel, force=args.force))
            print(f"[{channel.id}] poses: {', '.join(mascot.pose_names(channel))}"
                  f"\n  review: {mascot.library_dir(channel) / '_preview.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
