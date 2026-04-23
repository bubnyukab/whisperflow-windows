"""WhisperFlow Windows — entry point."""

from __future__ import annotations

import argparse
import logging

from dotenv import load_dotenv

from src.config.settings import Settings
from src.tray.tray_app import TrayApp


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="WhisperFlow Windows")
    parser.add_argument("--dev", action="store_true", help="Enable verbose debug logging")
    return parser.parse_args()


def _setup_logging(debug: bool) -> None:
    """Configure root logger."""
    level = logging.DEBUG if debug else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main() -> None:
    """Application entry point — tray app owns the main thread."""
    load_dotenv()
    args = _parse_args()
    _setup_logging(args.dev)
    settings = Settings.load()
    app = TrayApp(settings)
    app.run()


if __name__ == "__main__":
    main()
