"""Application settings — loaded from ~/.whisperflow/config.json, API key from .env."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal, Optional

FormatterBackend = Literal["fast", "ollama", "claude"]

SETTINGS_PATH = Path.home() / ".whisperflow" / "config.json"


@dataclass
class Settings:
    """All user-configurable settings for WhisperFlow."""

    whisper_model: str = "tiny.en"
    hotkey: str = "win+shift+space"
    formatter_backend: FormatterBackend = "ollama"
    ollama_model: str = "llama3.1:8b"
    ollama_base_url: str = "http://localhost:11434"
    llm_word_threshold: int = 10
    inject_delay_ms: int = 500

    def get_anthropic_api_key(self) -> Optional[str]:
        """Return ANTHROPIC_API_KEY from environment — never stored in config.json."""
        return os.getenv("ANTHROPIC_API_KEY") or None

    def save(self) -> None:
        """Persist settings to ~/.whisperflow/config.json."""
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls) -> "Settings":
        """Load settings from disk, returning defaults for any missing or invalid file."""
        if not SETTINGS_PATH.exists():
            return cls()
        try:
            data = json.loads(SETTINGS_PATH.read_text())
            valid = {f.name for f in fields(cls)}
            return cls(**{k: v for k, v in data.items() if k in valid})
        except Exception:
            return cls()
