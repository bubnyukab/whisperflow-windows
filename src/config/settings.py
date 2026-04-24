"""Application settings — loaded from ~/.whisperflow/config.json, API key from .env."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import httpx
from dotenv import load_dotenv

SETTINGS_PATH = Path.home() / ".whisperflow" / "config.json"

_PATH_FIELDS = {"models_dir", "log_dir"}


@dataclass
class Settings:
    """All user-configurable settings for WhisperFlow."""

    whisper_model: str = "tiny.en"
    hotkey: str = "win+shift+space"
    formatter_backend: str = "ollama"    # "fast" | "ollama" | "claude"
    ollama_model: str = "llama3.1:8b"
    ollama_url: str = "http://localhost:11434"
    ollama_timeout: float = 15.0
    anthropic_api_key: str = ""          # populated at runtime from env, never stored
    llm_word_threshold: int = 10
    vad_silence_ms: int = 400
    language: str = "en"
    recording_mode: str = "hold"         # "hold" | "toggle"
    history_max: int = 50
    models_dir: Path = field(default_factory=lambda: Path.home() / ".whisperflow" / "models")
    log_dir: Path = field(default_factory=lambda: Path.home() / ".whisperflow" / "logs")


def load_settings() -> Settings:
    """Read ~/.whisperflow/config.json; API key from .env/environment only."""
    load_dotenv(override=False)

    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    defaults = Settings()

    if not SETTINGS_PATH.exists():
        save_settings(defaults)
        return Settings(anthropic_api_key=api_key)

    try:
        data = json.loads(SETTINGS_PATH.read_text())
        valid = {f.name for f in fields(Settings) if f.name != "anthropic_api_key"}
        filtered: dict = {k: v for k, v in data.items() if k in valid}
        for name in _PATH_FIELDS:
            if name in filtered:
                filtered[name] = Path(filtered[name])
        return Settings(**filtered, anthropic_api_key=api_key)
    except Exception:
        return Settings(anthropic_api_key=api_key)


def save_settings(settings: Settings) -> None:
    """Write settings to config.json, excluding anthropic_api_key."""
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw = asdict(settings)
    raw.pop("anthropic_api_key", None)
    for name in _PATH_FIELDS:
        if name in raw:
            raw[name] = str(raw[name])
    SETTINGS_PATH.write_text(json.dumps(raw, indent=2))


def check_ollama(url: str) -> bool:
    """Return True if Ollama is reachable at {url}/api/tags; never raises."""
    try:
        httpx.get(f"{url}/api/tags", timeout=2.0)
        return True
    except Exception:
        return False
