"""Application settings — loaded from ~/.whisperflow/config.json, API key from .env."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

import httpx
from dotenv import load_dotenv

SETTINGS_PATH = Path.home() / ".whisperflow" / "config.json"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

_PATH_FIELDS = {"models_dir", "log_dir", "local_model_path", "training_pairs_path"}

# Env vars that override the corresponding Settings fields (lower priority than config.json is
# intentionally reversed here: env vars win over config.json so the user can override without
# editing the file — same semantics as ANTHROPIC_API_KEY).
_ENV_FIELD_MAP: dict[str, str] = {
    "HOTKEY": "hotkey",
    "WHISPER_MODEL": "whisper_model",
    "FORMATTER_BACKEND": "formatter_backend",
    "OLLAMA_MODEL": "ollama_model",
}


@dataclass
class Settings:
    """All user-configurable settings for WhisperFlow."""

    whisper_model: str = "tiny.en"
    hotkey: str = "win+shift+space"
    formatter_backend: str = "ollama"    # "fast" | "ollama" | "claude" | "local"
    ollama_model: str = "llama3.1:8b"
    ollama_url: str = "http://localhost:11434"
    ollama_timeout: float = 15.0
    anthropic_api_key: str = ""          # populated at runtime from env, never stored
    local_model_path: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "models" / "whisperflow-cleaner" / "model.gguf"
    )
    llm_word_threshold: int = 10
    vad_silence_ms: int = 400
    language: str = "en"
    recording_mode: str = "hold"         # "hold" | "toggle"
    history_max: int = 50
    models_dir: Path = field(default_factory=lambda: Path.home() / ".whisperflow" / "models")
    log_dir: Path = field(default_factory=lambda: Path.home() / ".whisperflow" / "logs")
    training_pairs_path: Path = field(
        default_factory=lambda: Path.home() / ".whisperflow" / "training_pairs.jsonl"
    )


def _apply_env_overrides(settings: Settings) -> Settings:
    """Overlay HOTKEY / WHISPER_MODEL / FORMATTER_BACKEND / OLLAMA_MODEL from environment."""
    overrides: dict = {}
    for env_var, field_name in _ENV_FIELD_MAP.items():
        val = os.getenv(env_var)
        if val:
            overrides[field_name] = val
    return replace(settings, **overrides) if overrides else settings


def load_settings() -> Settings:
    """Read ~/.whisperflow/config.json; env vars overlay config; API key from env only."""
    load_dotenv(override=False)

    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    defaults = Settings()

    if not SETTINGS_PATH.exists():
        save_settings(defaults)
        return _apply_env_overrides(Settings(anthropic_api_key=api_key))

    try:
        data = json.loads(SETTINGS_PATH.read_text())
        valid = {f.name for f in fields(Settings) if f.name != "anthropic_api_key"}
        filtered: dict = {k: v for k, v in data.items() if k in valid}
        for name in _PATH_FIELDS:
            if name in filtered:
                filtered[name] = Path(filtered[name])
        return _apply_env_overrides(Settings(**filtered, anthropic_api_key=api_key))
    except Exception:
        return _apply_env_overrides(Settings(anthropic_api_key=api_key))


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
        resp = httpx.get(f"{url}/api/tags", timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False
