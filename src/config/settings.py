"""Application settings — loaded from ~/.whisperflow/config.json."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

log = logging.getLogger(__name__)

SETTINGS_PATH = Path.home() / ".whisperflow" / "config.json"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

_PATH_FIELDS = {"models_dir", "log_dir", "local_model_path", "training_pairs_path"}

# Env vars that override the corresponding Settings fields so the user can
# override without editing config.json.
_ENV_FIELD_MAP: dict[str, str] = {
    "HOTKEY": "hotkey",
    "WHISPER_MODEL": "whisper_model",
    "FORMATTER_BACKEND": "formatter_backend",
}


@dataclass
class Settings:
    """All user-configurable settings for WhisperFlow."""

    whisper_model: str = "medium.en"
    hotkey: str = "ctrl+shift+space"
    formatter_backend: str = "local"     # "fast" | "local"
    local_model_path: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "models" / "whisperflow-cleaner" / "model.gguf"
    )
    llm_word_threshold: int = 4
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
    """Overlay HOTKEY / WHISPER_MODEL / FORMATTER_BACKEND from environment."""
    overrides: dict = {}
    for env_var, field_name in _ENV_FIELD_MAP.items():
        val = os.getenv(env_var)
        if val:
            overrides[field_name] = val
    return replace(settings, **overrides) if overrides else settings


def load_settings() -> Settings:
    """Read ~/.whisperflow/config.json; env vars overlay config."""
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)

    defaults = Settings()

    if not SETTINGS_PATH.exists():
        save_settings(defaults)
        return _apply_env_overrides(defaults)

    try:
        data = json.loads(SETTINGS_PATH.read_text())
        valid = {f.name for f in fields(Settings)}
        filtered: dict = {k: v for k, v in data.items() if k in valid}
        for name in _PATH_FIELDS:
            if name in filtered:
                filtered[name] = Path(filtered[name])
        return _apply_env_overrides(Settings(**filtered))
    except Exception:
        log.warning("Failed to parse %s — falling back to defaults", SETTINGS_PATH, exc_info=True)
        return _apply_env_overrides(defaults)


def save_settings(settings: Settings) -> None:
    """Write settings to config.json."""
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw = asdict(settings)
    for name in _PATH_FIELDS:
        if name in raw:
            raw[name] = str(raw[name])
    SETTINGS_PATH.write_text(json.dumps(raw, indent=2))


