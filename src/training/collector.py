"""Collects raw→cleaned transcript pairs for fine-tuning."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)


class TrainingCollector:
    """Appends (raw, cleaned) pairs to a JSONL file for future fine-tuning."""

    def __init__(self, output_path: Path) -> None:
        self._path = output_path

    def save_pair(self, raw: str, cleaned: str) -> None:
        """Append one input/output pair with a timestamp to the JSONL file."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "input": raw,
                "output": cleaned,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except Exception:
            log.warning("Failed to save training pair", exc_info=True)

    def load_pairs(self) -> list[dict]:
        """Return all pairs as a list of dicts; returns [] on any error."""
        if not self._path.exists():
            return []
        pairs: list[dict] = []
        try:
            for line in self._path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        pairs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception:
            log.warning("Failed to read training pairs", exc_info=True)
        return pairs
