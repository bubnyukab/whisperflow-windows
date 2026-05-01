"""Claude API formatter — optional paid formatter using the Anthropic SDK."""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a voice transcription formatter. The user dictated text using "
    "speech-to-text. Your job:\n"
    "1. Remove filler words (um, uh, like, you know)\n"
    "2. Fix grammar, punctuation, capitalisation\n"
    "3. Infer the user's actual intent — e.g. 'send email to sarah about meeting' "
    "becomes 'Send an email to Sarah about the meeting.'\n"
    "4. Preserve tone (casual stays casual, formal stays formal)\n"
    "5. Format as a list if it sounds like one\n"
    "Return ONLY the formatted text. No explanation, no preamble, no quotes."
)

_MODEL = "claude-haiku-4-5-20251001"


class ClaudeFormatter:
    """Formats voice transcripts via the Claude API."""

    def __init__(self, api_key: str, model: str = _MODEL, timeout: float = 10.0) -> None:
        import anthropic
        self._model = model
        # Only create the client when a key is provided; avoids network calls with empty key.
        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout) if api_key else None

    def format(self, raw_transcript: str, context_hint: str = "") -> str:
        """Send transcript to Claude API and return cleaned text.

        Falls back to raw_transcript on any error — never raises.
        """
        if self._client is None:
            return raw_transcript
        system = _SYSTEM_PROMPT
        if context_hint:
            system = f"{system}\n\nContext: {context_hint}"
        try:
            message = self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=system,
                messages=[{"role": "user", "content": raw_transcript}],
            )
            text = message.content[0].text.strip()
            return text or raw_transcript
        except Exception:
            log.exception("ClaudeFormatter.format failed")
            return raw_transcript

    def is_available(self) -> bool:
        """Return True if a client was constructed (no network call)."""
        return self._client is not None
