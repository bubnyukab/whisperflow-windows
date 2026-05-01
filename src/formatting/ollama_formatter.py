"""Ollama LLM formatter — calls local Ollama API to clean up transcripts."""

from __future__ import annotations

import logging

import httpx

log = logging.getLogger(__name__)

# Completion-style prompt — model fills in "Output:" lines, never enters chat/answer mode.
# Rules from the user-provided prompt; framing keeps the model in pattern-completion mode.
_COMPLETION_PROMPT_TEMPLATE = """\
You are a dictation post-processor. You receive raw speech-to-text output and return clean text ready to be typed into an application.

Your job:
- Remove filler words (um, uh, you know, like) unless they carry meaning.
- Fix spelling, grammar, and punctuation errors.
- Preserve the speaker's intent, tone, and meaning exactly.

Output rules:
- Return ONLY the cleaned transcript text, nothing else.
- If the transcription is empty or blank, return exactly: EMPTY
- Do not add words or content that are not in the transcription.
- Do not change the meaning of what was said.

Each block below shows a raw Input and the correct Output. Follow this pattern exactly.

Input: hey um so i just wanted to like follow up on the meating from yesterday i think we should definately move the dedline to next friday becuz the desine team still needs more time to finish the mock ups and um yeah let me know if that works for you ok thanks
Output: Hey, I just wanted to follow up on the meeting from yesterday. I think we should definitely move the deadline to next Friday because the design team still needs more time to finish the mockups. Let me know if that works for you. Thanks.

Input: what is one plus three
Output: What is one plus three?

Input: um explain to me how neural networks work
Output: Explain to me how neural networks work.

Input: what's the capital of france
Output: What's the capital of France?

Input: can you uh write me an email to like john about the meeting on friday
Output: Can you write me an email to John about the meeting on Friday?

Input: {transcript}
Output:"""


class OllamaFormatter:
    """Formats voice transcripts via a local Ollama instance."""

    def __init__(self, url: str, model: str, timeout: float = 15.0) -> None:
        self._url = url.rstrip("/")
        self._model = model
        self._timeout = timeout

    def format(self, raw_transcript: str, context_hint: str = "") -> str:
        """Send transcript to Ollama and return cleaned text.

        Uses /api/generate with a completion prompt so the model fills in a pattern
        rather than entering chat/answer mode. Falls back to raw_transcript on any error.
        """
        prompt = _COMPLETION_PROMPT_TEMPLATE.format(transcript=raw_transcript)
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 200},
        }
        try:
            resp = httpx.post(
                f"{self._url}/api/generate",
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "").strip()
            # Take only the first line — model must not continue the pattern
            result = raw.splitlines()[0].strip() if raw else ""
            return result or raw_transcript
        except Exception:
            log.exception("OllamaFormatter.format failed")
            return raw_transcript

    def is_available(self) -> bool:
        """Return True if the Ollama server responds at /api/tags."""
        try:
            resp = httpx.get(f"{self._url}/api/tags", timeout=2.0)
            return resp.status_code == 200
        except Exception:
            return False
