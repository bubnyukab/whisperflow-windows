# WhisperFlow Windows — CLAUDE.md

## Project Overview
A Windows desktop app that replicates Wispr Flow:
- Hold a hotkey to record voice
- Transcribe locally with faster-whisper via RealtimeSTT (no internet, no account)
- Format the transcript with a local LLM via Ollama (free, no account)
  OR optionally via Claude API (paid, best quality)
- Inject the result into the currently focused application via clipboard

Target latency: <700ms end-to-end on GPU, <1.5s on CPU-only machines.

## Core Principles
1. **100% free by default** — Ollama + Whisper, no accounts, no API keys required
2. **Privacy first** — audio never leaves the machine; transcript only leaves
   if Claude API formatter is explicitly enabled by the user
3. **Low latency** — two-tier formatter skips the LLM for short inputs;
   RealtimeSTT transcribes while you speak rather than after

## Tech Stack
- Python 3.11+
- RealtimeSTT (streaming audio + VAD + faster-whisper integration)
- faster-whisper (CTranslate2 Whisper backend)
- Ollama (local LLM server — must be installed separately from ollama.com)
- httpx (Ollama API calls)
- anthropic SDK (optional Claude API formatter)
- keyboard (global hotkeys, Windows-compatible)
- pyperclip (clipboard management)
- pystray + Pillow (system tray)
- tkinter (settings UI, Python stdlib — no extra install)
- python-dotenv (optional API key loading)
- PyInstaller (packaging to .exe)

## Environment
- Always use the `.venv` virtual environment in the project root
- Activate with: `source .venv/bin/activate`
- Install deps with: `.venv/bin/pip install -r requirements.txt`
- Run all Python commands through `.venv/bin/python`, not system python3

## Linux System Dependencies
Before installing Python requirements, ensure these are installed:
```bash
sudo apt-get install portaudio19-dev python3-pyaudio python3-dev
```

## Available MCP Servers
- **Superpowers** — filesystem ops, running scripts, reading logs
- **Context7** — look up latest docs for RealtimeSTT, faster-whisper,
  Ollama REST API, anthropic SDK, pystray, keyboard. ALWAYS use Context7
  before writing any library API calls — never rely on memory alone.
- **Playwright** — E2E testing the settings UI
- **GitHub MCP** — create repo, push commits, manage issues and releases

## Key Commands
```bash
# Install Python deps
pip install -r requirements.txt

# Pre-warm Whisper model (downloads ~75MB on first run for tiny.en)
python -c "from faster_whisper import WhisperModel; WhisperModel('tiny.en')"

# Install and start Ollama (done manually by user outside Python)
# Download from https://ollama.com then run:
ollama pull llama3.1:8b     # recommended (4.7GB)
ollama pull phi3:mini        # lighter option for CPU-only (2.3GB)
ollama serve                 # start the local API server

# Run in development
python main.py --dev

# Run tests
python -m pytest tests/ -v

# Run latency benchmark
python tools/benchmark.py --model tiny.en --backend ollama

# Build executable
pyinstaller build/build.spec
```

## Architecture Rules
1. **RealtimeSTT is the primary audio pipeline** — it handles mic capture, VAD,
   and Whisper transcription in one integrated streaming loop. Do not use raw
   sounddevice for the main recording path.
2. **Two-tier formatter is mandatory** — always run FastFormatter first.
   Only call Ollama/Claude if word count >= llm_word_threshold (default 10)
   AND the LLM backend is enabled in settings.
3. **Whisper model: tiny.en by default** — fastest, accurate enough for short
   inputs. Preload at startup, not on first keypress.
4. **Check Ollama at startup** — ping localhost:11434/api/tags on launch.
   If unreachable, show a tray warning but do not crash — fall back to
   fast_formatter only.
5. **Text injection uses clipboard** — save old clipboard content, inject new
   text, restore after 500ms. Handle non-text clipboard content gracefully.
6. **Settings at `~/.whisperflow/config.json`** — never hardcode paths.
7. **API key (if used) from `.env` only** — never written to config.json.
8. **Tray app owns the main thread** — all other components are daemon threads.
9. **Never crash silently** — all exceptions must surface as tray notifications.

## Formatter Selection Logic
user speaks
→ fast_formatter always runs first (<5ms)
word count < llm_word_threshold (10)?
→ inject fast_formatter result, done
word count >= threshold?
→ check settings.formatter_backend
"fast"   → inject fast_formatter result
"ollama" → call ollama_formatter, fallback to fast on error
"claude" → call claude_formatter, fallback to fast on error

## Latency Targets
- Hotkey press to recording start: <50ms
- Recording end (VAD) to transcript: <200ms GPU / <700ms CPU (tiny.en)
- Short input formatting (fast path): <5ms
- Long input formatting (Ollama GPU): <400ms
- Text injection: <50ms
- Total target: <700ms GPU / <1.5s CPU

## Whisper Model Reference
| Model | Size | RAM | CPU latency | GPU latency |
|-------|------|-----|-------------|-------------|
| tiny.en | 75MB | 1GB | ~400ms | ~50ms |
| base.en | 150MB | 1GB | ~700ms | ~100ms |
| medium.en | 1.5GB | 5GB | ~3s | ~300ms |
| large-v3 | 3GB | 10GB | ~8s | ~600ms |

Default: tiny.en

## Ollama Model Reference
| Model | Size | RAM | CPU latency | GPU latency |
|-------|------|-----|-------------|-------------|
| phi3:mini | 2.3GB | 4GB | ~800ms | ~150ms |
| mistral:7b | 4.1GB | 8GB | ~1.5s | ~250ms |
| llama3.1:8b | 4.7GB | 8GB | ~2s | ~350ms |
| gemma2:9b | 5.5GB | 10GB | ~2.5s | ~400ms |

Default: llama3.1:8b

## Error Handling
- Ollama unreachable at startup: warn in tray, fall back to fast_formatter
- Whisper transcription fails: show tray error, do not inject anything
- Ollama call fails or times out: fall back to fast_formatter result + tray warning
- Claude API fails: fall back to fast_formatter result + tray warning
- Text injection fails: show floating toast with text for manual copy
- Never crash silently — always surface errors via tray notification

## Testing Strategy
- Unit test fast_formatter with known inputs and expected outputs
- Unit test ollama_formatter with mocked httpx responses
- Unit test RealtimeSTT wrapper with mocked audio input
- Integration test full pipeline with tests/fixtures/sample.wav
- Use Playwright for settings window UI smoke test

## Code Style
- Type hints on all function signatures
- Docstrings on all public methods
- No global mutable state — pass Settings objects
- Max line length: 100 chars
- Use dataclasses for all config and result objects
