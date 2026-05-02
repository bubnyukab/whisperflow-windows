# WhisperFlow Windows — CLAUDE.md

## Project Overview
A Windows desktop app that replicates Wispr Flow:
- Hold a hotkey to record voice
- Transcribe locally with faster-whisper via RealtimeSTT (no internet, no account)
- Format the transcript with a fine-tuned local GGUF model (fast, private, free)
- Inject the result into the currently focused application via clipboard

Target latency: <700ms end-to-end on GPU, <1.5s on CPU-only machines.

## Core Principles
1. **100% free, no accounts** — local Whisper + local GGUF model, nothing leaves the machine
2. **Privacy first** — audio and transcripts never leave the machine
3. **Low latency** — two-tier formatter skips the LLM for short inputs;
   RealtimeSTT transcribes while you speak rather than after

## Tech Stack
- Python 3.12
- RealtimeSTT (streaming audio + VAD + faster-whisper integration)
- faster-whisper (CTranslate2 Whisper backend)
- pynput (global hotkeys and keyboard input)
- pyperclip (clipboard management)
- pystray + Pillow (system tray)
- tkinter (settings UI, Python stdlib — no extra install)
- torch (CUDA DLL loader for llama-cpp-python on Windows)
- llama-cpp-python (local GGUF inference for fine-tuned formatter)
- PyInstaller (packaging to .exe)

## Environment
- Always use the `.venv` virtual environment in the project root
- Activate with: `.venv\Scripts\activate`
- Install deps with: `.venv\Scripts\pip install -r requirements.txt`
- Run all Python commands through `.venv\Scripts\python`, not system python

## Available MCP Servers
- **Superpowers** — filesystem ops, running scripts, reading logs
- **Context7** — look up latest docs for RealtimeSTT, faster-whisper,
  pystray, pynput. ALWAYS use Context7 before writing any library API
  calls — never rely on memory alone.
- **Playwright** — E2E testing the settings UI
- **GitHub MCP** — create repo, push commits, manage issues and releases

## Key Commands
```powershell
# Install Python deps
.venv\Scripts\pip install -r requirements.txt

# Install llama-cpp-python with CUDA support (required for local formatter on GPU)
.venv\Scripts\pip install --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 llama-cpp-python==0.3.4

# Pre-warm Whisper model (downloads ~75MB on first run for tiny.en)
.venv\Scripts\python -c "from faster_whisper import WhisperModel; WhisperModel('tiny.en')"

# Run in development
.venv\Scripts\python main.py --dev

# Run tests
.venv\Scripts\python -m pytest tests/ -v

# Run latency benchmark
.venv\Scripts\python tools/benchmark.py --model tiny.en --backend local

# Build executable
.venv\Scripts\pyinstaller build/build.spec
```

## Architecture Rules
1. **RealtimeSTT is the primary audio pipeline** — it handles mic capture, VAD,
   and Whisper transcription in one integrated streaming loop. Do not use raw
   sounddevice for the main recording path.
2. **Two-tier formatter is mandatory** — always run FastFormatter first.
   Only call LocalLLMFormatter if word count >= llm_word_threshold (default 4)
   AND the backend is "local".
3. **Whisper model: medium.en by default** — good balance of speed and accuracy.
   Preload at startup, not on first keypress.
4. **Text injection uses clipboard** — save old clipboard content, inject new
   text, restore after 500ms. Handle non-text clipboard content gracefully.
5. **Settings at `~/.whisperflow/config.json`** — never hardcode paths.
6. **Tray app owns the main thread** — all other components are daemon threads.
7. **Never crash silently** — all exceptions must surface as tray notifications.
8. **Local model path** — default is `<project_root>/models/whisperflow-cleaner/model.gguf`.
   The GGUF file must exist for the "local" backend to activate; missing model
   falls back to fast formatter silently.

## Formatter Selection Logic
```
user speaks
→ fast_formatter always runs first (<5ms)
word count < llm_word_threshold (4)?
→ inject fast_formatter result, done
word count >= threshold?
→ check settings.formatter_backend
"fast"  → inject fast_formatter result
"local" → call local_llm_formatter (GGUF), fallback to fast on error
```

## Settings Dataclass Fields
| Field | Default | Notes |
|-------|---------|-------|
| `whisper_model` | `"medium.en"` | |
| `hotkey` | `"ctrl+shift+space"` | |
| `formatter_backend` | `"local"` | `"fast"` or `"local"` |
| `local_model_path` | `<project>/models/whisperflow-cleaner/model.gguf` | Path |
| `llm_word_threshold` | `4` | Skip LLM below this word count |
| `vad_silence_ms` | `400` | |
| `language` | `"en"` | |
| `recording_mode` | `"hold"` | `"hold"` or `"toggle"` |
| `history_max` | `50` | |
| `models_dir` | `~/.whisperflow/models` | Path, reserved for future use |
| `log_dir` | `~/.whisperflow/logs` | Path, reserved for future use |
| `training_pairs_path` | `~/.whisperflow/training_pairs.jsonl` | Path |

Env-var overrides: `HOTKEY`, `WHISPER_MODEL`, `FORMATTER_BACKEND`.

## Training Mode
When Training Mode is toggled on from the tray menu:
- Tray icon shows a small orange dot (RGB 255, 140, 0)
- After each transcription, a `ReviewWindow` popup appears (bottom-right, non-blocking)
- User can accept the pair as-is, edit the cleaned text, or skip
- Auto-saves as-is after 15 seconds if no action is taken
- Pairs saved to `settings.training_pairs_path` (default `~/.whisperflow/training_pairs.jsonl`) in JSONL format: `{"input": ..., "output": ..., "timestamp": ...}`
- "View training pairs" tray menu item shows count, last 10 pairs, and an export button (copies path to clipboard)
- Injection is never delayed — review happens asynchronously while the user keeps dictating

## Latency Targets
- Hotkey press to recording start: <50ms
- Recording end (VAD) to transcript: <200ms GPU / <700ms CPU (tiny.en)
- Short input formatting (fast path): <5ms
- Long input formatting (local GGUF, GPU): ~105ms
- Text injection: <50ms
- Total target: <700ms GPU / <1.5s CPU

## Whisper Model Reference
| Model | Size | RAM | CPU latency | GPU latency |
|-------|------|-----|-------------|-------------|
| tiny.en | 75MB | 1GB | ~400ms | ~50ms |
| base.en | 150MB | 1GB | ~700ms | ~100ms |
| medium.en | 1.5GB | 5GB | ~3s | ~300ms |
| large-v3 | 3GB | 10GB | ~8s | ~600ms |

Default: medium.en

## Error Handling
- Whisper transcription fails: show tray error, do not inject anything
- Local GGUF load fails: fall back to fast_formatter (sticky-cached failure state)
- Text injection fails: show floating toast with text for manual copy
- Never crash silently — always surface errors via tray notification

## Testing Strategy
- Unit test fast_formatter with known inputs and expected outputs
- Unit test LocalLLMFormatter load failure → fast fallback path
- Unit test RealtimeSTT wrapper with mocked audio input
- Unit test TrainingCollector with real filesystem (tmp_path)
- Unit test ReviewWindow actions (accept/skip/correct) without rendering GUI
- Integration test full pipeline with tests/fixtures/sample.wav
- Use Playwright for settings window UI smoke test

## Code Style
- Type hints on all function signatures
- Docstrings on all public methods
- No global mutable state — pass Settings objects
- Max line length: 100 chars
- Use dataclasses for all config and result objects
