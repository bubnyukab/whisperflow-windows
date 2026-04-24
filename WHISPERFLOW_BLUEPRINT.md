# WhisperFlow Windows — Project Blueprint

> A local, privacy-first Wispr Flow clone for Windows.
> Voice → local Whisper transcription → local LLM formatting → text injected into any app.
> 100% free. 100% offline. No accounts. No telemetry.

---

## Latency Goal

Wispr Flow achieves ~700ms end-to-end using cloud APIs on fast servers.
We can match or beat this locally — but it depends on your hardware.

### Where the time goes

| Stage | CPU only | With GPU (any NVIDIA) |
|---|---|---|
| VAD silence detection | +150–300ms | +150–300ms |
| Whisper `tiny.en` transcription | 400–700ms | 50–120ms |
| LLM formatting (phi3:mini via Ollama) | 1500–3000ms | 200–400ms |
| Text injection | ~50ms | ~50ms |
| **Total (naive)** | **2–4 seconds ❌** | **450–870ms ✅** |
| **Total (optimized — see below)** | **~700ms–1.5s ✅** | **~300–500ms ✅** |

The optimized path gets CPU-only machines into an acceptable range by skipping
the LLM for short inputs (which covers ~80% of real usage).

### Optimization Stack (apply all of these)

**1. Use `tiny.en` Whisper model**
5× faster than `base.en`, still very accurate for short voice inputs (5–15 words).
The accuracy difference is barely noticeable for typical dictation use.

**2. Two-tier formatter — biggest win on CPU**
Skip the LLM entirely for short inputs. A fast regex pass handles the majority
of real-world cases in under 5ms:

```python
import re

FILLERS = r'\b(um+|uh+|like|you know|so|basically|literally|actually)\b'

def fast_format(text: str) -> str:
    text = re.sub(FILLERS, '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text[0].upper() + text[1:] if text else text
    if text and text[-1] not in '.!?,;:':
        text += '.'
    return text

def format_text(text: str, use_llm: bool = True) -> str:
    # Only call the LLM for longer, more complex inputs
    if len(text.split()) < 10 or not use_llm:
        return fast_format(text)
    return ollama_format(text)  # ~200–400ms on GPU, ~1.5s on CPU
```

**3. Aggressive VAD silence threshold**
Default VAD waits 1–1.5s of silence before stopping. Drop it to 400ms.
Most people naturally pause ~300ms between utterances, so 400ms catches the
end of speech reliably without feeling slow.

**4. Use `RealtimeSTT` instead of batch Whisper**
This is the biggest architectural upgrade. Instead of record-then-transcribe,
RealtimeSTT transcribes while you are still speaking using a fast small model,
then corrects after silence is detected. Fires ~400ms after you stop speaking.

```python
from RealtimeSTT import AudioToTextRecorder

recorder = AudioToTextRecorder(
    model="tiny.en",
    language="en",
    silero_sensitivity=0.4,             # aggressive VAD
    post_speech_silence_duration=0.4,
    on_realtime_transcription_update=lambda t: print(f"interim: {t}")
)

final_text = recorder.text()  # fires ~400ms after you stop speaking
```

**5. Pre-warm everything at startup**
Both Whisper and Ollama have a cold-start cost (loading model weights into RAM/VRAM).
Load both before the first keypress so that cost is paid once at launch, not per use.

### Realistic expectations

| Hardware | Best achievable latency | Strategy |
|---|---|---|
| CPU only | ~700ms–1.5s | tiny.en + two-tier (LLM skipped for short inputs) |
| Any NVIDIA GPU | ~400–700ms | tiny.en + phi3:mini on GPU + RealtimeSTT |
| RTX 3080+ | ~200–400ms | medium.en + llama3.1:8b on GPU |

---

## LLM Formatter Options (all free, all local)

The formatting step is what turns "um send email to john about the meeting"
into "Send an email to John about the meeting." You have three free options
plus one optional paid option.

### Option 1: Ollama (recommended)
Install from [ollama.com](https://ollama.com), pull a model, done.
Runs a local REST API on `localhost:11434`. Easy to switch models.
No Python dependencies beyond `httpx`.

```python
import httpx

response = httpx.post(
    "http://localhost:11434/api/generate",
    json={"model": "llama3.1:8b", "prompt": full_prompt, "stream": False},
    timeout=30.0
)
return response.json()["response"].strip()
```

**Model recommendations:**

| Model | Size | RAM needed | Quality | Speed (CPU) | Speed (GPU) |
|---|---|---|---|---|---|
| `phi3:mini` | 2.3 GB | 4 GB | Good for basic cleanup | ~800ms | ~150ms |
| `mistral:7b` | 4.1 GB | 8 GB | Very good, fast | ~1.5s | ~250ms |
| `llama3.1:8b` | 4.7 GB | 8 GB | Excellent — recommended | ~2s | ~350ms |
| `gemma2:9b` | 5.5 GB | 10 GB | Best quality in this range | ~2.5s | ~400ms |
| `llama3.1:70b` | 40 GB | 64 GB | Claude-level quality | very slow | needs A100 |

For most people: `llama3.1:8b` on GPU, `phi3:mini` on CPU-only.

### Option 2: LM Studio
Similar to Ollama but has a graphical model browser and downloader.
Exposes an OpenAI-compatible local server at `localhost:1234`.
Good if you want a visual model manager.

### Option 3: llama-cpp-python
Runs GGUF models directly inside your Python process — no external server needed.
More complex setup but truly self-contained (useful for packaging into a single `.exe`).

```python
from llama_cpp import Llama

llm = Llama(model_path="./models/llama-3.1-8b.Q4_K_M.gguf", n_gpu_layers=-1)
output = llm(prompt, max_tokens=256)
return output["choices"][0]["text"].strip()
```

### Option 4: Claude API (paid, best quality)
If you have an Anthropic API key and want the highest formatting quality,
the formatter backend is pluggable — Claude API can be selected in Settings.
Uses `claude-sonnet-4-20250514`. Only the transcript text is sent, never audio.

---

## Tech Stack

| Layer | Tool | Why |
|---|---|---|
| Language | Python 3.11+ | Best ecosystem for audio + AI glue |
| Audio pipeline | `RealtimeSTT` | Streaming transcription, built-in VAD, lowest latency |
| Whisper backend | `faster-whisper` (via RealtimeSTT) | Local, no login, GPU optional |
| Default Whisper model | `tiny.en` | Fastest, accurate enough for short inputs |
| LLM formatter (free) | Ollama + `llama3.1:8b` or `phi3:mini` | Local, free, no account |
| LLM formatter (paid) | Claude API (optional) | Best quality, needs API key |
| Fast formatter | regex pipeline | Handles short inputs in <5ms, no LLM needed |
| Global hotkey | `keyboard` library | Works across all Windows apps |
| Text injection | `pyperclip` + `keyboard` | Clipboard paste into any focused app |
| System tray | `pystray` + `Pillow` | Background process with icon |
| Settings UI | `tkinter` | Simple config window, no extra deps |
| Config storage | `json` + `.env` | Preferences + optional API key |
| Packaging | `PyInstaller` | Single `.exe` distribution |

---

## Project File Structure

```
whisperflow/
├── CLAUDE.md                      ← AI assistant instructions (copy from below)
├── README.md
├── requirements.txt
├── .env.example
├── main.py                        ← Entry point (tray app)
├── src/
│   ├── __init__.py
│   ├── audio/
│   │   ├── __init__.py
│   │   └── realtime_recorder.py   ← RealtimeSTT wrapper
│   ├── transcription/
│   │   ├── __init__.py
│   │   └── whisper_engine.py      ← faster-whisper (batch fallback + CLI)
│   ├── formatting/
│   │   ├── __init__.py            ← Formatter router + two-tier logic
│   │   ├── fast_formatter.py      ← Regex-based, <5ms, no LLM
│   │   ├── ollama_formatter.py    ← Ollama local LLM
│   │   └── claude_formatter.py    ← Claude API (optional paid)
│   ├── injection/
│   │   ├── __init__.py
│   │   └── text_injector.py       ← Clipboard + Ctrl+V
│   ├── hotkey/
│   │   ├── __init__.py
│   │   └── listener.py            ← Global hotkey handler
│   ├── tray/
│   │   ├── __init__.py
│   │   ├── tray_app.py            ← pystray system tray
│   │   └── settings_ui.py         ← tkinter settings window
│   └── config/
│       ├── __init__.py
│       └── settings.py            ← Load/save user preferences
├── assets/
│   └── icon.png
├── tests/
│   ├── __init__.py
│   ├── fixtures/
│   │   └── sample.wav             ← 3s test audio file
│   ├── test_audio.py
│   ├── test_transcription.py
│   └── test_formatting.py
├── tools/
│   └── benchmark.py               ← Latency benchmarking script
└── build/
    └── build.spec
```

---

## CLAUDE.md

Copy this file verbatim to the root of your project.
Claude Code reads it automatically at the start of every session.

```markdown
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
```
user speaks
  → fast_formatter always runs first (<5ms)
      word count < llm_word_threshold (10)?
        → inject fast_formatter result, done
      word count >= threshold?
        → check settings.formatter_backend
            "fast"   → inject fast_formatter result
            "ollama" → call ollama_formatter, fallback to fast on error
            "claude" → call claude_formatter, fallback to fast on error
```

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
```

---

## Claude Code Prompts

Run these **in order**. Each builds on the previous one.
Commit to GitHub after prompts 2, 5, 7, 9, 11, and 12.

---

### Prompt 1 — Repo + scaffold

```
Use the GitHub MCP to create a new public repository called "whisperflow-windows"
with description "Local, privacy-first Wispr Flow clone for Windows — free, offline, no accounts".

Then scaffold the full project structure. Create every file listed (even if
empty with just a docstring) so the import tree is complete from the start.

Structure:
whisperflow/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .env.example
├── main.py
├── src/__init__.py
├── src/audio/__init__.py
├── src/audio/realtime_recorder.py
├── src/transcription/__init__.py
├── src/transcription/whisper_engine.py
├── src/formatting/__init__.py
├── src/formatting/fast_formatter.py
├── src/formatting/ollama_formatter.py
├── src/formatting/claude_formatter.py
├── src/injection/__init__.py
├── src/injection/text_injector.py
├── src/hotkey/__init__.py
├── src/hotkey/listener.py
├── src/tray/__init__.py
├── src/tray/tray_app.py
├── src/tray/settings_ui.py
├── src/config/__init__.py
├── src/config/settings.py
├── assets/.gitkeep
├── tests/__init__.py
├── tests/fixtures/.gitkeep
├── tests/test_audio.py
├── tests/test_transcription.py
├── tests/test_formatting.py
├── tools/benchmark.py
└── build/build.spec

requirements.txt must include:
RealtimeSTT
faster-whisper
httpx
anthropic
keyboard
pyperclip
pystray
Pillow
python-dotenv
pytest
pyinstaller

Use Context7 to look up the latest stable versions of RealtimeSTT and
faster-whisper before writing requirements.txt — pin to current versions.

.env.example:
# Optional — only needed if using Claude API formatter
ANTHROPIC_API_KEY=your_key_here

# Whisper model: tiny.en / base.en / medium.en / large-v3
WHISPER_MODEL=tiny.en

# Global hotkey
HOTKEY=win+shift+space

# Formatter backend: fast / ollama / claude
FORMATTER_BACKEND=ollama

# Ollama model
OLLAMA_MODEL=llama3.1:8b

Commit everything with message "chore: initial project scaffold".
```

---

### Prompt 2 — Settings & config system

```
Implement src/config/settings.py.

Use Context7 to look up the python-dotenv API before writing.

Settings dataclass fields and defaults:
  whisper_model: str = "tiny.en"
  hotkey: str = "win+shift+space"
  formatter_backend: str = "ollama"    # "fast" | "ollama" | "claude"
  ollama_model: str = "llama3.1:8b"
  ollama_url: str = "http://localhost:11434"
  ollama_timeout: float = 15.0
  anthropic_api_key: str = ""          # from .env only, never stored in json
  llm_word_threshold: int = 10         # skip LLM for inputs shorter than this
  vad_silence_ms: int = 400            # ms of silence before auto-stop
  language: str = "en"
  recording_mode: str = "hold"         # "hold" | "toggle"
  history_max: int = 50
  models_dir: Path = ~/.whisperflow/models
  log_dir: Path = ~/.whisperflow/logs

Implement:
  load_settings() → Settings
    Reads ~/.whisperflow/config.json, falls back to .env and defaults.
    Creates config file and dirs if missing.
    API key is ONLY read from .env / environment, never from json.

  save_settings(settings: Settings) → None
    Writes to config.json, excluding the api_key field.

  check_ollama(url: str) → bool
    GET {url}/api/tags with 2s timeout.
    Returns True if reachable, False otherwise. Never raises.

Write unit tests in tests/test_settings.py using tmp_path fixture.
Run the tests. Commit to GitHub.
```

---

### Prompt 3 — Fast regex formatter

```
Implement src/formatting/fast_formatter.py.

Zero external dependencies. Must return in under 5ms.

Filler words to strip (whole-word, case-insensitive):
  um, umm, uh, uhh, like, you know, so, basically,
  literally, actually, right, I mean, kind of, sort of

FastFormatter class:
  format(text: str) → str
    1. Strip filler words
    2. Collapse multiple spaces to single space
    3. Trim whitespace
    4. Capitalise first letter
    5. Append period if no terminal punctuation (.!?,;:) exists
    6. Return result

No side effects on import — no model loading, no network calls.

Write thorough tests in tests/test_formatting.py:
  "um send an email to john"
    → "Send an email to john."
  "so basically i think we should uh meet tomorrow"
    → "I think we should meet tomorrow."
  "Hello, how are you?"
    → "Hello, how are you?"   (unchanged, already has punctuation)
  "" → ""
  "um uh like" → ""  (empty after stripping — no period added to empty string)

Run tests. All must pass.
```

---

### Prompt 4 — Ollama LLM formatter

```
Implement src/formatting/ollama_formatter.py.

Use Context7 to look up the Ollama REST API (/api/generate endpoint) before writing.
Use httpx for HTTP calls.

OllamaFormatter class:
  Constructor: url: str, model: str, timeout: float = 15.0

  format(raw_transcript: str, context_hint: str = "") → str
    System prompt:
      "You are a voice transcription formatter. The user dictated text using
       speech-to-text. Your job:
       1. Remove filler words (um, uh, like, you know)
       2. Fix grammar, punctuation, capitalisation
       3. Infer the user's actual intent — e.g. 'send email to sarah about meeting'
          becomes 'Send an email to Sarah about the meeting.'
       4. Preserve tone (casual stays casual, formal stays formal)
       5. Format as a list if it sounds like one
       Return ONLY the formatted text. No explanation, no preamble, no quotes."
    Append context_hint to system prompt if provided.
    POST {url}/api/generate with model, system + user prompt, stream=False.
    Return response text stripped of whitespace.
    On any exception: log the error and return raw_transcript unchanged. NEVER raise.

  is_available() → bool
    GET {url}/api/tags, return True if 200 OK, else False.

Tests (mock httpx.post):
  - Happy path: formatted text is returned
  - Timeout: raw_transcript returned unchanged
  - Connection error: raw_transcript returned unchanged
  - context_hint present in the prompt body

Run tests.
```

---

### Prompt 5 — Claude API formatter (optional)

```
Implement src/formatting/claude_formatter.py.

Use Context7 to look up the anthropic Python SDK Messages API before writing.

ClaudeFormatter class:
  Constructor: api_key: str, model: str = "claude-sonnet-4-20250514", timeout: float = 10.0

  format(raw_transcript: str, context_hint: str = "") → str
    Same system prompt as OllamaFormatter.
    On any exception: log and return raw_transcript unchanged.

  is_available() → bool
    Returns True if api_key is non-empty string, else False.
    Does NOT make a network call — just validates key exists.

This formatter is optional. Only instantiated when settings.formatter_backend == "claude".

Write tests mocking anthropic.Anthropic client.
Run tests. Commit everything to GitHub.
```

---

### Prompt 6 — Formatter router (two-tier logic)

```
Implement src/formatting/__init__.py as the formatter router.

create_formatter(settings: Settings) → object
  Returns the appropriate formatter instance:
    "fast"   → FastFormatter()
    "ollama" → OllamaFormatter(settings.ollama_url, settings.ollama_model, settings.ollama_timeout)
    "claude" → ClaudeFormatter(settings.anthropic_api_key)
  Logs which formatter backend is active.

format_text(text: str, settings: Settings, context_hint: str = "") → str
  Implements the two-tier logic:
    1. fast_result = FastFormatter().format(text)   (always runs, <5ms)
    2. If len(text.split()) < settings.llm_word_threshold:
         return fast_result                          (skip LLM)
    3. If settings.formatter_backend == "fast":
         return fast_result
    4. Otherwise:
         result = llm_formatter.format(text, context_hint)
         return result if result else fast_result    (fallback on empty)

Write a test verifying that inputs with fewer than llm_word_threshold words
never trigger a call to the LLM formatter. Run tests.
```

---

### Prompt 7 — RealtimeSTT audio pipeline

```
Implement src/audio/realtime_recorder.py.

Use Context7 to look up the RealtimeSTT library (AudioToTextRecorder) before writing.
Pay attention to: model, language, silero_sensitivity, post_speech_silence_duration,
on_realtime_transcription_update, on_recording_start, on_recording_stop.

RealtimeRecorder class:
  Constructor: settings: Settings, on_transcript: Callable[[str], None]
    on_transcript is called with the final transcript string when VAD stops.

  start() → None
    Initialises AudioToTextRecorder:
      model = settings.whisper_model
      language = settings.language
      post_speech_silence_duration = settings.vad_silence_ms / 1000
      silero_sensitivity = 0.4
    Runs in a background daemon thread.
    Calls on_transcript(text) when a transcription is ready.

  stop() → None  — stops the recorder cleanly
  is_ready: bool property — True after successful initialisation

Also implement src/transcription/whisper_engine.py as a batch fallback:
  WhisperEngine.transcribe(wav_path: str) → str
  Used only in tests and CLI mode. Not used in the main live pipeline.

Write tests mocking AudioToTextRecorder.
Run tests. Commit to GitHub.
```

---

### Prompt 8 — Text injection

```
Implement src/injection/text_injector.py.

Use Context7 to look up pyperclip and the keyboard library APIs.

TextInjector class:
  inject(text: str) → bool
    1. Save clipboard: old = pyperclip.paste() — wrap in try/except
       (clipboard may contain non-text, images, files)
    2. pyperclip.copy(text)
    3. Sleep 50ms
    4. keyboard.send("ctrl+v")
    5. Sleep 500ms
    6. Restore old clipboard if it was text; skip restore if it was non-text
    7. Return True on success, False on any error

  show_fallback_toast(text: str) → None
    Borderless tkinter Toplevel window when injection fails:
    - Positioned bottom-right of primary screen
    - Scrollable label with the transcript text
    - "Copy" button that runs pyperclip.copy(text)
    - Auto-closes after 15 seconds
    - Always on top (wm_attributes -topmost true)

Write tests mocking pyperclip and keyboard.
```

---

### Prompt 9 — Global hotkey listener

```
Implement src/hotkey/listener.py.

Use Context7 to look up the keyboard library hotkey and on_release APIs.

HotkeyListener class:
  Constructor: hotkey: str, on_press: Callable, on_release: Callable, mode: str = "hold"

  HOLD mode: on_press fires on keydown, on_release fires on keyup
  TOGGLE mode: on_press fires on first keydown, on_release fires on next keydown

  start() → registers hotkey in a background thread, does not block
  stop() → unregisters hotkey cleanly
  is_listening: bool property

  Debounce: ignore re-fires within 150ms of previous event.
  All callback exceptions are caught and logged — never crash the listener.

Write tests using keyboard library mocking.
```

---

### Prompt 10 — System tray + settings UI

```
Implement src/tray/tray_app.py and src/tray/settings_ui.py.

Use Context7 to look up pystray and Pillow APIs before writing.

TrayApp:
  Icon states — draw with Pillow as 64×64 circles:
    idle       → grey
    recording  → red
    processing → blue
    done       → green (auto-resets to idle after 1.5s)

  Tray menu:
    "WhisperFlow" (disabled title)
    "Status: Ready" (dynamic label, disabled)
    separator
    "Settings..."
    "View history" (scrollable tkinter window, last 50 transcriptions from history.json)
    separator
    "Quit"

  set_state(state) — thread-safe icon + status update
  show_notification(title, message) — tkinter toast, bottom-right, 4s auto-close

SettingsWindow (tk.Toplevel) with three tabs using ttk.Notebook:

  General tab:
    Hotkey (entry field), Recording mode (HOLD / TOGGLE radio), Language (dropdown)

  Formatter tab:
    Backend dropdown: "Fast only" / "Ollama (local, free)" / "Claude API (paid)"
    Ollama section (visible when Ollama selected):
      URL entry (default http://localhost:11434)
      Model dropdown: phi3:mini / mistral:7b / llama3.1:8b / gemma2:9b
      "Check connection" button → shows OK / FAIL + latency
    Claude section (visible when Claude selected):
      API key password entry
      "Test key" button → shows OK / FAIL
    LLM word threshold: spinbox (range 3–50, default 10)
    Label explaining: "Inputs shorter than this skip the LLM for speed"

  Advanced tab:
    Whisper model dropdown (tiny.en / base.en / medium.en / large-v3)
    VAD silence slider (200–1500ms, default 400ms)
    Label showing estimated latency based on current settings

  Save button → save_settings(), close window
  "Test mic" button → records 3s, shows raw transcript in a label

Use Context7 for tkinter ttk.Notebook and Toplevel patterns if needed.
```

---

### Prompt 11 — Main orchestrator + full pipeline

```
Implement main.py — the entry point that wires all components together.

Startup sequence:
  1. load_settings()
  2. If config.json did not exist (first run), open a first-run wizard:
       - Check if Ollama is reachable
       - If yes: show model picker, explain hotkey, save settings
       - If no: explain how to install Ollama, offer "Fast only" mode as fallback
  3. check_ollama() if backend is ollama:
       warn via tray notification if down, continue in fast-only mode
  4. Pre-warm: start WhisperModel load in a daemon thread so first use is instant
  5. Initialise formatter via create_formatter(settings)
  6. Start HotkeyListener(settings.hotkey, on_press, on_release, settings.recording_mode)
  7. Start TrayApp — this blocks the main thread

Pipeline on hotkey press (HOLD mode example):
  on_press:
    tray.set_state("recording")
    recorder.start()

  RealtimeSTT fires on_transcript(raw_text) automatically when VAD detects silence.

  on_transcript(raw_text):
    Run entirely in a daemon thread:
      tray.set_state("processing")
      final_text = format_text(raw_text, settings)
      success = injector.inject(final_text)
      if not success:
          injector.show_fallback_toast(final_text)
      append {timestamp, raw_text, final_text} to ~/.whisperflow/history.json
        (keep last settings.history_max entries, overwrite oldest)
      tray.set_state("done")   (auto-resets to idle after 1.5s)

  on_release (HOLD mode):
    recorder.stop()   (RealtimeSTT will finish current transcription)

CLI flags:
  --dev        verbose logging to stdout
  --model X    override whisper model
  --backend X  override formatter backend (fast / ollama / claude)

After implementing, run a dry-run integration test with tests/fixtures/sample.wav,
mocking the text injection step, and verify the full pipeline executes without errors.
Commit everything to GitHub.
```

---

### Prompt 12 — Latency benchmark + packaging + release

```
Implement tools/benchmark.py:
  - Runs tests/fixtures/sample.wav through the full pipeline
  - Times each stage individually: VAD/recording end, transcription, formatting, injection
  - Runs 5 iterations, reports min/avg/max per stage
  - Prints a table:

      Stage                | min (ms) | avg (ms) | max (ms)
      ---------------------|----------|----------|----------
      Whisper transcribe   |          |          |
      Formatter (fast)     |          |          |
      Formatter (LLM)      |          |          |
      Text injection       |          |          |
      TOTAL                |          |          |

  - Accepts --model and --backend CLI flags

Set up PyInstaller packaging:
  build/build.spec — one-file spec:
    Entry point: main.py
    Include: assets/icon.png, all src/ modules
    Hidden imports: faster_whisper, RealtimeSTT, sounddevice, pystray, keyboard
    Exclude: tests/, tools/, .env files
    Output: dist/WhisperFlow.exe

  Build and verify:
    pyinstaller build/build.spec --clean
    Confirm dist/WhisperFlow.exe launches and shows the tray icon.

Use Playwright to write a UI smoke test in tests/test_ui.py:
  - Launch the settings window
  - Verify the formatter backend dropdown is present
  - Select "Ollama (local, free)"
  - Verify the Ollama model dropdown appears
  - Verify "Check connection" button is present
  - Close the window and assert no errors

Create a GitHub Release using the GitHub MCP:
  Tag: v0.1.0
  Upload dist/WhisperFlow.exe as release asset
  Release notes must cover:
    - Hardware requirements and expected latency (CPU vs GPU table)
    - How to install Ollama and pull a model (one-command setup)
    - Privacy model (audio never leaves machine, transcript only if Claude enabled)
    - How to optionally enable Claude API formatter

Update README.md with:
  - Quick start: install Ollama → pull model → run .exe
  - Hardware + latency expectations table
  - Whisper and Ollama model comparison tables
  - Hotkey usage (HOLD and TOGGLE modes)
  - Privacy statement
  - Troubleshooting: Ollama not detected, slow transcription on CPU, hotkey conflicts
```

---

## Notes for Claude Code sessions

- **Use Context7 before every library API call** — RealtimeSTT, faster-whisper,
  Ollama REST API, pystray, and keyboard all have APIs that may differ from
  Claude's training data. This is non-negotiable.
- **Commit cadence via GitHub MCP**: after prompts 2, 5, 7, 9, 11, 12.
- **Pre-warm Whisper before testing**:
  `python -c "from faster_whisper import WhisperModel; WhisperModel('tiny.en')"`
- **Ollama must be running** before testing the LLM formatter.
  Start with `ollama serve`. The app handles Ollama being offline gracefully.
- **On CPU-only machines**, use `phi3:mini` during development — fastest Ollama model,
  good enough for testing the pipeline end-to-end.
- **The two-tier formatter is the most important optimization** — verify it works
  correctly before worrying about GPU acceleration or model size tuning.
- **Superpowers** is useful for reading benchmark output, checking config files,
  and running the test suite across sessions.
- **Playwright** is used in Prompt 12 for the settings UI smoke test. If tkinter
  windows aren't detectable, fall back to tkinter's own test utilities.
- **RealtimeSTT replaces both sounddevice and webrtcvad** — do not add those as
  separate components; RealtimeSTT wraps them internally.
