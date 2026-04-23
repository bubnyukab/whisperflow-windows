# WhisperFlow Windows

> Local, privacy-first Wispr Flow clone for Windows — 100% free, fully offline, no accounts required.

Hold a hotkey → speak → formatted text appears in your focused app.

## Features

- **100% free by default** — Ollama + Whisper, zero accounts, zero API keys
- **Privacy first** — audio never leaves your machine
- **Low latency** — <700ms end-to-end on GPU, <1.5s on CPU
- **Two-tier formatting** — instant path for short inputs, LLM for longer text

## Requirements

- Windows 10/11
- Python 3.11+
- NVIDIA GPU recommended (CPU works, slower)
- [Ollama](https://ollama.com) (optional — for LLM formatting)

## Setup

```bash
pip install -r requirements.txt

# Pre-warm the Whisper model (~75 MB download on first run)
python -c "from faster_whisper import WhisperModel; WhisperModel('tiny.en')"
```

### Optional: LLM Formatting with Ollama

```bash
# Install Ollama from https://ollama.com, then:
ollama pull llama3.1:8b
ollama serve
```

### Optional: Claude API Formatting

Copy `.env.example` to `.env` and set your `ANTHROPIC_API_KEY`.

## Usage

```bash
python main.py
```

A tray icon appears. Press the hotkey (default `Win+Shift+Space`), speak, release —
the transcribed and formatted text is injected into your current app.

Right-click the tray icon to open Settings or quit.

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `WHISPER_MODEL` | `tiny.en` | Whisper model size |
| `HOTKEY` | `win+shift+space` | Global recording hotkey |
| `FORMATTER_BACKEND` | `ollama` | `fast` / `ollama` / `claude` |
| `OLLAMA_MODEL` | `llama3.1:8b` | Ollama model name |

All settings are editable via the tray icon → Settings.

## Development

```bash
python main.py --dev        # verbose logging
python -m pytest tests/ -v  # run tests
python tools/benchmark.py   # latency benchmark
```

## License

MIT
