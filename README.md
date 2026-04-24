# WhisperFlow Windows

> Local, privacy-first Wispr Flow clone for Windows — 100% free, fully offline, no accounts required.

Hold a hotkey → speak → formatted text appears in your currently focused app.

---

## Quick Start

1. **Install Ollama** — download from [ollama.com](https://ollama.com) and run the installer
2. **Pull a model** — open a terminal and run:
   ```
   ollama pull llama3.1:8b
   ```
3. **Run WhisperFlow** — double-click `WhisperFlow.exe`

A setup wizard appears on first launch to confirm your Ollama model and hotkey.

---

## Hardware & Latency

WhisperFlow achieves Wispr Flow–level latency locally on any NVIDIA GPU.

| Hardware | Whisper (`tiny.en`) | Formatter (Ollama) | Total |
|---|---|---|---|
| CPU only | ~400 ms | ~2 s (skipped for short inputs) | **~700 ms–1.5 s** ✅ |
| Any NVIDIA GPU | ~50 ms | ~350 ms | **~400–500 ms** ✅ |
| RTX 3080+ | ~50 ms | ~250 ms | **~300 ms** ✅ |

> **Two-tier formatter:** inputs shorter than 10 words skip the LLM entirely (<5 ms). This covers ~80% of real dictation use cases and keeps CPU-only machines responsive.

---

## Whisper Model Options

| Model | Size | RAM | CPU latency | GPU latency |
|---|---|---|---|---|
| `tiny.en` *(default)* | 75 MB | 1 GB | ~400 ms | ~50 ms |
| `base.en` | 150 MB | 1 GB | ~700 ms | ~100 ms |
| `medium.en` | 1.5 GB | 5 GB | ~3 s | ~300 ms |
| `large-v3` | 3 GB | 10 GB | ~8 s | ~600 ms |

Change the model in **Settings → Advanced**.

---

## Ollama Model Options

| Model | Size | RAM | CPU latency | GPU latency | Quality |
|---|---|---|---|---|---|
| `phi3:mini` | 2.3 GB | 4 GB | ~800 ms | ~150 ms | Good |
| `mistral:7b` | 4.1 GB | 8 GB | ~1.5 s | ~250 ms | Very good |
| `llama3.1:8b` *(default)* | 4.7 GB | 8 GB | ~2 s | ~350 ms | Excellent |
| `gemma2:9b` | 5.5 GB | 10 GB | ~2.5 s | ~400 ms | Best |

For CPU-only machines, `phi3:mini` gives the best speed/quality tradeoff.

---

## Hotkey Usage

| Mode | How it works |
|---|---|
| **Hold** *(default)* | Press and hold hotkey while speaking; release to transcribe |
| **Toggle** | Press once to start recording; press again to stop |

Default hotkey: `Win+Shift+Space`

Change in **Settings → General**.

---

## Privacy

- **Audio never leaves your machine.** Recording, VAD, and Whisper transcription run entirely locally.
- **Transcript stays local by default.** The Ollama formatter runs locally with no network calls.
- **The only exception:** if you enable the Claude API formatter in Settings, the transcript text (not audio) is sent to Anthropic's API. Audio is never sent.

---

## Optional: Claude API Formatter

For the highest formatting quality, WhisperFlow supports Anthropic's Claude API.

1. Copy `.env.example` to `.env`
2. Add your key: `ANTHROPIC_API_KEY=sk-ant-...`
3. In Settings → Formatter, select **Claude API (paid)**

Only the transcript text is sent — never audio.

---

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run from source
python main.py --dev          # verbose logging
python main.py --model base.en --backend fast

# Tests
python -m pytest tests/ -v

# Latency benchmark (uses tests/fixtures/sample.wav)
python tools/benchmark.py --model tiny.en --backend fast --iterations 5

# Build .exe (Windows, requires PyInstaller)
pyinstaller build/build.spec --clean
```

---

## Troubleshooting

**Ollama not detected on startup**

- Make sure Ollama is running: open a terminal and run `ollama serve`
- Check the URL in Settings → Formatter → "Check connection"
- WhisperFlow falls back to Fast-only mode automatically if Ollama is unreachable

**Transcription is slow on CPU**

- Switch to `tiny.en` in Settings → Advanced (it's the default and fastest)
- Enable the two-tier formatter — short inputs skip the LLM entirely
- Consider `phi3:mini` as the Ollama model for faster formatting

**Hotkey not working / conflicts with another app**

- Change the hotkey in Settings → General
- Try `ctrl+shift+space` or `alt+shift+space` if the default conflicts
- Some apps (e.g., games in exclusive mode) block global hotkeys

**Text injected in the wrong place**

- Click once in the target field before pressing the hotkey
- The injection uses clipboard paste (`Ctrl+V`); ensure the target app accepts it

---

## License

MIT
