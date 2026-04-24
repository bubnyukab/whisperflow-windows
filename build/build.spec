# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller one-file spec for WhisperFlow Windows.

Build (from project root on Windows):
    pyinstaller build/build.spec --clean

Output: dist/WhisperFlow.exe
"""

import sys
from pathlib import Path

ROOT = Path(SPECPATH).parent  # noqa: F821  (SPECPATH injected by PyInstaller)

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[
        (str(ROOT / "assets"), "assets"),
        (str(ROOT / "src"), "src"),
    ],
    hiddenimports=[
        # Audio / transcription
        "faster_whisper",
        "faster_whisper.transcribe",
        "RealtimeSTT",
        "sounddevice",
        "soundfile",
        "webrtcvad",
        # GUI / tray
        "pystray",
        "pystray._win32",
        "PIL._tkinter_finder",
        "tkinter",
        "tkinter.ttk",
        # Input / output
        "keyboard",
        "pyperclip",
        # Network / LLM
        "httpx",
        "anthropic",
        # Config
        "dotenv",
    ],
    excludes=[
        "tests",
        "tools",
        ".env",
        "pytest",
        "unittest",
    ],
    hookspath=[],
    runtime_hooks=[],
    noarchive=False,
)

pyz = PYZ(a.pure)  # noqa: F821  (PYZ injected by PyInstaller)

exe = EXE(  # noqa: F821  (EXE injected by PyInstaller)
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="WhisperFlow",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    icon=str(ROOT / "assets" / "icon.png"),
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
)
