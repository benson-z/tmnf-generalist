"""Reading the in-game TMInterface console log.

``copy_log`` puts the whole console into the clipboard, which is the only way
to get at it from outside the game.  Used for diagnosing runs; nothing in the
collection path depends on it.

On Windows that is the system clipboard.  Under Wine the game's clipboard is
bridged to the X server and from there to the Wayland compositor, so it is
read with ``wl-paste`` (or ``xclip`` on a plain X desktop).
"""

from __future__ import annotations

import shutil
import subprocess
import time

from .hostos import IS_WINDOWS

if IS_WINDOWS:
    import ctypes
    from ctypes import wintypes

    CF_UNICODETEXT = 13

    _user32 = ctypes.windll.user32
    _kernel32 = ctypes.windll.kernel32
    _user32.GetClipboardData.argtypes = [wintypes.UINT]
    _user32.GetClipboardData.restype = wintypes.HANDLE
    _kernel32.GlobalLock.argtypes = [wintypes.HANDLE]
    _kernel32.GlobalLock.restype = ctypes.c_void_p
    _kernel32.GlobalUnlock.argtypes = [wintypes.HANDLE]


def _read_windows_clipboard(retries: int, delay: float) -> str:
    for _ in range(retries):
        if _user32.OpenClipboard(None):
            try:
                handle = _user32.GetClipboardData(CF_UNICODETEXT)
                if not handle:
                    return ""
                pointer = _kernel32.GlobalLock(handle)
                if not pointer:
                    return ""
                try:
                    return ctypes.c_wchar_p(pointer).value or ""
                finally:
                    _kernel32.GlobalUnlock(handle)
            finally:
                _user32.CloseClipboard()
        time.sleep(delay)
    return ""


def _read_x11_clipboard(retries: int, delay: float) -> str:
    if shutil.which("wl-paste"):
        command = ["wl-paste", "--no-newline", "--type", "text"]
    elif shutil.which("xclip"):
        command = ["xclip", "-o", "-selection", "clipboard"]
    else:
        return ""
    for _ in range(retries):
        completed = subprocess.run(
            command, capture_output=True, text=True, check=False
        )
        if completed.returncode == 0:
            return completed.stdout
        time.sleep(delay)
    return ""


def read_clipboard_text(retries: int = 20, delay: float = 0.1) -> str:
    """Read the clipboard as text, retrying while another process holds it."""
    if IS_WINDOWS:
        return _read_windows_clipboard(retries, delay)
    return _read_x11_clipboard(retries, delay)
