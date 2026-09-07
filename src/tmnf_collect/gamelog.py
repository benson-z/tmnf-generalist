"""Reading the in-game TMInterface console log.

``copy_log`` puts the whole console into the Windows clipboard, which is the
only way to get at it from outside the game.  Used for diagnosing runs; nothing
in the collection path depends on it.
"""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes

CF_UNICODETEXT = 13

_user32 = ctypes.windll.user32
_kernel32 = ctypes.windll.kernel32
_user32.GetClipboardData.argtypes = [wintypes.UINT]
_user32.GetClipboardData.restype = wintypes.HANDLE
_kernel32.GlobalLock.argtypes = [wintypes.HANDLE]
_kernel32.GlobalLock.restype = ctypes.c_void_p
_kernel32.GlobalUnlock.argtypes = [wintypes.HANDLE]


def read_clipboard_text(retries: int = 20, delay: float = 0.1) -> str:
    """Read the clipboard as text, retrying while another process holds it."""
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
