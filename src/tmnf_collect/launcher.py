"""Launch and manage TMNF game instances through TrackMania ModLoader.

TMLoader's CLI is::

    TMLoader.exe run <game> <profile> <extra args passed to the game>

The loader process exits as soon as it has spawned the game, so the game is a
grandchild we have to find by matching a unique token we put on its command
line.  That token is also how each in-game plugin learns which controller port
it belongs to (``IO::GetCommandLineArgs`` inside AngelScript).
"""

from __future__ import annotations

import ctypes
import json
import subprocess
import time
import uuid
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path

from .paths import Layout, detect

GAME_EXE = "TmForever.exe"

# A minimized Direct3D 9 window stops rendering, which starves the collector's
# natural-frame barrier.  Moving it beyond the virtual desktop leaves it shown
# and rendering without letting it cover (or accidentally receive clicks from)
# the desktop the collector is running on.
SW_SHOWNOACTIVATE = 4
SWP_NOSIZE = 0x0001
SWP_NOZORDER = 0x0004
SWP_NOACTIVATE = 0x0010
SM_XVIRTUALSCREEN = 76
SM_CXVIRTUALSCREEN = 78
WNDENUMPROC = ctypes.WINFUNCTYPE(
    wintypes.BOOL, wintypes.HWND, wintypes.LPARAM
)


def _user32() -> ctypes.WinDLL:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetWindowThreadProcessId.argtypes = [
        wintypes.HWND,
        ctypes.POINTER(wintypes.DWORD),
    ]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    user32.EnumWindows.argtypes = [WNDENUMPROC, wintypes.LPARAM]
    user32.EnumWindows.restype = wintypes.BOOL
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.ShowWindow.restype = wintypes.BOOL
    user32.SetWindowPos.argtypes = [
        wintypes.HWND,
        wintypes.HWND,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.UINT,
    ]
    user32.SetWindowPos.restype = wintypes.BOOL
    user32.GetSystemMetrics.argtypes = [ctypes.c_int]
    user32.GetSystemMetrics.restype = ctypes.c_int
    return user32


def _windows_for(pid: int) -> list[int]:
    """Return visible top-level windows owned by ``pid``."""
    user32 = _user32()
    found: list[int] = []

    @WNDENUMPROC
    def visit(hwnd: int, _lparam: int) -> bool:
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid and user32.IsWindowVisible(hwnd):
            found.append(hwnd)
        return True

    user32.EnumWindows(visit, 0)
    return found


def render_offscreen(pid: int) -> int:
    """Move a process's windows beyond the desktop without minimizing them."""
    user32 = _user32()
    right = (
        user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
        + user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
    )
    windows = _windows_for(pid)
    for hwnd in windows:
        user32.ShowWindow(hwnd, SW_SHOWNOACTIVATE)
        user32.SetWindowPos(
            hwnd,
            0,
            right + 64,
            0,
            0,
            0,
            SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE,
        )
    return len(windows)


def _powershell(script: str) -> str:
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            script,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout


def _processes(exe_name: str = GAME_EXE) -> list[dict]:
    out = _powershell(
        f"Get-CimInstance Win32_Process -Filter \"Name='{exe_name}'\" "
        "| Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    ).strip()
    if not out:
        return []
    data = json.loads(out)
    return data if isinstance(data, list) else [data]


@dataclass
class GameInstance:
    """A running TMNF process launched by us."""

    pid: int
    token: str
    port: int
    instance_id: int
    layout: Layout

    def is_alive(self) -> bool:
        return any(p["ProcessId"] == self.pid for p in _processes())

    def terminate(self, timeout: float = 10.0) -> None:
        subprocess.run(
            ["taskkill", "/PID", str(self.pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and self.is_alive():
            time.sleep(0.2)


def launch(
    *,
    port: int,
    instance_id: int = 0,
    layout: Layout | None = None,
    profile: str | None = None,
    timeout: float = 120.0,
) -> GameInstance:
    """Start one game instance and return it once its process exists.

    ``port`` and ``instance_id`` are handed to the game on its command line so
    the in-game plugin can dial back to the right controller socket.
    """
    layout = layout or detect()
    profile = profile or layout.profile
    token = uuid.uuid4().hex[:12]
    game_args = (
        f"/tmnfml_token={token} /tmnfml_port={port} /tmnfml_id={instance_id}"
    )

    before = {p["ProcessId"] for p in _processes()}
    subprocess.Popen(
        [
            str(layout.tmloader_exe),
            "run",
            layout.game,
            profile,
            game_args,
        ],
        cwd=str(layout.tmloader_root),
        creationflags=subprocess.DETACHED_PROCESS,
    )

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for proc in _processes():
            cmdline = proc.get("CommandLine") or ""
            if token in cmdline:
                return GameInstance(
                    pid=proc["ProcessId"],
                    token=token,
                    port=port,
                    instance_id=instance_id,
                    layout=layout,
                )
            # TMLoader may drop the argument string in some configurations;
            # fall back to "a process that wasn't there before".
            if proc["ProcessId"] not in before and cmdline == "":
                continue
        time.sleep(0.5)

    raise TimeoutError(
        f"game process did not appear within {timeout:.0f}s "
        f"(launched via {layout.tmloader_exe} run {layout.game} {profile})"
    )


def kill_all() -> int:
    """Terminate every running TMNF process. Returns how many were killed."""
    procs = _processes()
    for proc in procs:
        subprocess.run(
            ["taskkill", "/PID", str(proc["ProcessId"]), "/T", "/F"],
            capture_output=True,
            check=False,
        )
    return len(procs)
