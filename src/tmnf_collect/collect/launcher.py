"""Launch and manage TMNF game instances through TrackMania ModLoader.

TMLoader's CLI is::

    TMLoader.exe run <game> <profile> <extra args passed to the game>

The loader process exits as soon as it has spawned the game, so the game is a
grandchild we have to find by matching a unique token we put on its command
line.  That token is also how each in-game plugin learns which controller port
it belongs to (``IO::GetCommandLineArgs`` inside AngelScript).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from ..common.hostos import DESKTOP_SIZE, IS_WINDOWS, game_command, wine_desktop
from ..common.paths import Layout, detect

if IS_WINDOWS:
    import ctypes
    from ctypes import wintypes

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


def _user32() -> "ctypes.WinDLL":
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
    enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    @enum_proc
    def visit(hwnd: int, _lparam: int) -> bool:
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid and user32.IsWindowVisible(hwnd):
            found.append(hwnd)
        return True

    user32.EnumWindows(visit, 0)
    return found


def render_offscreen(pid: int, slot: int = 0) -> int:
    """Move a process's windows beyond the desktop without minimizing them.

    ``slot`` keeps parallel instances from stacking on top of each other where
    that matters (a Wayland compositor stops sending frames to a window that
    is fully covered, and the game stops rendering with them).
    """
    if not IS_WINDOWS:
        return _sway_render_offscreen(pid, slot)
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


def _swaymsg(*args: str) -> str:
    completed = subprocess.run(
        ["swaymsg", *args], capture_output=True, text=True, check=False
    )
    return completed.stdout


def _sway_windows_for(pid: int, slot: int) -> list[dict]:
    """sway tree nodes (windows) belonging to ``pid`` or to its Wine desktop.

    The desktop window belongs to Wine's explorer, not the game, so it is
    matched by the name Wine gives it.
    """
    try:
        tree = json.loads(_swaymsg("-t", "get_tree") or "{}")
    except json.JSONDecodeError:
        return []
    desktop = f"{wine_desktop(slot)} - "
    found: list[dict] = []

    def visit(node: dict) -> None:
        if node.get("type") in ("con", "floating_con") and (
            node.get("pid") == pid or (node.get("name") or "").startswith(desktop)
        ):
            found.append(node)
        for child in node.get("nodes", []) + node.get("floating_nodes", []):
            visit(child)

    visit(tree)
    return found


# Tile size for parked windows: one Wine desktop each.
SLOT_WIDTH, SLOT_HEIGHT = (int(v) for v in DESKTOP_SIZE.split("x"))
SLOTS_PER_ROW = 4


def _sway_render_offscreen(pid: int, slot: int, timeout: float = 15.0) -> int:
    """Under Wine on a sway desktop: tile the windows so none covers another.

    Not actually off screen. A Wayland compositor stops sending frames to a
    fully covered window, and with them the game stops rendering, so what
    matters is that the instances never overlap. ``TMNF_GAME_OUTPUT`` names
    the output they are tiled on (the console's own, so they can be watched);
    without it they go past the right edge of the rightmost output instead.
    """
    output = os.environ.get("TMNF_GAME_OUTPUT")
    deadline = time.monotonic() + timeout
    windows = _sway_windows_for(pid, slot)
    while not windows and time.monotonic() < deadline:
        time.sleep(0.5)
        windows = _sway_windows_for(pid, slot)
    if not windows:
        return 0

    x = (slot % SLOTS_PER_ROW) * SLOT_WIDTH
    y = (slot // SLOTS_PER_ROW) * SLOT_HEIGHT
    if output:
        target = f"move container to output {output}, move position {x} {y}"
    else:
        outputs = json.loads(_swaymsg("-t", "get_outputs") or "[]")
        right = max(
            (o["rect"]["x"] + o["rect"]["width"] for o in outputs if o.get("active")),
            default=0,
        )
        target = f"move absolute position {right + 64 + x} {y}"
    for window in windows:
        _swaymsg(f'[con_id={window["id"]}] floating enable, {target}')
    return len(windows)


def _linux_processes(exe_name: str) -> list[dict]:
    """Wine processes running ``exe_name``, by scanning /proc command lines."""
    needle = exe_name.lower()
    found: list[dict] = []
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            raw = Path(entry.path, "cmdline").read_bytes()
        except OSError:
            continue
        argv = [a.decode("utf-8", "replace") for a in raw.split(b"\0") if a]
        if not argv or needle not in argv[0].lower():
            continue
        found.append({"ProcessId": int(entry.name), "CommandLine": " ".join(argv)})
    return found


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
    if not IS_WINDOWS:
        return _linux_processes(exe_name)
    out = _powershell(
        f"Get-CimInstance Win32_Process -Filter \"Name='{exe_name}'\" "
        "| Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    ).strip()
    if not out:
        return []
    data = json.loads(out)
    return data if isinstance(data, list) else [data]


def _kill(pid: int, token: str | None = None) -> None:
    if IS_WINDOWS:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
        return
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    # The Wine desktop the game lived in outlives it; its explorer carries the
    # same launch command line.
    for proc in _linux_processes("explorer.exe"):
        cmdline = proc["CommandLine"]
        if "/desktop=tmnf-" not in cmdline:
            continue
        if token is None or token in cmdline:
            try:
                os.kill(proc["ProcessId"], signal.SIGKILL)
            except ProcessLookupError:
                pass


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
        _kill(self.pid, self.token)
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
    command = game_command(
        layout.tmloader_exe,
        "run",
        layout.game,
        profile,
        game_args,
        instance_id=instance_id,
    )
    if IS_WINDOWS:
        subprocess.Popen(
            command,
            cwd=str(layout.tmloader_root),
            creationflags=subprocess.DETACHED_PROCESS,
        )
    else:
        subprocess.Popen(
            command,
            cwd=str(layout.tmloader_root),
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
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
        _kill(proc["ProcessId"])
    return len(procs)
