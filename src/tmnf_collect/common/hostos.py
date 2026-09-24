"""Where the game runs: natively on Windows, or under Wine on Linux.

The game is a Windows program either way, so anything handed to it (``/userdir``,
TMLoader profile args) must be a Windows path. Under Wine that means the
prefix's ``C:`` drive; anything outside it is reached through Wine's ``Z:``
mapping of the whole filesystem.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

IS_WINDOWS = sys.platform == "win32"


def wine_prefix() -> Path:
    return Path(os.environ.get("WINEPREFIX") or Path.home() / ".wine")


def wine_user_dir() -> Path:
    """``C:\\users\\<name>`` of the prefix, as a host path."""
    name = os.environ.get("USER") or os.environ.get("LOGNAME") or "user"
    return wine_prefix() / "drive_c" / "users" / name


def windows_path(path: Path) -> str:
    """``path`` as the game will see it."""
    if IS_WINDOWS:
        return str(path)
    resolved = path.resolve()
    drive_c = (wine_prefix() / "drive_c").resolve()
    try:
        relative = resolved.relative_to(drive_c)
    except ValueError:
        return "Z:" + str(resolved).replace("/", "\\")
    return "C:\\" + str(relative).replace("/", "\\")


# Size of the Wine desktop each instance lives in: the 640x480 game window
# plus Wine's own title bar and borders. Wine only honours standard display
# sizes here; anything odd (660x520, say) becomes a full-screen desktop.
DESKTOP_SIZE = "800x600"


def wine_desktop(instance_id: int) -> str:
    return f"tmnf-{instance_id}"


def game_command(exe: Path, *args: str, instance_id: int = 0) -> list[str]:
    """Command line that starts a Windows executable on this host.

    Under Wine every instance gets its own virtual desktop. Wine keeps the
    foreground window per desktop, and that is what TMInterface's input
    injection keys off: arming a run needs the instance to be foreground,
    while another instance losing foreground mid-run releases its inputs and
    the re-drive diverges. One desktop each and neither can happen to the
    other.
    """
    if IS_WINDOWS:
        return [str(exe), *args]
    return [
        "wine",
        "explorer",
        f"/desktop={wine_desktop(instance_id)},{DESKTOP_SIZE}",
        str(exe),
        *args,
    ]


def program_files_x86() -> Path:
    """Where 32-bit programs install: the game's stock content lives there."""
    if IS_WINDOWS:
        return Path(os.environ.get("ProgramFiles(x86)") or r"C:\Program Files (x86)")
    return wine_prefix() / "drive_c" / "Program Files (x86)"


def campaign_root() -> Path:
    """The stock campaign maps, honouring ``TMNF_CAMPAIGN_DIR``."""
    override = os.environ.get("TMNF_CAMPAIGN_DIR")
    if override:
        return Path(override)
    return program_files_x86() / "TmNationsForever" / "GameData" / "Tracks" / "Campaigns"
