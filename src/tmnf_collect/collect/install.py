"""Installing the AngelScript plugin into TMInterface, and the car skin."""

from __future__ import annotations

import os
from pathlib import Path

from ..common.paths import Layout, detect

PLUGIN_NAME = "TMNFCollect.as"
SKIN_NAME = "Magenta.zip"


def source_plugin() -> Path:
    """The plugin shipped in this repository."""
    return Path(__file__).resolve().parents[3] / "plugin" / PLUGIN_NAME


def source_skin() -> Path:
    """The car skin shipped in this repository; see skins/make_magenta.py."""
    return Path(__file__).resolve().parents[3] / "skins" / SKIN_NAME


def skin_target(layout: Layout) -> Path:
    return layout.user_dir / "Skins" / "Vehicles" / "StadiumCar" / SKIN_NAME


def _replace_if_changed(target: Path, wanted: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        if target.read_bytes() == wanted:
            return
    except OSError:
        pass

    # Same directory, so the replace is atomic rather than across volumes.
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    temporary.write_bytes(wanted)
    os.replace(temporary, target)


def install(layout: Layout | None = None) -> Path:
    """Copy the plugin into ``Documents/TMInterface/Plugins``.

    TMInterface reloads a plugin whenever its file changes, so overwriting an
    installed copy is enough to update a running game -- and equally, rewriting
    it under a game that is already running restarts its plugin mid-run. Every
    collector calls this at startup, so with several collecting at once the
    later ones would otherwise reload the plugin out from under the earlier
    ones' games. Unchanged content is therefore left alone.

    The write itself replaces the file rather than truncating and refilling it,
    because a game reading a half-written plugin fails to compile it and never
    connects at all. Measured: eight collectors starting 1.5s apart lost six of
    eight instances to `plugin closed the connection` before this.

    The magenta car skin goes into the game's user directory the same way.
    Which skin the car wears is the player profile's choice, and the seeded
    profile (docker/seed) picks this one; on Windows, pick it once under
    Profile > Vehicles. The profile pins the zip's MD5, so the file has to be
    byte for byte the one it was chosen as.
    """
    layout = layout or detect()
    target = layout.plugins_dir / PLUGIN_NAME
    _replace_if_changed(target, source_plugin().read_bytes())
    _replace_if_changed(skin_target(layout), source_skin().read_bytes())
    return target


def is_installed(layout: Layout | None = None) -> bool:
    layout = layout or detect()
    target = layout.plugins_dir / PLUGIN_NAME
    if not target.is_file():
        return False
    return target.read_bytes() == source_plugin().read_bytes()
