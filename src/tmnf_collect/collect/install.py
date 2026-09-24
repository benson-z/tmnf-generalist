"""Installing the AngelScript plugin into TMInterface."""

from __future__ import annotations

import os
from pathlib import Path

from ..common.paths import Layout, detect

PLUGIN_NAME = "TMNFCollect.as"


def source_plugin() -> Path:
    """The plugin shipped in this repository."""
    return Path(__file__).resolve().parents[3] / "plugin" / PLUGIN_NAME


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
    """
    layout = layout or detect()
    layout.plugins_dir.mkdir(parents=True, exist_ok=True)
    target = layout.plugins_dir / PLUGIN_NAME
    wanted = source_plugin().read_bytes()
    try:
        if target.read_bytes() == wanted:
            return target
    except OSError:
        pass

    # Same directory, so the replace is atomic rather than across volumes.
    temporary = target.with_name(f"{PLUGIN_NAME}.{os.getpid()}.tmp")
    temporary.write_bytes(wanted)
    os.replace(temporary, target)
    return target


def is_installed(layout: Layout | None = None) -> bool:
    layout = layout or detect()
    target = layout.plugins_dir / PLUGIN_NAME
    if not target.is_file():
        return False
    return target.read_bytes() == source_plugin().read_bytes()
