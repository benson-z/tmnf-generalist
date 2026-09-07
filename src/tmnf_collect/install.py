"""Installing the AngelScript plugin into TMInterface."""

from __future__ import annotations

import shutil
from pathlib import Path

from .paths import Layout, detect

PLUGIN_NAME = "TMNFCollect.as"


def source_plugin() -> Path:
    """The plugin shipped in this repository."""
    return Path(__file__).resolve().parents[2] / "plugin" / PLUGIN_NAME


def install(layout: Layout | None = None) -> Path:
    """Copy the plugin into ``Documents/TMInterface/Plugins``.

    TMInterface reloads a plugin whenever its file changes, so overwriting an
    installed copy is enough to update a running game.
    """
    layout = layout or detect()
    layout.plugins_dir.mkdir(parents=True, exist_ok=True)
    target = layout.plugins_dir / PLUGIN_NAME
    shutil.copyfile(source_plugin(), target)
    return target


def is_installed(layout: Layout | None = None) -> bool:
    layout = layout or detect()
    target = layout.plugins_dir / PLUGIN_NAME
    if not target.is_file():
        return False
    return target.read_bytes() == source_plugin().read_bytes()
