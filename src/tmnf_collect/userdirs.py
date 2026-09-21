"""Per-instance game user directories.

Several game instances sharing one ``Documents/TmForever`` tread on each
other's profile, and an instance can end up unable to inject any input at all
("no binding for Accelerate found" in its console) which leaves the car parked
on the start line for the whole run.

The game takes a ``/userdir=<path>`` switch, so each instance gets its own copy
of the small, mutable parts (Profiles, Config) and shares the big read-only
part (Tracks) through a junction.

``Config`` and ``Profiles`` are mirrored from the real user directory on every
start, not copied once. That makes ``Documents/TmForever`` the single place to
change a setting -- graphics detail, resolution, keybinds, camera -- and every
instance picks it up next launch. Copying once was measured to go wrong: the
instances ran for days on graphics settings that had been changed in the
launcher, because the change never reached their copies, and a benchmark of
"minimum settings" quietly measured the old ones.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import yaml

from .hostos import IS_WINDOWS, windows_path
from .paths import Layout

# Mirrored from the master on every start: the settings a person edits.
SYNCED_DIRS = ("Profiles", "Config")
# Copied once: per-instance state nobody edits by hand.
PRIVATE_DIRS = ("Scores",)
# Shared read-only: this is where staged maps and replays live.
SHARED_DIRS = ("Tracks",)


class UserDirError(RuntimeError):
    pass


def root(layout: Layout) -> Path:
    return layout.tmloader_root / "tmnf-collect-userdirs"


def _link_directory(link: Path, target: Path) -> None:
    """Point ``link`` at ``target`` with a junction (no admin rights needed)."""
    if link.is_dir():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    if not IS_WINDOWS:
        # Wine follows symlinks like any other directory.
        link.symlink_to(target, target_is_directory=True)
        return
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    if not link.is_dir():
        raise UserDirError(
            f"could not link {link} -> {target}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )


def prepare(layout: Layout, instance_id: int, *, refresh: bool = False) -> Path:
    """Build (or reuse) the user directory for one instance."""
    target = root(layout) / str(instance_id)
    if " " in windows_path(target):
        # TMLoader passes profile args through unquoted, so a space would split
        # the switch in half.
        raise UserDirError(
            f"user directory path contains a space and cannot be passed to the "
            f"game: {target}"
        )

    if refresh and target.exists():
        shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True, exist_ok=True)

    for name in SYNCED_DIRS:
        source = layout.user_dir / name
        if source.is_dir():
            # Overwrites whatever the game wrote on its last exit, which is the
            # point: an instance should look like the master, not drift.
            shutil.copytree(source, target / name, dirs_exist_ok=True)

    for name in PRIVATE_DIRS:
        source = layout.user_dir / name
        destination = target / name
        if source.is_dir() and not destination.exists():
            shutil.copytree(source, destination)

    for name in SHARED_DIRS:
        source = layout.user_dir / name
        if source.is_dir():
            _link_directory(target / name, source)

    return target


def tmloader_profile(layout: Layout, instance_id: int, user_dir: Path) -> str:
    """Write a TMLoader profile that launches with this user directory.

    It copies the program and mods of the configured profile, so the instance
    still gets TMInterface, and only adds the ``/userdir`` switch.
    """
    base_path = layout.profile_yaml()
    try:
        base = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise UserDirError(f"could not read profile {base_path}: {exc}") from exc

    name = f"tmnf-collect-{instance_id}"
    derived = {
        "program": base.get("program", {"id": layout.game}),
        "mods": base.get("mods", []),
        "args": f"/userdir={windows_path(user_dir)}",
    }
    (base_path.parent / f"{name}.yaml").write_text(
        yaml.safe_dump(derived, sort_keys=False), encoding="utf-8"
    )
    return name


def setup(layout: Layout, instance_id: int, *, refresh: bool = False) -> str:
    """Prepare one instance's user directory. Returns its TMLoader profile."""
    user_dir = prepare(layout, instance_id, refresh=refresh)
    return tmloader_profile(layout, instance_id, user_dir)
