"""Discovery of TrackMania / TMLoader / TMInterface locations on this machine.

Everything is resolved at runtime so the tool keeps working if the user moves an
install; each value can be overridden with an environment variable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from .hostos import IS_WINDOWS, wine_user_dir

if IS_WINDOWS:
    import winreg


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def _documents_dir() -> Path:
    if not IS_WINDOWS:
        # Wine's shell folders live in the prefix; the game never sees $HOME.
        return wine_user_dir() / "Documents"
    # The Gbx games always use the shell "Personal" folder, not %USERPROFILE%\Documents,
    # which differ when OneDrive redirection is on.
    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders",
        ) as key:
            return Path(winreg.QueryValueEx(key, "Personal")[0])
    except OSError:
        return Path.home() / "Documents"


@dataclass(frozen=True)
class Layout:
    """Absolute locations this tool needs."""

    tmloader_exe: Path
    tmloader_root: Path
    game: str
    profile: str
    tmi_dir: Path  # Documents/TMInterface
    plugins_dir: Path  # Documents/TMInterface/Plugins
    scripts_dir: Path  # Documents/TMInterface/Scripts
    user_dir: Path  # Documents/TmForever
    tracks_dir: Path  # Documents/TmForever/Tracks
    replays_dir: Path  # Documents/TmForever/Tracks/Replays
    challenges_dir: Path  # Documents/TmForever/Tracks/Challenges

    def profile_yaml(self) -> Path:
        return (
            self.tmloader_root
            / "database"
            / self.game
            / "profiles"
            / f"{self.profile}.yaml"
        )


def _find_tmloader() -> Path:
    override = _env_path("TMNF_TMLOADER_EXE")
    if override:
        return override
    if IS_WINDOWS:
        candidates = [
            Path(os.environ.get("LOCALAPPDATA", "")) / "TMLoader" / "TMLoader.exe",
            Path(os.environ.get("APPDATA", "")) / "TMLoader" / "TMLoader.exe",
        ]
    else:
        appdata = wine_user_dir() / "AppData"
        candidates = [
            appdata / "Local" / "TMLoader" / "TMLoader.exe",
            appdata / "Roaming" / "TMLoader" / "TMLoader.exe",
        ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "TMLoader.exe not found. Set TMNF_TMLOADER_EXE to its full path."
    )


def _user_dir_for(game: str) -> Path:
    docs = _documents_dir()
    # TmForever == Nations Forever, TmUnited == United Forever.
    return docs / ("TmForever" if game == "TmForever" else "TrackMania")


def detect(game: str = "TmForever", profile: str | None = None) -> Layout:
    """Resolve every path the collector needs, honouring env overrides."""
    tmloader_exe = _find_tmloader()
    tmloader_root = tmloader_exe.parent

    if profile is None:
        profile = os.environ.get("TMNF_PROFILE") or _default_profile(
            tmloader_root, game
        )

    tmi_dir = _env_path("TMNF_TMI_DIR") or (_documents_dir() / "TMInterface")
    user_dir = _env_path("TMNF_USER_DIR") or _user_dir_for(game)
    tracks_dir = user_dir / "Tracks"

    return Layout(
        tmloader_exe=tmloader_exe,
        tmloader_root=tmloader_root,
        game=game,
        profile=profile,
        tmi_dir=tmi_dir,
        plugins_dir=tmi_dir / "Plugins",
        scripts_dir=tmi_dir / "Scripts",
        user_dir=user_dir,
        tracks_dir=tracks_dir,
        replays_dir=tracks_dir / "Replays",
        challenges_dir=tracks_dir / "Challenges",
    )


def _default_profile(tmloader_root: Path, game: str) -> str:
    config = tmloader_root / "config.yaml"
    try:
        data = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
        return data.get("default_profiles", {}).get(game) or "default"
    except OSError:
        return "default"
