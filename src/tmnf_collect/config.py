"""Settings from a file, so a collection run is one command again.

There are enough choices now -- instances, camera, intro stripping, the input
filter, tag exclusions -- that passing them every time invites getting one
wrong, and getting one wrong is not obvious until `verify` runs an hour later.

The file sets defaults; an explicit flag still wins, because it is applied with
``set_defaults`` before parsing rather than after. Sections are per command,
since ``limit`` means a different thing to `harvest` than to `collect`.

An unknown key is an error rather than a shrug: a typo that silently does
nothing is exactly the failure this file exists to prevent.
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_NAME = "tmnf-collect.yaml"


class ConfigError(RuntimeError):
    pass


def find(explicit: str | None = None) -> Path | None:
    """The config to use: the one asked for, else one beside the project."""
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise ConfigError(f"no config file at {path}")
        return path
    here = Path.cwd() / DEFAULT_NAME
    if here.is_file():
        return here
    # Installed as a tool rather than run from a checkout: fall back to the
    # config that ships with the source.
    beside = Path(__file__).resolve().parent.parent.parent / DEFAULT_NAME
    return beside if beside.is_file() else None


def load(path: Path | None) -> dict:
    if path is None:
        return {}
    import yaml

    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path}: expected a mapping of commands to settings")
    return loaded


def option_names(parser: argparse.ArgumentParser) -> set[str]:
    """Which keys a command understands, as their config spellings."""
    return {
        action.dest
        for action in parser._actions
        if action.dest not in ("help", "==SUPPRESS==")
    }


def apply(
    config: dict,
    parsers: dict[str, argparse.ArgumentParser],
    *,
    source: Path | None,
) -> None:
    """Push each section's values in as that command's defaults."""
    for command, values in config.items():
        parser = parsers.get(command)
        if parser is None:
            raise ConfigError(
                f"{source}: no command called {command!r}; "
                f"known commands are {', '.join(sorted(parsers))}"
            )
        if values is None:
            continue
        if not isinstance(values, dict):
            raise ConfigError(f"{source}: {command} should be a mapping")

        known = option_names(parser)
        unknown = set(values) - known
        if unknown:
            raise ConfigError(
                f"{source}: {command} has no option "
                f"{', '.join(repr(u) for u in sorted(unknown))}; "
                f"try one of {', '.join(sorted(known))}"
            )
        parser.set_defaults(**values)


def effective(args: argparse.Namespace, parser: argparse.ArgumentParser) -> dict:
    """What this command is actually running with, for the dataset to record."""
    return {
        name: getattr(args, name)
        for name in sorted(option_names(parser))
        if name not in ("func", "config")
        and isinstance(getattr(args, name, None), (str, int, float, bool, type(None)))
    }
