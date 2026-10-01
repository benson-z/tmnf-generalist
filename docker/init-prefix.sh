#!/bin/bash
# One-time Wine prefix: the game, TMLoader (which bundles TMInterface),
# and a TMLoader profile that runs the game with TMInterface. Idempotent.
set -euo pipefail

MARK="$WINEPREFIX/.tmnf-ready"
if [ -f "$MARK" ]; then
    echo "[init-prefix] prefix ready ($(cat "$MARK"))"
    exit 0
fi

log() { echo "[init-prefix] $*"; }
USERDIR="$WINEPREFIX/drive_c/users/$USER"

log "creating prefix"
wineboot -u
wineserver -w

# Wine's own Direct3D 9 (see the Dockerfile for why not DXVK).
wine reg add 'HKCU\Software\Wine\DllOverrides' /v d3d9 /t REG_SZ /d builtin /f >/dev/null
# The game sizes its texture budget from reported VRAM; tell Wine how much the
# iGPU has carved out rather than letting it guess.
wine reg add 'HKCU\Software\Wine\Direct3D' /v VideoMemorySize /t REG_SZ /d "${TMNF_VRAM_MB:-3072}" /f >/dev/null

if [ ! -f "$WINEPREFIX/drive_c/Program Files (x86)/TmNationsForever/TmForever.exe" ]; then
    log "installing TrackMania Nations Forever (silent)"
    wine /dl/tmnationsforever_setup.exe /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /SP- || true
    wineserver -w
    if [ ! -f "$WINEPREFIX/drive_c/Program Files (x86)/TmNationsForever/TmForever.exe" ]; then
        log "silent install did not produce TmForever.exe; run the installer by hand:"
        log "  tmnf-launch installer"
    fi
fi

TML="$USERDIR/AppData/Local/TMLoader"
if [ ! -f "$TML/TMLoader.exe" ]; then
    log "unpacking TMLoader into $TML"
    mkdir -p "$TML"
    unzip -q -o /dl/TMLoader-latest.zip -d "$TML"
fi

# Profile the collector launches: the modded game with TMInterface enabled.
mkdir -p "$TML/database/TmForever/profiles"
if [ ! -f "$TML/database/TmForever/profiles/default.yaml" ]; then
    cat > "$TML/database/TmForever/profiles/default.yaml" <<'YAML'
program:
  id: TmForever
mods:
- id: TMInterface
YAML
fi
# Where the game is installed. Without it TMLoader's first launch stops on a
# dialog asking you to start the original game once, so it can find it.
mkdir -p "$TML/database/TmForever/products/TmForever"
if [ ! -f "$TML/database/TmForever/products/TmForever/settings.yaml" ]; then
    printf "install: 'C:/Program Files (x86)/TmNationsForever'\n" \
        > "$TML/database/TmForever/products/TmForever/settings.yaml"
fi
if [ ! -f "$TML/config.yaml" ]; then
    cat > "$TML/config.yaml" <<'YAML'
default_profiles:
  TmForever: default
YAML
fi

# The game keeps its user data under Documents/TmForever. Seed it with a known
# good state so nothing has to be clicked through on the console: a local
# player profile ("tmnf", no account), its score file, and a system config
# that is windowed 640x480 (the collector's 2:1 capture assumes that) on the
# Minimum Quality preset. PC3 shaders render every lit surface black on Mesa,
# so do not let the launcher's benchmark pick the graphics settings again.
if [ ! -d "$USERDIR/Documents/TmForever/Profiles" ]; then
    log "seeding game profile and settings"
    mkdir -p "$USERDIR/Documents"
    cp -rn /work/docker/seed/TmForever "$USERDIR/Documents/"
fi

# TMInterface must pick the player profile by itself, or the game parks on the
# profile screen and the collector never sees the main menu: `autologin` is
# the account login, or the 1-based index in the profile list for a local
# profile without one.
mkdir -p "$USERDIR/Documents/TMInterface/Plugins" "$USERDIR/Documents/TMInterface/Scripts"
if [ ! -s "$USERDIR/Documents/TMInterface/config.txt" ]; then
    printf 'set "autologin" "1";\n' > "$USERDIR/Documents/TMInterface/config.txt"
fi

date -Is > "$MARK"
log "done"
