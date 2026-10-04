#!/bin/bash
# Bring up the headless desktop, the VNC exporter, the browser console, then
# the Wine prefix (first run installs the game). Stays in the foreground.
set -euo pipefail

# Stale sockets from a previous run of this container would fool the checks below.
rm -rf "$XDG_RUNTIME_DIR"
# /tmp survives a container restart. Xwayland's lock files and sockets from the
# last run would push it to :1, :2, ... while the game uses DISPLAY=:0, and
# wined3d with no X server fails as TmForever's "Could not find DirectX9".
rm -f /tmp/.X*-lock /tmp/.X11-unix/X*
mkdir -p "$XDG_RUNTIME_DIR" && chmod 700 "$XDG_RUNTIME_DIR"
mkdir -p "$HOME/.config/wayvnc"

log() { echo "[entrypoint] $*"; }

log "starting sway (headless, $WLR_RENDER_DRM_DEVICE)"
sway -c /etc/tmnf/sway.config >"$XDG_RUNTIME_DIR/sway.log" 2>&1 &
SWAY_PID=$!
for _ in $(seq 1 100); do
    swaymsg -q -t get_version 2>/dev/null && break
    kill -0 "$SWAY_PID" 2>/dev/null || { cat "$XDG_RUNTIME_DIR/sway.log"; exit 1; }
    sleep 0.2
done
swaymsg -q -t get_version 2>/dev/null || { log "sway never came up"; cat "$XDG_RUNTIME_DIR/sway.log"; exit 1; }

log "starting wayvnc on 127.0.0.1:5900"
wayvnc -o HEADLESS-1 127.0.0.1 5900 >"$XDG_RUNTIME_DIR/wayvnc.log" 2>&1 &

: "${NOVNC_USER:=tmnf}"
if [ -z "${NOVNC_PASSWORD:-}" ]; then
    log "NOVNC_PASSWORD is not set; refusing to expose the console without one"
    exit 1
fi
log "starting noVNC on :6080 (HTTP basic auth, user $NOVNC_USER)"
websockify --web /usr/share/novnc \
    --web-auth --auth-plugin BasicHTTPAuth --auth-source "$NOVNC_USER:$NOVNC_PASSWORD" \
    0.0.0.0:6080 127.0.0.1:5900 >"$XDG_RUNTIME_DIR/websockify.log" 2>&1 &

init-prefix.sh

log "ready. console: http://<host>:6080/vnc.html?autoconnect=1&resize=scale"
wait "$SWAY_PID"
