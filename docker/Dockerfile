# TMNF collection rig for Linux: Wine under a headless GPU compositor,
# with a Proxmox-style noVNC console served from inside the container. The
# host only lends /dev/dri; everything else lives here.
FROM debian:trixie-slim

ARG DEBIAN_FRONTEND=noninteractive

RUN dpkg --add-architecture i386 \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates curl wget gnupg unzip xz-utils procps less nano git \
      # headless Wayland desktop, X bridge for Wine, VNC + browser console
      sway xwayland wayvnc novnc python3-websockify foot wl-clipboard grim xdotool \
      # encodes collected frames (frame_codec: hevc), and `tmnf-collect
      # video` renders a recorded run back into a video
      ffmpeg \
      # VAAPI video encoding, for frame_codec: hevc-vaapi
      mesa-va-drivers \
      # GPU: Mesa GL for wined3d (32-bit game needs the i386 driver too);
      # Vulkan for the compositor and for anything else that wants it
      libgl1-mesa-dri libgl1-mesa-dri:i386 libgl1 libgl1:i386 \
      mesa-vulkan-drivers mesa-vulkan-drivers:i386 \
      libegl1 libgles2 vulkan-tools \
      fonts-dejavu-core fonts-liberation \
 && rm -rf /var/lib/apt/lists/*

# WineHQ stable for trixie (multiarch build; 32-bit TMNF needs wine32)
RUN mkdir -pm755 /etc/apt/keyrings \
 && wget -qO /etc/apt/keyrings/winehq-archive.key https://dl.winehq.org/wine-builds/winehq.key \
 && wget -qNP /etc/apt/sources.list.d/ https://dl.winehq.org/wine-builds/debian/dists/trixie/winehq-trixie.sources \
 && apt-get update \
 && apt-get install -y --no-install-recommends winehq-stable \
 && rm -rf /var/lib/apt/lists/*

# No DXVK: TMNF's lit geometry renders black through it on this stack (sky
# and emissives fine, everything else unlit). Wine's own d3d9 over Mesa GL is
# correct, and the community Linux guides run TMF on wined3d too.

# uv + a managed Python
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_PROJECT_ENVIRONMENT=/home/tmnf/venv \
    UV_LINK_MODE=copy
RUN uv python install 3.11 && chmod -R a+rX /opt/uv

RUN useradd -m -u 1000 -s /bin/bash tmnf

COPY sway.config /etc/tmnf/sway.config
COPY entrypoint.sh init-prefix.sh tmnf-launch /usr/local/bin/
RUN chmod 755 /usr/local/bin/entrypoint.sh /usr/local/bin/init-prefix.sh /usr/local/bin/tmnf-launch

# Static session environment, so `docker exec` shells see the same desktop
# and prefix that the entrypoint started.
ENV HOME=/home/tmnf \
    USER=tmnf \
    XDG_RUNTIME_DIR=/tmp/xdg \
    WAYLAND_DISPLAY=wayland-1 \
    SWAYSOCK=/tmp/xdg/sway-ipc.sock \
    DISPLAY=:0 \
    WINEPREFIX=/home/tmnf/.wine \
    WINEDEBUG=-all \
    WINEDLLOVERRIDES=mscoree,mshtml= \
    WLR_BACKENDS=headless \
    WLR_LIBINPUT_NO_DEVICES=1 \
    WLR_RENDER_DRM_DEVICE=/dev/dri/renderD128 \
    TMNF_GAME_OUTPUT=HEADLESS-1 \
    TMNF_CONFIG=/work/docker/tmnf-collect.yaml

USER tmnf
WORKDIR /work
EXPOSE 6080
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
