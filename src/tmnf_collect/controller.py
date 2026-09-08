"""Controller side of the plugin bridge.

One :class:`Controller` owns one TCP port and therefore one game instance. It
accepts the plugin's connection, hands out console commands, and yields the
messages the plugin streams back.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from contextlib import contextmanager

from . import protocol
from .protocol import Event, Hello, MessageReader, Pong, Sample


class Controller:
    """A listening socket that one game instance connects back to."""

    def __init__(self, port: int, host: str = "127.0.0.1") -> None:
        self.host = host
        self.port = port
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((host, port))
        self._server.listen(1)
        self._conn: socket.socket | None = None
        self._reader: MessageReader | None = None
        self.hello: Hello | None = None

    # -- lifecycle ---------------------------------------------------------

    def accept(self, timeout: float = 120.0) -> Hello:
        """Wait for the plugin to dial in and return its HELLO."""
        self._server.settimeout(timeout)
        conn, _ = self._server.accept()
        conn.settimeout(None)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._conn = conn
        self._reader = MessageReader(conn)

        message = self._reader.read()
        if not isinstance(message, Hello):
            raise protocol.ProtocolError(f"expected HELLO, got {message!r}")
        if message.proto_version != protocol.PROTO_VERSION:
            raise protocol.ProtocolError(
                f"plugin speaks protocol {message.proto_version}, "
                f"controller speaks {protocol.PROTO_VERSION} "
                "(reinstall the plugin with `tmnf-collect install-plugin`)"
            )
        self.hello = message
        return message

    def close(self) -> None:
        for sock in (self._conn, self._server):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        self._conn = None
        self._reader = None

    # -- outbound ----------------------------------------------------------

    def _send(self, payload: bytes) -> None:
        if self._conn is None:
            raise ConnectionError("no plugin connected")
        self._conn.sendall(payload)

    def command(self, text: str) -> None:
        """Run a TMInterface console command inside the game."""
        self._send(protocol.encode_command(text))

    def configure(
        self,
        *,
        collect: bool | None = None,
        period_ms: int | None = None,
        width: int | None = None,
        height: int | None = None,
        force_render: bool | None = None,
    ) -> None:
        settings: dict[str, object] = {}
        if collect is not None:
            settings["collect"] = collect
        if period_ms is not None:
            settings["period"] = period_ms
        if width is not None:
            settings["width"] = width
        if height is not None:
            settings["height"] = height
        if force_render is not None:
            settings["force_render"] = force_render
        if settings:
            self._send(protocol.encode_config(**settings))

    def play(self, script_name: str) -> None:
        """Have the plugin replay an input script, bypassing key bindings."""
        self._send(protocol.encode_play(script_name))

    def focus(self) -> None:
        """Bring this instance's window forward."""
        self._send(protocol.encode_focus())

    def ping(self, payload: str = "ping") -> None:
        self._send(protocol.encode_ping(payload))

    # -- inbound -----------------------------------------------------------

    def messages(self) -> Iterator[Sample | Event | Pong | Hello]:
        """Yield messages until the plugin disconnects."""
        if self._reader is None:
            raise ConnectionError("no plugin connected")
        while True:
            try:
                yield self._reader.read()
            except (ConnectionError, OSError):
                return

    def poll(self, timeout: float) -> Sample | Event | Pong | Hello | None:
        """Read one message, or return None if none arrived within ``timeout``.

        A timeout leaves any partial message buffered, so the next call resumes
        cleanly.
        """
        if self._reader is None or self._conn is None:
            raise ConnectionError("no plugin connected")
        self._conn.settimeout(timeout)
        try:
            return self._reader.read()
        except (TimeoutError, socket.timeout):
            return None
        finally:
            self._conn.settimeout(None)


@contextmanager
def controller(port: int) -> Iterator[Controller]:
    ctrl = Controller(port)
    try:
        yield ctrl
    finally:
        ctrl.close()
