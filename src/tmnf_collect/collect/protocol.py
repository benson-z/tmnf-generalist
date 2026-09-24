"""Wire protocol between the in-game AngelScript plugin and this controller.

The plugin dials in, so the controller is the TCP server. Everything is
little-endian and self-delimiting; see ``plugin/TMNFCollect.as`` for the
writer side.
"""

from __future__ import annotations

import socket
import struct
from dataclasses import dataclass

PROTO_VERSION = 5

# plugin -> controller
MSG_HELLO = 0x01
MSG_SAMPLE = 0x02
MSG_EVENT = 0x03
MSG_TICK = 0x06

# controller -> plugin
CMD_COMMAND = 0x10
CMD_CONFIG = 0x11
CMD_FOCUS = 0x14

# event kinds
EV_RUN_START = 1
EV_CHECKPOINT = 2
EV_FINISH = 3
EV_GAMESTATE = 4
EV_RUN_RESET = 5

EVENT_NAMES = {
    EV_RUN_START: "run_start",
    EV_CHECKPOINT: "checkpoint",
    EV_FINISH: "finish",
    EV_GAMESTATE: "gamestate",
    EV_RUN_RESET: "run_reset",
}

_HELLO = struct.Struct("<IiIH")
_TICK = struct.Struct("<iB")
_EVENT = struct.Struct("<Bii")
_SAMPLE = struct.Struct("<IiI12f4Bii3fI2Bi7fiIHHI")
assert _SAMPLE.size == 138, _SAMPLE.size


@dataclass(frozen=True)
class Hello:
    proto_version: int
    instance_id: int
    pid: int
    token: str


@dataclass(frozen=True)
class Event:
    kind: int
    race_time: int
    arg: int

    @property
    def name(self) -> str:
        return EVENT_NAMES.get(self.kind, f"unknown({self.kind})")


@dataclass(frozen=True)
class Tick:
    """One physics step: what was held, without a frame.

    The simulation steps every 10ms, so these arrive at 100Hz -- five times the
    frame rate. Keyboard driving is full of taps shorter than a 50ms sample
    period, and at 20Hz those either vanish or get attributed to the wrong
    moment.
    """

    race_time: int
    up: bool
    down: bool
    left: bool
    right: bool


@dataclass(frozen=True)
class Sample:
    """One 20 Hz sample: telemetry, inputs and the frame that goes with them."""

    seq: int
    race_time: int
    display_speed: int  # km/h, as shown in game
    velocity: tuple[float, float, float]  # m/s, world space
    position: tuple[float, float, float]
    yaw_pitch_roll: tuple[float, float, float]
    local_speed: tuple[float, float, float]  # m/s, car space
    up: bool
    down: bool
    left: bool
    right: bool
    gas: int  # analog gas as seen by the game, [-65536, 65536]
    steer: int  # analog steer, [-65536, 65536]
    gas_f: float  # the car's resolved gas/brake/steer, [-1, 1]
    brake_f: float
    steer_f: float
    checkpoints: int
    finished: bool
    sliding: bool
    gearbox: int
    camera_position: tuple[float, float, float]
    camera_yaw_pitch_roll: tuple[float, float, float]
    camera_fov: float
    render_race_time: int  # tick the game had reached when the frame was drawn
    dropped: int  # cumulative sample points that never became a frame
    width: int
    height: int
    pixels: bytes  # BGRA, top-to-bottom, width * height * 4 bytes


class ProtocolError(RuntimeError):
    pass


class MessageReader:
    """Buffered reader for one plugin connection.

    Parsing never consumes from the buffer until a whole message is present, so
    a read that times out part-way through a frame can be resumed safely.
    """

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._buf = bytearray()

    def read(self) -> Hello | Sample | Tick | Event:
        while True:
            parsed = self._try_parse()
            if parsed is not None:
                message, consumed = parsed
                del self._buf[:consumed]
                return message
            # Large enough to lift a whole frame out in one call, so the
            # plugin's blocked write is released as early as possible.
            chunk = self._sock.recv(1 << 20)
            if not chunk:
                raise ConnectionError("plugin closed the connection")
            self._buf += chunk

    def _try_parse(self) -> tuple[Hello | Sample | Tick | Event, int] | None:
        """Parse one message from the head of the buffer, if it is all there."""
        buf = self._buf
        if len(buf) < 1:
            return None
        kind = buf[0]

        if kind == MSG_SAMPLE:
            end = 1 + _SAMPLE.size
            if len(buf) < end:
                return None
            fields = _SAMPLE.unpack_from(buf, 1)
            pixel_bytes = fields[-1]
            if len(buf) < end + pixel_bytes:
                return None
            pixels = bytes(buf[end : end + pixel_bytes])
            sample = Sample(
                seq=fields[0],
                race_time=fields[1],
                display_speed=fields[2],
                velocity=fields[3:6],
                position=fields[6:9],
                yaw_pitch_roll=fields[9:12],
                local_speed=fields[12:15],
                up=bool(fields[15]),
                down=bool(fields[16]),
                left=bool(fields[17]),
                right=bool(fields[18]),
                gas=fields[19],
                steer=fields[20],
                gas_f=fields[21],
                brake_f=fields[22],
                steer_f=fields[23],
                checkpoints=fields[24],
                finished=bool(fields[25]),
                sliding=bool(fields[26]),
                gearbox=fields[27],
                camera_position=fields[28:31],
                camera_yaw_pitch_roll=fields[31:34],
                camera_fov=fields[34],
                render_race_time=fields[35],
                dropped=fields[36],
                width=fields[37],
                height=fields[38],
                pixels=pixels,
            )
            return sample, end + pixel_bytes

        if kind == MSG_TICK:
            end = 1 + _TICK.size
            if len(buf) < end:
                return None
            race_time, keys = _TICK.unpack_from(buf, 1)
            tick = Tick(
                race_time=race_time,
                up=bool(keys & 1),
                down=bool(keys & 2),
                left=bool(keys & 4),
                right=bool(keys & 8),
            )
            return tick, end

        if kind == MSG_EVENT:
            end = 1 + _EVENT.size
            if len(buf) < end:
                return None
            event_kind, race_time, arg = _EVENT.unpack_from(buf, 1)
            return Event(kind=event_kind, race_time=race_time, arg=arg), end

        if kind == MSG_HELLO:
            end = 1 + _HELLO.size
            if len(buf) < end:
                return None
            proto, instance_id, pid, token_len = _HELLO.unpack_from(buf, 1)
            if len(buf) < end + token_len:
                return None
            token = bytes(buf[end : end + token_len]).decode("utf-8", "replace")
            hello = Hello(
                proto_version=proto,
                instance_id=instance_id,
                pid=pid,
                token=token,
            )
            return hello, end + token_len

        raise ProtocolError(f"unknown message id 0x{kind:02x}")


def encode_command(text: str) -> bytes:
    """A TMInterface console command for the plugin to run."""
    body = text.encode("utf-8")
    return struct.pack("<BH", CMD_COMMAND, len(body)) + body


def encode_config(**settings: object) -> bytes:
    """Runtime plugin settings: collect, period, width, height, hide_ui."""
    parts = []
    for key, value in settings.items():
        if isinstance(value, bool):
            value = int(value)
        parts.append(f"{key}={value}")
    body = ";".join(parts).encode("utf-8")
    return struct.pack("<BH", CMD_CONFIG, len(body)) + body


def encode_focus() -> bytes:
    """Bring this instance's game window to the foreground."""
    return struct.pack("<BH", CMD_FOCUS, 0)
