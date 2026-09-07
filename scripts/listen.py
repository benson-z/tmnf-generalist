"""Dev helper: hexdump whatever the plugin sends on a port.

    uv run python scripts/listen.py [port] [seconds]
"""

import socket
import sys
import time

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8477
seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("127.0.0.1", port))
server.listen(1)
server.settimeout(seconds)
print(f"listening on {port} for {seconds}s", flush=True)

conn, addr = server.accept()
print(f"connection from {addr}", flush=True)
conn.settimeout(seconds)

total = 0
deadline = time.monotonic() + seconds
try:
    while time.monotonic() < deadline:
        chunk = conn.recv(65536)
        if not chunk:
            print(f"EOF after {total} bytes", flush=True)
            break
        if total < 256:
            print(f"+{len(chunk)}: {chunk[:128].hex(' ')}", flush=True)
        total += len(chunk)
except TimeoutError:
    print("recv timeout", flush=True)
print(f"total {total} bytes", flush=True)
