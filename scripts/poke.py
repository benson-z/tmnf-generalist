"""Dev helper: attach to a running game instance and send it commands.

    uv run python scripts/poke.py 8477 'log "hi"' 'map A01-Race.Challenge.Gbx'
"""

import sys
import time

sys.path.insert(0, "src")

from tmnf_collect.controller import Controller  # noqa: E402
from tmnf_collect.protocol import Event, Sample  # noqa: E402

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8477
commands = sys.argv[2:]

controller = Controller(port)
print(f"waiting for plugin on {port}", flush=True)
hello = controller.accept(timeout=120)
print(f"hello: {hello}", flush=True)

for command in commands:
    if command.startswith("sleep:"):
        time.sleep(float(command.split(":", 1)[1]))
        continue
    print(f"-> {command}", flush=True)
    controller.command(command)
    time.sleep(0.5)

deadline = time.monotonic() + 20
for message in controller.messages():
    if isinstance(message, Sample):
        print(
            f"sample t={message.race_time} speed={message.display_speed} "
            f"{message.width}x{message.height} {len(message.pixels)}B",
            flush=True,
        )
    elif isinstance(message, Event):
        print(f"event {message.name} t={message.race_time} arg={message.arg}", flush=True)
    else:
        print(message, flush=True)
    if time.monotonic() > deadline:
        break
controller.close()
