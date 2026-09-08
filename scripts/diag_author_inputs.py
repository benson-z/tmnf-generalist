"""Dev helper: can dump_inputs pull the author run straight out of a map?

TMX maps embed the author's validation ghost. If TMInterface's dump_inputs
accepts a .Challenge.Gbx the way it accepts a .Replay.Gbx, then a map download
is a complete training sample on its own -- map plus demonstration, already
paired -- and no separate replay harvesting is needed.

    uv run python scripts/diag_author_inputs.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

from tmnf_collect import install, staging  # noqa: E402
from tmnf_collect.paths import detect  # noqa: E402
from tmnf_collect.session import Session  # noqa: E402

layout = detect()
install.install(layout)

MAP = "tmnf-collect/tmx-3358000.Challenge.Gbx"
candidates = [
    MAP,
    f"{layout.challenges_dir / 'tmnf-collect' / 'tmx-3358000.Challenge.Gbx'}",
    "tmx-3358000.Challenge.Gbx",
]

session = Session(port=8477, layout=layout)
session.start()
try:
    # Keep the console visible so its replies land in the log we read back.
    session.prepare(speed=1.0, hide_console=False)

    for index, path in enumerate(candidates):
        out_name = f"diag_author_{index}.txt"
        target = layout.scripts_dir / out_name
        target.unlink(missing_ok=True)
        print(f"\n>>> dump_inputs {path} {out_name}", flush=True)
        session._command(f"dump_inputs {path} {out_name}", settle=1.0)
        session._drain(4.0)
        if target.is_file() and target.stat().st_size > 0:
            text = target.read_text(encoding="utf-8")
            lines = text.splitlines()
            print(f"    WROTE {target.stat().st_size} bytes, {len(lines)} lines")
            print("    first:", lines[0] if lines else "(empty)")
            print("    last: ", lines[-1] if lines else "(empty)")
            analog = sum(1 for line in lines if " steer " in line)
            print(f"    analog steer commands: {analog} "
                  f"({'PAD' if analog else 'KEYBOARD'})")
        else:
            print("    no file written")

    time.sleep(1)
    log = session.console_log()
    Path("out").mkdir(exist_ok=True)
    Path("out/diag_author_inputs.log").write_text(log, encoding="utf-8")
    tail = [
        line for line in log.splitlines()
        if "dump_inputs" in line or "replay" in line.lower()
        or "input" in line.lower() or "error" in line.lower()
    ]
    print("\nconsole says:")
    for line in tail[-15:]:
        print("   ", line)
finally:
    session.close()
