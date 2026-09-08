"""Dev helper: will the game write out a map that is only inside a replay?

TMNF can play a downloaded replay without having its map installed, because
the challenge is embedded in the replay file. If validating such a replay makes
the game materialise the map on disk, the collector can use that instead of
needing every map up front.

    uv run python scripts/diag_extract_map.py <replay.Gbx>
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

from tmnf_collect import install, replays, staging  # noqa: E402
from tmnf_collect.paths import detect  # noqa: E402
from tmnf_collect.session import Session  # noqa: E402

layout = detect()
install.install(layout)

source = Path(sys.argv[1] if len(sys.argv) > 1 else "testdata/test").resolve()
if source.is_dir():
    source = replays.discover_replays(source)[0]
info = replays.read_replay(source)
print(f"replay {source.name}\n  uid {info.map_uid}  time {info.race_time} ms")

roots = [layout.tracks_dir, layout.user_dir]


def snapshot() -> dict[Path, float]:
    seen = {}
    for root in roots:
        for path in root.rglob("*"):
            if path.is_file() and path.name.lower().endswith(".challenge.gbx"):
                seen[path] = path.stat().st_mtime
    return seen


before = snapshot()
print(f"  {len(before)} challenge files under the user Tracks folder before")

staged = staging.stage_replay(source, layout)
session = Session(port=8477, layout=layout)
session.start()
try:
    session.prepare(speed=1.0, hide_console=False)
    print(f"  validate_replay {staged}", flush=True)
    session._command(f"validate_replay {staged}")
    session._drain(25.0)
    print(f"  game state after validation: {session.game_state}", flush=True)
    time.sleep(2)
finally:
    session.close()

time.sleep(1)
after = snapshot()
new = sorted(set(after) - set(before))
print(f"  {len(after)} challenge files after; {len(new)} new")
for path in new:
    parsed = replays.read_challenge(path)
    match = parsed and parsed[0] == info.map_uid
    print(f"    {'MATCH' if match else '     '} {path}")
    if parsed:
        print(f"           uid={parsed[0]} name={parsed[1]!r}")
if not new:
    print("  the game did not write the map out")
