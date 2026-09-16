"""Repeat a deterministic drive with different render schedules.

Run with the project's Python: experiments/render_probe.py --out out/render-probe
Installs the current plugin, stages a stock map/script, and owns one game process.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tmnf_collect import install, staging, protocol
from tmnf_collect.dataset import sample_record
from tmnf_collect.paths import detect
from tmnf_collect.session import Session
from tmnf_collect.smoke import SMOKE_SCRIPT, _save_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--modes', default='natural1,natural1b,barrier5')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    layout = detect()
    install.install(layout)
    _, track = staging.stage_bootstrap(layout)
    script = staging.write_script('tmnf_render_probe.txt', SMOKE_SCRIPT, layout)
    modes = {'natural1': 1, 'natural1b': 1, 'barrier5': 5}
    results = {}
    with Session(port=8499, instance_id=19) as session:
        session.prepare()
        for name in args.modes.split(','):
            speed = modes[name]
            session._command(f'set speed {speed}')
            session._ctrl._send(protocol.encode_config(
                frame_barrier=name.startswith('barrier')))
            begin = time.monotonic()
            captured, arrival = [], []
            def receive(frame):
                captured.append(frame)
                arrival.append(time.monotonic())
            run = session.record_map_run(track, script, max_samples=120, timeout=90,
                                         on_sample=receive,
                                         on_reset=lambda: (captured.clear(), arrival.clear()))
            run.samples = captured
            wall = time.monotonic() - begin
            rows = [sample_record(s, i, 0, 0) for i, s in enumerate(run.samples)]
            (args.out / f'{name}.json').write_text(json.dumps(rows), encoding='utf-8')
            for s in run.samples[::30]:
                _save_frame(s, args.out / f'{name}-{s.race_time}.png')
            summary = dict(samples=len(rows), wall_seconds=wall, dropped=run.dropped,
                           driving=run.driving, clean_start=run.clean_start,
                           exact=sum(s.race_time == s.render_race_time for s in run.samples))
            if len(arrival) > 1:
                summary['capture_seconds'] = arrival[-1] - arrival[0]
                summary['capture_speedup'] = ((captured[-1].race_time - captured[0].race_time)
                                              / 1000 / summary['capture_seconds'])
            results[name] = summary
            print(name, json.dumps(summary), flush=True)
        session._command('set speed 1')
    baseline = json.loads((args.out / 'natural1.json').read_text())
    baseline = {r['race_time']: r for r in baseline if r['race_time'] == r['render_race_time']}
    for name, summary in results.items():
        rows = json.loads((args.out / f'{name}.json').read_text())
        pairs = [(baseline[r['race_time']], r) for r in rows
                 if r['race_time'] in baseline and r['race_time'] == r['render_race_time']]
        summary['matched_exact_frames'] = len(pairs)
        for key in ('position', 'camera_position'):
            distances = [np.linalg.norm(np.array(a[key]) - b[key]) for a, b in pairs]
            if distances:
                summary[key] = dict(mean=float(np.mean(distances)), max=float(np.max(distances)))
        angles = [np.linalg.norm((np.array(a['camera_yaw_pitch_roll'])
                                 - b['camera_yaw_pitch_roll'] + np.pi) % (2 * np.pi) - np.pi)
                  * 180 / np.pi for a, b in pairs]
        if angles:
            summary['camera_angles_degrees'] = dict(mean=float(np.mean(angles)), max=float(np.max(angles)))
    (args.out / 'summary.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
