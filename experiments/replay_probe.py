"""Validate full replay finishes, 100 Hz inputs, and camera agreement."""
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from tmnf_collect import collect, install, protocol
from tmnf_collect.paths import detect
from tmnf_collect.session import Session


class TimedSession(Session):
    def record_map_run(self, *args, **kwargs):
        arrivals = []
        callback = kwargs['on_sample']
        reset = kwargs.get('on_reset')
        def receive(sample):
            arrivals.append((sample.race_time, time.monotonic()))
            callback(sample)
        def restart():
            arrivals.clear()
            if reset:
                reset()
        kwargs.update(on_sample=receive, on_reset=restart)
        result = super().record_map_run(*args, **kwargs)
        self.capture_speedup = None
        if len(arrivals) > 1:
            self.capture_speedup = ((arrivals[-1][0] - arrivals[0][0]) / 1000
                                    / (arrivals[-1][1] - arrivals[0][1]))
        return result


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('replays', nargs='+', type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    layout = detect()
    install.install(layout)
    plan = collect.plan(args.replays, layout=layout, strip_intros=True)
    if plan.skipped:
        raise RuntimeError(plan.skipped)
    results = {}
    with TimedSession(port=8499, instance_id=19) as session:
        session.prepare()
        for mode, speed in [('natural1', 1), ('barrier5', 5), ('barrier20', 20)]:
            session._command(f'set speed {speed}')
            session._ctrl._send(protocol.encode_config(
                frame_barrier=mode.startswith('barrier')))
            for job in plan.jobs:
                result = collect.run_job(session, job, args.out / mode, timeout=240)
                value = asdict(result)
                value['capture_speedup'] = session.capture_speedup
                results[f'{mode}/{job.output_name}'] = value
                print(mode, json.dumps(value), flush=True)
                (args.out / 'results.json').write_text(json.dumps(results, indent=2))
        session._command('set speed 1')
    for job in plan.jobs:
        base = args.out / 'natural1' / job.output_name
        reference = {r['race_time']: r for r in rows(base / 'samples.jsonl')
                     if r['race_time'] == r['render_race_time']}
        base_ticks = rows(base / 'inputs.jsonl')
        for mode in ('barrier5', 'barrier20'):
            root = args.out / mode / job.output_name
            samples = rows(root / 'samples.jsonl')
            info = results[f'{mode}/{job.output_name}']
            info['identical_input_stream'] = rows(root / 'inputs.jsonl') == base_ticks
            info['exact_frames'] = sum(r['race_time'] == r['render_race_time'] for r in samples)
            paired = [(reference[r['race_time']], r) for r in samples
                      if r['race_time'] in reference and r['render_race_time'] == r['race_time']]
            info['matched_frames'] = len(paired)
            for key in ('position', 'camera_position'):
                delta = [np.linalg.norm(np.array(a[key])-b[key]) for a,b in paired]
                if delta:
                    info[key] = dict(mean=float(np.mean(delta)), max=float(np.max(delta)))
    (args.out / 'results.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
