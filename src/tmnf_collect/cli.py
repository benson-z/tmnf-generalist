"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

from . import (
    camera_check,
    config as config_mod,
    collect as collect_mod,
    inputs as inputs_mod,
    install,
    launcher,
    smoke,
    stats as stats_mod,
    tmx,
    verify,
    video,
)
from .paths import detect
from . import staging

# Set when the parser is built; used to record what a run was configured with.
_COLLECT_PARSER: argparse.ArgumentParser | None = None


def _cmd_paths(args: argparse.Namespace) -> int:
    layout = detect(game=args.game, profile=args.profile)
    for field, value in vars(layout).items():
        print(f"{field:16} {value}")
    return 0


def _cmd_launch(args: argparse.Namespace) -> int:
    layout = detect(game=args.game, profile=args.profile)
    instance = launcher.launch(
        port=args.port, instance_id=args.id, layout=layout
    )
    print(f"pid={instance.pid} port={instance.port} token={instance.token}")
    if args.wait:
        while instance.is_alive():
            time.sleep(1.0)
        print("game exited")
    return 0


def _cmd_kill(_: argparse.Namespace) -> int:
    print(f"killed {launcher.kill_all()} game process(es)")
    return 0


def _cmd_install_plugin(args: argparse.Namespace) -> int:
    layout = detect(game=args.game, profile=args.profile)
    print(f"installed {install.install(layout)}")
    return 0


def _cmd_collect(args: argparse.Namespace) -> int:
    summary = collect_mod.collect(
        Path(args.replays),
        Path(args.out),
        port=args.port,
        width=args.width,
        height=args.height,
        period_ms=args.period,
        speed=args.speed,
        force_render=args.force_render,
        hide_ui=not args.show_ui,
        unfocused_fps_limit=args.fps_limit,
        hide_console=not args.show_console,
        camera=args.camera,
        settings=config_mod.effective(args, _COLLECT_PARSER),
        strip_intros=not args.keep_intros,
        fetch_maps=args.fetch_maps,
        limit=args.limit,
        image_format=args.format,
        quality=args.quality,
        capture_log=args.log,
        instances=args.instances,
        resume=not args.no_resume,
        progress=lambda r: print(
            f"  [{r.instance}] {r.status:14} {r.output_name} "
            f"samples={r.samples} {r.seconds}s "
            f"attempts={r.attempts}"
            + (f" arming_retries={r.preroll_restarts}" if r.preroll_restarts else "")
            + (f" {r.detail}" if r.detail else ""),
            flush=True,
        ),
    )
    trimmed = {k: v for k, v in summary.items() if k != "results"}
    print(json.dumps(trimmed, indent=2))
    return 0 if not summary["by_status"].get("error") else 1


def _cmd_filter(args: argparse.Namespace) -> int:
    layout = detect(game=args.game, profile=args.profile)
    result = inputs_mod.filter_replays(
        Path(args.replays),
        want=args.inputs,
        max_seconds=args.max_seconds,
        port=args.port,
        layout=layout,
        dry_run=args.dry_run,
    )
    print(
        json.dumps(
            {
                "want": args.inputs,
                "max_seconds": args.max_seconds,
                "counts": result.counts,
                "kept": len(result.kept),
                "moved": len(result.moved),
                "read_offline": result.read_offline,
                "seconds": result.seconds,
                "dry_run": args.dry_run,
            },
            indent=2,
        )
    )
    for entry in result.moved[:20]:
        print(f"  {entry['kind']:11} {Path(entry['replay']).name}")
    return 0 if result.kept else 1


def _by_tag(picked: list) -> dict:
    """Maps and gameplay seconds per tag, heaviest first."""
    maps: Counter[str] = Counter()
    seconds: Counter[str] = Counter()
    for item in picked:
        name = tmx.tag_names(item.track.tags)
        maps[name] += 1
        seconds[name] += item.replay.time_ms / 1000
    return {
        name: {"maps": maps[name], "seconds": round(seconds[name], 1)}
        for name in sorted(maps, key=lambda n: -seconds[n])
    }


def _cmd_harvest(args: argparse.Namespace) -> int:
    layout = detect(game=args.game, profile=args.profile)
    replays_dir = Path(args.out)
    try:
        include_tags = tmx.parse_tags(args.tags)
        exclude_tags = tmx.parse_tags(args.exclude_tags)
    except tmx.TmxError as exc:
        print(exc)
        return 2
    result = tmx.harvest(
        maps_into=staging.challenges_dir(layout),
        replays_into=replays_dir,
        limit=args.limit,
        min_author_time=int(args.min_seconds * 1000) if args.min_seconds else None,
        max_author_time=int(args.max_seconds * 1000) if args.max_seconds else None,
        min_awards=args.min_awards,
        include_tags=include_tags,
        exclude_tags=exclude_tags,
        prefer=args.prefer,
        dry_run=args.dry_run,
    )

    for item in result.picked:
        kind = "author run" if item.is_author_run else item.replay.user
        print(
            f"  {item.track.awards:5} awards  "
            f"run {item.replay.time_ms / 1000:6.2f}s  "
            f"(author {item.track.author_time / 1000:6.2f}s)  "
            f"{tmx.tag_names(item.track.tags)[:16]:16} "
            f"{kind[:18]:18} {item.track.name[:40]}",
            flush=True,
        )
    print(
        json.dumps(
            {
                "tracks_considered": result.tracks_considered,
                "picked": len(result.picked),
                # Seconds are the chosen replays' own times, so this is the
                # gameplay each tag will cost to record, before overheads.
                "by_tag": _by_tag(result.picked),
                "seconds_to_record": round(
                    sum(i.replay.time_ms for i in result.picked) / 1000, 1
                ),
                "author_runs": sum(1 for i in result.picked if i.is_author_run),
                "skipped": len(result.skipped),
                "dry_run": args.dry_run,
                "replays_dir": str(replays_dir),
                "maps_dir": str(staging.challenges_dir(layout)),
            },
            indent=2,
        )
    )
    for entry in result.skipped[:10]:
        print(f"  skipped {entry.get('track')}: {entry['reason']}")
    return 0 if result.picked else 1


def _cmd_video(args: argparse.Namespace) -> int:
    summary = video.render_run(
        Path(args.run),
        Path(args.out),
        fps=args.fps,
        scale=args.scale,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    report = verify.check_dataset(Path(args.dataset), period_ms=args.period)
    checks = report.pop("checks")
    print(json.dumps(report, indent=2))
    for check in checks:
        mark = "PASS" if check.ok else "FAIL"
        print(
            f"  {mark} {check.name:28} {check.rows:5} rows "
            f"{check.frames:5} frames  {check.duration_ms / 1000:7.2f}s"
        )
        for problem in check.problems:
            print(f"        - {problem}")
    return 0 if report["failed"] == 0 else 1


def _cmd_stats(args: argparse.Namespace) -> int:
    root = Path(args.dataset)
    runs, unreadable = stats_mod.read_runs(root, only_ok=not args.all)
    summary = stats_mod.summarize(runs, bin_seconds=args.bin)
    print(stats_mod.histogram(summary, width=args.width))
    if not args.no_tags:
        tags = stats_mod.tag_counts(root, runs, refresh=args.refresh_tags)
        summary["tag_counts"] = tags
        print()
        print(stats_mod.format_tags(tags))
    if unreadable:
        print(f"\n{len(unreadable)} directories had no readable meta.json")
    if args.png:
        out = stats_mod.render_png(summary, Path(args.png))
        print(f"\nwrote {out}")
    if args.json:
        Path(args.json).write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(f"wrote {args.json}")
    return 0


def _cmd_clean(args: argparse.Namespace) -> int:
    summary = verify.clean_dataset(
        Path(args.dataset),
        period_ms=args.period,
        delete=args.delete,
        dry_run=args.dry_run,
    )
    details = summary.pop("details")
    print(json.dumps(summary, indent=2))
    for entry in details:
        print(f"  {entry['reason']:12} {entry['run']:24} {entry['problems'][0]}")
    return 0


def _cmd_camera(args: argparse.Namespace) -> int:
    summary = camera_check.run(
        out_dir=Path(args.out),
        port=args.port,
        width=args.width,
        height=args.height,
        samples=args.samples,
        speed_up=args.speed_up,
    )
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_smoke(args: argparse.Namespace) -> int:
    summary = smoke.run(
        out_dir=Path(args.out),
        port=args.port,
        width=args.width,
        height=args.height,
        period_ms=args.period,
        force_render=args.force_render,
        max_samples=args.samples,
        speed=args.speed,
        keep_open=args.keep_open,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["samples"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tmnf-collect")
    parser.add_argument("--game", default="TmForever")
    parser.add_argument("--profile", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    p_paths = sub.add_parser("paths", help="show detected install locations")
    p_paths.set_defaults(func=_cmd_paths)

    p_launch = sub.add_parser("launch", help="start one game instance")
    p_launch.add_argument("--port", type=int, default=8477)
    p_launch.add_argument("--id", type=int, default=0)
    p_launch.add_argument("--wait", action="store_true")
    p_launch.set_defaults(func=_cmd_launch)

    p_kill = sub.add_parser("kill", help="terminate all game processes")
    p_kill.set_defaults(func=_cmd_kill)

    p_install = sub.add_parser(
        "install-plugin", help="copy the collector plugin into TMInterface"
    )
    p_install.set_defaults(func=_cmd_install_plugin)

    p_smoke = sub.add_parser(
        "smoke", help="end-to-end check of the game -> controller bridge"
    )
    p_smoke.add_argument("--out", default="out/smoke")
    p_smoke.add_argument("--port", type=int, default=8477)
    p_smoke.add_argument("--width", type=int, default=320)
    p_smoke.add_argument("--height", type=int, default=240)
    p_smoke.add_argument("--period", type=int, default=50)
    p_smoke.add_argument("--samples", type=int, default=120)
    p_smoke.add_argument("--speed", type=float, default=1.0)
    p_smoke.add_argument(
        "--force-render",
        action="store_true",
        help="drive rendering from ticks instead of using the game's own frames",
    )
    p_smoke.add_argument("--keep-open", action="store_true")
    p_smoke.set_defaults(func=_cmd_smoke)

    p_camera = sub.add_parser(
        "camera-check",
        help="verify ForceGameRender does not disturb the camera",
    )
    p_camera.add_argument("--out", default="out/camera")
    p_camera.add_argument("--port", type=int, default=8477)
    p_camera.add_argument("--width", type=int, default=320)
    p_camera.add_argument("--height", type=int, default=240)
    p_camera.add_argument("--samples", type=int, default=100)
    p_camera.add_argument("--speed-up", type=float, default=5.0)
    p_camera.set_defaults(func=_cmd_camera)

    p_collect = sub.add_parser(
        "collect", help="record every replay under a folder into a dataset"
    )
    p_collect.add_argument("replays", help="a replay file, or a folder of them")
    p_collect.add_argument("--out", default="out/dataset")
    p_collect.add_argument("--port", type=int, default=8477)
    p_collect.add_argument("--width", type=int, default=320)
    p_collect.add_argument("--height", type=int, default=240)
    p_collect.add_argument("--period", type=int, default=50)
    p_collect.add_argument("--speed", type=float, default=1.0)
    p_collect.add_argument("--limit", type=int, default=None)
    p_collect.add_argument("--format", default="jpeg", choices=["jpeg", "png"])
    p_collect.add_argument("--quality", type=int, default=80)
    p_collect.add_argument(
        "--instances",
        type=int,
        default=1,
        help="game instances to run in parallel, one port each",
    )
    p_collect.add_argument(
        "--no-resume",
        action="store_true",
        help="re-record replays that already have a successful run",
    )
    p_collect.add_argument(
        "--fps-limit",
        action="store_true",
        help="keep the game's unfocused FPS limit on (slower, but paces rendering normally)",
    )
    p_collect.add_argument(
        "--show-console",
        action="store_true",
        help="leave the TMInterface console on screen during collection",
    )
    p_collect.add_argument(
        "--fetch-maps",
        action="store_true",
        help="download missing maps from tmnf.exchange by UID",
    )
    p_collect.add_argument(
        "--keep-intros",
        action="store_true",
        help=(
            "stage maps untouched. By default their MediaTracker clips are "
            "stripped from the staged copy, which removes the intro flythrough "
            "(about 20s a map) and the in-race clips that hijack the camera; "
            "the map keeps its UID and its blocks"
        ),
    )
    p_collect.add_argument(
        "--camera",
        type=int,
        default=1,
        help=(
            "race camera 1-9. 1 sits further back and higher than 2 (7.37m vs "
            "6.24m behind, 3.29m vs 1.70m above); 0 leaves whatever the "
            "profile last used"
        ),
    )
    p_collect.add_argument("--force-render", action="store_true")
    p_collect.add_argument(
        "--show-ui",
        action="store_true",
        help="keep the in-game speedometer and clock in the frames",
    )
    p_collect.add_argument(
        "--log",
        action="store_true",
        help="save the in-game console log to gamelog.txt",
    )
    p_collect.set_defaults(func=_cmd_collect)
    global _COLLECT_PARSER
    _COLLECT_PARSER = p_collect

    p_verify = sub.add_parser(
        "verify", help="check a recorded dataset on disk is complete and in order"
    )
    p_verify.add_argument("dataset", help="a dataset directory produced by collect")
    p_verify.add_argument("--period", type=int, default=50)
    p_verify.set_defaults(func=_cmd_verify)

    p_stats = sub.add_parser(
        "stats", help="graph how a dataset's recorded time is distributed"
    )
    p_stats.add_argument("dataset", help="a dataset directory produced by collect")
    p_stats.add_argument(
        "--bin", type=float, default=10.0, help="histogram bin width in seconds"
    )
    p_stats.add_argument(
        "--width", type=int, default=48, help="text histogram width in columns"
    )
    p_stats.add_argument("--png", default=None, help="also write the graph here")
    p_stats.add_argument("--json", default=None, help="also write the numbers here")
    p_stats.add_argument(
        "--all",
        action="store_true",
        help="include runs that did not reproduce their replay",
    )
    p_stats.add_argument(
        "--no-tags",
        action="store_true",
        help="skip the map-tag breakdown, which asks TMX the first time",
    )
    p_stats.add_argument(
        "--refresh-tags",
        action="store_true",
        help="re-ask TMX for tags instead of using the cached ones",
    )
    p_stats.set_defaults(func=_cmd_stats)

    p_clean = sub.add_parser(
        "clean",
        help="move runs that fail verify out of a dataset",
    )
    p_clean.add_argument("dataset", help="a dataset directory produced by collect")
    p_clean.add_argument("--period", type=int, default=50)
    p_clean.add_argument(
        "--delete",
        action="store_true",
        help="remove them instead of moving them to <dataset>.rejected",
    )
    p_clean.add_argument(
        "--dry-run", action="store_true", help="report without touching anything"
    )
    p_clean.set_defaults(func=_cmd_clean)

    p_video = sub.add_parser(
        "video",
        help="render one recorded run to video with its inputs drawn on",
    )
    p_video.add_argument("run", help="a single run directory inside a dataset")
    p_video.add_argument("--out", default="out/run.mp4")
    p_video.add_argument("--fps", type=int, default=20)
    p_video.add_argument("--scale", type=int, default=3)
    p_video.add_argument("--limit", type=int, default=None)
    p_video.set_defaults(func=_cmd_video)

    p_harvest = sub.add_parser(
        "harvest",
        help="pick maps on TMX and download a demonstration replay for each",
    )
    p_harvest.add_argument("--out", default="testdata/harvest")
    p_harvest.add_argument("--limit", type=int, default=25)
    p_harvest.add_argument(
        "--min-awards",
        type=int,
        default=5,
        help="award count is the best available quality signal",
    )
    p_harvest.add_argument("--min-seconds", type=float, default=25.0)
    p_harvest.add_argument(
        "--max-seconds",
        type=float,
        default=75.0,
        help="long maps cost proportionally more to record and teach less",
    )
    p_harvest.add_argument(
        "--tags",
        default="",
        help=(
            "keep only maps carrying one of these tags, by name or id "
            "(e.g. Tech,SpeedTech,FullSpeed)"
        ),
    )
    p_harvest.add_argument(
        "--exclude-tags",
        default="",
        help=(
            "drop maps carrying any of these; LOL,PressForward,RPG,Trial,Maze "
            "is the set that teaches least about racing"
        ),
    )
    p_harvest.add_argument(
        "--prefer",
        default="median",
        choices=["median", "best", "author"],
        help=(
            "which run to learn from: median of the leaderboard (default), the "
            "fastest, or the map author's validation lap (often far too slow)"
        ),
    )
    p_harvest.add_argument(
        "--dry-run",
        action="store_true",
        help="show what would be downloaded without downloading it",
    )
    p_harvest.set_defaults(func=_cmd_harvest)

    p_filter = sub.add_parser(
        "filter",
        help="sort replays by input device before collecting, in one pass",
    )
    p_filter.add_argument("replays", help="a folder of replays to sort in place")
    p_filter.add_argument(
        "--inputs",
        default=inputs_mod.KEYBOARD,
        choices=[inputs_mod.KEYBOARD, inputs_mod.PAD],
        help="which device to keep; everything else is moved aside",
    )
    p_filter.add_argument(
        "--max-seconds",
        type=float,
        default=180.0,
        help="drop runs longer than this; cost is linear in race time",
    )
    p_filter.add_argument("--port", type=int, default=8477)
    p_filter.add_argument(
        "--dry-run", action="store_true", help="report without moving anything"
    )
    p_filter.set_defaults(func=_cmd_filter)

    parser.add_argument(
        "--config",
        default=None,
        help=f"settings file to read defaults from (default: {config_mod.DEFAULT_NAME})",
    )

    # Config values become defaults, so anything given on the command line
    # still overrides them.
    known = {
        "collect": p_collect,
        "verify": p_verify,
        "stats": p_stats,
        "clean": p_clean,
        "video": p_video,
        "harvest": p_harvest,
        "filter": p_filter,
        "smoke": p_smoke,
        "camera-check": p_camera,
    }
    try:
        explicit = None
        if argv is not None and "--config" in argv:
            explicit = argv[argv.index("--config") + 1]
        elif "--config" in sys.argv:
            explicit = sys.argv[sys.argv.index("--config") + 1]
        source = config_mod.find(explicit)
        config_mod.apply(config_mod.load(source), known, source=source)
    except (config_mod.ConfigError, IndexError) as exc:
        print(f"config: {exc}")
        return 2

    args = parser.parse_args(argv)
    return args.func(args)
