"""Command-line entry point: `python3 -m crop_mot <subcommand>`. [B1/B2/B3]

A module entry point rather than a console script or a shell wrapper, deliberately: it
works with no install step (PYTHONPATH=/workspace in the Dockerfile) and it needs no
executable bit, which Windows bind mounts cannot express anyway.

Kept thin on purpose. Argument parsing belongs here; everything else belongs in
`crop_mot.runner` and `crop_mot.analysis`, so that phase 2's ROS node can call the same
functions without going through a CLI.

Subcommands:
    simulate  --config configs/b1_two_rows.yaml          [B1]
    track     --config configs/b2_bernoulli_phantom.yaml [B2]
    analyse   --run runs/<stamp>_...                     [B1/B2/B3]
    candidates --run runs/<stamp>_... --min-distance 1.0 [B2] phantom seeds for the bank
    scaling   --config ... --sweep lambda_FA --values 1 2 4 8  [B4] time per scan
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from crop_mot.analysis.candidates import assumed_fov, format_candidates, phantom_candidates
from crop_mot.runner.analyse import analyse_run
from crop_mot.runner.run_dir import RunDir
from crop_mot.runner.scaling import scaling_from_config
from crop_mot.runner.simulate import simulate_from_config
from crop_mot.runner.track import track_from_config


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for all subcommands.

    Returns:
        A parser with `simulate`, `track`, `analyse` and `candidates` subcommands, each
        taking the arguments its runner function needs.
    """
    parser = argparse.ArgumentParser(prog="python3 -m crop_mot",
                                     description="Crop-row MOT simulator and filters.")
    commands = parser.add_subparsers(dest="command", required=True)

    simulate = commands.add_parser("simulate", help="[B1] scenario config -> run folder")
    simulate.add_argument("--config", type=Path, required=True, help="B1 scenario YAML")
    simulate.add_argument("--runs", type=Path, default=Path("runs"),
                          help="directory holding run folders (default: runs)")

    track = commands.add_parser("track", help="[B2] run a filter over recorded detections")
    track.add_argument("--config", type=Path, required=True, help="B2 run YAML")
    track.add_argument("--run", type=Path, default=None,
                       help="existing run folder to reuse; omit to simulate a fresh one")
    track.add_argument("--runs", type=Path, default=Path("runs"),
                       help="directory holding run folders (default: runs)")

    analyse = commands.add_parser("analyse", help="[B2/B3] plots and metrics for a run")
    analyse.add_argument("--run", type=Path, required=True, help="run folder to analyse")
    analyse.add_argument("--plots", nargs="+", default=None,
                         help="plots to render; default: the config's analysis.plots")
    analyse.add_argument("--scene-k", type=int, default=None,
                         help="scan the scene plot shows; default: the first scan a track "
                              "is reported, or 0")

    candidates = commands.add_parser(
        "candidates", help="[B2] list clutter detections usable as phantom seeds")
    candidates.add_argument("--run", type=Path, required=True, help="run folder to search")
    candidates.add_argument("--min-distance", type=float, default=1.0,
                            help="minimum distance to any plant in m (default: 1.0)")
    candidates.add_argument("--min-separation", type=float, default=1.0,
                            help="minimum distance between candidates in m (default: 1.0)")
    scaling = commands.add_parser(
        "scaling", help="[B4] update time per scan against problem size (roadmap 4c)")
    scaling.add_argument("--config", type=Path, required=True, help="run YAML to sweep")
    scaling.add_argument("--sweep", required=True,
                         choices=["row_length", "spacing", "lambda_FA", "weed_density"],
                         help="the scenario parameter to vary")
    scaling.add_argument("--values", type=float, nargs="+", required=True,
                         help="the parameter's values")
    scaling.add_argument("--seeds", type=int, default=3, help="seeds per value (default: 3)")
    scaling.add_argument("--runs", type=Path, default=Path("runs"),
                         help="directory holding run folders (default: runs)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the appropriate runner.

    Args:
        argv: command-line arguments, or None to use sys.argv[1:].

    Returns:
        A process exit code: 0 on success, non-zero on a handled error.
    """
    args = build_parser().parse_args(argv)
    try:
        if args.command == "simulate":
            run = simulate_from_config(args.config, args.runs)
        elif args.command == "track":
            run = track_from_config(args.config, args.runs, args.run)
        elif args.command == "scaling":
            run = scaling_from_config(args.config, args.sweep, args.values, args.seeds,
                                      args.runs)
        elif args.command == "candidates":
            run = RunDir(args.run)
            print(format_candidates(phantom_candidates(
                run, assumed_fov(run), args.min_distance, args.min_separation)))
            return 0
        else:
            run = RunDir(args.run)
            analyse_run(run, plots=args.plots, scene_k=args.scene_k)
    except (ValueError, KeyError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(run.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
