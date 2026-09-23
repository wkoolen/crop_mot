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
    analyse   --run runs/<stamp>_...                     [B3]
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for all three subcommands.

    Returns:
        A parser with `simulate`, `track` and `analyse` subcommands, each taking the
        arguments its runner function needs.
    """
    raise NotImplementedError


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the appropriate runner.

    Args:
        argv: command-line arguments, or None to use sys.argv[1:].

    Returns:
        A process exit code: 0 on success, non-zero on a handled error.
    """
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
