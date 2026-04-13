from __future__ import annotations

import argparse
from pathlib import Path

from pre_snap_motion.config import load_config
from pre_snap_motion.pipeline import fetch, prepare, run, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Framework for modeling the value of NFL pre-snap motion."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in [
        ("fetch", "Download and cache raw datasets."),
        ("prepare", "Build the processed modeling table."),
        ("train", "Fit configured models and write metrics."),
        ("run", "Execute fetch, prepare, and train."),
    ]:
        subparser = subparsers.add_parser(command, help=help_text)
        subparser.add_argument(
            "--config",
            default="configs/default.yaml",
            help="Path to the YAML config file.",
        )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(Path(args.config))

    if args.command == "fetch":
        outputs = fetch(config)
    elif args.command == "prepare":
        outputs = {"processed_dataset": prepare(config)}
    elif args.command == "train":
        outputs = train(config)
    else:
        outputs = run(config)

    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
