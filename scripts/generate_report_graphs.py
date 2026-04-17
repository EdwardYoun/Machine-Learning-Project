from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pre_snap_motion.plotting.pipeline import generate_all_graphs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate presentation-ready charts from final experiment metrics."
    )
    parser.add_argument(
        "--metrics-dir",
        default="artifacts/metrics",
        help="Directory containing the final metrics CSV and JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/figures",
        help="Directory where the generated charts will be written.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = generate_all_graphs(
        metrics_dir=args.metrics_dir,
        output_dir=args.output_dir,
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
