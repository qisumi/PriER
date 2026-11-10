#!/usr/bin/env python3
"""Collect reviewer answers on the calibration split and fit L(k | y)."""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prier import pipeline
from prier.reviewer import Reviewer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--cache", default=None, help="reviewer answer cache (jsonl)")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    datasets = args.datasets or cfg["datasets"] + [cfg["transfer_dataset"]]
    reviewer = Reviewer(cfg["reviewer_path"], cfg["device"],
                        cfg["reviewer_max_input_tokens"], cache_path=args.cache)
    for dataset in datasets:
        channel = pipeline.calibrate_channel("data/prepared", cfg["work"], dataset, reviewer,
                                             pairs=cfg["calibration_pairs"],
                                             seed=cfg["calibration_seed"])
        print(f"{dataset}: cuts={channel.cuts.tolist()}, "
              f"max log-ratio={channel.max_log_ratio:.3f}")


if __name__ == "__main__":
    main()
