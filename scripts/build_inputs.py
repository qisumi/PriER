#!/usr/bin/env python3
"""Encode every split, block it, and fit the shared anchor prior p^c."""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prier import data, pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--datasets", nargs="*", default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    datasets = args.datasets or cfg["datasets"] + [cfg["transfer_dataset"]]
    for dataset in datasets:
        print(f"building {dataset} ...")
        pipeline.build_inputs("data/prepared",
                              cfg["work"], dataset, cfg["encoder_path"], cfg["device"])


if __name__ == "__main__":
    main()
