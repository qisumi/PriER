#!/usr/bin/env python3
"""Fit the identity prior (and leave-one-group-out variants) plus the readout."""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prier import pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--ablations", action="store_true",
                        help="also fit the leave-one-group-out variants")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    datasets = args.datasets or cfg["datasets"] + [cfg["transfer_dataset"]]
    for dataset in datasets:
        for seed in cfg["fit_seeds"]:
            pipeline.fit_prior("data/prepared", cfg["work"], dataset, seed)
            if args.ablations:
                groups = [g for g in cfg["feature_groups"]
                          if g != "institution" or dataset == "affiliations"]
                for group in groups:
                    pipeline.fit_prior("data/prepared", cfg["work"], dataset, seed, drop=(group,))
            pipeline.fit_readout("data/prepared", cfg["work"], dataset, seed)
            print(f"fitted {dataset} seed {seed}")


if __name__ == "__main__":
    main()
