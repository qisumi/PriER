#!/usr/bin/env python3
"""Run methods x budgets on the evaluation split and write metrics.csv."""
import argparse
import csv
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
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--ablations", action="store_true")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    datasets = args.datasets or cfg["datasets"]
    methods = args.methods or cfg["methods"]
    if args.ablations:
        groups = [g for g in cfg["feature_groups"] if g != "institution"]
        methods = methods + [f"prier_drop_{g}" for g in groups]
    reviewer = Reviewer(cfg["reviewer_path"], cfg["device"],
                        cfg["reviewer_max_input_tokens"], cache_path=args.cache)
    rows = []
    for dataset in datasets:
        for fit_seed, run_seed in zip(cfg["fit_seeds"], cfg["run_seeds"]):
            runs = pipeline.run_dataset("data/prepared", cfg["work"], dataset, fit_seed, run_seed,
                                        methods, cfg["rates"], reviewer,
                                        readout=dataset in cfg["readout_datasets"])
            rows += pipeline.evaluate_runs("data/prepared", cfg["work"], dataset, runs)
            print(f"{dataset} seed {fit_seed}: {len(rows)} rows so far")
    out = Path(args.output or Path(cfg["work"]) / "metrics.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
