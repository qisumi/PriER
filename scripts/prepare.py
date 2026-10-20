#!/usr/bin/env python3
"""Normalize the raw benchmarks into records, labels and entity-disjoint splits."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prier import data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data")
    parser.add_argument("--out", default="data/prepared")
    parser.add_argument("--datasets", nargs="*", default=list(data.DATASETS))
    args = parser.parse_args()
    summary = data.prepare(args.raw, args.out, args.datasets)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
