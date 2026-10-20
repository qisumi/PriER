#!/usr/bin/env python3
"""Cyclic order-permutation probe for set-level reviewer position bias.

For fixed (center, candidate-set) samples from the evaluation split, every
cyclic rotation of the candidate list is asked once. A record chosen from
slot 2 but not from other slots measures the causal position effect.
"""
import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prier import data
from prier.inputs import Split
from prier.reviewer import Reviewer, parse_set_answer


def sample_sets(records, blocks, per_domain=160, seed=20261019):
    """Deterministic sample of center/candidate-set pairs (hash order, no labels)."""
    import hashlib
    sets = []
    for block in blocks:
        for center in block:
            others = [r for r in block if r != center]
            if 2 <= len(others) <= 4:
                sets.append((center, tuple(others)))
    order = sorted(range(len(sets)),
                   key=lambda i: hashlib.sha256(f"{seed}:{i}".encode()).hexdigest())
    return [sets[i] for i in order[:per_domain]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--limit", type=int, default=160)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    task = data.load_task("data/prepared", args.dataset)
    split = Split.load(Path(cfg["work"]) / args.dataset / "evaluation")
    reviewer = Reviewer(cfg["reviewer_path"], cfg["device"],
                        cfg["reviewer_max_input_tokens"])
    texts = {r["id"]: r["text"] for r in split.records}
    rows = []
    for center, candidates in sample_sets(split.records, split.blocks, args.limit):
        for k in range(len(candidates)):
            order = candidates[k:] + candidates[:k]
            answer = reviewer.ask_set(texts[center], [texts[c] for c in order])
            chosen = parse_set_answer(answer["text"], len(order))
            rows.append({"center": split.records[center]["id"], "rotation": k,
                         "candidates": [split.records[c]["id"] for c in order],
                         "chosen": None if chosen is None else split.records[order[chosen]]["id"],
                         "chosen_slot": None if chosen is None else chosen + 1,
                         "tokens": answer["tokens"]})
    out = Path(args.output or Path(cfg["work"]) / args.dataset / "order_probe.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    slot2 = [r for r in rows if r["chosen_slot"] == 2]
    other = [r for r in rows if r["chosen_slot"] not in (None, 2)]
    print(f"wrote {out}: {len(rows)} answers, "
          f"slot-2 selection rate {len(slot2) / max(1, len(rows) - sum(r['chosen_slot'] is None for r in rows)):.3f}")


if __name__ == "__main__":
    main()
