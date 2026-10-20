"""Pair precision/recall/F1 and B^3 on a whole evaluation split.

Predictions are given per block; records in different blocks are never in the
same predicted cluster, so true pairs separated by blocking count as false
negatives for every method.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np


def _ratio(a, b):
    return a / b if b else 1.0


def _f1(p, r):
    return 2 * p * r / (p + r) if p + r else 0.0


def _choose2(x):
    return x * (x - 1) // 2


def partition_metrics(predicted, gold):
    """``predicted`` and ``gold`` are cluster labels aligned by record."""
    joint = Counter(zip(gold, predicted))
    gc, pc = Counter(gold), Counter(predicted)
    tp = sum(_choose2(v) for v in joint.values())
    positives = sum(_choose2(v) for v in gc.values())
    proposed = sum(_choose2(v) for v in pc.values())
    precision, recall = _ratio(tp, proposed), _ratio(tp, positives)
    bp = np.mean([joint[g, p] / pc[p] for g, p in zip(gold, predicted)])
    br = np.mean([joint[g, p] / gc[g] for g, p in zip(gold, predicted)])
    return {"tp": tp, "fp": proposed - tp, "fn": positives - tp,
            "pair_precision": precision, "pair_recall": recall, "pair_f1": _f1(precision, recall),
            "b3_precision": float(bp), "b3_recall": float(br), "b3_f1": _f1(bp, br)}


def cross_source_metrics(ids, sources, predicted, positive_pairs, universe):
    """Only pairs of records from the two sources are scored."""
    labels = dict(zip(ids, predicted))
    left, right = sorted(universe)
    per_cluster = defaultdict(Counter)
    for rid in ids:
        per_cluster[labels[rid]][sources[rid]] += 1
    proposed = sum(c[left] * c[right] for c in per_cluster.values())
    idset = set(ids)
    positives = [p for p in positive_pairs if set(p) <= idset]
    tp = sum(labels[a] == labels[b] for a, b in positives)
    precision, recall = _ratio(tp, proposed), _ratio(tp, len(positives))
    return {"tp": int(tp), "fp": int(proposed - tp), "fn": int(len(positives) - tp),
            "pair_precision": precision, "pair_recall": recall, "pair_f1": _f1(precision, recall)}


def evaluate_split(records, labels, task, blocks, partitions):
    """Score per-block partitions against the split's ground truth."""
    ids, predicted = [], []
    for b, (block, part) in enumerate(zip(blocks, partitions)):
        ids.extend(records[i]["id"] for i in block)
        predicted.extend((b, int(c)) for c in part)
    if task["task"] == "cross_source":
        sources = {r["id"]: r["source"] for r in records}
        return cross_source_metrics(ids, sources, predicted, labels["positive_pairs"], labels["source_universe"])
    gold = [labels["record_to_entity"][rid] for rid in ids]
    return partition_metrics(predicted, gold)
