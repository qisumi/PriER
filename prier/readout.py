"""Task-aware readout: re-decode the final posterior with a schema-specific
false-merge cost lambda_task = theta / (1 - theta).

For schemas whose entities fragment into many surface forms (institution
names), theta is chosen on the held-out 20% of fit entities by maximising the
decoded pair-F1; other schemas keep lambda = 1.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from .decoder import PartitionRisk, decode

THRESHOLDS = (.05, .1, .15, .2, .25, .3, .35, .4, .45, .5)


def apply_readout(posterior, partition, theta, pair_weight=None):
    risk = PartitionRisk(posterior, theta / (1 - theta), pair_weight)
    _, labels = decode(risk, np.atleast_2d(partition))
    return labels


def heldout_positive_pairs(records, labels, held, cross_source):
    """Number of true pairs among held-out fit records (including cross-block ones)."""
    selected = [r for r, h in zip(records, held) if h]
    if labels["partition_available"]:
        entity = labels["record_to_entity"]
        if not cross_source:
            return sum(n * (n - 1) // 2 for n in Counter(entity[r["id"]] for r in selected).values())
        counts = defaultdict(Counter)
        for r in selected:
            counts[entity[r["id"]]][r["source"]] += 1
        return sum((sum(c.values()) ** 2 - sum(v * v for v in c.values())) // 2 for c in counts.values())
    ids = {r["id"] for r in selected}
    return sum(set(p) <= ids for p in labels["positive_pairs"])


def select_threshold(blocks, positives):
    """``blocks``: list of (prior matrix, targets over triu pairs, legal mask, weight).

    Returns the theta with the best decoded pair-F1 (ties: fewer false
    positives, then the larger theta) together with all candidates.
    """
    prepared = []
    for p, y, legal, weight in blocks:
        _, initial = decode(PartitionRisk(p, 1.0, weight))
        prepared.append((p, y, legal, weight, initial))
    candidates = []
    for theta in THRESHOLDS:
        tp = fp = 0
        for p, y, legal, weight, initial in prepared:
            part = apply_readout(p, initial, theta, weight)
            tri = np.triu_indices(len(p), 1)
            predicted = (part[tri[0]] == part[tri[1]]) & legal
            tp += int(np.sum(predicted & (y == 1)))
            fp += int(np.sum(predicted & (y == 0)))
        fn = positives - tp
        candidates.append({"theta": theta, "tp": tp, "fp": fp, "fn": fn,
                           "pair_f1": 2 * tp / max(1, 2 * tp + fp + fn)})
    best = max(candidates, key=lambda c: (c["pair_f1"], -c["fp"], c["theta"]))
    return best["theta"], candidates
