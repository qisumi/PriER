"""BoostER (WWW 2024 demo), reimplemented from the paper.

A fixed bank of candidate partitions (connected components of the anchor
graph at a grid of thresholds, plus all-singletons and all-in-one) receives a
product-Bernoulli prior. Each round greedily picks pairs with the highest
joint-answer entropy per token, updates the bank posterior with the reviewer
channel, and reads out the MAP partition.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp, softmax, xlogy

from ..decoder import PartitionRisk, canonical
from .simple import threshold_components

ROUNDS = 5
BANK_SIZE = 32


def candidate_bank(p, pair_weight=None, size=BANK_SIZE):
    n = len(p)
    rows = {tuple(np.arange(n)): np.arange(n), tuple(np.zeros(n, int)): np.zeros(n, int)}
    for threshold in np.r_[.5, np.linspace(.025, .975, max(1, size - 3))]:
        labels = threshold_components(p, pair_weight, threshold)
        rows.setdefault(tuple(labels), labels)
    return np.asarray([canonical(rows[k]) for k in sorted(rows)], dtype=int)


def _relations(bank):
    i, j = np.triu_indices(bank.shape[1], 1)
    return (bank[:, i] == bank[:, j]).astype(np.int8)


def _conditional_gains(y, w, groups):
    """H(Y_e | answers already planned) for every pair e."""
    count = int(groups.max()) + 1
    mass = np.bincount(groups, weights=w, minlength=count)[:, None]
    yes = np.zeros((count, y.shape[1]))
    no = np.zeros_like(yes)
    np.add.at(yes, groups, w[:, None] * y)
    np.add.at(no, groups, w[:, None] * (1 - y))
    ry = np.divide(yes, mass, out=np.zeros_like(yes), where=mass > 0)
    rn = np.divide(no, mass, out=np.zeros_like(no), where=mass > 0)
    return np.maximum(0.0, (-xlogy(yes, ry) - xlogy(no, rn)).sum(axis=0))


def _plan_round(y, w, quotes, allowance, eligible, min_gain=1e-12):
    w = w / w.sum()
    available = {int(e) for e in eligible if quotes[e] <= allowance}
    groups = np.zeros(len(y), dtype=int)
    selected, reserved = [], 0
    while available:
        affordable = sorted(e for e in available if quotes[e] <= allowance - reserved)
        if not affordable:
            break
        gain = _conditional_gains(y, w, groups)
        e = min(affordable, key=lambda q: (-gain[q] / quotes[q], -gain[q], q))
        if gain[e] <= min_gain:
            break
        selected.append(e)
        available.remove(e)
        reserved += int(quotes[e])
        _, groups = np.unique(groups * 2 + y[:, e], return_inverse=True)
    return selected


def run(p, channel, quotes, reviewer, budget, pair_weight=None):
    risk = PartitionRisk(p, 1.0, pair_weight)
    quotes = np.asarray(quotes, dtype=float)
    bank = candidate_bank(risk.probability, pair_weight)
    y = _relations(bank)
    mask = risk.weight > 0
    q = np.clip(risk.p[mask], 1e-12, 1 - 1e-12)
    logw = y[:, mask] @ np.log(q) + (1 - y[:, mask]) @ np.log1p(-q)
    logw -= logsumexp(logw)
    weights = softmax(logw)
    legal = [e for e in range(len(risk.tri)) if np.isfinite(quotes[e]) and mask[e]]
    used, seen = 0, set()
    for turn in range(ROUNDS):
        allowance = ((turn + 1) * int(budget)) // ROUNDS - used
        plan = _plan_round(y, weights, quotes, allowance, [e for e in legal if e not in seen])
        for e in plan:
            i, j = map(int, risk.tri[e])
            answer = reviewer.review_pair(i, j, budget - used)
            k = int(channel.encode(answer["score"]))
            logw = logw + np.log(channel.likelihood[y[:, e], k])
            logw -= logsumexp(logw)
            weights = softmax(logw)
            used += int(answer["tokens"])
            seen.add(e)
        if not any(e not in seen and quotes[e] <= budget - used for e in legal):
            break
    return canonical(bank[int(weights.argmax())])
