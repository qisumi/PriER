"""Cosine-prior review baselines: U/t (uncertainty per token) and Risk (risk mass).

Both start from the anchor prior, update a reviewed pair with the same
calibrated channel as PriER, and read out connected components of pairs with
posterior above 0.5.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.special import expit, logit

from ..decoder import PartitionRisk, canonical


def threshold_components(p, pair_weight=None, threshold=0.5):
    p = np.asarray(p, dtype=float)
    legal = np.ones_like(p, bool) if pair_weight is None else np.asarray(pair_weight) > 0
    graph = (p > threshold) & legal
    np.fill_diagonal(graph, False)
    return canonical(connected_components(csr_matrix(graph), directed=False)[1])


def split_merge_risk(p, clusters, tri, false_merge_cost=1.0, pair_weight=None):
    """Expected error mass that a merge or a spanning-tree cut would touch."""
    n = len(p)
    risk = np.zeros((n, n))
    weights = np.ones((n, n)) if pair_weight is None else np.asarray(pair_weight, dtype=float)
    groups = [np.flatnonzero(clusters == c) for c in np.unique(clusters)]
    for ai, a in enumerate(groups):
        for b in groups[ai + 1:]:
            mass = float((p[np.ix_(a, b)] * weights[np.ix_(a, b)]).sum())
            risk[np.ix_(a, b)] = mass
            risk[np.ix_(b, a)] = mass
        if len(a) < 2:
            continue
        distance = np.maximum(1 - p[np.ix_(a, a)], 1e-12)
        np.fill_diagonal(distance, 0)
        tree = minimum_spanning_tree(csr_matrix(distance)).toarray()
        tree = (tree + tree.T) > 0
        for i, j in zip(*np.triu(tree, 1).nonzero()):
            cut = tree.copy()
            cut[i, j] = cut[j, i] = False
            _, side = connected_components(cut, directed=False)
            left, right = a[side == 0], a[side != 0]
            cross = p[np.ix_(left, right)]
            w = weights[np.ix_(left, right)]
            support = max(1, int(((cross >= .5) & (w > 0)).sum()))
            mass = false_merge_cost * float(((1 - cross) * w).sum()) / support
            risk[np.ix_(left, right)] = np.maximum(risk[np.ix_(left, right)], mass)
            risk[np.ix_(right, left)] = np.maximum(risk[np.ix_(right, left)], mass)
    return risk[tri[:, 0], tri[:, 1]]


def run(method, p, channel, quotes, reviewer, budget, pair_weight=None):
    """``method`` is "ut" or "risk"."""
    risk = PartitionRisk(p, pair_weight=pair_weight)
    q = np.asarray(quotes, dtype=float)
    log_odds = logit(np.clip(np.asarray(p, float), 1e-8, 1 - 1e-8))
    current = threshold_components(risk.probability, pair_weight)
    used, seen = 0, set()
    while True:
        affordable = [e for e in range(len(risk.tri))
                      if e not in seen and risk.weight[e] > 0 and q[e] <= budget - used]
        if not affordable:
            break
        uncertainty = risk.p * (1 - risk.p)
        if method == "ut":
            scores = uncertainty / q
        else:
            mass = split_merge_risk(risk.probability, current, risk.tri, 1.0, pair_weight)
            scores = (mass * uncertainty + 1e-9 * uncertainty) / q
        e = min(affordable, key=lambda k: (-scores[k], -uncertainty[k] / q[k], -uncertainty[k], k))
        i, j = map(int, risk.tri[e])
        answer = reviewer.review_pair(i, j, budget - used)
        ratio = channel.log_ratio(answer["score"])
        log_odds[i, j] += ratio
        log_odds[j, i] += ratio
        posterior = expit(log_odds)
        np.fill_diagonal(posterior, 0.0)
        risk = PartitionRisk(posterior, pair_weight=pair_weight)
        current = threshold_components(posterior, pair_weight)
        used += int(answer["tokens"])
        seen.add(e)
    return current
