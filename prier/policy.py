"""Budgeted review loop (Algorithm 1 in the paper).

Within one block, the controller repeatedly picks the affordable, unreviewed
pair with the highest uncertainty per prompt token, asks the reviewer, folds
the answer into the pair's log-odds with the channel's likelihood ratio, and
re-decodes the block with a warm start.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.special import expit, logit

from .decoder import PartitionRisk, canonical, decode


def review_loop(prior, channel, quotes, review, budget, pair_weight=None):
    """Run the U/t review loop on one block.

    Args:
        prior: (n, n) symmetric matrix of prior match probabilities.
        channel: a fitted :class:`prier.channel.ReviewChannel`.
        quotes: prompt-token cost of every pair in ``np.triu_indices(n, 1)``
            order; ``inf`` marks pairs whose prompt does not fit the reviewer.
        review: callable ``review(i, j) -> {"score": float, "tokens": int}``.
        budget: token budget of the block.
        pair_weight: optional (n, n) task weights (cross-source tasks use 0 for
            same-source pairs, which are never reviewed).

    Returns:
        (partition, trace, stats)
    """
    started, cpu_started = time.perf_counter(), time.process_time()
    review_wall = review_cpu = 0.0
    log_odds = logit(np.clip(np.asarray(prior, dtype=float), 1e-8, 1 - 1e-8))

    def posterior():
        p = expit(log_odds)
        np.fill_diagonal(p, 0)
        return p

    risk = PartitionRisk(posterior(), 1.0, pair_weight)
    quotes = np.asarray(quotes, dtype=float)
    table, current = decode(risk)
    used, seen, trace = 0, set(), []
    while True:
        affordable = [e for e in range(len(risk.tri))
                      if e not in seen and risk.weight[e] > 0 and quotes[e] <= budget - used]
        if not affordable:
            break
        uncertainty = risk.p * (1 - risk.p)
        score = uncertainty / quotes
        e = min(affordable, key=lambda k: (-score[k], -uncertainty[k], k))
        i, j = map(int, risk.tri[e])

        t0, c0 = time.perf_counter(), time.process_time()
        answer = review(i, j)
        review_wall += time.perf_counter() - t0
        review_cpu += time.process_time() - c0

        ratio = channel.log_ratio(answer["score"])
        before = float(risk.probability[i, j])
        log_odds[i, j] += ratio
        log_odds[j, i] += ratio
        used += int(answer["tokens"])
        seen.add(e)

        risk = PartitionRisk(posterior(), 1.0, pair_weight)
        table, current = decode(risk, table)
        trace.append({"pair": [i, j], "tokens": int(answer["tokens"]),
                      "score": float(answer["score"]), "bin": int(channel.encode(answer["score"])),
                      "log_ratio": ratio, "p_before": before,
                      "p_after": float(risk.probability[i, j])})

    stats = {"tokens": used, "reviews": len(trace),
             "controller_wall_s": time.perf_counter() - started - review_wall,
             "controller_cpu_s": time.process_time() - cpu_started - review_cpu}
    return canonical(current), trace, stats


def reviewed_posterior(prior, trace):
    """Prior matrix with every reviewed pair replaced by its updated probability."""
    p = np.asarray(prior, dtype=float).copy()
    for row in trace:
        i, j = row["pair"]
        p[i, j] = p[j, i] = row["p_after"]
    return p
