"""ACC (TMLR 2024): the authors' max-min query strategy and dynamic-K
correlation-clustering decoder, driven by our reviewer and a token budget.

Pairwise similarities start from the anchor prior (2p - 1). A reviewed pair
contributes 2q - 1, where q is the calibrated posterior of the reviewer's
answer, and repeated reads are averaged as in the authors' code (at most
``tau`` reads per pair). Batches of ceil(|E| / 1000) pairs are bought until
the budget is exhausted.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from ..decoder import canonical
from ._vendor import acc, seeded

EPSILON, TAU, BETA, RESTARTS = .3, 5, 1.0, 5


def run(p, reviewer, budget, seed):
    queries, decoder = acc()
    n = len(p)
    count = n * (n - 1) // 2
    batch_size = max(1, math.ceil(count / 1000))
    spent = 0
    with seeded(int(seed) + 317421, python=False):
        similarity = 2 * np.asarray(p, dtype=float).copy() - 1
        np.fill_diagonal(similarity, 0.0)
        state = SimpleNamespace(
            N=n, pairwise_similarities=similarity, feedback_freq=np.ones((n, n)),
            eps=EPSILON, tau=TAU, beta=BETA, num_maxmin_edges=-1.0,
            clustering_solution=np.zeros(n, dtype=int), violations=np.zeros((n, n)), num_clusters=1)
        strategy = queries.QueryStrategy(state)

        def decode():
            if n < 2:
                state.clustering_solution = np.zeros(n, dtype=int)
            else:
                state.clustering_solution, _ = decoder.max_correlation_dynamic_K(
                    state.pairwise_similarities, state.num_clusters, RESTARTS)
            state.num_clusters = int(state.clustering_solution.max()) + 1 if n else 0
            same = state.clustering_solution[:, None] == state.clustering_solution[None, :]
            violated = ((~same) & (state.pairwise_similarities >= 0)) | (same & (state.pairwise_similarities < 0))
            state.violations = np.where(violated, np.abs(state.pairwise_similarities), 0.0)
            np.fill_diagonal(state.violations, 0.0)

        decode()
        while spent < budget and count:
            remaining = budget - spent
            # Hide pairs that cannot be afforded (or were read tau times) from
            # the author's selection, without changing its scores.
            frequency = state.feedback_freq.copy()
            affordable = 0
            for i in range(n):
                for j in range(i):
                    if frequency[i, j] > TAU:
                        continue
                    if reviewer.quote_pair(j, i) > remaining:
                        state.feedback_freq[i, j] = state.feedback_freq[j, i] = np.inf
                    else:
                        affordable += 1
            if not affordable:
                state.feedback_freq = frequency
                break
            selected = strategy.select_batch("maxmin", min(batch_size, affordable))
            state.feedback_freq = frequency
            bought = 0
            for left, right in selected:
                i, j = sorted((int(left), int(right)))
                if reviewer.quote_pair(i, j) > budget - spent:
                    continue
                answer = reviewer.review_pair(i, j)
                state.feedback_freq[i, j] += 1
                state.feedback_freq[j, i] += 1
                f = state.feedback_freq[i, j]
                value = ((f - 1) * state.pairwise_similarities[i, j] + (2 * answer["probability"] - 1)) / f
                state.pairwise_similarities[i, j] = state.pairwise_similarities[j, i] = value
                spent += int(answer["tokens"])
                bought += 1
            if not bought:
                break
            decode()
    return canonical(state.clustering_solution)
