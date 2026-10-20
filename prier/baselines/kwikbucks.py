"""KwikBucks (ICLR 2023) and Pivot on the authors' reference code.

The "strong signal" of KwikBucks is our reviewer: every read of a pair is a
paid review (score > 0 means "same"). Three variants are provided:

* ``kwikbucks`` (Kwik-Orig): the authors' defaults (200 pivots, batch 100,
  top-100 neighbours, 20 merge samples). 90% of the token budget is used for
  the main pass and the rest for merging. A sampled merge group is only read
  if all of its reads fit the remaining budget.
* ``kwikbucks_budget`` (Kwik-Adap): converts tokens into author query units
  with the median pair prompt length and limits the number of pivots so that
  every pivot can be compared with every record of the block.
* ``pivot``: classic Pivot (Ailon et al.) on the anchor graph p > 0.5, no reviews.
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout
from functools import partial
from types import MethodType

import numpy as np

from ..decoder import canonical
from ._vendor import kwikbucks, seeded


class OutOfBudget(Exception):
    pass


class PaidSignal:
    """Index-able pair oracle that bills every read to the reviewer."""

    def __init__(self, n, reviewer, cap):
        self.shape = (n, n)
        self.reviewer = reviewer
        self.cap = int(cap)
        self.tokens = 0

    def _read(self, i, j):
        i, j = int(i), int(j)
        if i == j:
            return 1.0
        quote = self.reviewer.quote_pair(i, j)
        if not np.isfinite(quote) or quote > self.cap - self.tokens:
            raise OutOfBudget
        answer = self.reviewer.review_pair(i, j)
        self.tokens += int(answer["tokens"])
        return float(answer["score"] > 0.0)

    def __getitem__(self, key):
        rows, cols = key
        if np.isscalar(rows) and np.isscalar(cols):
            return self._read(rows, cols)
        a, b = np.broadcast_arrays(np.asarray(rows, int), np.asarray(cols, int))
        out = np.empty(a.shape, float)
        for index in np.ndindex(a.shape):
            out[index] = self._read(a[index], b[index])
        return out

    def read_group(self, pairs):
        """Read a whole sampled merge group, or nothing if it does not fit."""
        quotes = [0.0 if i == j else float(self.reviewer.quote_pair(*sorted((int(i), int(j)))))
                  for i, j in pairs]
        if not all(np.isfinite(q) for q in quotes) or sum(quotes) > self.cap - self.tokens:
            raise OutOfBudget
        return sum(self._read(i, j) for i, j in pairs)


def _merge_average(model, dataset, cluster_ids_to_indices, len_cluster_ids, c1, c2):
    """Author's merge estimate, reading its sampled pairs as one group."""
    head = model.randomly_sample_indices(len_cluster_ids[c1], model.num_pairs_to_query)
    tail = model.randomly_sample_indices(len_cluster_ids[c2], model.num_pairs_to_query)
    pairs = [(cluster_ids_to_indices[c1][head[t]], cluster_ids_to_indices[c2][tail[t]])
             for t in range(model.num_pairs_to_query)]
    return dataset.strong_signal.read_group(pairs) / model.num_pairs_to_query


def _dataset(data_utils, weak, strong):
    dataset = object.__new__(data_utils.Dataset)
    dataset.weak_signal, dataset.strong_signal = weak, strong
    dataset.is_graph = dataset.is_sparse = False
    return dataset


def _model(n, num_pivots):
    model_utils, _ = kwikbucks()
    model = model_utils.CorrelationClustering()
    model.num_queries_multiplier = 1
    model.set_num_pivots(n, num_pivots=num_pivots)
    model.set_merge_post_processing_pairs(num_pairs_to_query=20)
    model.assign_to_pivots_given_ordering = partial(model.assign_to_pivots_given_ordering,
                                                    boost_by_neighbors_assignment=True)
    model.approximate_strong_signal_between_clusters = MethodType(_merge_average, model)
    return model


def _cluster(n, cosine, reviewer, budget, seed, num_pivots, main_cap, main_queries, merge_queries):
    _, data_utils = kwikbucks()
    model = _model(n, num_pivots)
    signal = PaidSignal(n, reviewer, main_cap)
    dataset = _dataset(data_utils, np.asarray(cosine, float).copy(), signal)
    with seeded(seed), redirect_stdout(io.StringIO()):
        try:
            clusters, _ = model.qwick_cluster_using_ordering(
                dataset, main_queries(signal), ordering_style="weak_signal", order_non_pivots=True,
                batch_size=100, max_neighbors_to_query=100)
        except OutOfBudget:
            clusters = model.cluster_pairs.convert_cluster_pairs_to_clusters(n)
        signal.cap = int(budget)
        try:
            clusters, _ = model.merge_post_process(dataset, clusters, merge_queries(signal), merge_threshold=.5)
        except OutOfBudget:
            clusters = model.cluster_pairs.convert_cluster_pairs_to_clusters(n)
    return canonical([clusters.assignments[i] for i in range(n)])


def run_original(cosine, reviewer, budget, seed):
    n = len(cosine)
    if n < 2:
        return np.zeros(n, int)
    cap = int(np.floor(.9 * budget))
    return _cluster(n, cosine, reviewer, budget, seed, min(n, 200), cap,
                    main_queries=lambda s: max(1, cap),
                    merge_queries=lambda s: max(0, budget - s.tokens))


def run_budgeted(cosine, reviewer, budget, seed):
    n = len(cosine)
    quotes = [reviewer.quote_pair(i, j) for i in range(n) for j in range(i + 1, n)]
    quotes = [q for q in quotes if np.isfinite(q) and q > 0]
    if n < 2 or budget == 0 or not quotes:
        return canonical(np.arange(n))
    reference = float(np.median(quotes))
    cap = 9 * int(budget) // 10
    main = int(np.floor(cap / reference))
    pivots = max(1, min(n - 1, 200, main // max(1, n - 1)))
    return _cluster(n, cosine, reviewer, budget, seed, pivots, cap,
                    main_queries=lambda s: main,
                    merge_queries=lambda s: int(np.floor((budget - s.tokens) / reference)))


def run_pivot(cosine, p, seed, pair_weight=None):
    model_utils, data_utils = kwikbucks()
    n = len(p)
    if n < 2:
        return np.zeros(n, int)
    legal = np.ones((n, n), bool) if pair_weight is None else np.asarray(pair_weight) > 0
    strong = ((np.asarray(p) > .5) & legal).astype(float)
    np.fill_diagonal(strong, 1.0)
    model = model_utils.CorrelationClustering()
    model.num_queries_multiplier = 1
    with seeded(seed), redirect_stdout(io.StringIO()):
        clusters, _ = model.qwick_cluster_original(_dataset(data_utils, np.asarray(cosine), strong),
                                                   n * (n - 1) // 2 + 1)
    return canonical([clusters.assignments[i] for i in range(n)])
