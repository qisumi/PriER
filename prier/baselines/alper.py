"""Alper (KDD 2026) on the authors' AlperFramework with our local reviewer.

The framework is unchanged; we supply exact cosine top-k search over the
block, a set-level oracle backed by the reviewer, and a token budget. The
author scheduler works in dollars, so one token is mapped to a fixed unit
price; queries are only issued when their prompt fits the remaining budget.
Hyper-parameters follow the authors' ``run_table_experiments.sh``.
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout

import numpy as np

from ..decoder import canonical
from ._vendor import alper, seeded

K, TOP_M, MAX_ITERATIONS = 15, 5, 50
ALPHA, THETA, XI_LLM = .8, .6, .95
LOWER, UPPER = 20.0, 1000.0
UNIT_PRICE = .25 / 1_000_000  # dollars per token, used only by the scheduler


class CosineSearch:
    def __init__(self, cosine):
        self.cosine = cosine

    def get_all_topk(self, k):
        n = len(self.cosine)
        result = {}
        for i in range(n):
            ids = sorted((j for j in range(n) if j != i), key=lambda j: (-self.cosine[i, j], j))
            ids = ids[:min(k, max(0, n - 1))]
            result[i] = (ids, [float(self.cosine[i, j]) for j in ids])
        return result

    def compute_similarity(self, i, j):
        return float(self.cosine[i, j])


def run(cosine, texts, reviewer, budget, seed):
    framework_module = alper()

    class Oracle:
        def __init__(self):
            self.total_input_tokens = self.total_output_tokens = 0
            self.total_cost = 0.0
            self.spent = 0

        def query_ppl(self, center, candidates):
            answer = reviewer.review_set(center, candidates)
            self.total_input_tokens += answer["input_tokens"]
            self.total_output_tokens += answer["output_tokens"]
            self.spent += answer["tokens"]
            self.total_cost += UNIT_PRICE * (answer["input_tokens"] + answer["output_tokens"])
            chosen = answer["chosen"]
            return None if chosen is None else (chosen, [chosen])

    oracle = Oracle()

    class Framework(framework_module.AlperFramework):
        def _compute_label_distribution(self, center):
            self._center = center
            self._distribution = super()._compute_label_distribution(center)
            return self._distribution

        def _signal_select(self, confidence):
            selected, estimate = super()._signal_select(confidence)
            if not selected:
                return selected, estimate
            # Recompute the representatives _local_update would ask about and
            # skip the query if its prompt does not fit the remaining tokens.
            center, reps = self._center, []
            top = sorted(self._distribution.items(), key=lambda x: x[1], reverse=True)[:self.top_m]
            for label, _ in top:
                best, best_sim = None, -1.0
                for j in self.neighbors.get(center, []):
                    if self.labels.get(j) == label:
                        sim = self.topk_search.compute_similarity(center, j)
                        if sim > best_sim:
                            best, best_sim = j, sim
                if best is not None:
                    reps.append(best)
            if not reps or reviewer.quote_set(center, reps) > budget - oracle.spent:
                return False, 0.0
            return selected, estimate

    framework = Framework(
        records=dict(enumerate(texts)), data_loader=None, topk_search=CosineSearch(cosine),
        llm_oracle=oracle, budget=budget * UNIT_PRICE, k=K, max_iterations=MAX_ITERATIONS,
        top_m=TOP_M, alpha=ALPHA, theta=THETA, xi_llm=XI_LLM, L=LOWER, U=UPPER, seed=int(seed),
        input_price=UNIT_PRICE, output_price=UNIT_PRICE)
    with seeded(seed, python=False), redirect_stdout(io.StringIO()):
        clusters, _, _ = framework.run()
    labels = np.empty(len(texts), dtype=int)
    for label, members in enumerate(clusters):
        labels[members] = label
    return canonical(labels)
