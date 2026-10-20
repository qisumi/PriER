"""Cost-sensitive correlation-clustering risk and a local-search decoder.

For a block with pairwise match probabilities p and false-merge cost lam, the
expected pair loss of a partition P is

    R(P) = sum_e w_e [ lam (1 - p_e) r_e(P) + p_e (1 - r_e(P)) ],

where r_e(P) = 1 when both endpoints of e share a cluster. The decoder starts
from the connected components of positive-pull edges, reassigns single records
(node descent), and then tries a neighbourhood of edits (cluster merges,
single-record moves, and spanning-tree splits) until no edit lowers the risk.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree


def canonical(labels) -> np.ndarray:
    """Relabel clusters 0, 1, 2, ... in order of first appearance."""
    mapping: dict[int, int] = {}
    return np.array([mapping.setdefault(int(c), len(mapping)) for c in labels], dtype=int)


class PartitionRisk:
    """Expected pairwise loss of candidate partitions for one block."""

    def __init__(self, probability, false_merge_cost=1.0, pair_weight=None):
        self.probability = np.asarray(probability, dtype=float)
        self.n = n = len(self.probability)
        self.lam = float(false_merge_cost)
        self.tri = np.array(np.triu_indices(n, 1)).T
        self.p = self.probability[self.tri[:, 0], self.tri[:, 1]]
        weight = np.ones((n, n)) if pair_weight is None else np.asarray(pair_weight, dtype=float)
        self.weight = weight[self.tri[:, 0], self.tri[:, 1]]
        # Positive entries pull the two records together, negative ones push apart.
        self.signed = self.weight * ((self.lam + 1) * self.p - self.lam)

    def relations(self, partitions):
        parts = np.atleast_2d(partitions)
        return parts[:, self.tri[:, 0]] == parts[:, self.tri[:, 1]]

    def losses(self, partitions):
        return float(self.weight @ self.p) - self.relations(partitions) @ self.signed

    def initial(self):
        graph = np.zeros((self.n, self.n), dtype=bool)
        positive = self.tri[self.signed > 0]
        graph[positive[:, 0], positive[:, 1]] = True
        return connected_components(csr_matrix(graph), directed=False)[1]

    def node_descent(self, initial=None, passes=50):
        c = canonical(self.initial() if initial is None else initial)
        weights = np.zeros((self.n, self.n))
        weights[self.tri[:, 0], self.tri[:, 1]] = self.signed
        weights += weights.T
        for _ in range(passes):
            changed = False
            for u in range(self.n):
                used = np.unique(c)
                scores = np.bincount(c, weights=weights[u], minlength=self.n)
                target = int(used[np.argmax(scores[used])])
                value = scores[target]
                free = np.setdiff1d(np.arange(self.n), used)
                if value <= 0 and len(free):
                    target, value = int(free[0]), 0.0
                if value > scores[c[u]] + 1e-12:
                    c[u] = target
                    changed = True
            if not changed:
                break
        return canonical(c)

    def edits(self, current):
        """The current partition plus merges, single-node moves and MST splits."""
        current = canonical(current)
        unique = {tuple(current): current}

        def add(labels):
            labels = canonical(labels)
            unique.setdefault(tuple(labels), labels)

        groups = [np.flatnonzero(current == label) for label in np.unique(current)]
        for ai, a in enumerate(groups):
            for b in groups[ai + 1:]:
                labels = current.copy()
                labels[b] = labels[a[0]]
                add(labels)
            if len(a) > 1:
                distance = np.maximum(1 - self.probability[np.ix_(a, a)], 1e-12)
                np.fill_diagonal(distance, 0)
                tree = minimum_spanning_tree(csr_matrix(distance)).toarray()
                tree = (tree + tree.T) > 0
                for i, j in zip(*np.triu(tree, 1).nonzero()):
                    cut = tree.copy()
                    cut[i, j] = cut[j, i] = False
                    side = connected_components(csr_matrix(cut), directed=False)[1]
                    labels = current.copy()
                    labels[a[side != 0]] = len(groups)
                    add(labels)
        for u in range(self.n):
            for target in range(len(groups) + 1):
                if target != current[u]:
                    labels = current.copy()
                    labels[u] = target
                    add(labels)
        return np.array(list(unique.values()), dtype=int)

    def repair(self, initial=None, steps=8):
        current = self.node_descent(initial)
        best = float(self.losses(current)[0])
        for _ in range(steps):
            choices = self.edits(current)
            risks = self.losses(choices)
            k = int(np.argmin(risks))
            if risks[k] >= best - 1e-10:
                break
            current = self.node_descent(choices[k])
            best = float(self.losses(current)[0])
        return self.edits(current)


def decode(risk: PartitionRisk, candidates=None):
    """Decode a block, warm-starting from the best of the previous candidates.

    Returns the new candidate table and the chosen partition. Keeping the
    previous best candidate in the table makes the risk non-increasing across
    consecutive decodes (Proposition 2 in the paper).
    """
    warm = None
    if candidates is not None:
        candidates = np.asarray(candidates, dtype=int)
        warm = candidates[int(np.argmin(risk.losses(candidates)))].copy()
    keep = {tuple(canonical(row)): canonical(row) for row in risk.repair(initial=warm)}
    if warm is not None:
        keep.setdefault(tuple(canonical(warm)), canonical(warm))
    table = np.asarray([keep[key] for key in sorted(keep)], dtype=int)
    best = int(np.argmin(risk.losses(table)))
    return table, table[best].copy()
