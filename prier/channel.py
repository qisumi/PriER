"""Reviewer observation channel: L(k | y) over tertile bins of the logit gap."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class ReviewChannel:
    """Categorical likelihood table for binned reviewer scores.

    ``cuts`` are the two empirical tertile boundaries of the calibration scores
    and ``likelihood[y, k]`` is the Laplace-smoothed probability of bin k given
    the true relation y.
    """

    def __init__(self, cuts, likelihood):
        self.cuts = np.asarray(cuts, dtype=float)
        self.likelihood = np.asarray(likelihood, dtype=float)

    @classmethod
    def fit(cls, scores, labels, smoothing=1.0):
        scores = np.asarray(scores, dtype=float)
        labels = np.asarray(labels, dtype=int)
        cuts = np.unique(np.quantile(scores, [1 / 3, 2 / 3]))
        bins = np.searchsorted(cuts, scores, side="right")
        table = np.full((2, len(cuts) + 1), float(smoothing))
        np.add.at(table, (labels, bins), 1)
        table /= table.sum(axis=1, keepdims=True)
        return cls(cuts, table)

    def encode(self, scores):
        return np.searchsorted(self.cuts, scores, side="right")

    def log_ratio(self, score) -> float:
        k = int(self.encode(score))
        return float(np.log(self.likelihood[1, k] / self.likelihood[0, k]))

    @property
    def max_log_ratio(self) -> float:
        """Lambda in Proposition 1: the largest single-review log-odds shift."""
        return float(np.max(np.abs(np.log(self.likelihood[1] / self.likelihood[0]))))

    def to_dict(self):
        return {"cuts": self.cuts.tolist(), "likelihood": self.likelihood.tolist()}

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path):
        value = json.loads(Path(path).read_text())
        return cls(value["cuts"], value["likelihood"])
