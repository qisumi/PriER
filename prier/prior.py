"""Identity prior: a residual logistic correction around the anchor p^c.

    logit p_e = logit p^c_e + w^T z_e + b

z_e are the standardised identity features. The anchor log-odds is also one
of the features, so the model spans plain logistic regression; the offset only
centres the L2 shrinkage of the anchor weight at 1 instead of 0.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from .data import entity_groups, pair_targets
from .features import GROUPS, identity_features


def train_roles(records, labels, seed, fraction=0.8):
    """True for records whose entity is used for training (80% of fit entities)."""
    groups = entity_groups(records, labels)
    return np.asarray([
        int.from_bytes(hashlib.sha256(f"{seed}:{g}".encode()).digest()[:8], "big") / 2**64 < fraction
        for g in groups])


class IdentityPrior:
    def __init__(self, mean, scale, coef, intercept, institution=False):
        self.mean = np.asarray(mean, dtype=float)
        self.scale = np.asarray(scale, dtype=float)
        self.coef = np.asarray(coef, dtype=float)
        self.intercept = float(intercept)
        self.institution = bool(institution)

    @classmethod
    def fit(cls, x, anchor, y, *, positive_weight=2.0, l2=1e-4, drop=(), institution=False):
        """Weighted L2-regularised fit; ``drop`` names feature groups to leave out."""
        x, anchor, y = np.asarray(x, float), np.asarray(anchor, float), np.asarray(y, int)
        weights = np.where(y == 1, positive_weight, 1.0)
        mean = np.average(x, axis=0, weights=weights)
        scale = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=weights))
        scale[scale < 1e-6] = 1.0
        active = np.array([c for c in range(x.shape[1])
                           if not any(GROUPS[g][0] <= c < GROUPS[g][1] for g in drop)], dtype=int)
        z = ((x - mean) / scale)[:, active]
        offset = logit(np.clip(anchor, 1e-8, 1 - 1e-8))
        weights = weights / weights.sum()

        def objective(theta):
            logits = offset + z @ theta[:-1] + theta[-1]
            residual = weights * (expit(logits) - y)
            loss = float(weights @ (np.logaddexp(0.0, logits) - y * logits)
                         + 0.5 * l2 * (theta[:-1] @ theta[:-1]))
            return loss, np.r_[z.T @ residual + l2 * theta[:-1], residual.sum()]

        result = minimize(objective, np.zeros(len(active) + 1), jac=True, method="L-BFGS-B",
                          options={"maxiter": 500, "ftol": 1e-10})
        coef = np.zeros(x.shape[1])
        coef[active] = result.x[:-1]
        return cls(mean, scale, coef, result.x[-1], institution)

    def predict_features(self, x, anchor):
        z = (np.asarray(x, float) - self.mean) / self.scale
        return expit(logit(np.clip(np.asarray(anchor, float), 1e-8, 1 - 1e-8)) + z @ self.coef + self.intercept)

    def predict_block(self, records, embeddings, anchor_matrix):
        """Prior matrix for one block from its anchor matrix."""
        n = len(records)
        tri = np.triu_indices(n, 1)
        pairs = np.asarray(tri).T
        x = identity_features(records, embeddings, pairs, anchor_matrix[tri], self.institution)
        values = self.predict_features(x, anchor_matrix[tri])
        p = np.asarray(anchor_matrix, float).copy()
        p[tri] = values
        p[(tri[1], tri[0])] = values
        return p

    def to_dict(self):
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist(), "coef": self.coef.tolist(),
                "intercept": self.intercept, "institution": self.institution}

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict()) + "\n")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


def training_pairs(split, labels, anchor, fit_pairs, seed):
    """Features, anchor values, targets and the 80% train mask on the fit split."""
    records = split.records
    y = np.asarray(pair_targets(records, labels, fit_pairs))
    known = y >= 0
    pairs, y = np.asarray(fit_pairs)[known], y[known]
    from .inputs import predict_anchor

    p = predict_anchor(split.anchor_features.pairs(pairs), anchor)
    roles = train_roles(records, labels, seed)
    train = roles[pairs].all(axis=1)
    return pairs, p, y, train
