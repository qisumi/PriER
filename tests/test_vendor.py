"""The vendored baseline kernels load and run with a fake reviewer."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prier.baselines import acc, alper, kwikbucks


class FakeReviewer:
    def __init__(self, texts):
        self.texts = texts
        self.calls = 0

    def quote_pair(self, i, j, repeat=0):
        return 40 + (int(i) * 7 + int(j) * 13) % 50

    def review_pair(self, i, j, remaining=None, repeat=0):
        self.calls += 1
        return {"score": float(np.sin(3 * i + 5 * j) * 5), "tokens": self.quote_pair(i, j),
                "input_tokens": self.quote_pair(i, j), "output_tokens": 0,
                "probability": float((np.sin(3 * i + 5 * j) + 1) / 2)}

    def quote_set(self, center, candidates):
        return 60 + sum(c % 7 for c in candidates)

    def review_set(self, center, candidates, remaining=None):
        self.calls += 1
        n = len(candidates)
        pick = (center + n) % (n + 1)
        return {"text": "NONE" if pick == n else str(pick + 1), "tokens": self.quote_set(center, candidates),
                "input_tokens": self.quote_set(center, candidates) - 16, "output_tokens": 16,
                "chosen": None if pick == n else int(candidates[pick])}


@pytest.fixture
def block():
    rng = np.random.default_rng(6)
    n = 10
    p = np.triu(rng.uniform(0, 1, (n, n)) ** 3, 1)
    p = p + p.T
    np.fill_diagonal(p, 0)
    cosine = 2 * p - .5
    np.fill_diagonal(cosine, 1)
    return p, cosine, [f"record {i}" for i in range(n)]


def test_alper_runs(block):
    p, cosine, texts = block
    reviewer = FakeReviewer(texts)
    labels = alper.run(cosine, texts, reviewer, budget=len(texts) * 64, seed=17)
    assert labels.shape == (len(texts),)


def test_acc_runs(block):
    p, _, texts = block
    labels = acc.run(p, FakeReviewer(texts), budget=len(texts) * 64, seed=17)
    assert labels.shape == (len(texts),)


def test_kwikbucks_variants_run(block):
    p, cosine, texts = block
    for runner in (kwikbucks.run_original, kwikbucks.run_budgeted):
        labels = runner(cosine, FakeReviewer(texts), budget=len(texts) * 128, seed=17)
        assert labels.shape == (len(texts),)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
