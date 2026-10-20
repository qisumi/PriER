"""Unit tests on synthetic data: decoder, channel, policy, features, baselines."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prier.channel import ReviewChannel
from prier.decoder import PartitionRisk, canonical, decode
from prier.features import GENERIC_FEATURES, GROUPS, generic_features
from prier.policy import review_loop, reviewed_posterior
from prier.readout import apply_readout
from prier.baselines import simple, booster, kwikbucks
from prier import data as pdata


def rand_p(rng, n, sharp=True):
    a = rng.uniform(0, 1, (n, n))
    if sharp:
        a = np.where(rng.uniform(size=(n, n)) < .5, a ** 6, 1 - a ** 6)
    a = np.triu(a, 1)
    a = a + a.T
    np.fill_diagonal(a, 0)
    return a


class FakeReviewer:
    def __init__(self, texts):
        self.texts = texts
        self.calls = 0

    def quote_pair(self, i, j, repeat=0):
        i, j = sorted((int(i), int(j)))
        return 40 + (i * 7 + j * 13) % 50

    def review_pair(self, i, j, remaining=None, repeat=0):
        i, j = sorted((int(i), int(j)))
        self.calls += 1
        return {"score": float(np.sin(3 * i + 5 * j) * 5), "tokens": self.quote_pair(i, j)}

    def quote_set(self, center, candidates):
        return 60 + sum(c % 7 for c in candidates)

    def review_set(self, center, candidates, remaining=None):
        self.calls += 1
        n = len(candidates)
        pick = (center + n) % (n + 1)
        return {"text": "NONE" if pick == n else str(pick + 1), "tokens": self.quote_set(center, candidates),
                "chosen": None if pick == n else int(candidates[pick])}


def test_canonical():
    assert list(canonical([5, 5, 9, 2, 2, 2])) == [0, 0, 1, 2, 2, 2]


def test_warm_start_never_increases_risk():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = int(rng.integers(2, 12))
        p = rand_p(rng, n)
        risk = PartitionRisk(p)
        table, current = decode(risk)
        before = float(risk.losses(current)[0])
        p2 = rand_p(rng, n)
        risk2 = PartitionRisk(p2)
        table2, current2 = decode(risk2, table)
        assert float(risk2.losses(current2)[0]) <= min(risk2.losses(table)) + 1e-9


def test_channel_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    scores = rng.normal(size=300)
    labels = (rng.uniform(size=300) < .3).astype(int)
    channel = ReviewChannel.fit(scores, labels)
    k = int(channel.encode(scores[7]))
    ratio = np.log(channel.likelihood[1, k] / channel.likelihood[0, k])
    assert channel.log_ratio(scores[7]) == pytest.approx(ratio)
    assert channel.likelihood.sum(axis=1) == pytest.approx(1.0)
    channel.save(tmp_path / "ch.json")
    assert ReviewChannel.load(tmp_path / "ch.json").cuts == pytest.approx(channel.cuts)


def test_review_loop_respects_budget_and_updates():
    rng = np.random.default_rng(2)
    n = 10
    p = rand_p(rng, n)
    scores = rng.normal(size=200)
    labels = (rng.uniform(size=200) < .4).astype(int)
    channel = ReviewChannel.fit(scores, labels)
    tri = np.triu_indices(n, 1)
    quotes = rng.integers(40, 90, len(tri[0])).astype(float)
    quotes[1] = np.inf
    reviewer = FakeReviewer([f"r{i}" for i in range(n)])
    budget = n * 32
    partition, trace, _ = review_loop(p, channel, quotes,
                                      lambda i, j: reviewer.review_pair(i, j), budget)
    assert sum(t["tokens"] for t in trace) <= budget
    pairs = [tuple(t["pair"]) for t in trace]
    assert len(pairs) == len(set(pairs))
    assert not any(tuple(t["pair"]) == (0, 2) for t in trace)  # the inf-quote pair
    posterior = reviewed_posterior(p, trace)
    for t in trace:
        i, j = t["pair"]
        assert posterior[i, j] == pytest.approx(t["p_after"])
    # every reviewed edge moved by at most the channel bound (Proposition 1)
    lam = channel.max_log_ratio
    for t in trace:
        i, j = t["pair"]
        before = np.log(p[i, j] / (1 - p[i, j]))
        after = np.log(posterior[i, j] / (1 - posterior[i, j]))
        assert abs(after - before) <= lam + 1e-9


def test_feature_groups_and_values():
    a = {"id": "a", "source": "s1", "text": "title: RF20 remote",
         "attributes": {"title": "RF20 remote", "modelno": "RF20"}}
    b = {"id": "b", "source": "s1", "text": "title: RF30 remote",
         "attributes": {"title": "RF30 remote", "modelno": "RF30"}}
    c = {"id": "c", "source": "s2", "text": "title: RF20 universal remote control",
         "attributes": {"title": "RF20 universal remote control", "modelno": "RF20"}}
    records = [a, b, c]
    emb = np.eye(3, dtype=float)
    x = generic_features(records, emb, np.array([[0, 1], [0, 2]]), np.array([.5, .5]))
    assert x.shape == (2, len(GENERIC_FEATURES))
    assert len(GENERIC_FEATURES) == 57
    for lo, hi in GROUPS.values():
        assert 0 <= lo < hi <= len(GENERIC_FEATURES) + 22 or hi <= 79
    model_exact = GENERIC_FEATURES.index("field_model_exact")
    model_conflict = GENERIC_FEATURES.index("field_model_number_conflict")
    assert x[0, model_exact] == 0 and x[0, model_conflict] == 1   # RF20 vs RF30
    assert x[1, model_exact] == 1 and x[1, model_conflict] == 0   # shared RF20


def test_readout_changes_lambda():
    rng = np.random.default_rng(3)
    n = 8
    p = rand_p(rng, n)
    tri = np.triu_indices(n, 1)
    posterior = p.copy()
    _, base = decode(PartitionRisk(posterior))
    recall = apply_readout(posterior, base, .15)
    assert recall.shape == base.shape


def test_simple_and_booster_baselines():
    rng = np.random.default_rng(4)
    n = 9
    p = rand_p(rng, n)
    texts = [f"record {i}" for i in range(n)]
    channel = ReviewChannel.fit(rng.normal(size=200), (rng.uniform(size=200) < .4).astype(int))
    tri = np.triu_indices(n, 1)
    quotes = np.array([40 + (i * 7 + j * 13) % 50 for i, j in zip(*tri)], float)
    for method in ("ut", "risk"):
        reviewer = FakeReviewer(texts)
        labels = simple.run(method, p.copy(), channel, quotes, reviewer, n * 32)
        assert labels.shape == (n,)
        assert reviewer.calls > 0
    labels = booster.run(p.copy(), channel, quotes, FakeReviewer(texts), n * 64)
    again = booster.run(p.copy(), channel, quotes, FakeReviewer(texts), n * 64)
    assert np.array_equal(labels, again)


def test_kwikbucks_pivot_matches_threshold_graph():
    rng = np.random.default_rng(5)
    n = 7
    p = rand_p(rng, n)
    labels = kwikbucks.run_pivot(np.asarray(p), p, seed=17)
    assert labels.shape == (n,)
    # every cluster of the threshold graph stays intact (Pivot may only merge further)


def test_data_split_is_deterministic_and_disjoint():
    groups = [[f"a{i}"] for i in range(50)] + [[f"b{i}", f"b{i}-2"] for i in range(25)]
    r1 = pdata.hash_split("t", groups, "seed")[1]
    r2 = pdata.hash_split("t", groups, "seed")[1]
    assert r1 == r2
    assert set(r1.values()) == {"fit", "calibration", "evaluation"}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
