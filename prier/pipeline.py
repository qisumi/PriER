"""End-to-end steps: build inputs, fit models, run methods, evaluate.

Directory layout under ``work``::

    <work>/<dataset>/<split>/{records.jsonl, embeddings.npy, cosine.npy, blocks.json}
    <work>/<dataset>/anchor.json, fit_pairs.npy        # anchor prior p^c
    <work>/<dataset>/channel.json, calibration.jsonl   # reviewer channel
    <work>/<dataset>/models/prior_s<seed>[_<variant>].json, readout_s<seed>.json
    <work>/<dataset>/runs/s<seed>.jsonl                # one line per block x method x budget
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import data
from .baselines import run_baseline
from .channel import ReviewChannel
from .evaluate import evaluate_split
from .features import has_institution_schema, identity_features
from .inputs import (Split, block_pairs, encode, fit_anchor, public_text,
                     sample_cross_block_pairs)
from .policy import review_loop, reviewed_posterior
from .prior import IdentityPrior, train_roles, training_pairs
from .readout import apply_readout, heldout_positive_pairs, select_threshold
from .reviewer import BlockReviewer

ANCHOR_SEED = 20261008
CROSS_BLOCK_PAIRS = 50_000


def _write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def _read_json(path):
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------------------
# 1. shared inputs

def build_inputs(data_root, work, dataset, encoder_path, device="cuda:0"):
    """Encode every split, block it, and fit the anchor prior on the fit split."""
    base = Path(work) / dataset
    for split in data.SPLITS:
        records = [{**r, "text": public_text(r)} for r in data.load_records(data_root, dataset, split)]
        records.sort(key=lambda r: r["id"])
        embeddings = encode([r["text"] for r in records], encoder_path, device)
        Split(records, embeddings).save(base / split)

    fit = Split.load(base / "fit")
    labels = data.load_labels(data_root, dataset, "fit")
    pairs = np.concatenate((block_pairs(fit.blocks),
                            sample_cross_block_pairs(fit.blocks, len(fit.records), CROSS_BLOCK_PAIRS, ANCHOR_SEED)))
    y = np.asarray(data.pair_targets(fit.records, labels, pairs))
    known = y >= 0
    anchor = fit_anchor(fit.anchor_features.pairs(pairs[known]), y[known], seed=ANCHOR_SEED)
    _write_json(base / "anchor.json", anchor)
    np.save(base / "fit_pairs.npy", pairs)


# ---------------------------------------------------------------------------
# 2. fitted models

def fit_prior(data_root, work, dataset, seed, drop=()):
    base = Path(work) / dataset
    fit = Split.load(base / "fit")
    labels = data.load_labels(data_root, dataset, "fit")
    anchor = _read_json(base / "anchor.json")
    fit_pairs = np.load(base / "fit_pairs.npy") if (base / "fit_pairs.npy").exists() else block_pairs(fit.blocks)
    pairs, p, y, train = training_pairs(fit, labels, anchor, fit_pairs, seed)
    institution = has_institution_schema(fit.records)
    x = identity_features(fit.records, fit.embeddings, pairs[train], p[train], institution)
    prior = IdentityPrior.fit(x, p[train], y[train], drop=drop, institution=institution)
    name = "prior_s%d%s.json" % (seed, "".join("_drop_" + g for g in drop))
    (base / "models").mkdir(parents=True, exist_ok=True)
    prior.save(base / "models" / name)
    return prior


def fit_readout(data_root, work, dataset, seed):
    """Choose lambda_task on the held-out 20% of fit entities."""
    base = Path(work) / dataset
    task = data.load_task(data_root, dataset)
    fit = Split.load(base / "fit")
    labels = data.load_labels(data_root, dataset, "fit")
    anchor = _read_json(base / "anchor.json")
    prior = IdentityPrior.load(base / "models" / f"prior_s{seed}.json")
    held = ~train_roles(fit.records, labels, seed)
    positives = heldout_positive_pairs(fit.records, labels, held, task["task"] == "cross_source")
    blocks = []
    for block in fit.blocks:
        ix = [i for i in block if held[i]]
        if not ix:
            continue
        records = [fit.records[i] for i in ix]
        p = prior.predict_block(records, fit.embeddings[ix], fit.anchor_matrix(ix, anchor))
        tri = np.triu_indices(len(ix), 1)
        y = np.asarray(data.pair_targets(fit.records, labels, [(ix[a], ix[b]) for a, b in zip(*tri)]))
        weight = data.pair_weight(task, records)
        legal = np.ones(len(y), bool) if weight is None else weight[tri] > 0
        blocks.append((p, y, legal, weight))
    theta, candidates = select_threshold(blocks, positives)
    _write_json(base / "models" / f"readout_s{seed}.json", {"theta": theta, "candidates": candidates})
    return theta


def calibrate_channel(data_root, work, dataset, reviewer, pairs=768, seed=17):
    """Review a uniform sample of calibration-split block pairs and fit L(k | y)."""
    base = Path(work) / dataset
    task = data.load_task(data_root, dataset)
    cal = Split.load(base / "calibration")
    labels = data.load_labels(data_root, dataset, "calibration")
    candidates = block_pairs(cal.blocks)
    if task["task"] == "cross_source":
        same = np.array([cal.records[i]["source"] == cal.records[j]["source"] for i, j in candidates], bool)
        candidates = candidates[~same]
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(len(candidates), size=min(pairs, len(candidates)), replace=False).tolist())
    chosen = sorted(tuple(map(int, candidates[k])) for k in chosen)
    membership = {i: b for b, block in enumerate(cal.blocks) for i in block}
    rows = []
    for b, block in enumerate(cal.blocks):
        active = [pair for pair in chosen if membership[pair[0]] == b]
        if not active:
            continue
        local = {g: k for k, g in enumerate(block)}
        br = BlockReviewer(reviewer, [cal.records[i]["text"] for i in block], task["domain"])
        for i, j in active:
            if not np.isfinite(br.quote_pair(local[i], local[j])):
                continue  # prompt longer than the reviewer context; skipped
            answer = br.review_pair(local[i], local[j])
            rows.append({"pair": [i, j], "score": answer["score"], "tokens": answer["tokens"]})
    y = data.pair_targets(cal.records, labels, [r["pair"] for r in rows])
    channel = ReviewChannel.fit([r["score"] for r in rows], y)
    channel.save(base / "channel.json")
    (base / "calibration.jsonl").write_text("".join(json.dumps({**r, "label": t}) + "\n" for r, t in zip(rows, y)))
    return channel


# ---------------------------------------------------------------------------
# 3. running methods

@dataclass
class Block:
    indices: list
    records: list
    texts: list
    embeddings: np.ndarray
    cosine: np.ndarray
    anchor: np.ndarray
    quotes: np.ndarray
    pair_weight: np.ndarray | None
    channel: ReviewChannel


def prepare_blocks(split, task, anchor, channel, reviewer):
    for indices in split.blocks:
        records = [split.records[i] for i in indices]
        texts = [r["text"] for r in records]
        n = len(indices)
        quoting = BlockReviewer(reviewer, texts, task["domain"])
        quotes = np.array([quoting.quote_pair(i, j) for i, j in zip(*np.triu_indices(n, 1))], dtype=float)
        yield Block(indices=list(indices), records=records, texts=texts,
                    embeddings=split.embeddings[indices],
                    cosine=np.asarray(split.cosine[np.ix_(indices, indices)]),
                    anchor=split.anchor_matrix(indices, anchor), quotes=quotes,
                    pair_weight=data.pair_weight(task, records), channel=channel)


def run_prier(block, reviewer, budget, prior, theta=None):
    if len(block.indices) < 2:
        return np.zeros(len(block.indices), int), {}
    p = prior.predict_block(block.records, block.embeddings, block.anchor)

    def review(i, j):
        return reviewer.review_pair(i, j)

    partition, trace, stats = review_loop(p, block.channel, block.quotes, review, budget, block.pair_weight)
    if theta is not None:
        partition = apply_readout(reviewed_posterior(p, trace), partition, theta, block.pair_weight)
    return partition, stats


def run_dataset(data_root, work, dataset, seed, run_seed, methods, rates, reviewer, readout=False, out=None):
    """Run every method at every budget rate on the evaluation split.

    ``methods`` may contain "prier", "prier_drop_<group>" and baseline names.
    Budgets are ``rate * block size`` prompt tokens per block.
    """
    base = Path(work) / dataset
    task = data.load_task(data_root, dataset)
    split = Split.load(base / "evaluation")
    anchor = _read_json(base / "anchor.json")
    channel = ReviewChannel.load(base / "channel.json")
    priors, theta = {}, None
    for m in methods:
        if m.startswith("prier"):
            priors[m] = IdentityPrior.load(base / "models" / f"prior_s{seed}{m[len('prier'):]}.json")
    if readout:
        theta = _read_json(base / "models" / f"readout_s{seed}.json")["theta"]
    out = Path(out or base / "runs" / f"s{seed}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for b, block in enumerate(prepare_blocks(split, task, anchor, channel, reviewer)):
            n = len(block.indices)
            for method in methods:
                for rate in ((0,) if method == "pivot" else rates):
                    budget = n * rate
                    meter = BlockReviewer(reviewer, block.texts, task["domain"],
                                          prior=block.anchor, channel=channel)
                    started = time.process_time()
                    if method.startswith("prier"):
                        partition, _ = run_prier(block, meter, budget, priors[method],
                                                 theta if method == "prier" else None)
                    elif n < 2:
                        partition = np.zeros(n, int)
                    else:
                        partition = run_baseline(method, block, meter, budget, run_seed)
                    row = {"block": b, "method": method, "rate": rate, "partition": [int(c) for c in partition],
                           "tokens": meter.tokens, "reviews": meter.reviews,
                           "cpu_s": time.process_time() - started}
                    if method == "alper":
                        row["set_queries"] = [e["key"] for e in meter.log if e["kind"] == "set"]
                    f.write(json.dumps(row) + "\n")
    return out


# ---------------------------------------------------------------------------
# 4. evaluation

def evaluate_runs(data_root, work, dataset, runs):
    """Metrics per (method, rate) for one run file."""
    base = Path(work) / dataset
    task = data.load_task(data_root, dataset)
    split = Split.load(base / "evaluation")
    labels = data.load_labels(data_root, dataset, "evaluation")
    grouped = {}
    for line in Path(runs).read_text().splitlines():
        row = json.loads(line)
        grouped.setdefault((row["method"], row["rate"]), {})[row["block"]] = row
    results = []
    for (method, rate), rows in sorted(grouped.items()):
        partitions = [rows[b]["partition"] for b in range(len(split.blocks))]
        metrics = evaluate_split(split.records, labels, task, split.blocks, partitions)
        results.append({"dataset": dataset, "method": method, "rate": rate, **metrics,
                        "tokens": sum(r["tokens"] for r in rows.values()),
                        "reviews": sum(r["reviews"] for r in rows.values()),
                        "cpu_s": sum(r["cpu_s"] for r in rows.values())})
    return results
