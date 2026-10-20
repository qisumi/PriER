"""Shared inputs: MiniLM embeddings, blocking and the 6-feature anchor prior.

All methods (PriER and every baseline) use the same blocks and the same
anchor probabilities p^c, so they only differ in how they use the reviewer.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ANCHOR_FEATURES = ("cosine", "token_jaccard", "number_jaccard",
                   "nonempty_number_equality", "token_length_ratio", "source_equality")
TEXT_WORDS = 160


def public_text(record) -> str:
    """Text shown to the encoder and the reviewer (first 160 words)."""
    return " ".join(record["text"].split()[:TEXT_WORDS])


# ---------------------------------------------------------------------------
# encoder

def encode(texts, model_path, device="cuda:0", batch_size=64, max_length=256):
    """Mean-pooled, L2-normalised all-MiniLM-L6-v2 sentence embeddings."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device).eval()
    chunks = []
    with torch.inference_mode():
        for lo in range(0, len(texts), batch_size):
            batch = tokenizer(list(texts[lo:lo + batch_size]), padding=True, truncation=True,
                              max_length=max_length, return_tensors="pt").to(device)
            mask = batch["attention_mask"]
            hidden = model(**batch).last_hidden_state
            pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
            chunks.append(torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy())
    return np.concatenate(chunks).astype(np.float32) if chunks else np.empty((0, 384), np.float32)


def cosine_matrix(embeddings, tile=512):
    """Exactly symmetric float32 cosine matrix, computed tile by tile."""
    n = len(embeddings)
    cosine = np.empty((n, n), dtype=np.float32)
    for left in range(0, n, tile):
        a = slice(left, min(n, left + tile))
        for right in range(left, n, tile):
            b = slice(right, min(n, right + tile))
            values = np.clip(embeddings[a] @ embeddings[b].T, -1.0, 1.0)
            if left == right:
                values = np.triu(values) + np.triu(values, 1).T
            cosine[a, b] = values
            if left != right:
                cosine[b, a] = values.T
    np.fill_diagonal(cosine, 1.0)
    return cosine


# ---------------------------------------------------------------------------
# blocking

def block_records(cosine, top_k=10, min_cosine=0.5, max_size=64):
    """Link each record to its top-k neighbours with cosine >= min_cosine and
    merge components greedily in descending cosine order, never exceeding
    ``max_size`` records per block."""
    n = len(cosine)
    k = min(top_k, max(0, n - 1))
    edges = {}
    for i in range(n):
        if not k:
            continue
        scores = np.array(cosine[i], dtype=np.float32, copy=True)
        scores[i] = -np.inf
        kth = np.partition(scores, n - k)[n - k]
        above = np.flatnonzero(scores > kth)
        tied = np.flatnonzero(scores == kth)[:k - len(above)]  # lowest ids win ties
        for j in np.concatenate((above, tied)):
            if scores[j] >= min_cosine:
                edge = (min(i, int(j)), max(i, int(j)))
                edges[edge] = max(float(scores[j]), edges.get(edge, -np.inf))

    parent, size = list(range(n)), [1] * n

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for (i, j), _ in sorted(edges.items(), key=lambda item: (-item[1], *item[0])):
        a, b = find(i), find(j)
        if a != b and size[a] + size[b] <= max_size:
            a, b = min(a, b), max(a, b)
            parent[b] = a
            size[a] += size[b]
    grouped = {}
    for i in range(n):
        grouped.setdefault(find(i), []).append(i)
    return sorted(grouped.values(), key=lambda block: block[0])


def block_pairs(blocks):
    return np.asarray([(b[i], b[j]) for b in blocks for i in range(len(b)) for j in range(i + 1, len(b))],
                      dtype=np.int64).reshape(-1, 2)


# ---------------------------------------------------------------------------
# anchor prior p^c: logistic regression on six coarse features

class AnchorFeatures:
    def __init__(self, records, cosine):
        self.records = records
        self.cosine = cosine
        self.tokens = [set(re.findall(r"\w+", r["text"].casefold())) for r in records]
        self.numbers = [set(re.findall(r"\d+(?:\.\d+)?", r["text"])) for r in records]

    def pairs(self, pairs):
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        out = np.empty((len(pairs), len(ANCHOR_FEATURES)), dtype=np.float32)
        for k, (i, j) in enumerate(pairs):
            ta, tb, na, nb = self.tokens[i], self.tokens[j], self.numbers[i], self.numbers[j]
            out[k] = (self.cosine[i, j],
                      len(ta & tb) / max(1, len(ta | tb)),
                      len(na & nb) / max(1, len(na | nb)),
                      float(bool(na) and na == nb),
                      min(len(ta), len(tb)) / max(1, max(len(ta), len(tb))),
                      float(self.records[i]["source"] == self.records[j]["source"]))
        return out


def sample_cross_block_pairs(blocks, n, count, seed):
    """Uniform sample of unordered pairs whose records lie in different blocks."""
    membership = np.empty(n, dtype=np.int32)
    for b, block in enumerate(blocks):
        membership[list(block)] = b
    population = n * (n - 1) // 2 - sum(len(b) * (len(b) - 1) // 2 for b in blocks)
    target = min(count, population)
    if not target:
        return np.empty((0, 2), dtype=np.int32)
    rng = np.random.default_rng(seed)
    if population <= 2 * target:
        possible = [(i, j) for i in range(n) for j in range(i + 1, n) if membership[i] != membership[j]]
        chosen = rng.choice(len(possible), size=target, replace=False)
        return np.asarray(sorted(possible[i] for i in chosen), dtype=np.int32).reshape(-1, 2)
    sampled = set()
    while len(sampled) < target:
        size = min(65536, max(1024, 2 * (target - len(sampled))))
        left = rng.integers(0, n, size=size)
        right = rng.integers(0, n - 1, size=size)
        right += right >= left
        for i, j in zip(left, right):
            if membership[i] != membership[j]:
                sampled.add((min(int(i), int(j)), max(int(i), int(j))))
                if len(sampled) == target:
                    break
    return np.asarray(sorted(sampled), dtype=np.int32).reshape(-1, 2)


def fit_anchor(features, targets, seed=20261008):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=seed)
    model.fit(features, targets)
    return {"coef": model.coef_.ravel().tolist(), "intercept": float(model.intercept_[0])}


def predict_anchor(features, anchor):
    z = np.asarray(features, dtype=np.float64) @ np.asarray(anchor["coef"], dtype=np.float64) + anchor["intercept"]
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


# ---------------------------------------------------------------------------
# per-split container

class Split:
    """Records, embeddings, cosine matrix and blocks of one data split."""

    def __init__(self, records, embeddings, cosine=None, blocks=None):
        self.records = records
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.cosine = cosine_matrix(self.embeddings) if cosine is None else cosine
        self.blocks = block_records(self.cosine) if blocks is None else blocks
        self.anchor_features = AnchorFeatures(records, self.cosine)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "records.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in self.records))
        np.save(directory / "embeddings.npy", self.embeddings)
        np.save(directory / "cosine.npy", self.cosine)
        (directory / "blocks.json").write_text(json.dumps(self.blocks) + "\n")

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        records = [json.loads(x) for x in (directory / "records.jsonl").read_text().splitlines() if x]
        return cls(records, np.load(directory / "embeddings.npy"),
                   np.load(directory / "cosine.npy", mmap_mode="r"),
                   json.loads((directory / "blocks.json").read_text()))

    def anchor_matrix(self, indices, anchor):
        """Anchor probabilities for all pairs among ``indices`` (zero diagonal)."""
        ix = np.asarray(indices, dtype=np.int64)
        n = len(ix)
        tri = np.triu_indices(n, 1)
        p = np.zeros((n, n))
        if n > 1:
            values = predict_anchor(self.anchor_features.pairs(np.column_stack((ix[tri[0]], ix[tri[1]]))), anchor)
            p[tri] = values
            p[(tri[1], tri[0])] = values
        return p
