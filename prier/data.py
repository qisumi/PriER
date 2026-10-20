"""Benchmark preparation: normalized records, entity-disjoint splits and labels.

Every benchmark is converted into the same layout::

    <root>/<dataset>/task.json
    <root>/<dataset>/<split>/records.jsonl   # id, source, text, attributes
    <root>/<dataset>/<split>/labels.json     # only read for fitting/evaluation

Records receive opaque ids, and whole entities (or, for pair-only benchmarks,
whole known-match components) are assigned to fit / calibration / evaluation
with a fixed 50/20/30 hash split.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

SPLITS = ("fit", "calibration", "evaluation")

# task: "partition" scores every intra-block pair, "cross_source" scores only
# pairs whose records come from two different sources.
DATASETS = {
    "abt_buy": {"task": "partition", "domain": "product"},
    "amazon_google": {"task": "cross_source", "domain": "product"},
    "restaurant": {"task": "partition", "domain": "generic"},
    "dblp_acm": {"task": "cross_source", "domain": "bibliographic"},
    "musicbrainz_20k": {"task": "partition", "domain": "music"},
    "affiliations": {"task": "partition", "domain": "organization"},
    "walmart_amazon": {"task": "cross_source", "domain": "product"},
}
PRIMARY = ("abt_buy", "amazon_google", "restaurant", "dblp_acm", "musicbrainz_20k", "affiliations")

# Split seeds of the original preparation runs; changing them changes the splits.
LEIPZIG_SEED = "e2-benchmarks-20261008-v1"
EXTENSION_SEED = "e2-paper-extension-20261010-v1"
WALMART_SEED = "e2-unseen-walmart-amazon-20261019-v1"


# ---------------------------------------------------------------------------
# helpers

def digest(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()


def read_csv(data: bytes, header: bool = True):
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin1")
    reader = csv.DictReader(io.StringIO(text)) if header else csv.reader(io.StringIO(text))
    return list(reader)


def make_record(dataset, source, original_id, attributes) -> dict:
    attrs = {str(k): " ".join(str(v or "").split()) for k, v in attributes.items()}
    return {
        "id": digest("record", dataset, source, str(original_id)),
        "source": source,
        "text": "; ".join(f"{k}: {v}" for k, v in attrs.items() if v),
        "attributes": attrs,
    }


def components(ids, edges):
    """Connected components of ``edges`` over ``ids`` (sorted lists)."""
    parent = {rid: rid for rid in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)
    groups = defaultdict(list)
    for rid in ids:
        groups[find(rid)].append(rid)
    return sorted(sorted(g) for g in groups.values())


def hash_split(dataset, groups, seed):
    """Assign whole groups to fit/calibration/evaluation (50/20/30)."""
    record_to_entity, record_to_split = {}, {}
    for group in groups:
        entity = digest("entity", dataset, *sorted(group))
        bucket = int(digest("split", seed, dataset, entity)[:16], 16) % 10000
        split = "fit" if bucket < 5000 else "calibration" if bucket < 7000 else "evaluation"
        for rid in group:
            record_to_entity[rid] = entity
            record_to_split[rid] = split
    return record_to_entity, record_to_split


# ---------------------------------------------------------------------------
# raw readers; each returns (records, groups, truth)

PAIR_FILES = {
    "abt_buy": ("Abt-Buy.zip", "Abt.csv", "Buy.csv", "abt_buy_perfectMapping.csv",
                "abt", "buy", "idAbt", "idBuy"),
    "dblp_acm": ("DBLP-ACM.zip", "DBLP2.csv", "ACM.csv", "DBLP-ACM_perfectMapping.csv",
                 "dblp", "acm", "idDBLP", "idACM"),
    "amazon_google": ("Amazon-GoogleProducts.zip", "Amazon.csv", "GoogleProducts.csv",
                      "Amzon_GoogleProducts_perfectMapping.csv",
                      "amazon", "google", "idAmazon", "idGoogleBase"),
}


def read_pair_benchmark(raw: Path, name: str):
    """Two-source benchmark with an exhaustive cross-source mapping."""
    archive, left_csv, right_csv, gold_csv, ls, rs, lc, rc = PAIR_FILES[name]
    with zipfile.ZipFile(raw / archive) as z:
        left, right, gold = (read_csv(z.read(f)) for f in (left_csv, right_csv, gold_csv))
    records, lookup = [], {}
    for source, rows in ((ls, left), (rs, right)):
        for row in rows:
            record = make_record(name, source, row["id"], {k: v for k, v in row.items() if k != "id"})
            lookup[source, row["id"]] = record["id"]
            records.append(record)
    edges = {(lookup[ls, row[lc]], lookup[rs, row[rc]]) for row in gold}
    groups = components([r["id"] for r in records], edges)
    # When every record has a cross-source partner, the mapping defines a full
    # entity partition (Abt-Buy); otherwise only cross-source pairs are known.
    covered = {rid for edge in edges for rid in edge}
    truth = {"partition_available": len(covered) == len(records),
             "positive_pairs": sorted([list(e) for e in edges]),
             "source_universe": [ls, rs]}
    return records, groups, truth


def read_affiliations(raw: Path):
    with zipfile.ZipFile(raw / "affiliationstrings.zip") as z:
        rows = read_csv(z.read("affiliationstrings_ids.csv"))
        mapping = read_csv(z.read("affiliationstrings_mapping.csv"), header=False)
    records = [make_record("affiliations", "affiliation_strings", row["id1"], {"affiliation": row["affil1"]})
               for row in rows]
    lookup = {row["id1"]: rec["id"] for row, rec in zip(rows, records)}
    edges = {tuple(sorted((lookup[a], lookup[b]))) for a, b in mapping}
    groups = components([r["id"] for r in records], edges)
    return records, groups, {"partition_available": True}


def read_musicbrainz(raw: Path):
    rows = read_csv((raw / "musicbrainz-20-A01.csv.dapo").read_bytes())
    fields = ("number", "title", "length", "artist", "album", "year", "language")
    records, grouped = [], defaultdict(list)
    for row in rows:
        record = make_record("musicbrainz_20k", "music_source_" + row["SourceID"], row["TID"],
                             {k: row[k] for k in fields})
        records.append(record)
        grouped[row["CID"]].append(record["id"])
    groups = sorted(sorted(g) for g in grouped.values())
    return records, groups, {"partition_available": True}


def read_restaurant(raw: Path):
    import pandas as pd
    import rdata  # only needed for this benchmark

    table = rdata.read_rda(raw / "restaurant.RData")["restaurant"]
    records, grouped = [], defaultdict(list)
    for row in table.to_dict(orient="records"):
        attrs = {k: "" if pd.isna(row[k]) else str(row[k]) for k in ("name", "addr", "city", "phone", "type")}
        record = make_record("restaurant", "restaurant", int(row["id"]), attrs)
        records.append(record)
        grouped[str(row["class"])].append(record["id"])
    groups = sorted(sorted(g) for g in grouped.values())
    return records, groups, {"partition_available": True}


# ---------------------------------------------------------------------------
# writing

def write_dataset(out: Path, name: str, records, record_to_split, truth, record_to_entity=None,
                   task=None):
    records = sorted(records, key=lambda r: r["id"])
    task = task or {"dataset": name, **DATASETS[name]}
    if truth.get("source_universe"):
        task["source_universe"] = truth["source_universe"]
    base = out / name
    base.mkdir(parents=True, exist_ok=True)
    (base / "task.json").write_text(json.dumps(task, indent=2) + "\n")
    summary = {}
    for split in SPLITS:
        rows = [r for r in records if record_to_split[r["id"]] == split]
        ids = {r["id"] for r in rows}
        labels = {"split": split, "partition_available": truth["partition_available"],
                  "source_universe": truth.get("source_universe"),
                  "positive_pairs": [p for p in truth.get("positive_pairs", [])
                                     if p[0] in ids and p[1] in ids]}
        if truth["partition_available"]:
            labels["record_to_entity"] = {rid: record_to_entity[rid] for rid in sorted(ids)}
        d = base / split
        d.mkdir(exist_ok=True)
        (d / "records.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))
        (d / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, sort_keys=True) + "\n")
        summary[split] = len(rows)
    return summary


def prepare_standard(raw: Path, out: Path, name: str):
    if name in PAIR_FILES:
        records, groups, truth = read_pair_benchmark(raw, name)
    elif name == "affiliations":
        records, groups, truth = read_affiliations(raw)
    elif name == "musicbrainz_20k":
        records, groups, truth = read_musicbrainz(raw)
    elif name == "restaurant":
        records, groups, truth = read_restaurant(raw)
    else:
        raise ValueError(name)
    seed = EXTENSION_SEED if name in ("restaurant", "amazon_google") else LEIPZIG_SEED
    record_to_entity, record_to_split = hash_split(name, groups, seed)
    return write_dataset(out, name, records, record_to_split, truth, record_to_entity)


# ---------------------------------------------------------------------------
# Walmart-Amazon (held-out transfer benchmark)

WALMART_FIELDS = ("title", "category", "brand", "modelno", "price")


def _key(value):
    return re.sub(r"[^\w]+", "", str(value).casefold())


def _signatures(attributes):
    """Exact title or brand+model keys, used to drop records seen in other benchmarks."""
    attrs = {str(k).casefold(): str(v) for k, v in attributes.items()}
    title = next((attrs[k] for k in ("title", "name", "product_name") if attrs.get(k)), "")
    brand = next((attrs[k] for k in ("brand", "manufacturer", "maker") if attrs.get(k)), "")
    model = next((attrs[k] for k in ("modelno", "model", "model_number", "mpn") if attrs.get(k)), "")
    keys = []
    if len(_key(title)) >= 20:
        keys.append(("title", _key(title)))
    if brand and model and any(c.isdigit() for c in model):
        keys.append(("brand_model", _key(brand) + "|" + _key(model)))
    return keys


def prepare_walmart(raw: Path, out: Path):
    """Needs the six primary benchmarks prepared first (for the overlap filter)."""
    seen = set()
    for name in PRIMARY:
        for split in SPLITS:
            for line in (out / name / split / "records.jsonl").read_text().splitlines():
                seen.update(_signatures(json.loads(line)["attributes"]))

    def norm(value):
        return " ".join(str(value or "").split())

    with zipfile.ZipFile(raw / "walmart_amazon_raw_data.zip") as z:
        table_a, table_b, matches = (read_csv(z.read(f)) for f in ("tableA.csv", "tableB.csv", "matches.csv"))
    records, lookup = [], {}
    for source, rows in (("walmart", table_a), ("amazon", table_b)):
        for row in rows:
            rid = hashlib.sha256(f"walmart_amazon:{source}:{row['custom_id']}".encode()).hexdigest()
            attrs = {"title": norm(row.get("title")),
                     "category": norm(row.get("groupname") if source == "walmart" else row.get("category1")),
                     "brand": norm(row.get("brand")), "modelno": norm(row.get("modelno")),
                     "price": norm(row.get("price"))}
            text = " | ".join(f"{k}: {attrs[k]}" for k in WALMART_FIELDS if attrs[k])
            records.append({"id": rid, "source": source, "text": text, "attributes": attrs})
            lookup[source, row["custom_id"]] = rid
    positives = sorted({tuple(sorted((lookup["walmart", m["id1"]], lookup["amazon", m["id2"]])))
                        for m in matches})
    groups = components(sorted(r["id"] for r in records), positives)
    by_id = {r["id"]: r for r in records}
    flagged = {tuple(g) for g in groups
               if any(sig in seen for rid in g for sig in _signatures(by_id[rid]["attributes"]))}
    record_to_split = {}
    for group in groups:
        member_hash = hashlib.sha256("\n".join(group).encode()).hexdigest()
        h = int.from_bytes(hashlib.sha256(f"{WALMART_SEED}:{member_hash}".encode()).digest()[:8], "big") / 2**64
        split = ("quarantine" if tuple(group) in flagged
                 else "fit" if h < .5 else "calibration" if h < .7 else "evaluation")
        for rid in group:
            record_to_split[rid] = split
    truth = {"partition_available": False, "positive_pairs": [list(p) for p in positives],
             "source_universe": ["walmart", "amazon"]}
    return write_dataset(out, "walmart_amazon", records, record_to_split, truth)


def prepare(raw: Path, out: Path, names=None):
    raw, out = Path(raw), Path(out)
    names = list(names or DATASETS)
    summary = {}
    for name in names:
        if name != "walmart_amazon":
            summary[name] = prepare_standard(raw, out, name)
    if "walmart_amazon" in names:
        summary["walmart_amazon"] = prepare_walmart(raw, out)
    return summary


# ---------------------------------------------------------------------------
# loading

def load_task(root, dataset) -> dict:
    return json.loads((Path(root) / dataset / "task.json").read_text())


def load_records(root, dataset, split) -> list[dict]:
    path = Path(root) / dataset / split / "records.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def load_labels(root, dataset, split) -> dict:
    return json.loads((Path(root) / dataset / split / "labels.json").read_text())


def pair_targets(records, labels, pairs):
    """1/0 for known relations, -1 for same-source pairs of pair-only benchmarks."""
    if labels["partition_available"]:
        entity = labels["record_to_entity"]
        return [int(entity[records[i]["id"]] == entity[records[j]["id"]]) for i, j in pairs]
    universe = set(labels["source_universe"])
    positives = {frozenset(p) for p in labels["positive_pairs"]}
    return [int(frozenset((records[i]["id"], records[j]["id"])) in positives)
            if {records[i]["source"], records[j]["source"]} == universe else -1
            for i, j in pairs]


def entity_groups(records, labels):
    """Entity id per record; pair-only benchmarks use known-match components."""
    if labels["partition_available"]:
        return [str(labels["record_to_entity"][r["id"]]) for r in records]
    parent = {r["id"]: r["id"] for r in records}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in labels["positive_pairs"]:
        x, y = find(a), find(b)
        parent[max(x, y)] = min(x, y)
    return [find(r["id"]) for r in records]


def pair_weight(task, records):
    """Task weights: 1 for evaluated pairs, 0 for same-source pairs in cross-source tasks."""
    if task["task"] != "cross_source":
        return None
    import numpy as np

    source = np.asarray([r["source"] for r in records])
    return (source[:, None] != source[None, :]).astype(float)
