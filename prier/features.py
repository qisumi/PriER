"""Pre-review identity features (57 generic + 22 for institution names).

Generic features (one extraction rule for every schema):

* 25 structured field-family features: attribute names are mapped to the
  families brand / model / capacity / version / other; per family we compute
  availability, token overlap, exact equality, number overlap and number conflict.
* 4 free-text identifier features: overlap and conflict of numbers and of
  alphanumeric model-code tokens.
* 4 anchor and provenance features: anchor log-odds, anchor variance,
  source equality and raw embedding cosine.
* 24 subspace embedding interactions: mean |a-b| and mean a*b over 12
  contiguous groups of embedding dimensions.
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

import numpy as np
from scipy.special import logit

FAMILIES = ("brand", "model", "capacity", "version", "other")
EMBED_GROUPS = 12
GENERIC_FEATURES = (
    [f"field_{f}_{s}" for f in FAMILIES
     for s in ("available", "token_overlap", "exact", "number_overlap", "number_conflict")]
    + ["text_number_overlap", "text_number_conflict", "text_model_overlap", "text_model_conflict",
       "anchor_logit", "anchor_uncertainty", "source_equal", "embedding_cosine"]
    + [f"embedding_{s}_{k}" for s in ("absdiff", "product") for k in range(EMBED_GROUPS)]
)

# Column ranges of the feature groups, used by the leave-one-group-out ablation.
GROUPS = {"fields": (0, 25), "identifiers": (25, 29), "anchor_source": (29, 33),
          "embedding_interactions": (33, 57), "institution": (57, 79)}

INSTITUTION_FIELDS = {"affiliation", "institution", "organization", "organisation"}


def _token_sets(text):
    text = str(text).casefold()
    words = set(re.findall(r"\w+", text))
    numbers = set(re.findall(r"\d+(?:\.\d+)?", text))
    codes = set(re.findall(r"\b(?=\w*\d)(?=\w*[a-z])[a-z0-9]+\b", text))
    return words, numbers, codes


def _overlap(a, b):
    return len(a & b) / max(1, len(a | b))


def _family(key):
    key = str(key).casefold()
    if any(s in key for s in ("brand", "manufacturer", "maker")):
        return "brand"
    if any(s in key for s in ("model", "mpn", "part", "sku", "isbn", "upc", "productid")):
        return "model"
    if any(s in key for s in ("capacity", "size", "memory", "storage", "weight", "volume")):
        return "capacity"
    if any(s in key for s in ("version", "edition", "release", "year")):
        return "version"
    return "other"


def generic_features(records, embeddings, pairs, anchor):
    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
    emb = np.asarray(embeddings, dtype=float)
    anchor = np.asarray(anchor, dtype=float)
    out = np.zeros((len(pairs), len(GENERIC_FEATURES)))
    text_sets = [_token_sets(r["text"]) for r in records]
    attrs = [{str(k): str(v).casefold().strip() for k, v in r["attributes"].items()} for r in records]
    for row, (i, j) in enumerate(pairs):
        left, right = attrs[i], attrs[j]
        stats = np.zeros((len(FAMILIES), 5))
        counts = np.zeros(len(FAMILIES))
        for key in sorted(set(left) | set(right)):
            g = FAMILIES.index(_family(key))
            counts[g] += 1
            a, b = left.get(key, ""), right.get(key, "")
            if not a or not b:
                continue
            ta, na, _ = _token_sets(a)
            tb, nb, _ = _token_sets(b)
            stats[g] += (1.0, _overlap(ta, tb), float(a == b), _overlap(na, nb),
                         float(bool(na) and bool(nb) and na != nb))
        stats /= np.maximum(counts, 1.0)[:, None]
        out[row, :25] = stats.ravel()
        _, na, ma = text_sets[i]
        _, nb, mb = text_sets[j]
        p = anchor[row]
        out[row, 25:33] = (_overlap(na, nb), float(bool(na) and bool(nb) and na != nb),
                           _overlap(ma, mb), float(bool(ma) and bool(mb) and ma != mb),
                           float(logit(np.clip(p, 1e-6, 1 - 1e-6))), 4 * p * (1 - p),
                           float(records[i]["source"] == records[j]["source"]),
                           float(emb[i] @ emb[j]))
    if len(pairs):
        a, b = emb[pairs[:, 0]], emb[pairs[:, 1]]
        for k, cols in enumerate(np.array_split(np.arange(emb.shape[1]), EMBED_GROUPS)):
            if len(cols):
                out[:, 33 + k] = np.abs(a[:, cols] - b[:, cols]).mean(axis=1)
                out[:, 33 + EMBED_GROUPS + k] = (a[:, cols] * b[:, cols]).mean(axis=1)
    return out


# ---------------------------------------------------------------------------
# institution names

INSTITUTION_FEATURES = [
    "aff_normalized_exact", "aff_token_jaccard", "aff_token_containment",
    "aff_char_trigram_jaccard", "aff_string_ratio", "aff_org_present_both",
    "aff_org_token_jaccard", "aff_org_token_containment", "aff_org_exact",
    "aff_acronym_jaccard", "aff_cross_acronym_any", "aff_cross_acronym_count",
    "aff_org_initials_jaccard", "aff_subunit_present_both", "aff_subunit_jaccard",
    "aff_subunit_conflict", "aff_geo_present_both", "aff_geo_jaccard",
    "aff_geo_conflict", "aff_country_agreement", "aff_country_conflict",
    "aff_length_ratio",
]
ALIASES = dict(univ="university", universite="university", universitat="university",
               universita="university", dept="department", dep="department",
               inst="institute", instit="institute", tech="technology",
               sci="science", comp="computer", eng="engineering", centre="center")
STOP = {"of", "the", "and", "for", "in", "at", "on", "a", "an", "de", "di", "du", "la", "le"}
ORG = {"university", "institute", "college", "hospital", "laboratory", "academy",
       "polytechnic", "technion", "corporation", "company", "research"}
SUB = {"department", "division", "school", "faculty", "group", "unit", "center",
       "centre", "program", "programme", "laboratories", "lab"}
COUNTRIES = {"usa", "uk", "canada", "china", "germany", "france", "italy", "japan",
             "australia", "austria", "switzerland", "netherlands", "sweden", "norway",
             "finland", "denmark", "spain", "portugal", "india", "israel", "korea",
             "singapore", "taiwan", "greece", "belgium", "ireland", "brazil", "russia",
             "poland", "turkey", "mexico", "argentina", "hungary", "romania", "iran",
             "egypt", "thailand", "malaysia", "pakistan", "bangladesh", "vietnam"}


def _plain(value):
    return "".join(c for c in unicodedata.normalize("NFKD", str(value)) if not unicodedata.combining(c))


def _words(value):
    value = _plain(value).casefold()
    value = re.sub(r"\bunited states(?: of america)?\b|\bu\.?s\.?a?\.?\b", " usa ", value)
    value = re.sub(r"\bunited kingdom\b|\bu\.?k\.?\b", " uk ", value)
    return [ALIASES.get(t, t) for t in re.findall(r"[a-z]+", value) if t not in STOP]


def _jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def _containment(a, b):
    return len(a & b) / min(len(a), len(b)) if a and b else 0.0


def _initials(tokens):
    # Prefixes keep abbreviations like "ETHZ" when a city follows the name.
    return {"".join(t[0] for t in tokens[:k]) for k in range(2, min(8, len(tokens)) + 1)}


def _institution_name(record):
    attributes = record.get("attributes", {})
    for key in ("affiliation", "institution", "organization", "organisation", "name"):
        if attributes.get(key):
            return str(attributes[key])
    return re.sub(r"^[^:]{1,30}:\s*", "", str(record["text"]))


def _institution_view(record):
    text = _institution_name(record)
    tokens = _words(text)
    segments = [_words(x) for x in re.split(r"[,;|\n]+", text)]
    org_segments = [s for s in segments if set(s) & ORG]
    # A lone abbreviation has no organization marker; keep it as the core name.
    core_segments = org_segments or [s for s in segments if not set(s) & SUB][:1]
    core = [t for s in core_segments for t in s if t not in COUNTRIES]
    sub = {t for s in segments if set(s) & SUB for t in s if t not in SUB | ORG}
    last_org = max((i for i, s in enumerate(segments) if set(s) & ORG), default=-1)
    geo = ({t for i, s in enumerate(segments) if i > last_org and not set(s) & (ORG | SUB)
            for t in s if t not in COUNTRIES} if last_org >= 0 else set())
    acronym = {_plain(t).casefold() for t in re.findall(r"\b[A-Z][A-Z0-9]{1,7}\b", text)
               if _plain(t).casefold() not in COUNTRIES | STOP | ORG | SUB}
    normalized = " ".join(tokens)
    compact = normalized.replace(" ", "")
    return dict(tokens=set(tokens), normalized=normalized,
                grams={compact[i:i + 3] for i in range(max(0, len(compact) - 2))},
                core=set(core), initials=set().union(*(_initials(s) for s in core_segments)),
                acronym=acronym, sub=sub, geo=geo, country=set(tokens) & COUNTRIES)


def institution_features(records, pairs):
    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
    views = [_institution_view(r) for r in records]
    out = np.zeros((len(pairs), len(INSTITUTION_FEATURES)))
    for row, (i, j) in enumerate(pairs):
        a, b = views[i], views[j]
        cross = (a["acronym"] & b["initials"]) | (b["acronym"] & a["initials"])
        out[row] = (
            float(bool(a["normalized"]) and a["normalized"] == b["normalized"]),
            _jaccard(a["tokens"], b["tokens"]), _containment(a["tokens"], b["tokens"]),
            _jaccard(a["grams"], b["grams"]),
            # SequenceMatcher is order dependent; average both directions.
            .5 * (SequenceMatcher(None, a["normalized"], b["normalized"]).ratio()
                  + SequenceMatcher(None, b["normalized"], a["normalized"]).ratio()),
            float(bool(a["core"]) and bool(b["core"])), _jaccard(a["core"], b["core"]),
            _containment(a["core"], b["core"]), float(bool(a["core"]) and a["core"] == b["core"]),
            _jaccard(a["acronym"], b["acronym"]), float(bool(cross)), min(3, len(cross)) / 3,
            _jaccard(a["initials"], b["initials"]),
            float(bool(a["sub"]) and bool(b["sub"])), _jaccard(a["sub"], b["sub"]),
            float(bool(a["sub"]) and bool(b["sub"]) and not a["sub"] & b["sub"]),
            float(bool(a["geo"]) and bool(b["geo"])), _jaccard(a["geo"], b["geo"]),
            float(bool(a["geo"]) and bool(b["geo"]) and not a["geo"] & b["geo"]),
            float(bool(a["country"] & b["country"])),
            float(bool(a["country"]) and bool(b["country"]) and not a["country"] & b["country"]),
            min(len(a["normalized"]), len(b["normalized"]))
            / max(1, len(a["normalized"]), len(b["normalized"])),
        )
    return out


def has_institution_schema(records) -> bool:
    fields = {str(k).casefold() for r in records for k in r.get("attributes", {})}
    return bool(fields & INSTITUTION_FIELDS)


def identity_features(records, embeddings, pairs, anchor, institution=False):
    x = generic_features(records, embeddings, pairs, anchor)
    if institution:
        x = np.column_stack((x, institution_features(records, pairs)))
    return x
