# Third-party code and data

## Code (vendor/)

| Component | Source | License |
|---|---|---|
| `vendor/kwikbucks/` | [google-research/kwikbucks](https://github.com/google-research/google-research/tree/5b09c22d73a9d35eb6c5d2a99b95677a45053466/kwikbucks) @ `5b09c22` | Apache-2.0 |
| `vendor/alper/alper_framework.py` | [HKBU-LAGAS/Alper](https://github.com/HKBU-LAGAS/Alper/tree/d4151ee5ce3ac7dabf377ffe2dc3877ad9b965fd) @ `d4151ee` | MIT © 2026 LAGAS, HKBU |
| `vendor/acc/` | [Linusaronsson/Active-Correlation-Clustering](https://github.com/Linusaronsson/Active-Correlation-Clustering/tree/313d989df570437d0b004e1e8a0f354a750fbe62) @ `313d989` | Apache-2.0 |

The files are unmodified; `prier/baselines/_vendor.py` loads them with small
stand-ins for driver-only imports. Each directory retains the original
`LICENSE` file.

## Models (downloaded separately, not redistributed)

* `sentence-transformers/all-MiniLM-L6-v2` — Apache-2.0.
* `Qwen/Qwen2.5-1.5B-Instruct` — Apache-2.0.

## Benchmark data (downloaded by `scripts/download.py`)

| Dataset | Source | License |
|---|---|---|
| Abt-Buy, DBLP-ACM, Amazon-Google, Affiliations, MusicBrainz | [Leipzig benchmark group](https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution) | CC-BY-4.0 |
| Restaurant | [cleanzr/restaurant](https://github.com/cleanzr/restaurant) | CC0 |
| Walmart–Amazon | [DeepMatcher data page](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md) | see source |

## Python dependencies

numpy, scipy, scikit-learn, pandas, pyyaml, rdata, torch, transformers,
sentence-transformers — see `requirements.txt` for versions.
