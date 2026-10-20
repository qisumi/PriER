# PriER

**PriER: Budgeted Entity Resolution with a Small LLM Reviewer**

Code for the ICDE 2027 submission. When entity resolution has to run on
premises with a small local LLM (here Qwen2.5-1.5B-Instruct) and a strict
prompt-token budget, query-driven pipelines break down: the reviewer rejects
match-free candidate lists only 29% of the time, fixates on the second
candidate slot, and the budget cannot grow clusters from scratch. PriER
resolves most candidate pairs *before* querying, with a supervised identity
prior fitted on observable record attributes, spends tokens only on pairs the
prior cannot settle, folds each answer into the pair posterior as a
calibrated likelihood ratio, and decodes the partition under a cost-sensitive
correlation-clustering risk with a task-aware false-merge penalty.

Across six benchmarks and six budgets (36 settings), PriER reaches **84.40%
macro pair-F1** versus **72.77%** for the per-setting best of eight baselines
(+11.62 pp), leads in all 36 settings, and transfers with a frozen design to
unseen Walmart–Amazon (**+22.35 pp**). The per-setting result tables behind
these numbers are in [`results/`](results/).

## Repository layout

```
prier/                  method and experiment code
  data.py               benchmark normalization, entity-disjoint splits, labels
  inputs.py             MiniLM encoding, blocking, 6-feature anchor prior p^c
  features.py           57 identity features (+22 for institution names)
  prior.py              residual logistic identity prior
  channel.py            reviewer channel L(k | y) over tertile bins
  policy.py             budgeted U/t review loop
  decoder.py            cost-sensitive correlation-clustering decoder
  readout.py            task-aware false-merge cost selection
  reviewer.py           local Qwen reviewer (pair scoring + set answers)
  evaluate.py           pair-F1 / B^3 / cross-source metrics
  pipeline.py           build → fit → calibrate → run → evaluate
  baselines/            Risk, U/t, BoostER, Pivot, KwikBucks, Alper, ACC
vendor/                 unmodified algorithm files of Alper, ACC, KwikBucks
scripts/                runnable pipeline entry points
configs/default.yaml    datasets, budgets, seeds, hyper-parameters
tests/                  unit tests (synthetic data, fake reviewer)
results/                result tables of the paper
```

## Setup

Python 3.12, CUDA GPU recommended (encoding and the reviewer).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # add torch matching your CUDA build
pytest tests                             # unit tests, no models needed
```

Download the two local models into `models/` (not redistributed here):

* encoder: [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
* reviewer: [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)

## Running

```bash
# 1. raw benchmark archives (~17 MB total, public research datasets)
python scripts/download.py --dest data

# 2. normalize into records, labels and 50/20/30 entity-disjoint splits
python scripts/prepare.py --raw data --out data/prepared

# 3. encode every split with MiniLM, block, fit the shared anchor prior
python scripts/build_inputs.py

# 4. fit the identity prior (per fit seed) and the task-aware readout
python scripts/fit_models.py                 # add --ablations for leave-one-group-out

# 5. reviewer answers on the calibration split -> channel L(k | y)
python scripts/calibrate.py                 # answers are cached in work/<ds>/cache.jsonl

# 6. all methods x all budgets on the evaluation split -> work/metrics.csv
python scripts/run_experiment.py

# optional: causal slot-permutation probe for set-level position bias
python scripts/alper_order_probe.py --dataset abt_buy
```

All methods share the same blocks, anchor prior, reviewer and token budget;
`r` tokens per record means each block of `n` records may spend `r * n`
prompt tokens. Reviewer answers are cached, so repeated runs add no cost.
Walmart–Amazon is prepared *after* the six primary benchmarks
(`scripts/prepare.py --datasets walmart_amazon`) because its preparation
quarantines records that also occur in them.

## Results in this repository

| Directory | Content |
|---|---|
| `results/primary/` | main leaderboard: per-setting and per-domain metrics of all 9 methods |
| `results/ablation/` | leave-one-group-out feature ablations (50 refit configurations) |
| `results/mechanism/` | Proposition 1 band audit, channel calibration, Alper order permutation |
| `results/walmart_amazon/` | frozen transfer benchmark: all 165 configurations |

## Notes

* The likelihood-ratio update treats the reviewer answer as approximately
  conditionally independent of the prior features given the true relation;
  Proposition 1 bounds the resulting per-pair shift by the channel's largest
  log-ratio.
* Baseline adapters feed our reviewer and token budget to the *authors'*
  algorithm code (`vendor/`); only data loading and configuration are ours.

## Citation

The paper is currently under submission (ICDE 2027 research track) and not
yet published; a citation entry will be added here upon acceptance.

## License

MIT for our code. Third-party code in `vendor/` keeps its original licenses
(Apache-2.0, MIT); benchmark data keeps the sources' licenses (CC-BY-4.0,
CC0). See [`LICENSE`](LICENSE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
