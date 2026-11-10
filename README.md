# PriER

**PriER: Budgeted Entity Resolution with a Small LLM Reviewer**

PriER learns a supervised identity prior from record attributes, identifiers,
and text embeddings. The prior supplies an initial partition and guides a
small local reviewer (Qwen2.5-1.5B-Instruct) toward uncertain pairs under a
prompt-token budget. A calibrated likelihood ratio updates the reviewed pair,
and a cost-sensitive decoder weighs that update alongside the other links in
the block.

Across six benchmarks and six review budgets, PriER improves pair-F1 by an average of **11.62 percentage points** over the strongest baseline in each of the 36 settings. Its macro pair-F1 is **83.90% before review** and **84.43% at 32 prompt tokens per record**. On identical queries and answers, calibrated updates outperform hard decisions on Abt-Buy and Amazon–Google.

## Setup

Python 3.12, CUDA GPU recommended (encoding and the reviewer).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # add torch matching your CUDA build
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

All methods use the same candidate blocks and budget allowance. A block of
`n` records receives `r * n` prompt tokens. PriER starts from the supervised
identity prior. In the main comparison, Risk, U/t, and BoostER start from the
shared base prior.

### Serial and batched review

The default is **serial review** (`review_batch_size: 1` in
[`configs/default.yaml`](configs/default.yaml)). Each purchased pair is scored in
one forward pass. To run the nine methods on Abt-Buy at the paper's timing
budget after completing steps 1–5:

```bash
python scripts/run_experiment.py --datasets abt_buy --rates 32 --output work/abt_buy_serial.csv
```

Candidate prompt lengths are tokenized in chunks before review. This step
uses the tokenizer alone. The decoder maintains cluster sizes as it moves
records during local search.

For optional batched PriER inference:

```bash
python scripts/run_experiment.py --datasets abt_buy --methods prier --rates 32 --review-batch-size 16 --output work/abt_buy_batched.csv
```

This mode groups the next admitted pair from independent blocks by prompt
length, using left padding and a cap of `review_batch_max_tokens` (8,192 by
default). For Qwen2 pair scoring, the batched path projects the final hidden
state onto the No and Yes output weights. Each answer is billed to its own
block and updates that block's partition. Baseline methods retain their own
query procedures.

Runs use fresh responses by default. `--cache PATH` enables replay. Cache
hits consume logical tokens and add zero GPU time. Walmart–Amazon uses
local fitting and calibration after overlapping primary-benchmark records
have been filtered during preparation.

## Paper measurements

Table III and Fig. 6 use Abt-Buy, 621 records, 32 prompt tokens per record and three paired seeds. With one query per forward pass, PriER achieves **84.72% pair-F1** and uses **31.14 tokens per record** and **3.955 reviewer GPU seconds per 1,000 records**. The timing covers reviewer forward passes and excludes model loading, warm-up, input construction, fitting and partition decoding.

## Authors

Ziqi Zhao, Run Xu, Lu Chen, and Qingjian Ni.

## License

MIT for our code. Third-party code in `vendor/` keeps its original licenses
(Apache-2.0, MIT); benchmark data keeps the sources' licenses (CC-BY-4.0,
CC0). See [`LICENSE`](LICENSE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
