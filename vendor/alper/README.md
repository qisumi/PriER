# Alper

This repository contains the source code and data in the paper.

## Installation

To set up the environment and install the required dependencies, please follow these steps：

```bash
# Step 1: Create conda environment
conda create -n Alper -c conda-forge python=3.12

# Step 2: Activate the environment
conda activate Alper

# Step 3: Install dependencies
pip install -r requirements.txt
```

For detailed installation instructions, please refer to `INSTALLATION.txt`.

## Usage

### Running Experiments

Use the provided script to run experiments on multiple datasets:

```bash
cd src
bash run_table_experiments.sh
```

### Running Individual Experiments

Run the main script with specific parameters:

```bash
python main.py --dataset <dataset_name> --budget <budget> --model gpt-5-mini [options]
```

## Output

Results are saved in the `alper_results/` directory with the following format:

- `{dataset_name}_alper_llm_{timestamp}.json` - Main results
- `{dataset_name}_alper_{timestamp}_llm_logs.json` - Detailed LLM logs
