# CausalLM-Flash

This project uses a Llama-based causal language model to decipher homophonic substitution ciphers of extreme lengths. The model uses flash attention to efficiently process long sequences.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Configuration](#configuration)
- [Token Visualisation](#token-visualisation)
- [Development](#development)

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/SW10/ciphers
    cd Llama-xFormers
    ```

2.  **Install dependencies:**

    This project uses `uv` for package management. If you haven't already, install `uv` [here](https://docs.astral.sh/uv/getting-started/installation/).
    
    To install all dependencies, run:

    ```bash
    uv sync
    ```

## Usage

### Training

1.  **Prepare Data:**

    Before training, ensure your JSON data is preprocessed. Also ensure that the `DATA_DIR` in `src/classes/config.py` is set to the path to your preprocessed data.

2.  **Start Training:**

    Training is initiated using SLURM. You can start a training job with:

    ```bash
    sbatch train.slurm
    ```

    To train with word boundaries (spaces), use:

    ```bash
    sbatch train.slurm --spaces
    ```

3.  **Monitor Training:**

    You can monitor the training process by tailing the log file:

    ```bash
    tail -f logs/train_live_<JOB_ID>.log
    ```

### Evaluation

To evaluate a trained model, use the `src/eval.py` script. You need to provide the path to the model and specify whether to use spaces.

```bash
python src/eval.py --model_path <path_to_your_model> [--spaces]
```

The evaluation script will output a JSONL file named `evaluation_results.jsonl` in the model directory, containing detailed results for each sample, including the Symbol Error Rate (SER).

## Configuration

All parameters for the model, training, and data are managed in `src/classes/config.py`.

## Token Visualisation

The following table illustrates the token representation used in this project:

| PAD | Cipher start | Cipher end | SEP | SPACE | BOS | EOS | a...  | ...z   |
| :-- | :----------- | :--------- | :-- | :---- | :-- | :-- | :---- | :----- |
| 0   | 1...         | ...N       | N+1 | N+2   | N+3 | N+4 | N+5.. | ..N+30 |

## Development

This project uses `ruff` for linting and formatting. The following GitHub Actions workflows are configured:

-   `lint.yml`: Lints the codebase.
-   `test.yml`: Runs tests.

To run the tests locally, use `pytest`:

```bash
uv run pytest
```

To run the linter, use `ruff`:

```bash
uv run ruff check .
```
Or in a minimal environment:

```bash
uvx ruff check .
```
