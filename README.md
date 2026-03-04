# CausalLM

Causal LM with xFormers attention

## Usage
1. Preprocess data using `uv run src/preprocess.py` and `uv run src/preprocess.py --spaces` to generate data for spaces as well.
 2. Queue up job with ``sbatch train.slurm`` or ``sbatch train.slurm --spaces``to train with word boundaries as well.
3. Monitor with ``﻿tail -f logs/train_live_<JOB_ID>.log``

## Configuration

All parameters are listed in `src/config.py`.

Make sure you change `DATA_DIR` parameter in `src/config.py` to the correct path to your json training data


### Rope-theta
Remember to adjust this parameter when changing cipher lengths... A good rule of thumb is 1mil for ciphers of length 8192.

## Token visualisation

| PAD | Cipher start | Cipher end | SEP | SPACE | a..   | ..z    |
|-----|--------------|------------|-----|-------|-------|--------|
| 0   | 1..          | ..N        | N+1 | N+2   | N+3.. | ..N+29 |
