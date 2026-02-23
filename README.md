# CausalLM

Causal LM with xFormers attention

## Usage
1. Queue up job with ``sbatch train.slurm``
2. Monitor with ``﻿tail -f logs/train_live_<JOB_ID>.log``

## Configuration

All parameters are listed in `src/config.py`.

Make sure you change `DATA_DIR` parameter in `src/config.py` to the correct path to your json training data


### Rope-theta
Remember to adjust this parameter when changing cipher lengths... A good rule of thumb is 1mil for ciphers of length 8192.
