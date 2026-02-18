# CausalLM

Causal LM with xFormers attention

## Usage
1. Queue up job with ``sbatch train.slurm``
2. Monitor with ``﻿tail -f logs/train_live_<JOB_ID>.log``

## Configuration

All parameters are listed in `src/config.py`.

### Changing cipher length
If you change cipher length, just change the variable `TEXT_LEN`. Remember to keep lengths at powers of 2, eg. 8192, 16384, etc..

### Changing Homophones
In order to handle increasing amounts of homophones, you need to change `the vocab_size` to be at least unique homophone count + each plaintext letter + 3 (tokens for padding, start, end)

### Rope-theta
Remember to adjust this parameter when changing cipher lengths... A good rule of thumb is 1mil for ciphers of length 8192.
