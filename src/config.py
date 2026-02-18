from dataclasses import dataclass

TEXT_LEN = 8_192
TOTAL_SEQ = TEXT_LEN * 2
OUTPUT_DIR = "./outputs"


@dataclass
class Config:
    # ARCHITECTURE

    # Vocab needs to be larger than unique homophone count + unique letter count + 3 (start/end/padding)
    vocab_size: int = 1024
    # Input is ciphertext + plaintext
    max_context: int = TOTAL_SEQ
    dims: int = 384
    layers: int = 16
    att_heads: int = 6
    kv_heads: int = 2
    rope_theta: float = 1_000_000.0

    # TRAINING
    batch_size: int = 1
    grad_accum: int = 16
    learning_rate: float = 3e-4
    epochs: int = 5

    # This should be more memory efficient https://deepwiki.com/huggingface/transformers/3.2-training-configuration
    bf16: bool = True
    optim: str = "adamw_8bit"
    grad_checkpoint: bool = True
    log_steps: int = 10
    save_steps = 500

    # SYSTEM
    output_dir = OUTPUT_DIR
