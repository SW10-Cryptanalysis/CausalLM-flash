from dataclasses import dataclass

TEXT_LEN = 8_192
UNIQUE_HOMOPHONE_COUNT = 500
UNIQUE_LETTER_COUNT = 26
TOTAL_SEQ = TEXT_LEN * 2
BUFFER = 8

OUTPUT_DIR = "./outputs"
DATA_DIR = "../../cipher.json"


@dataclass
class Config:
    # ARCHITECTURE

    # Vocab needs to be larger than unique homophone count + unique letter count + buffer (start/end/padding, etc)
    unique_homophones: int = UNIQUE_HOMOPHONE_COUNT
    unique_letters: int = UNIQUE_LETTER_COUNT
    vocab_size: int = UNIQUE_HOMOPHONE_COUNT + UNIQUE_LETTER_COUNT + BUFFER
    # Input is ciphertext + sep + plaintext
    max_context: int = TOTAL_SEQ + 1
    dims: int = 384
    layers: int = 16
    att_heads: int = 6
    kv_heads: int = 2
    rope_theta: float = 1_000_000.0

    # TRAINING
    batch_size: int = 1
    grad_accum: int = 16
    learning_rate: float = 3e-4
    epochs: int = 1
    log_steps: int = 10
    save_steps: int = 500

    # SYSTEM
    output_dir: str = OUTPUT_DIR
    data_dir: str = DATA_DIR
