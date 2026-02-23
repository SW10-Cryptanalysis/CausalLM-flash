import json
import os
from dataclasses import dataclass

TEXT_LEN = 8_192
UNIQUE_HOMOPHONE_COUNT = 500
UNIQUE_LETTER_COUNT = 26
TOTAL_SEQ = TEXT_LEN * 2
BUFFER = 8

OUTPUT_DIR = "./outputs"
DATA_DIR = "../../Ciphers/"
HOMOPHONE_FILE = "metadata.json"


@dataclass
class Config:
    # ARCHITECTURE

    # Default value is UNIQUE_HOMOPHONE_COUNT unless a HOMOPHONE_FILE exists
    unique_homophones: int = UNIQUE_HOMOPHONE_COUNT
    unique_letters: int = UNIQUE_LETTER_COUNT

    # Vocab needs to be larger than unique homophone count + unique letter count + buffer (start/end/padding, etc)
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

    def load_homophones(self):
        homophone_path = os.path.join(self.data_dir, HOMOPHONE_FILE)
        if os.path.exists(homophone_path):
            try:
                with open(homophone_path, "r") as f:
                    meta = json.load(f)
                    self.unique_homophones = int(
                        meta.get("max_symbol_id", UNIQUE_HOMOPHONE_COUNT)
                    )
            except (ValueError, IOError) as e:
                print(f"Warning - Could not read file: {HOMOPHONE_FILE}")
                print(f"Using default value: {self.unique_homophones}")
                print(f"{e}")

        self.vocab_size = self.unique_homophones + self.unique_letters + BUFFER
