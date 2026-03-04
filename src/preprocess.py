import argparse
import os
from datasets import load_dataset
from classes import Config

class RawToArrowConverter:
    """Encapsulates the tokenization logic for testing and execution."""
    def __init__(self, config: Config):
        self.cfg = config
        
        # TOKEN IDs
        self.sep_token = config.unique_homophones + 1
        self.space_token = self.sep_token + 1
        self.char_offset = self.space_token + 1

        # Key selection
        self.t_key = "plaintext_with_boundaries" if config.use_spaces else "plaintext"
        self.c_key = "ciphertext_with_boundaries" if config.use_spaces else "ciphertext"

    def tokenize_fn(self, example):
        # Cipher mapping (splitting and handling _)
        raw_cipher = example[self.c_key].split()
        cipher_ids = [self.space_token if x == "_" else int(x) for x in raw_cipher]
        
        # Plaintext mapping (char by char)
        plain_ids = []
        for char in example[self.t_key]:
            if char == "_":
                plain_ids.append(self.space_token)
            elif "a" <= char <= "z":
                plain_ids.append(ord(char) - ord("a") + self.char_offset)
        
        input_ids = (cipher_ids + [self.sep_token] + plain_ids)[:self.cfg.max_context]
        return {"input_ids": input_ids, "labels": input_ids}

def preprocess_data():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spaces", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    cfg.use_spaces = args.spaces
    cfg.load_homophones()

    # Initialize the converter
    converter = RawToArrowConverter(cfg)

    # Load Raw JSONs
    for split in ["Training", "Test"]:
        print(f"Converting {split} (Spaces: {cfg.use_spaces})...")
        
        # load_dataset returns a DatasetDict if split isn't specified
        raw_ds = load_dataset(
            "json", 
            data_files=f"{cfg.data_dir}/{split}/*.zip", 
            split="train"
        )
        
        tokenized_ds = raw_ds.map(
            converter.tokenize_fn,
            num_proc=8,
            remove_columns=raw_ds.column_names
        )
        
        save_path = cfg.tokenized_dir / split
        tokenized_ds.save_to_disk(str(save_path))
        print(f"Saved to {save_path}")

if __name__ == "__main__":
    preprocess_data()