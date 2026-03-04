import argparse
import os
from datasets import load_dataset, load_from_disk
from classes import Config

def preprocess_data():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spaces", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    cfg.use_spaces = args.spaces
    cfg.load_homophones()

    # TOKEN IDs
    sep_token = cfg.unique_homophones + 1
    space_token = sep_token + 1
    char_offset = space_token + 1

    # Spaces or no spaces
    t_key = "plaintext_with_boundaries" if cfg.use_spaces else "plaintext"
    c_key = "ciphertext_with_boundaries" if cfg.use_spaces else "ciphertext"

    def tokenize_fn(example):
        # Cipher mapping (splitting and handling _)
        raw_cipher = example[c_key].split()
        cipher_ids = [space_token if x == "_" else int(x) for x in raw_cipher]
        
        # Plaintext mapping (char by char)
        plain_ids = []
        for char in example[t_key]:
            if char == "_":
                plain_ids.append(space_token)
            elif "a" <= char <= "z":
                plain_ids.append(ord(char) - ord("a") + char_offset)
        
        input_ids = (cipher_ids + [sep_token] + plain_ids)[:cfg.max_context]
        return {"input_ids": input_ids, "labels": input_ids}

    # Load Raw JSONs
    for split in ["Training", "Test"]:
        print(f"Converting {split} (Spaces: {cfg.use_spaces})...")
        raw_ds = load_dataset("json", data_files=f"{cfg.data_dir}/{split}/*.zip", split="train")
        
        tokenized_ds = raw_ds.map(
            tokenize_fn,
            num_proc=8,
            remove_columns=raw_ds.column_names
        )
        
        save_path = cfg.tokenized_dir / split
        tokenized_ds.save_to_disk(str(save_path))
        print(f"Saved to {save_path}")

if __name__ == "__main__":
    preprocess_data()