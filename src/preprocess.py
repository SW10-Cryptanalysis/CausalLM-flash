import argparse
import json
import zipfile
import glob
from pathlib import Path
from datasets import Dataset as ArrowDataset
from classes import Config

class RawToArrowConverter:
    """Helper to crawl ZIPs and yield dictionary items for Arrow."""
    def __init__(self, config: Config, data_path: Path):
        self.config = config
        self.zip_files = glob.glob(str(data_path / "*.zip"))
        
        # Mapping logic mirrored from your requirements
        self.sep_token = config.unique_homophones + 1
        self.space_token = self.sep_token + 1
        self.char_offset = self.space_token + 1
        self.text_key = "plaintext_with_boundaries" if config.use_spaces else "plaintext"
        self.cipher_key = "ciphertext_with_boundaries" if config.use_spaces else "ciphertext"

    def __iter__(self):
        for zp in self.zip_files:
            with zipfile.ZipFile(zp, "r") as z:
                for file_name in [n for n in z.namelist() if n.endswith(".json")]:
                    with z.open(file_name) as f:
                        item = json.load(f)
                    yield self.map_item(item)

    def map_item(self, item):
        # Cipher mapping
        raw_cipher = item[self.cipher_key].split()
        cipher_ids = [self.space_token if x == "_" else int(x) for x in raw_cipher]
        
        # Plaintext mapping
        plain_ids = []
        for c in item[self.text_key]:
            if c == "_":
                plain_ids.append(self.space_token)
            elif "a" <= c <= "z":
                plain_ids.append(ord(c) - ord("a") + self.char_offset)

        input_ids = (cipher_ids + [self.sep_token] + plain_ids)[:self.config.max_context]
        labels = [x for x in input_ids]
        
        # We save raw lists to Arrow; padding happens dynamically in the Dataset
        return {"input_ids": input_ids, "labels": labels}

def run_conversion():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spaces", action="store_true")
    args = parser.parse_args()

    config = Config()
    config.use_spaces = args.spaces
    config.load_homophones()

    for split in ["Training", "Test"]:
        print(f"Converting {split}...")
        converter = RawToArrowConverter(config, config.data_dir / split)
        ds = ArrowDataset.from_generator(lambda: converter)
        
        suffix = "_spaced" if args.spaces else "_normal"
        save_path = config.data_dir / f"{split}_arrow{suffix}"
        ds.save_to_disk(save_path)

if __name__ == "__main__":
    run_conversion()