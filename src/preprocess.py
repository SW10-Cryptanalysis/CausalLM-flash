import argparse
import logging
from datasets import load_dataset
from classes import Config
from easy_logging import EasyFormatter
from typing import Any

handler = logging.StreamHandler()
handler.setFormatter(EasyFormatter())
logger = logging.getLogger("model.py")
logger.addHandler(handler)

class RawToArrowConverter:
	"""Encapsulates the tokenization logic for testing and execution."""

	def __init__(self, config: Config) -> None:
		"""Initialize the converter with model configuration.

		Args:
			config (Config): The configuration object containing token offsets.

		"""
		self.cfg = config

		# Key selection
		self.t_key = "plaintext_with_boundaries" if config.use_spaces else "plaintext"
		self.c_key = "ciphertext_with_boundaries" if config.use_spaces else "ciphertext"

	def tokenize_fn(self, example: dict[str, Any]) -> dict[str, Any]:
		"""Tokenize a single example from the dataset.

		Args:
			example (Dict[str, Any]): A raw dictionary of text and cipher strings.

		Returns:
			Dict[str, Any]: A dictionary containing 'input_ids' and 'labels'.

		"""
		# Cipher mapping (splitting and handling _)
		raw_cipher = example[self.c_key].split()
		cipher_ids = [self.cfg.space_token if x == "_" else int(x) for x in raw_cipher]

		# Plaintext mapping (char by char)
		plain_ids = []
		for char in example[self.t_key]:
			if char == "_":
				plain_ids.append(self.cfg.space_token)
			elif "a" <= char <= "z":
				plain_ids.append(ord(char) - ord("a") + self.cfg.char_offset)

		input_ids = (
			[self.cfg.bos_token_id]
			+ cipher_ids
			+ [self.cfg.sep_token]
			+ plain_ids
			+ [self.cfg.eos_token_id]
		)[: self.cfg.max_context]
		return {"input_ids": input_ids, "labels": input_ids}


def preprocess_data() -> None:
	"""Execute the entry point for preprocessing raw JSON data into Arrow format.

	Parses CLI arguments, loads the configuration, and iterates through
	data splits to save tokenized datasets to disk.
	"""
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
		logger.info("Converting %s (Spaces: %s)...", split, cfg.use_spaces)

		# load_dataset returns a DatasetDict if split isn't specified
		raw_ds = load_dataset(
			"json",
			data_files=f"{cfg.data_dir}/{split}/*.zip",
			split="train",
		)

		tokenized_ds = raw_ds.map(
			converter.tokenize_fn,
			num_proc=8,
			remove_columns=raw_ds.column_names,
		)

		save_path = cfg.tokenized_dir / split
		tokenized_ds.save_to_disk(str(save_path))
		logger.info("Saved to %s", save_path)


if __name__ == "__main__":
	preprocess_data()
