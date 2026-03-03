import torch
from pathlib import Path
from typing import TypedDict
from torch.utils.data import Dataset
from datasets import load_from_disk
from classes import Config

class CipherPlainDataItem(TypedDict):
    """TypedDict for CipherPlainDataItem."""

    input_ids: torch.Tensor
    labels: torch.Tensor


class CipherPlainData(Dataset):
	"""CipherPlainData dataset.

	This class is a subclass of torch.utils.data.Dataset and is used to load and
	iterate over the ciphertext-plaintext pairs in the Ciphers dataset.

	Attributes:
		config (Config): The config object containing the dataset parameters.
		sep_token (int): The token ID for the separator token.
		char_offset (int): The offset to add to the character IDs to avoid
			colliding with the cipher IDs.

	"""

	def __init__(self, config: Config, split: str = "Training") -> None:
		"""Initialize the CipherPlainData dataset.

		Args:
			config (Config): The config object containing the dataset parameters.
			split (str): The data split to load (e.g., 'Training', 'Test').

		"""
		self.config = config
		suffix = "_spaced" if config.use_spaces else "_normal"
		arrow_path = config.data_dir / f"{split}_arrow{suffix}"

		if not arrow_path.exists():
			raise FileNotFoundError(f"Missing Arrow Data: {arrow_path} - run preprocess.py first.")

		self.dataset = load_from_disk(str(arrow_path))

	def __len__(self) -> int:
		"""Get the length of the dataset.

		Returns:
			int: The length of the dataset.

		"""
		return len(self.dataset)

	def __getitem__(self, idx: int) -> CipherPlainDataItem:
		"""Get the item at the given index.

		Args:
			idx (int): The index of the item to get.

		Returns:
			tuple: A tuple containing the zip path and file name of the item.

		"""
		item = self.dataset[idx]

		# Ensure lists
		input_ids = list(item["input_ids"])
		labels = list(item["labels"])

		# Truncate to not exceed max content
		input_ids = input_ids[: self.config.max_context]
		labels = labels[: self.config.max_context]

		# Padding
		padding_len = self.config.max_context - len(input_ids)
		if padding_len > 0:
			input_ids += [0] * padding_len
			labels += [-100] * padding_len

		return {
			"input_ids": torch.tensor(input_ids, dtype=torch.long),
			"labels": torch.tensor(labels, dtype=torch.long),
		}