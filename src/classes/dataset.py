import torch
from typing import TypedDict
from torch.utils.data import Dataset
from datasets import load_from_disk
from classes import Config


class CipherPlainDataItem(TypedDict):
	"""TypedDict for CipherPlainDataItem."""

	input_ids: torch.Tensor
	labels: torch.Tensor
	attention_mask: torch.Tensor


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
		self.path = self.config.tokenized_dir / split

		if not self.path.exists():
			raise FileNotFoundError(
				f"Missing Arrow Data: {self.path} - run preprocess.py first.",
			)

		self.dataset = load_from_disk(str(self.path))

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

		# 1. Grab raw lists from Arrow
		input_ids = item["input_ids"]
		labels = item["labels"]

		# 2. Convert to tensors AND PAD immediately
		max_len = self.config.max_context

		# Convert to tensor and pad/truncate in one go
		input_tensor = torch.zeros(max_len, dtype=torch.long)
		# -100 is ignored by Loss
		label_tensor = torch.full((max_len,), -100, dtype=torch.long)

		# Fill with actual data (up to max_len)
		actual_len = min(len(input_ids), max_len)
		input_tensor[:actual_len] = torch.tensor(input_ids[:actual_len])
		label_tensor[:actual_len] = torch.tensor(labels[:actual_len])

		return {
			"input_ids": input_tensor,
			"attention_mask": (input_tensor != 0).long(),
			"labels": label_tensor,
		}
