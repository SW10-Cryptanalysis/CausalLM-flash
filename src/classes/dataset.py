import json
import glob
import os
import torch
import zipfile

from typing import TypedDict
from torch.utils.data import Dataset
from classes import Config

class CipherPlainDataItem(TypedDict):
    """TypedDict for CipherPlainDataItem."""

    input_ids: torch.Tensor
    labels: torch.Tensor

class CipherItem(TypedDict):
    """TypedDict for CipherItem."""

    ciphertext: str
    plaintext: str
    length: int
    num_symbols: int
    difficulty: int
    key: dict
    source_id: str
    source_name: str

class CipherPlainData(Dataset):
	"""CipherPlainData dataset.

	This class is a subclass of torch.utils.data.Dataset and is used to load and
	iterate over the ciphertext-plaintext pairs in the Ciphers dataset.

	Attributes:
		config (Config): The config object containing the dataset parameters.
		file_refs (list): A list of tuples containing the zip path and file name
			of each item in the dataset.
		handles (dict): A dictionary mapping zip paths to their corresponding
			zipfile.ZipFile objects.
		sep_token (int): The token ID for the separator token.
		char_offset (int): The offset to add to the character IDs to avoid
			colliding with the cipher IDs.

	"""

	def __init__(self, config: Config) -> None:
		"""Initialize the CipherPlainData dataset.

		Args:
			config (Config): The config object containing the dataset parameters.

		"""
		self.config = config
		self.file_refs = []

		zip_files = glob.glob(os.path.join(config.data_dir, "*.zip"))

		for zip_path in zip_files:
			with zipfile.ZipFile(zip_path, "r") as z:
				names = [n for n in z.namelist() if n.endswith(".json")]
				for file_name in names:
					self.file_refs.append((zip_path, file_name))

		self.handles = {}

		self.sep_token = config.unique_homophones + 1
		self.char_offset = self.sep_token + 1

	def __len__(self) -> int:
		"""Get the length of the dataset.

		Returns:
			int: The length of the dataset.

		"""
		return len(self.file_refs)

	def __getitem__(self, idx: int) -> CipherPlainDataItem:
		"""Get the item at the given index.

		Args:
			idx (int): The index of the item to get.

		Returns:
			tuple: A tuple containing the zip path and file name of the item.

		"""
		zip_path, file_name = self.file_refs[idx]

		if zip_path not in self.handles:
			self.handles[zip_path] = zipfile.ZipFile(zip_path, "r")

		with self.handles[zip_path].open(file_name) as f:
			item = json.load(f)

		self._validate_item(item)

		# Convert ciphertext string to integers
		cipher_ids = [int(x) for x in item["ciphertext"].split()]

		# Make a-z map to 0-25 and add the char_offset so it does
		# not collide with cipher_ids
		plain_ids = []
		for c in item["plaintext"]:
			if "a" <= c <= "z":
				token_id = ord(c) - ord("a") + self.char_offset
				plain_ids.append(token_id)
			else:
				raise ValueError(f"Unexpected char: {c} found in file {file_name}")

		# [Cipher] + [SEP] + [Plain]
		input_ids = cipher_ids + [self.sep_token] + plain_ids

		# Truncate to ensure it fits model context
		input_ids = input_ids[: self.config.max_context]

		# Copy input ids
		labels = list(input_ids)

		# Padding
		padding_len = self.config.max_context - len(input_ids)
		input_ids += [0] * padding_len
		labels += [-100] * padding_len

		return {
			"input_ids": torch.tensor(input_ids, dtype=torch.long),
			"labels": torch.tensor(labels, dtype=torch.long),
		}

	def _validate_item(self, item: dict) -> None:
		"""Validate that the item is a valid dictionary.

		Args:
			item (dict): The item to validate.

		Raises:
			ValueError: If the item is not a valid dictionary.

		"""
		for key in CipherItem.__annotations__:
			if key not in item:
				raise ValueError(f"Item is missing key: {key}")
