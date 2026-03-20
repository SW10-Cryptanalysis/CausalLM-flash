import pytest
import torch
from datasets import Dataset as ArrowDataset
from classes import CipherPlainData, Config
from unittest.mock import MagicMock

class RawToArrowConverter:
	def __init__(self, config: Config) -> None:
		self.cfg = config
		self.t_key = "plaintext_with_boundaries" if config.use_spaces else "plaintext"
		self.c_key = "ciphertext_with_boundaries" if config.use_spaces else "ciphertext"

	def tokenize_fn(self, example: dict) -> dict:
		raw_cipher = example[self.c_key].split()
		cipher_ids = [
			self.cfg.space_token_id if x == "_" else int(x) for x in raw_cipher
		]

		plain_ids = []
		for char in example[self.t_key]:
			if char == "_":
				plain_ids.append(self.cfg.space_token_id)
			elif "a" <= char <= "z":
				plain_ids.append(ord(char) - ord("a") + self.cfg.char_offset)

		special_tokens_count = 3
		max_content_budget = self.cfg.max_context - special_tokens_count
		total_content_len = len(cipher_ids) + len(plain_ids)

		if total_content_len > max_content_budget:
			budget_per_side = max_content_budget // 2
			cipher_ids = cipher_ids[:budget_per_side]
			plain_ids = plain_ids[:(max_content_budget - len(cipher_ids))]

		input_ids = (
			[self.cfg.bos_token_id] + cipher_ids +
			[self.cfg.sep_token_id] + plain_ids +
			[self.cfg.eos_token_id]
		)
		return {
			"input_ids": input_ids,
			"labels": list(input_ids), # Joint distribution training
			"redundancy": example["redundancy"]
		}


@pytest.fixture
def mock_arrow_dir(tmp_path):
	"""Creates a fake Arrow dataset structure on disk."""
	tokenized_path = tmp_path / "tokenized_normal" / "Training"
	tokenized_path.mkdir(parents=True)

	dummy_data = {
		"input_ids": [[10, 20, 501, 502, 503]],
		"labels": [[10, 20, 501, 502, 503]],
		"redundancy": [10]
	}
	ArrowDataset.from_dict(dummy_data).save_to_disk(str(tokenized_path))
	return tmp_path


class TestCipherPlainData:
	def test_init_and_len(self, mock_arrow_dir):
		cfg = Config(data_dir=mock_arrow_dir)
		ds = CipherPlainData(cfg, split="Training")
		assert len(ds) == 1

	def test_getitem_padding_and_masking(self, mock_arrow_dir):
		cfg = Config(data_dir=mock_arrow_dir)
		cfg.max_context = 8
		ds = CipherPlainData(cfg, split="Training")

		item = ds[0]
		# input_ids: [10, 20, 501, 502, 503, 0, 0, 0]
		# labels: [10, 20, 501, 502, 503, -100, -100, -100]
		assert item["input_ids"].tolist() == [10, 20, 501, 502, 503, 0, 0, 0]
		assert item["labels"].tolist() == [10, 20, 501, 502, 503, -100, -100, -100]

	def test_joint_distribution_labels(self, mock_arrow_dir):
		"""Verify that labels match input_ids for joint loss."""
		cfg = Config(data_dir=mock_arrow_dir)
		ds = CipherPlainData(cfg, split="Training")
		item = ds[0]

		input_ids = item["input_ids"]
		labels = item["labels"]

		mask = input_ids != 0
		assert torch.equal(input_ids[mask], labels[mask])

	from unittest.mock import MagicMock

	def test_sequence_order(self):
		"""Verify the [BOS] -> [Cipher] -> [SEP] -> [Plain] -> [EOS] structure."""
		cfg = MagicMock()

		# Define the IDs as standard attributes on the mock
		cfg.bos_token_id = 1
		cfg.sep_token_id = 2
		cfg.eos_token_id = 3
		cfg.space_token_id = 4
		cfg.char_offset = 10
		cfg.max_context = 100
		cfg.use_spaces = False

		example = {
			"ciphertext": "5 6",
			"plaintext": "xy",
			"redundancy": 5
		}

		# Initialize the converter with the mock
		converter = RawToArrowConverter(cfg)
		result = converter.tokenize_fn(example)
		ids = result["input_ids"]

		# Expected: [BOS], [5], [6], [SEP], [x], [y], [EOS]
		# 'x' -> ord('x') - ord('a') + 10 = 23 + 10 = 33
		# 'y' -> ord('y') - ord('a') + 10 = 24 + 10 = 34
		expected_ids = [1, 5, 6, 2, 33, 34, 3]

		assert ids == expected_ids
		assert result["labels"] == ids
