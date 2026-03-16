import pytest
import torch
from classes import CipherPlainData, Config
from datasets import Dataset as ArrowDataset
from preprocess import RawToArrowConverter


@pytest.fixture
def mock_arrow_dir(tmp_path):
	"""Creates a fake Arrow dataset structure on disk."""
	tokenized_path = tmp_path / "tokenized_normal" / "Training"
	tokenized_path.mkdir(parents=True)

	# Simulate what preprocess.py would have baked
	dummy_data = {
		"input_ids": [[10, 20, 501, 502, 503]],  # Cipher, SEP, Plain
		"labels": [[10, 20, 501, 502, 503]],
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
		cfg.max_context = 8  # Dummy data is 5 tokens
		ds = CipherPlainData(cfg, split="Training")

		item = ds[0]

		# Verify padding on input_ids (0) and labels (-100)
		assert item["input_ids"].tolist() == [10, 20, 501, 502, 503, 0, 0, 0]
		assert item["labels"].tolist() == [10, 20, 501, 502, 503, -100, -100, -100]

	def test_getitem_truncation(self, mock_arrow_dir):
		cfg = Config(data_dir=mock_arrow_dir)
		cfg.max_context = 3
		ds = CipherPlainData(cfg, split="Training")

		item = ds[0]
		assert item["input_ids"].tolist() == [10, 20, 501]
		assert len(item["input_ids"]) == 3

	def test_joint_distribution_labels(self, mock_arrow_dir):
		"""Verify that labels are not masked, supporting the CausalLM joint loss."""
		cfg = Config(data_dir=mock_arrow_dir)
		ds = CipherPlainData(cfg, split="Training")

		# Simulate a small item
		item = ds[0]

		# In the CausalLM approach, input_ids and labels must match exactly
		# (except for the -100 padding at the end) .
		input_ids = item["input_ids"]
		labels = item["labels"]

		# Check that where there is data, labels == input_ids
		mask = input_ids != 0
		assert torch.equal(input_ids[mask], labels[mask]), \
			"Labels must match input_ids to optimize P(X) and P(Y|X)"


	def test_sequence_order(self, mock_arrow_dir):
		"""Verify the [BOS] -> [Cipher] -> [SEP] -> [Plain] -> [EOS] structure."""
		# Force unique_homophones to a small number (10) for predictable testing
		cfg = Config(data_dir=mock_arrow_dir, unique_homophones=10, use_spaces=False)

		# With unique_homophones=10:
		# SEP = 11, SPACE = 12, BOS = 13, EOS = 14, char_offset = 15

		example = {
			"ciphertext": "1 2",
			"plaintext": "ab",
			"ciphertext_with_boundaries": "1 _ 2",
			"plaintext_with_boundaries": "a _ b",
			"difficulty": 20,
		}
		converter = RawToArrowConverter(cfg)
		result = converter.tokenize_fn(example)
		ids = result["input_ids"]

		assert ids[0] == cfg.bos_token_id

		# Contains Cipher IDs (split from "1 2")
		assert ids[1] == 1
		assert ids[2] == 2

		# Contains the Task Switch / SEP
		assert ids[3] == cfg.sep_token_id

		# Contains Plaintext IDs (starting at char_offset 15)
		# 'a' -> 15, 'b' -> 16
		assert ids[4] == 15
		assert ids[5] == 16

		# Ends with EOS (ID 14)
		assert ids[-1] == cfg.eos_token_id

		# Verify the model is also penalized for cipher prediction as per CausalLM
		assert result["labels"] == result["input_ids"]
