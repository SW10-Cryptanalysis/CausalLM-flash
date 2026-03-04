import pytest
from classes import CipherPlainData, Config
from datasets import Dataset as ArrowDataset


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
