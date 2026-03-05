import pytest
from preprocess import RawToArrowConverter
from classes import Config

@pytest.fixture
def sample_payload():
	"""Provide a sample dictionary as it would appear when loaded from JSON."""
	return {
		"ciphertext": "1 2",
		"plaintext": "ab",
		"ciphertext_with_boundaries": "1 _ 2",
		"plaintext_with_boundaries": "a _ b"
	}

def test_mapping_logic_no_spaces(sample_payload):
	# 1. Setup Config
	cfg = Config(unique_homophones=10, use_spaces=False)

	# 2. Manually set tokens (if your Config doesn't do this in __init__)
	cfg.bos_token_id = 100
	cfg.eos_token_id = 101
	cfg.sep_token = 11
	cfg.space_token = 12
	cfg.char_offset = 13
	cfg.max_context = 100

	converter = RawToArrowConverter(cfg)
	result = converter.tokenize_fn(sample_payload)
	ids = result["input_ids"]

	# Expected: [BOS] + [1, 2] + [SEP] + [13, 14] + [EOS]
	assert ids == [100, 1, 2, 11, 13, 14, 101]

def test_mapping_logic_with_spaces(sample_payload):
	cfg = Config(unique_homophones=10, use_spaces=True)
	cfg.bos_token_id = 100
	cfg.eos_token_id = 101
	cfg.sep_token = 11
	cfg.space_token = 12
	cfg.char_offset = 13
	cfg.max_context = 100

	converter = RawToArrowConverter(cfg)
	result = converter.tokenize_fn(sample_payload)
	ids = result["input_ids"]

	# Expected: [BOS] + [1, 12, 2] + [SEP] + [13, 12, 14] + [EOS]
	assert ids == [100, 1, 12, 2, 11, 13, 12, 14, 101]
