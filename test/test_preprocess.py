import pytest
from preprocess import RawToArrowConverter
from classes import Config

@pytest.fixture
def sample_payload():
    """Provides a sample dictionary as it would appear when loaded from JSON."""
    return {
        "ciphertext": "1 2",
        "plaintext": "ab",
        "ciphertext_with_boundaries": "1 _ 2",
        "plaintext_with_boundaries": "a _ b"
    }

def test_mapping_logic_no_spaces(sample_payload):
    # Setup config with 10 homophones
    cfg = Config(unique_homophones=10, use_spaces=False)
    converter = RawToArrowConverter(cfg)
    
    # Process the sample
    result = converter.tokenize_fn(sample_payload)
    ids = result["input_ids"]
    
    # IDs should be: [1, 2] (Cipher) + [11] (SEP) + [13, 14] (Plain 'a', 'b')
    # Note: SEP = 10 + 1 = 11. SPACE = 12. char_offset = 13.
    assert ids == [1, 2, 11, 13, 14]
    assert 12 not in ids  # Space token should not be present

def test_mapping_logic_with_spaces(sample_payload):
    # Setup config with 10 homophones and spaces enabled
    cfg = Config(unique_homophones=10, use_spaces=True)
    converter = RawToArrowConverter(cfg)
    
    # Process the sample
    result = converter.tokenize_fn(sample_payload)
    ids = result["input_ids"]
    
    # IDs should be: [1, 12, 2] (Cipher) + [11] (SEP) + [13, 12, 14] (Plain)
    # 12 is the space_token
    assert ids == [1, 12, 2, 11, 13, 12, 14]
    assert ids.count(12) == 2 # One in cipher, one in plain