import pytest
from classes import CipherPlainData, Config
import os
import zipfile
from pathlib import Path
import json


@pytest.fixture
def real_data_config(tmp_path):
	"""
	Reads static JSON files from test/test_data, zips them into a temporary
	directory, and returns a Config object pointed at that directory.
	"""
	test_data_dir = Path(__file__).parent.parent / "ciphers"
	zip_path = tmp_path / "ciphers.zip"

	with zipfile.ZipFile(zip_path, "w") as z:
		for json_file in test_data_dir.glob("*.json"):
			z.write(json_file, arcname=json_file.name)

	return Config(data_dir=tmp_path)


@pytest.fixture
def cipher():
	"""Fixture for a cipher."""
	return {
		"plaintext": "test",
		"length": 4,
		"num_symbols": 4,
		"difficulty": 1,
		"key": {
			"e": [2],
			"s": [3],
			"t": [1],
		},
		"source_id": "12345",
		"source_name": "test",
		"ciphertext": "1 2 3 1",
	}


def write_cipher(data_dir, cipher):
	zip1_path = data_dir / "batch_01.zip"
	with zipfile.ZipFile(zip1_path, "w") as z:
		z.writestr("cipher_a.json", json.dumps(cipher))


@pytest.fixture
def cipher_illegal_char(tmp_path, cipher):
	"""Fixture for a cipher with an illegal character."""
	data_dir = tmp_path / "data"
	data_dir.mkdir()

	cipher["ciphertext"] = "1 2 3 1 4"
	cipher["key"]["æ"] = [4]
	cipher["plaintext"] = "testæ"

	write_cipher(data_dir, cipher)

	return Config(data_dir=data_dir)


@pytest.fixture
def cipher_invalid_json(tmp_path, cipher):
	"""Fixture for a cipher with invalid JSON (missing key)."""
	data_dir = tmp_path / "data"
	data_dir.mkdir()

	del cipher["plaintext"]

	write_cipher(data_dir, cipher)

	return Config(data_dir=data_dir)


class TestCipherPlainDataInit:
	def test_init(self):
		data = CipherPlainData(Config())
		assert data.file_refs == []
		assert data.handles == {}
		assert data.sep_token == data.config.unique_homophones + 1
		assert data.char_offset == data.sep_token + 1

	def test_init_with_file_refs(self, tmp_path):
		"""Test that the CipherPlainData class can be initialized with file refs."""
		data_dir = tmp_path / "data"
		data_dir.mkdir()

		zip1_path = data_dir / "batch_01.zip"
		with zipfile.ZipFile(zip1_path, "w") as z:
			z.writestr("cipher_a.json", '{"dummy": "data"}')
			z.writestr("cipher_b.json", '{"dummy": "data"}')
			z.writestr("ignore_me.txt", "This should not be indexed")

		zip2_path = data_dir / "batch_02.zip"
		with zipfile.ZipFile(zip2_path, "w") as z:
			z.writestr("nested_folder/cipher_c.json", '{"dummy": "data"}')

		(data_dir / "not_a_zip.tar.gz").write_text("fake archive")

		dataset = CipherPlainData(Config(data_dir=data_dir))

		assert len(dataset.file_refs) == 3
		assert len(dataset) == 3  # Test length simultaneously

		actual_refs = {(os.path.basename(zp), fname) for zp, fname in dataset.file_refs}

		expected_refs = {
			("batch_01.zip", "cipher_a.json"),
			("batch_01.zip", "cipher_b.json"),
			("batch_02.zip", "nested_folder/cipher_c.json"),
		}

		assert actual_refs == expected_refs

	def test_length_with_empty_zip(self, tmp_path):
		"""Test that the length of the CipherPlainData class is correct when a zip is empty."""
		data_dir = tmp_path / "data"
		data_dir.mkdir()

		zip1_path = data_dir / "batch_01.zip"
		with zipfile.ZipFile(zip1_path, "w") as z:
			z.writestr("not_a_cipher.txt", "Doesn't matter")

		dataset = CipherPlainData(Config(data_dir=data_dir))

		assert len(dataset) == 0


class TestCipherPlainDataGetItem:
	def test_getitem(self, real_data_config):
		"""Test that the CipherPlainData class can be indexed."""
		dataset = CipherPlainData(real_data_config)

		item = dataset[0]

		assert item["input_ids"].shape == (Config().max_context,)
		assert item["labels"].shape == (Config().max_context,)

	def test_getitem_unexpected_char(self, cipher_illegal_char):
		"""Test that the CipherPlainData class raises an error when an
		unexpected character is encountered."""
		dataset = CipherPlainData(cipher_illegal_char)

		with pytest.raises(ValueError) as e:
			dataset[0]
		assert "Unexpected char" in str(e.value)

	def test_getitem_invalid_json(self, cipher_invalid_json):
		"""Test that the CipherPlainData class raises an error when an
		invalid JSON file is encountered."""
		dataset = CipherPlainData(cipher_invalid_json)

		with pytest.raises(ValueError) as e:
			dataset[0]
		assert "Item is missing key: plaintext" in str(e.value)
