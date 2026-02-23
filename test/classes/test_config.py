import json
import pytest
from classes import Config

class TestConfigInit:
	def test_init_defaults(self):
		config = Config()
		assert config.unique_homophones == 500
		assert config.unique_letters == 26
		assert config.vocab_size == 500 + 26 + 8
		assert config.max_context == 8192 * 2 + 1
		assert config.dims == 384
		assert config.layers == 16
		assert config.att_heads == 6
		assert config.kv_heads == 2
		assert config.rope_theta == 1_000_000.0
		assert config.batch_size == 1
		assert config.grad_accum == 16
		assert config.learning_rate == 3e-4
		assert config.epochs == 1
		assert config.log_steps == 10
		assert config.save_steps == 500
		assert config.output_dir == "./outputs"
		assert config.data_dir == "../../Ciphers/"
		
class TestConfigLoadHomophones:
	def test_load_homophones_success(self, tmp_path):
		data_dir = tmp_path / "data"
		data_dir.mkdir()
		meta_file = data_dir / "metadata.json"
		
		meta_file.write_text(json.dumps({"max_symbol_id": 999}))
		
		config = Config(data_dir=str(data_dir))
		config.load_homophones()

		assert config.unique_homophones == 999
		assert config.vocab_size == 999 + 26 + 8

	def test_load_homophones_fallback(self, tmp_path, caplog):
		data_dir = tmp_path / "data"
		data_dir.mkdir()
		meta_file = data_dir / "metadata.json"
  
		meta_file.write_text(json.dumps({"max_symb_id": 999}))
		
		config = Config(data_dir=str(data_dir))
		config.load_homophones()

		assert config.unique_homophones == 500
		assert config.vocab_size == 500 + 26 + 8
		assert "WARNING" in caplog.text
		assert "Invalid or missing data" in caplog.text
		assert "Using default value" in caplog.text
  
	def test_load_homophones_no_file(self, tmp_path, caplog):
		data_dir = tmp_path / "data"
		data_dir.mkdir()
  
		config = Config(data_dir=str(data_dir))
		config.load_homophones()

		assert config.unique_homophones == 500
		assert config.vocab_size == 500 + 26 + 8

	def test_load_homophones_invalid_file(self, tmp_path, caplog):
		data_dir = tmp_path / "data"
		data_dir.mkdir()
		meta_file = data_dir / "metadata.json"
  
		meta_file.write_text("invalid json")

		config = Config(data_dir=str(data_dir))
		config.load_homophones()

		assert config.unique_homophones == 500
		assert config.vocab_size == 500 + 26 + 8
		assert "WARNING" in caplog.text
		assert "Invalid or missing data" in caplog.text
		assert "Using default value" in caplog.text
  
	def test_load_homophones_error_reading_file(self, tmp_path, caplog, mocker):
		data_dir = tmp_path / "data"
		data_dir.mkdir()
		meta_file = data_dir / "metadata.json"
  
		meta_file.write_text(json.dumps({"max_symbol_id": 999}))
  
		mocker.patch("builtins.open", side_effect=OSError)

		config = Config(data_dir=str(data_dir))
		config.load_homophones()

		assert config.unique_homophones == 500
		assert config.vocab_size == 500 + 26 + 8
		assert "WARNING" in caplog.text
		assert "Could not read file" in caplog.text
		assert "Using default value" in caplog.text