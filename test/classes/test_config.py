import json
from classes import Config
from pathlib import Path


class TestConfigInit:
	def test_init_defaults(self):
		config = Config()

		assert isinstance(config.unique_letters, int)
		assert isinstance(config.vocab_size, int)
		assert isinstance(config.max_context, int)
		assert isinstance(config.dims, int)
		assert isinstance(config.layers, int)
		assert isinstance(config.att_heads, int)
		assert isinstance(config.kv_heads, int)
		assert isinstance(config.rope_theta, float)
		assert isinstance(config.batch_size, int)
		assert isinstance(config.grad_accum, int)
		assert isinstance(config.learning_rate, float)
		assert isinstance(config.epochs, int)
		assert isinstance(config.log_steps, int)
		assert isinstance(config.save_steps, int)
		assert isinstance(config.output_dir, Path)
		assert isinstance(config.data_dir, Path)


class TestConfigLoadHomophones:
	def test_load_homophones_success(self, tmp_path):
		data_dir = tmp_path / "data"
		data_dir.mkdir()
		meta_file = data_dir / "metadata.json"

		meta_file.write_text(json.dumps({"max_symbol_id": 999}))

		config = Config(data_dir=data_dir)
		config.load_homophones()

		assert config.unique_homophones == 999
		assert config.vocab_size == 999 + 26 + 8

	def test_load_homophones_fallback(self, tmp_path, caplog):
		data_dir = tmp_path / "data"
		data_dir.mkdir()
		meta_file = data_dir / "metadata.json"

		meta_file.write_text(json.dumps({"max_symb_id": 999}))

		config = Config(data_dir=data_dir)
		config.load_homophones()

		base_config = Config()

		assert config.unique_homophones == base_config.unique_homophones
		assert config.vocab_size == base_config.vocab_size
		assert "WARNING" in caplog.text
		assert "Invalid or missing data" in caplog.text
		assert "Using default value" in caplog.text

	def test_load_homophones_no_file(self, tmp_path, caplog):
		data_dir = tmp_path / "data"
		data_dir.mkdir()

		config = Config(data_dir=data_dir)
		config.load_homophones()

		base_config = Config()
		assert config.unique_homophones == base_config.unique_homophones
		assert config.vocab_size == base_config.vocab_size

	def test_load_homophones_invalid_file(self, tmp_path, caplog):
		data_dir = tmp_path / "data"
		data_dir.mkdir()
		meta_file = data_dir / "metadata.json"

		meta_file.write_text("invalid json")

		config = Config(data_dir=data_dir)
		config.load_homophones()

		base_config = Config()
		assert config.unique_homophones == base_config.unique_homophones
		assert config.vocab_size == base_config.vocab_size

		assert "WARNING" in caplog.text
		assert "Invalid or missing data" in caplog.text
		assert "Using default value" in caplog.text

	def test_load_homophones_error_reading_file(self, tmp_path, caplog, mocker):
		data_dir = tmp_path / "data"
		data_dir.mkdir()
		meta_file = data_dir / "metadata.json"

		meta_file.write_text(json.dumps({"max_symbol_id": 999}))

		mocker.patch("builtins.open", side_effect=OSError)

		config = Config(data_dir=data_dir)
		config.load_homophones()

		base_config = Config(data_dir=data_dir)
		assert config.unique_homophones == base_config.unique_homophones
		assert config.vocab_size == base_config.vocab_size

		assert "WARNING" in caplog.text
		assert "Could not read file" in caplog.text
		assert "Using default value" in caplog.text
