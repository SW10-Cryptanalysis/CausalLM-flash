from classes import Config


class TestConfigPaths:
	def test_tokenized_dir(self, tmp_path):
		cfg = Config(data_dir=tmp_path, use_spaces=False)
		assert cfg.tokenized_dir == tmp_path / "tokenized_normal"

		cfg.use_spaces = True
		assert cfg.tokenized_dir == tmp_path / "tokenized_spaced"


class TestConfigVocab:
	def test_vocab_size(self):
		cfg = Config(unique_homophones=500, unique_letters=26)
		buffer = 8
		u_homs = 500
		u_lett = 26
		assert cfg.vocab_size == buffer + u_homs + u_lett

	def test_vocab_capacity(self, tmp_path):
		"""Ensure the vocabulary size can accommodate the character 'z'."""
		cfg = Config(data_dir=tmp_path)
		cfg.load_homophones()

		# Calculate highest possible ID: offset + 25 (for 'z')
		highest_id = cfg.char_offset + 25

		assert cfg.vocab_size > highest_id, \
			f"Vocab size {cfg.vocab_size} is too small for character 'z' at {highest_id}."
