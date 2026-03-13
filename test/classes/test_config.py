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
