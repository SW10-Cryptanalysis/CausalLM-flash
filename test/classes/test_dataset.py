import pytest
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.classes.dataset import CipherPlainData
    from src.classes.config import Config


@pytest.fixture
def config_cls() -> type["Config"]:
    """Provides lazy-loaded Config class."""
    from src.classes.config import Config

    return Config


@pytest.fixture
def cipher_plain_data_cls() -> type["CipherPlainData"]:
    """Provides lazy-loaded CipherPlainData class."""
    from src.classes.dataset import CipherPlainData

    return CipherPlainData


class RawToArrowConverter:
    """Helper class to tokenize raw text pairs into Arrow dictionaries."""

    def __init__(self, config: "Config") -> None:
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
            plain_ids = plain_ids[: (max_content_budget - len(cipher_ids))]

        input_ids = (
            [self.cfg.bos_token_id]
            + cipher_ids
            + [self.cfg.sep_token_id]
            + plain_ids
            + [self.cfg.eos_token_id]
        )
        return {
            "input_ids": input_ids,
            "labels": list(input_ids),
            "redundancy": example.get("redundancy", 0),
        }


@pytest.fixture
def mock_arrow_dir(tmp_path: Path) -> Path:
    """Creates a fake Arrow dataset structure on disk for testing."""
    from datasets import Dataset as ArrowDataset

    tokenized_path = tmp_path / "tokenized_normal" / "Training"
    tokenized_path.mkdir(parents=True)

    dummy_data = {
        "input_ids": [[10, 20, 501, 502, 503]],
        "labels": [[10, 20, 501, 502, 503]],
        "redundancy": [10],
    }
    ArrowDataset.from_dict(dummy_data).save_to_disk(str(tokenized_path))
    return tmp_path


class TestCipherPlainData:
    """Tests for the PyTorch Dataset implementation."""

    def test_init_and_len(
        self,
        mock_arrow_dir: Path,
        config_cls: type["Config"],
        cipher_plain_data_cls: type["CipherPlainData"],
    ) -> None:
        """Ensure the dataset initializes and accurately reports its size."""
        cfg = config_cls(data_dir=mock_arrow_dir)
        ds = cipher_plain_data_cls(cfg, split="Training")

        assert len(ds) == 1

    def test_getitem_structure(
        self,
        mock_arrow_dir: Path,
        config_cls: type["Config"],
        cipher_plain_data_cls: type["CipherPlainData"],
    ) -> None:
        """Verify __getitem__ drops unused keys (like redundancy) and retains labels."""
        cfg = config_cls(data_dir=mock_arrow_dir)
        ds = cipher_plain_data_cls(cfg, split="Training")

        item = ds[0]

        assert item["input_ids"] == [10, 20, 501, 502, 503]
        assert item["labels"] == [10, 20, 501, 502, 503]
        assert "redundancy" not in item

    def test_missing_directory_raises_error(
        self,
        tmp_path: Path,
        config_cls: type["Config"],
        cipher_plain_data_cls: type["CipherPlainData"],
    ) -> None:
        """Verify that missing preprocessed data crashes early and cleanly."""
        cfg = config_cls(data_dir=tmp_path)

        with pytest.raises(FileNotFoundError, match="Missing Arrow Data"):
            cipher_plain_data_cls(cfg, split="Training")


class TestRawToArrowConverter:
    """Tests for the tokenization and truncation logic."""

    @pytest.fixture
    def mock_cfg(self, mocker: Any, config_cls: type["Config"]) -> Any:
        """Provides a standard mocked Config to control exact ID generation."""
        cfg = mocker.Mock(spec=config_cls)
        cfg.bos_token_id = 1
        cfg.sep_token_id = 2
        cfg.eos_token_id = 3
        cfg.space_token_id = 4
        cfg.char_offset = 10
        cfg.max_context = 100
        cfg.use_spaces = False
        return cfg

    def test_sequence_order(self, mock_cfg: Any) -> None:
        """Verify the arithmetic and exact ordering of sequence parts."""
        example = {"ciphertext": "5 6", "plaintext": "xy", "redundancy": 5}
        converter = RawToArrowConverter(mock_cfg)
        result = converter.tokenize_fn(example)

        """
        Expected calculation based on the mock:
        [BOS] -> 1
        [Cipher] -> 5, 6
        [SEP] -> 2
        [Plain] -> 'x' (23 + 10 = 33), 'y' (24 + 10 = 34)
        [EOS] -> 3
        """
        expected_ids = [1, 5, 6, 2, 33, 34, 3]

        assert result["input_ids"] == expected_ids
        assert result["labels"] == expected_ids

    @pytest.mark.parametrize(
        "use_spaces, example, expected_ids",
        [
            (
                True,
                {
                    "ciphertext_with_boundaries": "5 _ 6",
                    "plaintext_with_boundaries": "x_y",
                },
                [1, 5, 4, 6, 2, 33, 4, 34, 3],
            ),
        ],
    )
    def test_spaces_handling(
        self,
        mock_cfg: Any,
        use_spaces: bool,
        example: dict,
        expected_ids: list[int],
    ) -> None:
        """Verify that underscores are correctly translated into the configured space token ID."""
        mock_cfg.use_spaces = use_spaces
        converter = RawToArrowConverter(mock_cfg)
        result = converter.tokenize_fn(example)

        assert result["input_ids"] == expected_ids

    def test_truncation_logic(self, mock_cfg: Any) -> None:
        """Verify sequences exceeding max_context are correctly bounded."""
        mock_cfg.max_context = 7
        example = {"ciphertext": "10 20 30 40", "plaintext": "abcd", "redundancy": 0}

        converter = RawToArrowConverter(mock_cfg)
        result = converter.tokenize_fn(example)

        """
        Max context is 7. Special tokens budget is 3 (BOS, SEP, EOS).
        Content budget is 4. Budget per side is 2.
        Cipher gets truncated to [10, 20]. Plain gets truncated to ['a', 'b'].
        """
        expected_ids = [1, 10, 20, 2, 10, 11, 3]

        assert result["input_ids"] == expected_ids
