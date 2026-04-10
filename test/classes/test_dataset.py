import pytest
from pytest_mock import MockerFixture
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from src.classes.config import Config
from src.classes.dataset import CipherPlainData


class RawToArrowConverter:
    """Helper class to tokenize raw text pairs into Arrow dictionaries."""

    def __init__(self, config: Config) -> None:
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


@dataclass
class CipherDataTestCase:
    split: str
    path_exists: bool
    expected_exception: type[Exception] | None
    dataset_len: int
    mock_item: dict[str, Any] | None
    expected_item: dict[str, Any] | None


class TestCipherPlainData:
    """Tests for the PyTorch Dataset implementation."""

    @pytest.mark.parametrize(
        "test_case",
        [
            CipherDataTestCase(
                split="Training",
                path_exists=True,
                expected_exception=None,
                dataset_len=1,
                mock_item={
                    "input_ids": [10, 20, 501, 502, 503],
                    "labels": [10, 20, 501, 502, 503],
                    "redundancy": 10,
                },
                expected_item={
                    "input_ids": [10, 20, 501, 502, 503],
                    "labels": [10, 20, 501, 502, 503],
                },
            ),
            CipherDataTestCase(
                split="Test",
                path_exists=False,
                expected_exception=FileNotFoundError,
                dataset_len=0,
                mock_item=None,
                expected_item=None,
            ),
        ],
    )
    def test_dataset_lifecycle(
        self,
        test_case: CipherDataTestCase,
        tmp_path: Path,
        mocker: MockerFixture,
    ) -> None:
        cfg = Config(data_dir=tmp_path)
        tokenized_dir = cfg.tokenized_dir / test_case.split

        if test_case.path_exists:
            tokenized_dir.mkdir(parents=True, exist_ok=True)

        mock_hf_dataset = mocker.MagicMock()
        mock_hf_dataset.__len__.return_value = test_case.dataset_len

        if test_case.mock_item:
            mock_hf_dataset.__getitem__.return_value = test_case.mock_item

        mocker.patch(
            "src.classes.dataset.load_from_disk",
            return_value=mock_hf_dataset,
        )

        if test_case.expected_exception:
            with pytest.raises(
                test_case.expected_exception, match="Missing Arrow Data"
            ):
                CipherPlainData(cfg, split=test_case.split)
        else:
            ds = CipherPlainData(cfg, split=test_case.split)

            assert len(ds) == test_case.dataset_len

            item = ds[0]
            assert item == test_case.expected_item
            assert "redundancy" not in item


@dataclass
class ConverterTestCase:
    use_spaces: bool
    max_context: int
    example: dict[str, Any]
    expected_ids: list[int]


class TestRawToArrowConverter:
    """Tests for the tokenization and truncation logic."""

    @pytest.fixture
    def mock_cfg(self, mocker: MockerFixture) -> Any:
        cfg = mocker.Mock(spec=Config)
        cfg.bos_token_id = 1
        cfg.sep_token_id = 2
        cfg.eos_token_id = 3
        cfg.space_token_id = 4
        cfg.char_offset = 10
        return cfg

    @pytest.mark.parametrize(
        "test_case",
        [
            ConverterTestCase(
                use_spaces=False,
                max_context=100,
                example={"ciphertext": "5 6", "plaintext": "xy", "redundancy": 5},
                expected_ids=[1, 5, 6, 2, 33, 34, 3],
            ),
            ConverterTestCase(
                use_spaces=True,
                max_context=100,
                example={
                    "ciphertext_with_boundaries": "5 _ 6",
                    "plaintext_with_boundaries": "x_y",
                },
                expected_ids=[1, 5, 4, 6, 2, 33, 4, 34, 3],
            ),
            ConverterTestCase(
                use_spaces=False,
                max_context=7,
                example={
                    "ciphertext": "10 20 30 40",
                    "plaintext": "abcd",
                    "redundancy": 0,
                },
                expected_ids=[1, 10, 20, 2, 10, 11, 3],
            ),
        ],
    )
    def test_tokenize_fn(
        self,
        mock_cfg: Any,
        test_case: ConverterTestCase,
    ) -> None:
        mock_cfg.use_spaces = test_case.use_spaces
        mock_cfg.max_context = test_case.max_context

        converter = RawToArrowConverter(mock_cfg)
        result = converter.tokenize_fn(test_case.example)

        assert result["input_ids"] == test_case.expected_ids
        assert result["labels"] == test_case.expected_ids
