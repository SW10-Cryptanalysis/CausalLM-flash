import pytest
from dataclasses import dataclass
from typing import Any
import torch
from src.classes.pad_collator import PadCollator


@dataclass
class InitTestCase:
    """Defines parameters for testing PadCollator initialization."""

    pad_id: int
    max_ctx: int | None


class TestPadCollatorInit:
    """Tests for the initialization of PadCollator."""

    def test_default_init(self) -> None:
        """Ensure default values are set correctly."""
        collator = PadCollator()

        assert collator.pad_token_id == 0
        assert collator.max_context is None
        assert collator.ignore_index == -100

    @pytest.mark.parametrize(
        "test_case",
        [
            InitTestCase(pad_id=1, max_ctx=512),
            InitTestCase(pad_id=42, max_ctx=1024),
            InitTestCase(pad_id=-1, max_ctx=0),
        ],
    )
    def test_custom_init(self, test_case: InitTestCase) -> None:
        """Ensure custom values are applied during initialization."""
        collator = PadCollator(
            pad_token_id=test_case.pad_id,
            max_context=test_case.max_ctx,
        )

        assert collator.pad_token_id == test_case.pad_id
        assert collator.max_context == test_case.max_ctx


@dataclass
class TruncateTestCase:
    """Defines parameters for testing sequence truncation."""

    seq: list[int]
    max_ctx: int | None
    expected: list[int]


class TestPadCollatorTruncate:
    """Tests for the internal _truncate method."""

    @pytest.mark.parametrize(
        "test_case",
        [
            TruncateTestCase(
                seq=[1, 2, 3, 4, 5],
                max_ctx=None,
                expected=[1, 2, 3, 4, 5],
            ),
            TruncateTestCase(
                seq=[1, 2, 3, 4, 5],
                max_ctx=3,
                expected=[1, 2, 3],
            ),
            TruncateTestCase(
                seq=[1, 2],
                max_ctx=3,
                expected=[1, 2],
            ),
            TruncateTestCase(
                seq=[],
                max_ctx=3,
                expected=[],
            ),
            TruncateTestCase(
                seq=[1, 2, 3],
                max_ctx=0,
                expected=[],
            ),
        ],
    )
    def test_truncate(self, test_case: TruncateTestCase) -> None:
        """The sequence should be truncated strictly to max_context if provided."""
        collator = PadCollator(max_context=test_case.max_ctx)

        assert collator._truncate(test_case.seq) == test_case.expected


@dataclass
class CallTestCase:
    """Defines parameters for testing the __call__ method."""

    features: list[dict[str, Any]]
    pad_id: int
    max_ctx: int | None
    expected_input_ids: torch.Tensor
    expected_labels: torch.Tensor
    expected_mask: torch.Tensor


class TestPadCollatorCall:
    """Tests for the __call__ method to ensure proper batch processing."""

    def test_empty_features(self) -> None:
        """An empty input list should return empty tensors of shape (0, 0)."""
        collator = PadCollator()
        result = collator([])

        assert result["input_ids"].shape == (0, 0)
        assert result["labels"].shape == (0, 0)
        assert result["attention_mask"].shape == (0, 0)

    @pytest.mark.parametrize(
        "test_case",
        [
            CallTestCase(
                features=[{"input_ids": [1, 2, 3], "labels": [4, 5, 6]}],
                pad_id=0,
                max_ctx=None,
                expected_input_ids=torch.tensor([[1, 2, 3]]),
                expected_labels=torch.tensor([[4, 5, 6]]),
                expected_mask=torch.tensor([[1, 1, 1]]),
            ),
            CallTestCase(
                features=[
                    {"input_ids": [1, 2], "labels": [10, 20]},
                    {"input_ids": [1, 2, 3, 4], "labels": [10, 20, 30, 40]},
                ],
                pad_id=0,
                max_ctx=None,
                expected_input_ids=torch.tensor([[1, 2, 0, 0], [1, 2, 3, 4]]),
                expected_labels=torch.tensor([[10, 20, -100, -100], [10, 20, 30, 40]]),
                expected_mask=torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]]),
            ),
            CallTestCase(
                features=[
                    {"input_ids": [1], "labels": [10]},
                    {"input_ids": [1, 99, 3], "labels": [10, 20, 30]},
                ],
                pad_id=99,
                max_ctx=None,
                expected_input_ids=torch.tensor([[1, 99, 99], [1, 99, 3]]),
                expected_labels=torch.tensor([[10, -100, -100], [10, 20, 30]]),
                expected_mask=torch.tensor([[1, 0, 0], [1, 0, 1]]),
            ),
            CallTestCase(
                features=[
                    {"input_ids": [1], "labels": [10]},
                    {"input_ids": [1, 2, 3, 4, 5], "labels": [10, 20, 30, 40, 50]},
                ],
                pad_id=0,
                max_ctx=3,
                expected_input_ids=torch.tensor([[1, 0, 0], [1, 2, 3]]),
                expected_labels=torch.tensor([[10, -100, -100], [10, 20, 30]]),
                expected_mask=torch.tensor([[1, 0, 0], [1, 1, 1]]),
            ),
        ],
    )
    def test_call_processing(self, test_case: CallTestCase) -> None:
        """Batch processing applies padding, truncation, and masks concurrently."""
        collator = PadCollator(
            pad_token_id=test_case.pad_id,
            max_context=test_case.max_ctx,
        )
        result = collator(test_case.features)

        assert torch.equal(result["input_ids"], test_case.expected_input_ids)
        assert torch.equal(result["labels"], test_case.expected_labels)
        assert torch.equal(result["attention_mask"], test_case.expected_mask)
