import pytest
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.classes.pad_collator import PadCollator


@pytest.fixture
def collator_cls() -> type["PadCollator"]:
    """Provides lazy-loaded PadCollator class."""
    from src.classes.pad_collator import PadCollator

    return PadCollator


@pytest.fixture
def torch_lib() -> Any:
    """Provides lazy-loaded torch module."""
    import torch

    return torch


class TestPadCollatorInit:
    """Tests for the initialization of PadCollator."""

    def test_default_init(self, collator_cls: type["PadCollator"]) -> None:
        """Ensure default values are set correctly."""
        collator = collator_cls()

        assert collator.pad_token_id == 0
        assert collator.max_context is None
        assert collator.ignore_index == -100

    @pytest.mark.parametrize(
        "pad_id, max_ctx",
        [
            (1, 512),
            (42, 1024),
            (-1, 0),
        ],
    )
    def test_custom_init(
        self, pad_id: int, max_ctx: int, collator_cls: type["PadCollator"]
    ) -> None:
        """Ensure custom values are applied during initialization."""
        collator = collator_cls(pad_token_id=pad_id, max_context=max_ctx)

        assert collator.pad_token_id == pad_id
        assert collator.max_context == max_ctx


class TestPadCollatorTruncate:
    """Tests for the internal _truncate method."""

    def test_no_truncation(self, collator_cls: type["PadCollator"]) -> None:
        """When max_context is None, the sequence should remain entirely unchanged."""
        collator = collator_cls(max_context=None)
        seq = [1, 2, 3, 4, 5]

        assert collator._truncate(seq) == [1, 2, 3, 4, 5]

    @pytest.mark.parametrize(
        "seq, max_ctx, expected",
        [
            ([1, 2, 3, 4, 5], 3, [1, 2, 3]),
            ([1, 2], 3, [1, 2]),
            ([], 3, []),
            ([1, 2, 3], 0, []),
        ],
    )
    def test_with_truncation(
        self,
        seq: list[int],
        max_ctx: int,
        expected: list[int],
        collator_cls: type["PadCollator"],
    ) -> None:
        """The sequence should be truncated strictly to max_context."""
        collator = collator_cls(max_context=max_ctx)

        assert collator._truncate(seq) == expected


class TestPadCollatorCall:
    """Tests for the __call__ method to ensure proper batch processing."""

    def test_empty_features(self, collator_cls: type["PadCollator"]) -> None:
        """An empty input list should return empty tensors of shape (0, 0)."""
        collator = collator_cls()
        result = collator([])

        assert result["input_ids"].shape == (0, 0)
        assert result["labels"].shape == (0, 0)
        assert result["attention_mask"].shape == (0, 0)

    def test_single_feature(
        self, collator_cls: type["PadCollator"], torch_lib: Any
    ) -> None:
        """A single item should be converted to tensors without any padding applied."""
        collator = collator_cls()
        features = [{"input_ids": [1, 2, 3], "labels": [4, 5, 6]}]
        result = collator(features)

        assert torch_lib.equal(result["input_ids"], torch_lib.tensor([[1, 2, 3]]))
        assert torch_lib.equal(result["labels"], torch_lib.tensor([[4, 5, 6]]))
        assert torch_lib.equal(result["attention_mask"], torch_lib.tensor([[1, 1, 1]]))

    def test_multiple_features_padding(
        self, collator_cls: type["PadCollator"], torch_lib: Any
    ) -> None:
        """Multiple items of varying lengths should be padded to the longest sequence."""
        collator = collator_cls(pad_token_id=0)
        features = [
            {"input_ids": [1, 2], "labels": [10, 20]},
            {"input_ids": [1, 2, 3, 4], "labels": [10, 20, 30, 40]},
        ]
        result = collator(features)

        expected_input_ids = torch_lib.tensor([[1, 2, 0, 0], [1, 2, 3, 4]])
        expected_labels = torch_lib.tensor([[10, 20, -100, -100], [10, 20, 30, 40]])
        expected_attention_mask = torch_lib.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])

        assert torch_lib.equal(result["input_ids"], expected_input_ids)
        assert torch_lib.equal(result["labels"], expected_labels)
        assert torch_lib.equal(result["attention_mask"], expected_attention_mask)

    def test_attention_mask_custom_pad_token(
        self, collator_cls: type["PadCollator"], torch_lib: Any
    ) -> None:
        """The attention mask must correctly identify and mask custom pad tokens within the sequence."""
        collator = collator_cls(pad_token_id=99)
        features = [
            {"input_ids": [1], "labels": [10]},
            {"input_ids": [1, 99, 3], "labels": [10, 20, 30]},
        ]
        result = collator(features)

        expected_attention_mask = torch_lib.tensor([[1, 0, 0], [1, 0, 1]])

        assert torch_lib.equal(result["attention_mask"], expected_attention_mask)

    def test_call_with_truncation(
        self, collator_cls: type["PadCollator"], torch_lib: Any
    ) -> None:
        """Both padding and truncation should apply concurrently and correctly in a batch."""
        collator = collator_cls(pad_token_id=0, max_context=3)
        features = [
            {"input_ids": [1], "labels": [10]},
            {"input_ids": [1, 2, 3, 4, 5], "labels": [10, 20, 30, 40, 50]},
        ]
        result = collator(features)

        expected_input_ids = torch_lib.tensor([[1, 0, 0], [1, 2, 3]])
        expected_labels = torch_lib.tensor([[10, -100, -100], [10, 20, 30]])

        assert result["input_ids"].shape == (2, 3)
        assert torch_lib.equal(result["input_ids"], expected_input_ids)
        assert torch_lib.equal(result["labels"], expected_labels)
