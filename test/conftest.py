import sys
from unittest.mock import MagicMock

mock_transformers = MagicMock()
mock_transformers.__version__ = "5.2.0"


mock_torch = MagicMock()

class DummyDataset:
    """Lightweight base class to satisfy inheritance safely."""

mock_torch.utils.data.Dataset = DummyDataset

sys.modules["transformers"] = mock_transformers
sys.modules["datasets"] = MagicMock()
