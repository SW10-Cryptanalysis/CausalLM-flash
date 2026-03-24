from typing import TypedDict
from torch.utils.data import Dataset
from datasets import load_from_disk
from src.classes.config import Config


class CipherPlainDataItem(TypedDict):
    """TypedDict for CipherPlainDataItem."""

    input_ids: list[int]
    labels: list[int]


class CipherPlainData(Dataset):
    """CipherPlainData dataset.

    This class is a subclass of torch.utils.data.Dataset and is used to load and
    iterate over the ciphertext-plaintext pairs in the Ciphers dataset.

    Attributes:
        config (Config): The config object containing the dataset parameters.
        sep_token (int): The token ID for the separator token.
        char_offset (int): The offset to add to the character IDs to avoid
            colliding with the cipher IDs.

    """

    def __init__(self, config: Config, split: str = "Training") -> None:
        """Initialize the CipherPlainData dataset.

        Args:
            config (Config): The config object containing the dataset parameters.
            split (str): The data split to load (e.g., 'Training', 'Test').

        """
        self.config = config
        self.path = self.config.tokenized_dir / split

        if not self.path.exists():
            raise FileNotFoundError(
                f"Missing Arrow Data: {self.path} - run preprocess.py first.",
            )

        self.dataset = load_from_disk(str(self.path))

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            int: The length of the dataset.

        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> CipherPlainDataItem:
        """Get the raw, unpadded item at the given index."""
        item = self.dataset[idx]

        return {
            "input_ids": item["input_ids"],
            "labels": item["labels"],
        }
