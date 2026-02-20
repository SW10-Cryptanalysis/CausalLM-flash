import json
import os
import torch
from model import get_model
from transformers import Trainer, TrainingArguments
from torch.nn.attention import sdpa_kernel, SDPBackend

from torch.utils.data import Dataset
from config import Config

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class CipherPlainData(Dataset):
    def __init__(self, file_path):
        with open(file_path, "r") as f:
            self.data = json.load(f)

        self.sep_token = Config.unique_homophones + 1
        self.char_offset = self.sep_token + 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Convert ciphertext string to integers
        cipher_ids = [int(x) for x in item["ciphertext"].split()]

        # Make a-z map to 0-25 and add the char_offset so it does not collide with cipher_ids
        plain_ids = [(ord(c) - ord("a") + self.char_offset) for c in item["plaintext"]]

        # [Cipher] + [SEP] + [Plain]
        input_ids = cipher_ids + [self.sep_token] + plain_ids

        # Truncate to ensure it fits model context
        input_ids = input_ids[: Config.max_context]

        # Copy input ids
        labels = list(input_ids)

        # Padding
        padding_len = Config.max_context - len(input_ids)
        input_ids += [0] * padding_len
        labels += [-100] * padding_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def train():
    model = get_model()

    args = TrainingArguments(
        output_dir=Config.output_dir,
        num_train_epochs=Config.epochs,
        per_device_train_batch_size=Config.batch_size,
        gradient_accumulation_steps=Config.grad_accum,
        learning_rate=Config.learning_rate,
        # Faster to train without grad checkpoint
        gradient_checkpointing=False,
        logging_steps=Config.log_steps,
        save_steps=Config.save_steps,
        # OOM without below
        bf16=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=CipherPlainData(Config.data_dir),
    )

    print(f"Training on {torch.cuda.get_device_name(0)}...")

    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
        trainer.train()

    trainer.save_model(f"{Config.output_dir}/model")


if __name__ == "__main__":
    train()
