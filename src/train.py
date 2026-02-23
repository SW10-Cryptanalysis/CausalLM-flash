import json
import os
import torch
import zipfile
import glob
from model import get_model
from transformers import Trainer, TrainingArguments
from torch.nn.attention import sdpa_kernel, SDPBackend

from torch.utils.data import Dataset
from config import Config

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class CipherPlainData(Dataset):
    def __init__(self, config):
        self.config = config
        self.file_refs = []

        zip_files = glob.glob(os.path.join(config.data_dir, "*.zip"))

        for zip_path in zip_files:
            with zipfile.ZipFile(zip_path, "r") as z:
                names = [n for n in z.namelist() if n.endswith(".json")]
                for file_name in names:
                    self.file_refs.append((zip_path, file_name))

        self.handles = {}

        self.sep_token = config.unique_homophones + 1
        self.char_offset = self.sep_token + 1

    def __len__(self):
        return len(self.file_refs)

    def __getitem__(self, idx):
        zip_path, file_name = self.file_refs[idx]

        if zip_path not in self.handles:
            self.handles[zip_path] = zipfile.ZipFile(zip_path, "r")

        with self.handles[zip_path].open(file_name) as f:
            item = json.load(f)

        # Convert ciphertext string to integers
        cipher_ids = [int(x) for x in item["ciphertext"].split()]

        # Make a-z map to 0-25 and add the char_offset so it does not collide with cipher_ids
        plain_ids = []
        for c in item["plaintext"]:
            if "a" <= c <= "z":
                token_id = ord(c) - ord("a") + self.char_offset
                plain_ids.append(token_id)
            else:
                raise ValueError(f"Unexpected char: {c} found in file {file_name}")

        # [Cipher] + [SEP] + [Plain]
        input_ids = cipher_ids + [self.sep_token] + plain_ids

        # Truncate to ensure it fits model context
        input_ids = input_ids[: self.config.max_context]

        # Copy input ids
        labels = list(input_ids)

        # Padding
        padding_len = self.config.max_context - len(input_ids)
        input_ids += [0] * padding_len
        labels += [-100] * padding_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def train():
    config = Config()
    config.load_homophones()

    model = get_model(config)

    args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum,
        learning_rate=config.learning_rate,
        # Faster to train without grad checkpoint
        gradient_checkpointing=False,
        logging_steps=config.log_steps,
        save_steps=config.save_steps,
        # OOM without below
        bf16=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=CipherPlainData(config),
    )

    print(f"Training on {torch.cuda.get_device_name(0)}...")

    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
        trainer.train()

    trainer.save_model(f"{config.output_dir}/model")


if __name__ == "__main__":
    train()
