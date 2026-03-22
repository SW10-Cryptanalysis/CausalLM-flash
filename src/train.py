import os
import torch
import argparse
from model import get_model
from transformers import Trainer, TrainingArguments
import logging
from easy_logging import EasyFormatter
from pathlib import Path
from typing import Any

from classes import Config, CipherPlainData

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


handler = logging.StreamHandler()
handler.setFormatter(EasyFormatter())
logger = logging.getLogger("model.py")
logger.addHandler(handler)


class PadCollator:
	"""Dynamically pads ciphers to longest cipher in the batch."""

	def __init__(self, pad_token_id: int = 0) -> None:
		"""Initialize the collator with a padding token ID."""
		self.pad_token_id = pad_token_id


	def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
		"""Pad the batch to the maximum sequence length found in the features."""
		max_len = max(len(f["input_ids"]) for f in features)

		batch_input_ids = []
		batch_labels = []
		batch_att_mask = []

		for f in features:
			pad_length = max_len - len(f["input_ids"])

			# Pad inputs with pad token
			padded_inputs = f["input_ids"] + [self.pad_token_id] * pad_length

			# Pad with -100 so PyTorch ignores padded parts during loss calc
			padded_labels = f["labels"] + [-100] * pad_length

			# 1 means pay attention, 0 means ignore
			attention_mask = [1]*len(f["input_ids"]) + [0] * pad_length

			batch_input_ids.append(padded_inputs)
			batch_labels.append(padded_labels)
			batch_att_mask.append(attention_mask)

		return {
			"input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
			"labels": torch.tensor(batch_labels, dtype=torch.long),
			"attention_mask": torch.tensor(batch_att_mask, dtype=torch.long),
		}

def contains_checkpoint(output_dir: Path) -> bool:
	"""Check output dir for checkpoints."""
	if not output_dir.exists():
		return False

	for d in output_dir.iterdir():
		if d.is_dir() and d.name.startswith("checkpoint-") and any(d.iterdir()):
			logger.info("Found valid checkpoint: %s. Resuming...", d.name)
			return True

	logger.info("No checkpoints found. Starting training from scratch.")
	return False


def train() -> None:
	"""Start training the model with the given config."""
	parser = argparse.ArgumentParser()
	parser.add_argument("--spaces", action="store_true")
	cmd_args = parser.parse_args()

	config = Config()
	config.use_spaces = cmd_args.spaces
	config.load_homophones()

	# Path handling
	current_output_dir = config.final_output_dir
	current_output_dir.mkdir(parents=True, exist_ok=True)

	model = get_model(config)

	train_dataset = CipherPlainData(config, split="Training")
	eval_dataset = CipherPlainData(config, split="Validation")

	args = TrainingArguments(
		output_dir=str(current_output_dir),
		num_train_epochs=config.epochs,
		per_device_train_batch_size=config.batch_size,
		gradient_accumulation_steps=config.grad_accum,
		learning_rate=config.learning_rate,
		# Eval
		eval_strategy="steps",
		eval_steps=config.save_steps,
		per_device_eval_batch_size=config.batch_size,
		gradient_checkpointing=True,
		gradient_checkpointing_kwargs={"use_reentrant": False},
		eval_accumulation_steps=4,
		logging_steps=config.log_steps,
		save_steps=config.save_steps,
		# OOM without below
		fp16=False,
		bf16=True,
		tf32=True,
		dataloader_num_workers=8,
		dataloader_pin_memory=True,
		ddp_find_unused_parameters=False,
		# Checkpointing
		save_total_limit=2,
		load_best_model_at_end=True,
		metric_for_best_model="eval_loss",
		greater_is_better=False,
		ignore_data_skip=True,
		optim="adamw_torch_fused",
	)

	collator = PadCollator(pad_token_id=config.pad_token_id)

	trainer = Trainer(
		model=model,
		args=args,
		train_dataset=train_dataset,
		eval_dataset=eval_dataset,
		data_collator=collator,
	)

	checkpoint_exists = contains_checkpoint(current_output_dir)

	trainer.train(resume_from_checkpoint=checkpoint_exists)
	save_dest = f"{current_output_dir}/model"
	trainer.save_model(save_dest)


if __name__ == "__main__":
	train()
