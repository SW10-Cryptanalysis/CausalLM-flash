import os
import torch
import argparse
from model import get_model
from transformers import Trainer, TrainingArguments
import logging
from easy_logging import EasyFormatter

from classes import Config, CipherPlainData

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


handler = logging.StreamHandler()
handler.setFormatter(EasyFormatter())
logger = logging.getLogger("model.py")
logger.addHandler(handler)


def train() -> None:
	"""Start training the model with the given config."""
	parser = argparse.ArgumentParser()
	parser.add_argument("--spaces", action="store_true")
	cmd_args = parser.parse_args()

	config = Config()
	config.use_spaces = cmd_args.spaces
	config.load_homophones()

	model = get_model(config)

	train_dataset = CipherPlainData(config, split="Training")
	eval_dataset = CipherPlainData(config, split="Validation")

	args = TrainingArguments(
		output_dir=str(config.output_dir),
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

	trainer = Trainer(
		model=model,
		args=args,
		train_dataset=train_dataset,
		eval_dataset=eval_dataset,
	)

	checkpoint = None
	if os.path.isdir(config.output_dir) and any(config.output_dir.iterdir()):
		checkpoint = True
		logger.info(
			f"Checkpoint detected in {config.output_dir} - Resuming training...",
		)
	else:
		logger.info(f"Training on {torch.cuda.get_device_name(0)}...")

	trainer.train(resume_from_checkpoint=checkpoint)

	if config.use_spaces:
		save_dest = f"{config.output_dir}/model_with_spaces"
	else:
		save_dest = f"{config.output_dir}/model"

	trainer.save_model(save_dest)


if __name__ == "__main__":
	train()
