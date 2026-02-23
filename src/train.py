import os
import torch
from model import get_model
from transformers import Trainer, TrainingArguments
from torch.nn.attention import sdpa_kernel, SDPBackend
import logging
from easy_logging import EasyFormatter

from classes import Config, CipherPlainData

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

handler = logging.StreamHandler()
handler.setFormatter(EasyFormatter())
logger = logging.getLogger("model.py")
logger.addHandler(handler)

def train() -> None:
	"""Start training the model with the given config."""
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

	logger.info(f"Training on {torch.cuda.get_device_name(0)}...")

	with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
		trainer.train()

	trainer.save_model(f"{config.output_dir}/model")


if __name__ == "__main__":
	train()
