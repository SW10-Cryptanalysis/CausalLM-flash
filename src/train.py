import os
import argparse
from src.model import get_model
from transformers import Trainer, TrainingArguments
import logging
from easy_logging import EasyFormatter
from pathlib import Path

from src.classes.config import Config
from src.classes.dataset import CipherPlainData
from src.classes.pad_collator import PadCollator

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


handler = logging.StreamHandler()
handler.setFormatter(EasyFormatter())
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _is_checkpoint(d: Path) -> bool:
    """Check output dir for checkpoints."""
    if not d.is_dir():
        return False
    return d.name.startswith("checkpoint-") and any(d.iterdir())


def contains_checkpoint(output_dir: Path) -> bool:
    """Check output dir for checkpoints."""
    if not output_dir.exists():
        return False

    for d in output_dir.iterdir():
        if _is_checkpoint(d):
            logger.info("Found valid checkpoint: %s. Resuming...", d.name)
            return True

    logger.info("No checkpoints found. Starting training from scratch.")
    return False


def train() -> None:
    """Start training the model with the given config."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--spaces", action="store_true")
    cmd_args = parser.parse_args()

    config = Config(use_spaces=cmd_args.spaces)
    config.load_homophones()

    # Safety check
    if not config.is_valid_init:
        raise ValueError(
            f"CRITICAL CONFIG ERROR: dimension was not initialized properly!\n"
            f"vocab_size: {config.vocab_size}\n"
            f"max_context: {config.max_context}\n"
            f"unique_homophones: {config.unique_homophones}\n"
            f"Check the Config class and load_homophones() method.",
        )

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
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_accumulation_steps=4,
        logging_steps=config.log_steps,
        save_steps=config.save_steps,
        # OOM without below
        fp16=False,
        bf16=True,
        tf32=True,
        dataloader_num_workers=4,
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
