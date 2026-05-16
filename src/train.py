import os
import argparse
import logging
from pathlib import Path
import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments
from transformers.modeling_outputs import TokenClassifierOutput
from easy_logging import EasyFormatter

from src.model import get_model
from src.classes.config import Config
from src.classes.dataset import CipherPlainData
from src.classes.pad_collator import PadCollator

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


handler = logging.StreamHandler()
handler.setFormatter(EasyFormatter())
logger = logging.getLogger(__name__)
logger.addHandler(handler)


class AliceModelWrapper(nn.Module):
    """Wraps a causal transformer backbone into an ALICE homophonic encoder module."""

    def __init__(self, base_trunk: nn.Module, config: Config) -> None:
        super().__init__()
        self.base_trunk = base_trunk
        self.config = config

        # Expose config attribute directly to satisfy Hugging Face Trainer requirements
        self.config_hf = base_trunk.config

        # Classification head projecting pooled tokens directly to the plaintext vocabulary
        self.classifier = nn.Linear(base_trunk.config.hidden_size, config.vocab_size)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.base_trunk, "gradient_checkpointing_enable"):
            self.base_trunk.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.base_trunk, "gradient_checkpointing_disable"):
            self.base_trunk.gradient_checkpointing_disable()

    def save_pretrained(self, save_directory: str, **kwargs) -> None:
        """Saves the wrapper weights and the inner transformer configuration."""
        os.makedirs(save_directory, exist_ok=True)

        # Save the complete state dict (backbone weights + our linear classifier head)
        state_dict_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), state_dict_path)

        # Save the underlying transformer config so it can be reloaded later
        if hasattr(self.base_trunk, "config"):
            self.base_trunk.config.save_pretrained(save_directory)
        logger.info(
            "Model weights and configuration saved successfully to %s", save_directory
        )

    @property
    def config(self):
        # Tie both properties to prevent Hugging Face internals from throwing attribute errors
        return self.config_hf

    @config.setter
    def config(self, value):
        self.config_hf = value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        **kwargs,
    ) -> TokenClassifierOutput:
        # 1. Extract hidden states from the transformer trunk (processed bidirectionally via our collator mask)
        outputs = self.base_trunk(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )
        sequence_output = (
            outputs.last_hidden_state
        )  # Shape: (Batch, Seq_Len, Hidden_Dim)

        B, L, H = sequence_output.shape
        device = sequence_output.device
        dtype = sequence_output.dtype

        # 0 is reserved for padding; max possible token ID is config.unique_homophones
        num_cipher_tokens = self.config.unique_homophones + 1

        # 2. ALICE Symbol Pooling Layer
        pooled_states = torch.zeros(B, num_cipher_tokens, H, device=device, dtype=dtype)
        counts = torch.zeros(B, num_cipher_tokens, 1, device=device, dtype=dtype)

        # Accumulate hidden vectors into token buckets based on the recurrence IDs
        expanded_ids = input_ids.unsqueeze(-1).expand(-1, -1, H)
        pooled_states.scatter_add_(dim=1, index=expanded_ids, src=sequence_output)

        # Count occurrences of each token to calculate an accurate mean pool
        ones = torch.ones_like(input_ids).unsqueeze(-1).to(dtype=dtype)
        counts.scatter_add_(dim=1, index=input_ids.unsqueeze(-1), src=ones)

        # Clamp counts to 1 to avoid division-by-zero errors for tokens absent in a given sample
        counts = torch.clamp(counts, min=1.0)
        mean_pooled_symbols = (
            pooled_states / counts
        )  # Shape: (Batch, Num_Cipher_Tokens, Hidden_Dim)

        # 3. Global plain-character prediction per unique cipher symbol
        symbol_logits = self.classifier(
            mean_pooled_symbols
        )  # Shape: (Batch, Num_Cipher_Tokens, Alphabet_Size)

        # 4. Scatter predictions uniformly back to their original sequence coordinate indices
        expanded_logits_idx = input_ids.unsqueeze(-1).expand(
            -1, -1, symbol_logits.shape[-1]
        )
        logits = torch.gather(
            symbol_logits, dim=1, index=expanded_logits_idx
        )  # Shape: (Batch, Seq_Len, Alphabet_Size)

        # 5. Compute masked Cross-Entropy Loss during training and evaluation
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))

        return TokenClassifierOutput(loss=loss, logits=logits)


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

    raw_model = get_model(config)

    # Intercept and isolate the inner transformer block, bypassing the original CausalLM head
    base_trunk = getattr(
        raw_model, "model", getattr(raw_model, "transformer", raw_model)
    )

    # Wrap with our custom ALICE head
    model = AliceModelWrapper(base_trunk, config)

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
        dataloader_num_workers=2,
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
