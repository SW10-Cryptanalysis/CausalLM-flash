import torch
from transformers import LlamaConfig, LlamaForCausalLM
import logging
from easy_logging import EasyFormatter
from src.classes.config import Config

handler = logging.StreamHandler()
handler.setFormatter(EasyFormatter())
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def get_model(config: Config) -> LlamaForCausalLM:
    """Init model with params from config.

    Args:
            config (Config): The config object containing the model parameters.

    Returns:
            LlamaForCausalLM: The initialized model.

    """
    conf = LlamaConfig(
        vocab_size=config.vocab_size,
        max_position_embeddings=config.max_context,
        hidden_size=config.dims,
        num_hidden_layers=config.layers,
        intermediate_size=config.dims * 4,
        num_attention_heads=config.att_heads,
        num_key_value_heads=config.kv_heads,
        rope_theta=config.rope_theta,
        torch_dtype=torch.bfloat16,
        pad_token_id=config.pad_token_id,
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
        # Standard stuff
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-5,  # type: ignore
        attn_implementation="flash_attention_2",
        # Below saves memory during training, but dont know if it should be changed?
        use_cache=False,
    )

    model = LlamaForCausalLM(conf)
    logger.info("Llama Model loaded!")
    logger.info(f"Parameters:       {model.num_parameters():,}")
    logger.info(f"VRAM for Weights: {(model.get_memory_footprint() / 1e9):.4f} GB")

    return model
