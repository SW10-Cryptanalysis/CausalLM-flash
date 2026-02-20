from transformers import LlamaConfig, LlamaForCausalLM
from config import Config


def get_model():
    """Init model with params from config"""

    conf = LlamaConfig(
        vocab_size=Config.vocab_size,
        max_position_embeddings=Config.max_context,
        hidden_size=Config.dims,
        num_hidden_layers=Config.layers,
        intermediate_size=Config.dims * 4,
        num_attention_heads=Config.att_heads,
        num_key_value_heads=Config.kv_heads,
        rope_theta=Config.rope_theta,
        # Standard stuff
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        attn_implementation="sdpa",
        # Below saves memory during training, but dont know if it should be changed?
        use_cache=False,
    )

    model = LlamaForCausalLM(conf)
    print("Llama Model loaded!")
    print(f"Parameters:       {model.num_parameters():,}")
    print(f"VRAM for Weights: {(model.get_memory_footprint() / 1e9):.4f} GB")

    return model
