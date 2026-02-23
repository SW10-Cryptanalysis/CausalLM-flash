from transformers import LlamaConfig, LlamaForCausalLM


def get_model(config):
    """Init model with params from config"""

    conf = LlamaConfig(
        vocab_size=config.vocab_size,
        max_position_embeddings=config.max_context,
        hidden_size=config.dims,
        num_hidden_layers=config.layers,
        intermediate_size=config.dims * 4,
        num_attention_heads=config.att_heads,
        num_key_value_heads=config.kv_heads,
        rope_theta=config.rope_theta,
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
