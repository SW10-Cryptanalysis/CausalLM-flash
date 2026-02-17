import torch
from transformers import LlamaConfig, LlamaForCausalLM

CIPHERTEXT_LEN = 16_384

config = LlamaConfig(
    vocab_size=1024,  # We need enough for each homophone + 26 english letters + start/end/pad
    max_position_embeddings=CIPHERTEXT_LEN * 2,
    hidden_size=384,
    num_hidden_layers=16,
    intermediate_size=1536,
    num_attention_heads=6,
    num_key_value_heads=2,
    rope_theta=4_000_000.0,
    attn_implementation="sdpa",
    hidden_act="silu",
    initializer_rage=0.02,
    rms_norm_eps=1e-5,
)

model = LlamaForCausalLM(config).to(torch.bfloat16).to("cuda")


def test():
    print(f"Model Configured: Baby Llama 3 (Cipher Edition)")
    print(f"Parameters:       {model.num_parameters():,}")
    print(f"Vocab Capacity:   {config.vocab_size}")
    print(f"Context Window:   {config.max_position_embeddings}")

    # Quick sanity check on memory
    mem_params = model.get_memory_footprint() / 1e9
    print(f"VRAM for Weights: {mem_params:.4f} GB (Tiny!)")


if __name__ == "__main__":
    test()
