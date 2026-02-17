import torch
from transformers import LlamaConfig, LlamaForCausalLM

config = LlamaConfig(
    vocab_size=500,  # Your custom character/integer vocabulary
    max_position_embeddings=8192,  # Long context for your ciphers
    hidden_size=512,  # Small hidden size
    num_hidden_layers=6,  # Shallow depth
    num_attention_heads=8,
    intermediate_size=2048,
)

model = (
    LlamaForCausalLM._from_config(config, attn_implementation="flash_attention_2")
    .to(torch.bfloat16)
    .to("cuda")
)


def test():
    print(f"Parameters: {model.num_parameters():,}")


if __name__ == "__main__":
    test()
