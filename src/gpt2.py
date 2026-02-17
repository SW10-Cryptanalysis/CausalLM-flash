import torch
from transformers import GPT2Config, AutoModelForCausalLM

# 1. Define the size of your custom model and context window
# Make sure max_position_embeddings is large enough for your long ciphers!
config = GPT2Config(
    vocab_size=500,  # Your custom character/integer vocabulary size
    n_positions=8192,  # The maximum length of your cipher + plaintext
    n_embd=512,  # Hidden size (keep it small-ish to learn fast)
    n_layer=6,  # Number of transformer layers
    n_head=8,  # Number of attention heads
)

# 2. Instantiate the model from scratch WITH Flash Attention 2
model = (
    AutoModelForCausalLM.from_config(
        config,
        attn_implementation="sdpa",  # This is the default in newer HF versions anyway
    ).to(torch.bfloat16)
    # .to("cuda")
)


def test():
    print(f"Parameters: {model.num_parameters():,}")


if __name__ == "__main__":
    test()
