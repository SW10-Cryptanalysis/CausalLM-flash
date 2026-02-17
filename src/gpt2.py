import torch
from transformers import GPT2Config, AutoModelForCausalLM

config = GPT2Config(
    vocab_size=500,  # Your custom character/integer vocabulary size
    n_positions=8192,  # The maximum length of your cipher + plaintext
    n_embd=512,  # Hidden size (keep it small-ish to learn fast)
    n_layer=6,  # Number of transformer layers
    n_head=8,  # Number of attention heads
)

model = (
    AutoModelForCausalLM.from_config(
        config,
        attn_implementation="flash_attention_2",
    )
    .to(torch.bfloat16)
    .to("cuda")
)


def test():
    print(f"Parameters: {model.num_parameters():,}")


if __name__ == "__main__":
    test()
