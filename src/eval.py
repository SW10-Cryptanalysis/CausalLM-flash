import torch
import argparse
import Levenshtein
import logging
from datasets import load_from_disk
from transformers import LlamaForCausalLM
from easy_logging import EasyFormatter
from classes import Config

handler = logging.StreamHandler()
handler.setFormatter(EasyFormatter())
logger = logging.getLogger("evaluate.py")
logger.addHandler(handler)


def evaluate() -> None:
	"""Evaluate the SER of a saved model with the test set."""
	parser = argparse.ArgumentParser()
	parser.add_argument("--spaces", action="store_true")
	parser.add_argument(
		"--model_path",
		type=str,
		required=True,
		help="Path to saved model folder",
	)
	cmd_args = parser.parse_args()

	config = Config()
	config.use_spaces = cmd_args.spaces
	config.load_homophones()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# 1. Load your Llama model
	logger.info(f"Loading model from {cmd_args.model_path}...")
	model = LlamaForCausalLM.from_pretrained(
		cmd_args.model_path,
		torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
		device_map="auto",
	)

	# CRITICAL: Enable cache for fast generation
	model.config.use_cache = True
	model.eval()

	# 2. Decoding Helper
	def decode_prediction(ids: list[str]) -> str:
		chars = []
		for idx in ids:
			if idx == config.space_token_id:
				chars.append("_" if config.use_spaces else " ")
			elif idx >= config.char_offset:
				chars.append(chr(idx - config.char_offset + ord("a")))
			elif idx == config.eos_token_id:
				break
		return "".join(chars)

	# 3. Load from the Arrow TEST shards
	test_arrow_path = config.tokenized_dir / "Test"
	logger.info(f"Loading Test data from Arrow shards at {test_arrow_path}...")
	test_ds = load_from_disk(str(test_arrow_path))

	# Pick a subset to evaluate (e.g., first 50 examples)
	num_samples = min(50, len(test_ds))
	total_ser = 0.0

	logger.info(f"Starting generation on {num_samples} samples...")

	for i in range(num_samples):
		item = test_ds[i]
		all_ids = item["input_ids"]

		# In your preprocess.py, input_ids = [BOS] + cipher + [SEP] + plain + [EOS]
		# We need to find where the [SEP] token is to stop the input there.
		try:
			sep_idx = all_ids.index(config.sep_token_id)
			input_ids = all_ids[: sep_idx + 1]  # Include the SEP token

			# The ground truth is everything after the SEP and before the EOS
			true_ids = all_ids[sep_idx + 1 :]
			true_plain = decode_prediction(true_ids)
		except ValueError:
			logger.warning(f"Sample {i} missing SEP token. Skipping.")
			continue

		input_tensor = torch.tensor([input_ids]).to(device)

		with torch.no_grad():
			output_ids = model.generate(
				input_tensor,
				max_new_tokens=128,
				do_sample=False,
				use_cache=True,
				pad_token_id=0,
				eos_token_id=config.eos_token_id,
			)

		# Get only the newly generated part
		generated_part = output_ids[0][len(input_ids) :]
		pred_plain = decode_prediction(generated_part.tolist())

		# Calculate SER
		min_len = min(len(true_plain), len(pred_plain))
		if min_len > 0:
			dist = Levenshtein.distance(true_plain[:min_len], pred_plain[:min_len])
			ser = dist / min_len
			total_ser += ser

			if i % 10 == 0:  # Log every 10th sample to keep logs clean
				logger.info(f"Sample {i} | SER: {ser:.4f}")
				logger.info(f"  True: {true_plain[:60]}")
				logger.info(f"  Pred: {pred_plain[:60]}")

	avg_ser = total_ser / num_samples
	logger.info("=" * 30)
	logger.info(f"FINAL AVERAGE SYMBOL ERROR RATE (SER): {avg_ser:.4f}")


if __name__ == "__main__":
	evaluate()
