import torch
import argparse
import Levenshtein
import logging
import json
import time
from pathlib import Path
from datasets import load_from_disk
from transformers import LlamaForCausalLM
from easy_logging import EasyFormatter
from classes import Config

handler = logging.StreamHandler()
handler.setFormatter(EasyFormatter())
logger = logging.getLogger("evaluate.py")
logger.addHandler(handler)


def decode_prediction(ids: list[int], config: Config) -> str:
	"""Convert model token IDs back into a plaintext string."""
	chars = []
	for idx in ids:
		if idx == config.space_token_id:
			chars.append("_" if config.use_spaces else " ")
		elif idx >= config.char_offset:
			chars.append(chr(idx - config.char_offset + ord("a")))
		elif idx == config.eos_token_id:
			break
	return "".join(chars)


def decode_ciphertext(ids: list[int], config: Config) -> str:
	"""Convert integer cipher IDs back to a space-separated string."""
	excluded = {config.bos_token_id, config.sep_token_id}
	return " ".join(str(idx) for idx in ids if idx not in excluded)


def evaluate() -> None:
	"""Run evaluation on the test set and log results incrementally.

	Calculates SER and inference speed for each sample and saves to JSONL.
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument("--spaces", action="store_true")
	parser.add_argument("--model_path", type=str, required=True)
	cmd_args = parser.parse_args()

	config = Config()
	config.use_spaces = cmd_args.spaces
	config.load_homophones()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Load Model
	logger.info(f"Loading model from {cmd_args.model_path}...")
	model = LlamaForCausalLM.from_pretrained(
		cmd_args.model_path,
		torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
		device_map="auto",
	)
	model.config.use_cache = True
	model.eval()

	# Load Dataset
	test_arrow_path = config.tokenized_dir / "Test"
	test_ds = load_from_disk(str(test_arrow_path))
	num_samples = len(test_ds)

	output_log_path = Path(cmd_args.model_path) / "evaluation_results.jsonl"
	total_ser = 0.0
	processed_count = 0

	logger.info(f"Starting full evaluation on {num_samples} samples...")

	for i in range(num_samples):
		item = test_ds[i]
		all_ids = item["input_ids"]
		true_plain = item["raw_plaintext"]
		redundancy = item["redundancy"]

		try:
			sep_idx = all_ids.index(config.sep_token_id)
			# The prompt is [BOS] + cipher + [SEP]
			input_ids = all_ids[: sep_idx + 1]
			# Extract raw cipher IDs for logging (excluding BOS and SEP)
			raw_cipher_ids = all_ids[1:sep_idx]
		except ValueError:
			logger.warning(f"Sample {i} missing SEP token. Skipping.")
			continue

		input_tensor = torch.tensor([input_ids]).to(device)

		# length + EOS token
		target_length = len(raw_cipher_ids) + 1

		# --- TIMER START ---
		start_time = time.perf_counter()

		with torch.no_grad():
			output_ids = model.generate(
				input_tensor,
				max_new_tokens=target_length,
				min_new_tokens=target_length,
				do_sample=False,
				use_cache=True,
				pad_token_id=0,
				eos_token_id=config.eos_token_id,
			)

		generation_time = time.perf_counter() - start_time

		pred_ids = output_ids[0][len(input_ids) :].tolist()
		pred_plain = decode_prediction(pred_ids, config)

		dist = Levenshtein.distance(true_plain, pred_plain)
		ser = dist / len(true_plain) if len(true_plain) > 0 else 0
		total_ser += ser
		processed_count += 1

		# Incremental Save
		result_entry = {
			"index": i,
			"redundancy": int(redundancy),
			"ciphertext": decode_ciphertext(raw_cipher_ids, config),
			"plaintext": true_plain,
			"predicted_plaintext": pred_plain,
			"ser": float(ser),
			"inference_time_seconds": round(generation_time, 4),
		}

		with open(output_log_path, "a") as f:
			f.write(json.dumps(result_entry) + "\n")

		if i % 50 == 0:
			msg = (
				f"[{i + 1}/{num_samples}] SER: {ser:.4f} | Time: {generation_time:.2f}s"
			)
			logger.info(msg)

	if processed_count > 0:
		logger.info(f"DONE. Avg SER: {total_ser / processed_count:.4f}")


if __name__ == "__main__":
	evaluate()
