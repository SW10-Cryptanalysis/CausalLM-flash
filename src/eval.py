import argparse
import json
import logging
import time
from pathlib import Path

import Levenshtein
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from easy_logging import EasyFormatter
from transformers import LlamaForCausalLM

from classes import Config

handler = logging.StreamHandler()
handler.setFormatter(EasyFormatter())
logger = logging.getLogger("evaluate.py")
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class CipherEvaluator:
    """Orchestrates the evaluation of a Causal LM on cipher decoding tasks."""

    def __init__(self, model_path: str, use_spaces: bool) -> None:
        """Initialize state, sets up configuration, and loads required assets.

        Args:
            model_path (str): The path to the model checkpoint.
            use_spaces (bool): Whether to use spaces or not.

        """
        self.model_path = model_path
        self.config = Config()
        self.config.use_spaces = use_spaces
        self.config.load_homophones()

        self.output_log_path = Path(self.model_path) / "evaluation_results.jsonl"

        self.model = self._load_model()
        self.dataset = self._load_dataset()

    def _load_model(self) -> LlamaForCausalLM:
        """Instantiate the LLaMA model onto the appropriate device.

        Returns:
            LlamaForCausalLM: The loaded model.

        """
        logger.info(f"Loading model from {self.model_path}...")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = LlamaForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map="auto",
        )
        model.config.use_cache = True
        model.eval()
        return model

    def _load_dataset(self) -> Dataset | DatasetDict:
        """Retrieve the pre-tokenized dataset from the configured directory.

        Returns:
            Dataset | DatasetDict: The loaded dataset.

        """
        test_arrow_path = self.config.tokenized_dir / "Test"
        return load_from_disk(test_arrow_path)

    def decode_prediction(self, ids: list[int]) -> str:
        """Convert model token IDs back into a plaintext string based on config."""
        chars = []
        for idx in ids:
            if idx == self.config.space_token_id:
                chars.append("_" if self.config.use_spaces else " ")
            elif idx >= self.config.char_offset:
                chars.append(chr(idx - self.config.char_offset + ord("a")))
            elif idx == self.config.eos_token_id:
                break
        return "".join(chars)

    def decode_ciphertext(self, ids: list[int]) -> str:
        """Convert integer cipher IDs back to a space-separated string.

        Args:
            ids (list[int]): The list of token IDs to decode.

        Returns:
            str: The decoded string.

        """
        excluded = {self.config.bos_token_id, self.config.sep_token_id}
        return " ".join(str(idx) for idx in ids if idx not in excluded)

    def _evaluate_single_sample(self, item: dict, index: int) -> dict | None:
        """Extract targets, runs inference, and calculates metrics for one sample.

        Args:
            item (dict): The sample to evaluate.
            index (int): The index of the sample.

        Returns:
            dict | None: The evaluation result, or None if the sample failed.

        """
        all_ids = item["input_ids"]
        true_plain = item["raw_plaintext"]
        redundancy = int(item["redundancy"])

        try:
            sep_idx = all_ids.index(self.config.sep_token_id)
            input_ids = all_ids[: sep_idx + 1]
            raw_cipher_ids = all_ids[1:sep_idx]
        except ValueError:
            logger.warning(f"Sample {index} missing SEP token. Skipping.")
            return None

        input_tensor = torch.tensor([input_ids]).to(self.model.device)
        target_length = len(raw_cipher_ids)

        start_time = time.perf_counter()

        with torch.no_grad():
            output_ids = self.model.generate(
                input_tensor,
                attention_mask=torch.ones_like(input_tensor),
                max_new_tokens=target_length,
                min_new_tokens=target_length,
                do_sample=False,
                use_cache=True,
                pad_token_id=0,
                eos_token_id=self.config.eos_token_id,
            )

        generation_time = time.perf_counter() - start_time

        pred_ids = output_ids[0][len(input_ids) :].tolist()
        pred_plain = self.decode_prediction(pred_ids)

        dist = Levenshtein.distance(true_plain, pred_plain)
        ser = dist / len(true_plain) if len(true_plain) > 0 else 0.0

        return {
            "index": index,
            "redundancy": redundancy,
            "ciphertext": self.decode_ciphertext(raw_cipher_ids),
            "plaintext": true_plain,
            "predicted_plaintext": pred_plain,
            "ser": float(ser),
            "inference_time_seconds": round(generation_time, 4),
        }

    def run(self) -> None:
        """Execute the primary loop over all test samples and logs sequentially."""
        num_samples = len(self.dataset)
        logger.info(f"Starting full evaluation on {num_samples} samples...")

        total_ser = 0.0
        processed_count = 0

        for i in range(num_samples):
            result = self._evaluate_single_sample(
                self.dataset[i], # type: ignore
                i,
            )

            if result is None:
                continue

            total_ser += result["ser"]
            processed_count += 1

            with open(self.output_log_path, "a") as f:
                f.write(json.dumps(result) + "\n")

            if i % 50 == 0:
                msg = (
                    f"[{i + 1}/{num_samples}] SER: {result['ser']:.4f} | "
                    "Time: {result['inference_time_seconds']:.2f}s"
                )
                logger.info(msg)

        if processed_count > 0:
            logger.info(f"DONE. Avg SER: {total_ser / processed_count:.4f}")


def main() -> None:
    """Handle CLI arguments and acts as the entrypoint for execution."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--spaces", action="store_true")
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    evaluator = CipherEvaluator(model_path=args.model_path, use_spaces=args.spaces)
    evaluator.run()


if __name__ == "__main__":
    main()
