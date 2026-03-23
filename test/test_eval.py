import pytest
from pathlib import Path
from typing import Any


@pytest.fixture
def eval_module() -> Any:
    """Lazy-loads the evaluate module."""
    import eval

    return eval


@pytest.fixture
def torch_lib() -> Any:
    """Lazy-loads torch for safe local CPU execution."""
    import torch

    return torch


@pytest.fixture
def mock_config(mocker: Any) -> Any:
    """Mock config that mimics the actual Config class structure."""
    from classes import Config

    cfg = mocker.Mock(spec=Config)
    cfg.space_token_id = 4
    cfg.char_offset = 10
    cfg.eos_token_id = 3
    cfg.bos_token_id = 1
    cfg.sep_token_id = 2
    cfg.tokenized_dir = Path("/tmp/data") # noqa: S108
    cfg.use_spaces = False
    return cfg


class TestEvaluatorInitialization:
    """Tests for evaluator setup and asset loading."""

    def test_init_execution(self, mocker: Any, eval_module: Any) -> None:
        """Ensure config and properties are properly initialized."""
        mocker.patch.object(eval_module.CipherEvaluator, "_load_model")
        mocker.patch.object(eval_module.CipherEvaluator, "_load_dataset")
        mock_config_cls = mocker.patch("eval.Config")

        evaluator = eval_module.CipherEvaluator(
            model_path="/fake/path", use_spaces=True
        )

        assert evaluator.model_path == "/fake/path"
        mock_config_cls.return_value.load_homophones.assert_called_once()
        assert evaluator.config.use_spaces is True
        assert evaluator.output_log_path == Path("/fake/path/evaluation_results.jsonl")

    @pytest.mark.parametrize("test_cfg", [(True, "bfloat16"), (False, "float16")])
    def test_load_model_dtype_selection(
        self,
        mocker: Any,
        eval_module: Any,
        test_cfg: tuple[bool, str],
    ) -> None:
        """Verifies hardware-dependent dtype selection and model eval mode."""
        bf16_supported, expected_dtype = test_cfg
        mocker.patch("torch.cuda.is_bf16_supported", return_value=bf16_supported)
        mock_hf = mocker.patch("eval.LlamaForCausalLM.from_pretrained")

        evaluator = eval_module.CipherEvaluator.__new__(eval_module.CipherEvaluator)
        evaluator.model_path = "/fake/path"
        evaluator._load_model()

        actual_dtype = mock_hf.call_args[1]["torch_dtype"]
        assert str(actual_dtype).split(".")[-1] == expected_dtype

        assert mock_hf.return_value.config.use_cache is True
        mock_hf.return_value.eval.assert_called_once()

    def test_load_dataset_path(
        self, mocker: Any, eval_module: Any, mock_config: Any
    ) -> None:
        """Verifies dataset is loaded from the correct subfolder."""
        mock_load = mocker.patch("eval.load_from_disk")
        evaluator = eval_module.CipherEvaluator.__new__(eval_module.CipherEvaluator)
        evaluator.config = mock_config

        evaluator._load_dataset()
        mock_load.assert_called_once_with(Path("/tmp/data/Test")) # noqa: S108


class TestDecodingAndMetrics:
    """Tests for token decoding and metric calculations."""

    def test_decode_prediction_all_branches(
        self, eval_module: Any, mock_config: Any
    ) -> None:
        """Verifies sequence decoding handles offsets, spaces, and early stops."""
        evaluator = eval_module.CipherEvaluator.__new__(eval_module.CipherEvaluator)
        evaluator.config = mock_config

        evaluator.config.use_spaces = False
        ids = [10, 4, 11, 3, 12]
        assert evaluator.decode_prediction(ids) == "a b"

        evaluator.config.use_spaces = True
        assert evaluator.decode_prediction(ids) == "a_b"

    def test_decode_ciphertext(self, eval_module: Any, mock_config: Any) -> None:
        """Verifies structural tokens are excluded from cipher output."""
        evaluator = eval_module.CipherEvaluator.__new__(eval_module.CipherEvaluator)
        evaluator.config = mock_config

        ids = [1, 55, 56, 2]
        assert evaluator.decode_ciphertext(ids) == "55 56"

    def test_ser_all_branches(self, eval_module: Any) -> None:
        """Tests Symbol Error Rate boundaries and fractions."""
        evaluator = eval_module.CipherEvaluator.__new__(eval_module.CipherEvaluator)
        assert evaluator._ser("", "anything") == 1.0
        assert evaluator._ser("", "") == 0.0
        assert evaluator._ser("abc", "abc") == 0.0
        assert evaluator._ser("abc", "abd") == pytest.approx(1 / 3)
        assert evaluator._ser("abc", "abcd") == pytest.approx(1 / 3)


class TestOrchestrationLoop:
    """Tests for generation logic and file logging."""

    @pytest.fixture
    def mock_eval(self, mocker: Any, eval_module: Any, mock_config: Any) -> Any:
        """Provides a bootstrapped evaluator for inference loops."""
        mocker.patch.object(eval_module.CipherEvaluator, "_load_model")
        mocker.patch.object(eval_module.CipherEvaluator, "_load_dataset")
        mocker.patch("eval.Config", return_value=mock_config)

        ev = eval_module.CipherEvaluator(model_path="/tmp", use_spaces=False) # noqa: S108
        ev.output_log_path = Path("/tmp/results.jsonl") # noqa: S108
        return ev

    def test_evaluate_single_sample_success(
        self, mocker: Any, mock_eval: Any, torch_lib: Any
    ) -> None:
        """Ensures tensors and generation config flow correctly."""
        mock_eval.model = mocker.Mock()
        mock_eval.model.device = "cpu"

        mock_output_tensor = torch_lib.tensor([[1, 55, 2, 10, 11, 3]])
        mock_eval.model.generate.return_value = mock_output_tensor

        item = {
            "input_ids": [1, 55, 2],
            "raw_plaintext": "ab",
            "redundancy": 5,
        }

        result = mock_eval._evaluate_single_sample(item, index=0)

        assert result is not None
        assert result["plaintext"] == "ab"
        assert result["predicted_plaintext"] == "ab"
        assert result["ciphertext"] == "55"
        assert result["ser"] == 0.0
        assert "inference_time_seconds" in result

    def test_evaluate_missing_sep_warning(self, mocker: Any, mock_eval: Any) -> None:
        """Verifies broken token sequences are caught and skipped."""
        mock_logger = mocker.patch("eval.logger")
        bad_item = {"input_ids": [1, 10, 11], "raw_plaintext": "ab", "redundancy": 0}

        result = mock_eval._evaluate_single_sample(bad_item, index=5)

        assert result is None
        mock_logger.warning.assert_called_once_with(
            "Sample 5 missing SEP token. Skipping."
        )

    def test_run_logic_and_logging(self, mocker: Any, mock_eval: Any) -> None:
        """Verifies sequential evaluation, skipping failures, and disk writes."""
        mock_eval.dataset = [mocker.Mock()] * 2

        mocker.patch.object(
            mock_eval,
            "_evaluate_single_sample",
            side_effect=[
                {"ser": 0.5, "index": 0, "inference_time_seconds": 1.2},
                None,
            ],
        )
        mock_logger = mocker.patch("eval.logger")
        mock_open = mocker.patch("builtins.open", mocker.mock_open())

        mock_eval.run()

        log_messages = [call[0][0] for call in mock_logger.info.call_args_list]
        assert any("SER: 0.5000" in msg for msg in log_messages)
        assert mock_open().write.call_count == 1
        assert any("DONE. Avg SER: 0.5000" in msg for msg in log_messages)


class TestCLI:
    """Tests for the main entrypoint."""

    def test_main_execution(self, mocker: Any, eval_module: Any) -> None:
        """Verifies argparse mapping to evaluator instantiation."""
        mocker.patch("sys.argv", ["eval.py", "--model_path", "test_path", "--spaces"])
        mock_run = mocker.patch.object(eval_module.CipherEvaluator, "run")
        mocker.patch.object(eval_module.CipherEvaluator, "_load_model")
        mocker.patch.object(eval_module.CipherEvaluator, "_load_dataset")

        eval_module.main()
        mock_run.assert_called_once()
