import os
import sys
import pytest
from pathlib import Path
from typing import Any
from dataclasses import dataclass


@pytest.fixture
def train_module() -> Any:
    """Lazy-loads the train module to bypass pytest collection delay."""
    import train

    return train


class TestCheckpointDetection:
    """Tests for the custom checkpoint detection logic."""

    def test_is_checkpoint_valid(self, tmp_path: Path, train_module: Any) -> None:
        """A directory starting with 'checkpoint-' containing files should return True."""
        chk_dir = tmp_path / "checkpoint-500"
        chk_dir.mkdir()
        (chk_dir / "model.safetensors").touch()

        assert train_module._is_checkpoint(chk_dir) is True

    def test_is_checkpoint_empty(self, tmp_path: Path, train_module: Any) -> None:
        """An empty checkpoint directory should return False."""
        chk_dir = tmp_path / "checkpoint-1000"
        chk_dir.mkdir()

        assert train_module._is_checkpoint(chk_dir) is False

    def test_is_checkpoint_not_dir(self, tmp_path: Path, train_module: Any) -> None:
        """A file starting with 'checkpoint-' should return False."""
        chk_file = tmp_path / "checkpoint-file.txt"
        chk_file.touch()

        assert train_module._is_checkpoint(chk_file) is False

    def test_contains_checkpoint_true(self, tmp_path: Path, train_module: Any) -> None:
        """Should return True if at least one valid checkpoint exists in the output dir."""
        chk_dir = tmp_path / "checkpoint-100"
        chk_dir.mkdir()
        (chk_dir / "optimizer.pt").touch()

        assert train_module.contains_checkpoint(tmp_path) is True

    def test_contains_checkpoint_false_empty_dir(
        self, tmp_path: Path, train_module: Any
    ) -> None:
        """Should return False if the output dir is empty."""
        assert train_module.contains_checkpoint(tmp_path) is False

    def test_contains_checkpoint_missing_dir(
        self, tmp_path: Path, train_module: Any
    ) -> None:
        """Should return False if the output dir does not exist at all."""
        missing_dir = tmp_path / "does_not_exist"

        assert train_module.contains_checkpoint(missing_dir) is False


class TestTrainFunction:
    """Tests for the main train execution flow using pytest-mock."""

    @pytest.fixture
    def mock_dependencies(self, mocker: Any, tmp_path: Path) -> dict[str, Any]:
        """Fixtures to mock out all heavy HF, torch, and custom classes."""

        mock_config_cls = mocker.patch("train.Config")
        mock_config = mock_config_cls.return_value
        mock_config.final_output_dir = tmp_path / "outputs"
        mock_config.epochs = 1
        mock_config.batch_size = 2
        mock_config.grad_accum = 1
        mock_config.learning_rate = 3e-4
        mock_config.save_steps = 10
        mock_config.log_steps = 5
        mock_config.pad_token_id = 0

        """
        Mocking TrainingArguments prevents transformers from running hardware
        and torch version checks for bf16, tf32, and fused optimizers.
        """
        mocks = {
            "config": mock_config,
            "get_model": mocker.patch("train.get_model"),
            "dataset": mocker.patch("train.CipherPlainData"),
            "collator": mocker.patch("train.PadCollator"),
            "trainer_cls": mocker.patch("train.Trainer"),
            "training_args_cls": mocker.patch("train.TrainingArguments"),
            "contains_chk": mocker.patch(
                "train.contains_checkpoint", return_value=False
            ),
        }
        return mocks

    @dataclass
    class CliTestCase:
        """Test cases for the CLI arguments."""

        cli_args: list[str]
        expected_spaces: bool
        id: str

    cli_cases = [
        CliTestCase(
            cli_args=["train.py"],
            expected_spaces=False,
            id="no_spaces",
        ),
        CliTestCase(
            cli_args=["train.py", "--spaces"],
            expected_spaces=True,
            id="with_spaces",
        ),
    ]

    @pytest.mark.parametrize(
        "test_case",
        cli_cases,
        ids=[tc.id for tc in cli_cases],
    )
    def test_cli_arguments(
        self,
        mocker: Any,
        mock_dependencies: dict[str, Any],
        test_case: CliTestCase,
        train_module: Any,
    ) -> None:
        """Ensure argparse correctly parses the --spaces flag and applies it to Config."""
        cli_args = test_case.cli_args
        expected_spaces = test_case.expected_spaces

        mocker.patch.object(sys, "argv", cli_args)

        train_module.train()

        mock_config = mock_dependencies["config"]
        assert mock_config.use_spaces is expected_spaces
        mock_config.load_homophones.assert_called_once()

    def test_trainer_execution_flow(
        self, mocker: Any, mock_dependencies: dict[str, Any], train_module: Any
    ) -> None:
        """Ensure the Trainer is instantiated and train/save methods are called."""
        mocker.patch.object(sys, "argv", ["train.py"])

        mock_trainer_instance = mock_dependencies["trainer_cls"].return_value
        mock_dependencies["contains_chk"].return_value = True

        train_module.train()

        """Verify training is resumed if a checkpoint is found"""
        mock_trainer_instance.train.assert_called_once_with(resume_from_checkpoint=True)

        """Verify the model saves to the correct subdirectory"""
        expected_save_dest = f"{mock_dependencies['config'].final_output_dir}/model"
        mock_trainer_instance.save_model.assert_called_once_with(expected_save_dest)

    def test_environment_variable_set(self, train_module: Any) -> None:
        """Ensure the memory optimization environment variable is set."""
        """
        Requesting the train_module fixture ensures the file is executed,
        which triggers the top-level os.environ assignment.
        """
        assert os.environ.get("PYTORCH_ALLOC_CONF") == "expandable_segments:True"
