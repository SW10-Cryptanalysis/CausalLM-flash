import json
import pytest
from pathlib import Path
from typing import TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from src.classes.config import Config


@pytest.fixture
def config_cls() -> type["Config"]:
    from src.classes.config import Config

    return Config


@dataclass
class ConfigConstants:
    BUFFER: int
    UNIQUE_HOMOPHONE_COUNT: int
    UNIQUE_LETTER_COUNT: int
    HOMOPHONE_FILE: str


@pytest.fixture
def constants() -> ConfigConstants:
    from src.classes.config import (
        BUFFER,
        UNIQUE_HOMOPHONE_COUNT,
        UNIQUE_LETTER_COUNT,
        HOMOPHONE_FILE,
    )

    return ConfigConstants(
        BUFFER=BUFFER,
        UNIQUE_HOMOPHONE_COUNT=UNIQUE_HOMOPHONE_COUNT,
        UNIQUE_LETTER_COUNT=UNIQUE_LETTER_COUNT,
        HOMOPHONE_FILE=HOMOPHONE_FILE,
    )


@pytest.fixture
def class_and_constants(
    config_cls, constants
) -> tuple[type["Config"], ConfigConstants]:
    return config_cls, constants


class TestConfigTokens:
    """Tests for dynamic token IDs and offsets."""

    def test_token_id_chain(self) -> None:
        """Ensure token IDs increment sequentially based on unique_homophones."""
        from src.classes.config import Config

        cfg = Config(unique_homophones=100)

        assert cfg.sep_token_id == 101
        assert cfg.space_token_id == 102
        assert cfg.bos_token_id == 103
        assert cfg.eos_token_id == 104
        assert cfg.char_offset == 105


class TestConfigPaths:
    """Tests for path resolution and dynamic properties depending on flags."""

    @pytest.mark.parametrize(
        "test_cfg",
        [
            (False, "normal", "tokenized_normal"),
            (True, "spaces", "tokenized_spaced"),
        ],
    )
    def test_dynamic_directories(
        self,
        tmp_path: Path,
        test_cfg: tuple[bool, str, str],
        config_cls,
    ) -> None:
        """Ensure all dynamic directories correctly reflect the use_spaces flag."""

        cfg = config_cls(output_dir=tmp_path, data_dir=tmp_path, use_spaces=test_cfg[0])

        assert cfg.final_output_dir == tmp_path / test_cfg[1]
        assert cfg.tokenized_dir == tmp_path / test_cfg[2]


class TestConfigLoadHomophones:
    """Tests for the load_homophones method and vocabulary sizing."""

    def test_vocab_size_default(
        self, constants: ConfigConstants, config_cls: type["Config"]
    ) -> None:
        """Check default vocabulary size calculation upon initialization."""
        cfg = config_cls(
            unique_homophones=constants.UNIQUE_HOMOPHONE_COUNT,
            unique_letters=constants.UNIQUE_LETTER_COUNT,
        )
        expected_size = (
            constants.UNIQUE_HOMOPHONE_COUNT
            + constants.UNIQUE_LETTER_COUNT
            + constants.BUFFER
        )
        assert cfg.vocab_size == expected_size

    def test_vocab_capacity(self, tmp_path: Path, config_cls: type["Config"]) -> None:
        """Ensure the vocabulary size can accommodate the character 'z' after loading."""
        cfg = config_cls(data_dir=tmp_path)
        cfg.load_homophones()

        """Calculate highest possible ID: offset + 25 (for 'z')"""
        highest_id = cfg.char_offset + 25

        assert cfg.vocab_size > highest_id

    def test_load_homophones_success(
        self, tmp_path: Path, constants: ConfigConstants, config_cls: type["Config"]
    ) -> None:
        """Ensure max_symbol_id is loaded correctly from a valid metadata file."""
        meta_file = tmp_path / constants.HOMOPHONE_FILE
        meta_file.write_text(json.dumps({"max_symbol_id": 999}))

        cfg = config_cls(data_dir=tmp_path)
        cfg.load_homophones()

        assert cfg.unique_homophones == 999
        assert cfg.vocab_size == cfg.char_offset + 26 + 1

    def test_load_homophones_missing_file(
        self, tmp_path: Path, mocker, config_cls: type["Config"]
    ) -> None:
        """Ensure defaults are kept when the file is entirely missing."""
        mock_logger = mocker.patch("src.classes.config.logger")

        cfg = config_cls(data_dir=tmp_path, unique_homophones=500)
        cfg.load_homophones()

        assert cfg.unique_homophones == 500

        """The os.path.exists check prevents errors/warnings if the file is just absent"""
        mock_logger.warning.assert_not_called()

    @dataclass
    class LoadHomophonesErrors:
        file_content: str
        simulate_os_error: bool
        id: str

    load_homophone_cases = [
        LoadHomophonesErrors(
            file_content="{invalid_json:",
            simulate_os_error=False,
            id="invalid_json",
        ),
        LoadHomophonesErrors(
            file_content='{"wrong_key": 999}',
            simulate_os_error=False,
            id="missing_key",
        ),
        LoadHomophonesErrors(
            file_content="",
            simulate_os_error=True,
            id="os_error",
        ),
    ]

    @pytest.mark.parametrize(
        "test_cfg",
        load_homophone_cases,
        ids=[tc.id for tc in load_homophone_cases],
    )
    def test_load_homophones_errors(
        self,
        tmp_path: Path,
        mocker,
        test_cfg: LoadHomophonesErrors,
        class_and_constants,
    ) -> None:
        """Ensure errors are caught and warnings logged for various failure modes."""
        file_content, simulate_os_error = (
            test_cfg.file_content,
            test_cfg.simulate_os_error,
        )
        config_cls, constants = class_and_constants

        mock_logger = mocker.patch("src.classes.config.logger")
        meta_file = tmp_path / constants.HOMOPHONE_FILE

        if simulate_os_error:
            meta_file.touch()
            mocker.patch("builtins.open", side_effect=OSError("Permission denied"))
        else:
            meta_file.write_text(file_content)

        cfg = config_cls(data_dir=tmp_path, unique_homophones=500)
        cfg.load_homophones()

        assert cfg.unique_homophones == 500
        assert mock_logger.warning.call_count == 3
