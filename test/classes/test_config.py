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
    UNIQUE_LETTER_COUNT: int
    HOMOPHONE_FILE: str


@pytest.fixture
def constants() -> ConfigConstants:
    from src.classes.config import (
        BUFFER,
        UNIQUE_LETTER_COUNT,
        HOMOPHONE_FILE,
    )

    return ConfigConstants(
        BUFFER=BUFFER,
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

    def test_vocab_size_default(self, config_cls: type["Config"]) -> None:
        """Check that default vocabulary size is 0 before loading."""
        cfg = config_cls()
        assert cfg.vocab_size == 0

    def test_vocab_capacity(
        self, tmp_path: Path, constants: ConfigConstants, config_cls: type["Config"]
    ) -> None:
        """Ensure the vocabulary size can accommodate the character 'z' after loading."""
        meta_file = tmp_path / constants.HOMOPHONE_FILE
        meta_file.write_text(json.dumps({"max_symbol_id": 500}))

        cfg = config_cls(data_dir=tmp_path)
        cfg.load_homophones()

        # Calculate highest possible ID: offset + 25 (for 'z')
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
        self, tmp_path: Path, config_cls: type["Config"]
    ) -> None:
        """Ensure the script crashes loudly if the metadata file is missing."""
        cfg = config_cls(data_dir=tmp_path)

        # We EXPECT a FileNotFoundError because tmp_path is empty
        with pytest.raises(FileNotFoundError):
            cfg.load_homophones()

    @dataclass
    class LoadHomophonesErrors:
        file_content: str
        simulate_os_error: bool
        expected_exception: type[Exception]
        id: str

    load_homophone_cases = [
        LoadHomophonesErrors(
            file_content="{invalid_json:",
            simulate_os_error=False,
            expected_exception=ValueError,
            id="invalid_json",
        ),
        LoadHomophonesErrors(
            file_content='{"wrong_key": 999}',
            simulate_os_error=False,
            expected_exception=ValueError,
            id="missing_key",
        ),
        LoadHomophonesErrors(
            file_content="",
            simulate_os_error=True,
            expected_exception=OSError,
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
        config_cls, constants = class_and_constants
        meta_file = tmp_path / constants.HOMOPHONE_FILE

        if test_cfg.simulate_os_error:
            meta_file.touch()
            mocker.patch("builtins.open", side_effect=OSError("Permission denied"))
        else:
            meta_file.write_text(test_cfg.file_content)

        cfg = config_cls(data_dir=tmp_path)

        with pytest.raises(test_cfg.expected_exception):
            cfg.load_homophones()
