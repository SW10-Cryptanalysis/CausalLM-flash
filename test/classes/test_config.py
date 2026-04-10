import json
import pytest
from pathlib import Path
from dataclasses import dataclass
from pytest_mock import MockerFixture
from src.classes.config import Config


@dataclass
class ConfigConstants:
    """Constants required for configuration testing."""

    buffer_val: int
    unique_letter_count: int
    homophone_file: str
    max_plain_spaces: int
    max_plain_normal: int


@pytest.fixture
def constants() -> ConfigConstants:
    """Provides dynamic constants imported directly from the source module."""
    from src.classes.config import (
        BUFFER,
        UNIQUE_LETTER_COUNT,
        HOMOPHONE_FILE,
        MAX_PLAIN_SPACES,
        MAX_PLAIN_NORMAL,
    )

    return ConfigConstants(
        buffer_val=BUFFER,
        unique_letter_count=UNIQUE_LETTER_COUNT,
        homophone_file=HOMOPHONE_FILE,
        max_plain_spaces=MAX_PLAIN_SPACES,
        max_plain_normal=MAX_PLAIN_NORMAL,
    )


class TestConfigTokens:
    """Tests for dynamic token IDs and offsets."""

    def test_token_id_chain(self) -> None:
        """Ensure token IDs increment sequentially based on unique_homophones."""
        cfg = Config(unique_homophones=100)

        assert cfg.sep_token_id == 101
        assert cfg.space_token_id == 102
        assert cfg.bos_token_id == 103
        assert cfg.eos_token_id == 104
        assert cfg.char_offset == 105


@dataclass
class PathTestCase:
    """Defines parameters for dynamic path resolution tests."""

    use_spaces: bool
    expected_output: str
    expected_tokenized: str


class TestConfigPaths:
    """Tests for path resolution and dynamic properties depending on flags."""

    @pytest.mark.parametrize(
        "test_case",
        [
            PathTestCase(
                use_spaces=False,
                expected_output="normal",
                expected_tokenized="tokenized_normal",
            ),
            PathTestCase(
                use_spaces=True,
                expected_output="spaces",
                expected_tokenized="tokenized_spaced",
            ),
        ],
    )
    def test_dynamic_directories(
        self,
        tmp_path: Path,
        test_case: PathTestCase,
    ) -> None:
        """Ensure all dynamic directories correctly reflect the use_spaces flag."""
        cfg = Config(
            output_dir=tmp_path,
            data_dir=tmp_path,
            use_spaces=test_case.use_spaces,
        )

        assert cfg.final_output_dir == tmp_path / test_case.expected_output
        assert cfg.tokenized_dir == tmp_path / test_case.expected_tokenized


@dataclass
class ContextTestCase:
    """Defines parameters for max_context calculation tests."""

    use_spaces: bool


class TestConfigProperties:
    """Tests for logical properties and context calculations."""

    @pytest.mark.parametrize(
        "test_case",
        [
            ContextTestCase(use_spaces=False),
            ContextTestCase(use_spaces=True),
        ],
    )
    def test_max_context(
        self,
        test_case: ContextTestCase,
        constants: ConfigConstants,
    ) -> None:
        """Verify max context relies on the correct constant based on the spaces flag."""
        cfg = Config(use_spaces=test_case.use_spaces)

        if test_case.use_spaces:
            expected = (constants.max_plain_spaces * 2) + constants.buffer_val
        else:
            expected = (constants.max_plain_normal * 2) + constants.buffer_val

        assert cfg.max_context == expected

    def test_is_valid_init(self) -> None:
        """Verify the initialization validity check evaluates properly."""
        cfg = Config()

        assert cfg.is_valid_init is True


@dataclass
class LoadHomophonesTestCase:
    """Defines testing parameters for exception handling during homophone loads."""

    file_content: str
    simulate_os_error: bool
    expected_exception: type[Exception]


@dataclass
class HomophonesSuccessTestCase:
    """Defines parameters for successful homophone loading tests."""

    max_symbol_id: int


class TestConfigLoadHomophones:
    """Tests for the load_homophones method and vocabulary sizing."""

    def test_vocab_size_default(self) -> None:
        """Check that default vocabulary size is 0 before loading."""
        cfg = Config()

        assert cfg.vocab_size == 0

    @pytest.mark.parametrize(
        "test_case",
        [
            HomophonesSuccessTestCase(max_symbol_id=500),
            HomophonesSuccessTestCase(max_symbol_id=999),
        ],
    )
    def test_load_homophones_success(
        self,
        tmp_path: Path,
        constants: ConfigConstants,
        test_case: HomophonesSuccessTestCase,
    ) -> None:
        """Ensure homophones load correctly and vocabulary scales to accommodate all characters."""
        meta_file = tmp_path / constants.homophone_file
        meta_file.write_text(json.dumps({"max_symbol_id": test_case.max_symbol_id}))

        cfg = Config(data_dir=tmp_path)
        cfg.load_homophones()

        # Validates unique homophones are dynamically updated
        assert cfg.unique_homophones == test_case.max_symbol_id

        # Validates exact vocab arithmetic
        assert cfg.vocab_size == cfg.char_offset + 26 + 1

        # Validates capacity rule (covers the letter 'z')
        assert cfg.vocab_size > cfg.char_offset + 25

    def test_load_homophones_missing_file(
        self,
        tmp_path: Path,
    ) -> None:
        """Ensure the script crashes loudly if the metadata file is missing."""
        cfg = Config(data_dir=tmp_path)

        with pytest.raises(FileNotFoundError):
            cfg.load_homophones()

    @pytest.mark.parametrize(
        "test_case",
        [
            LoadHomophonesTestCase(
                file_content="{invalid_json:",
                simulate_os_error=False,
                expected_exception=ValueError,
            ),
            LoadHomophonesTestCase(
                file_content='{"wrong_key": 999}',
                simulate_os_error=False,
                expected_exception=ValueError,
            ),
            LoadHomophonesTestCase(
                file_content="",
                simulate_os_error=True,
                expected_exception=OSError,
            ),
        ],
    )
    def test_load_homophones_errors(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
        test_case: LoadHomophonesTestCase,
        constants: ConfigConstants,
    ) -> None:
        """Ensure errors are caught and correctly formatted for various failure modes."""
        meta_file = tmp_path / constants.homophone_file

        if test_case.simulate_os_error:
            meta_file.touch()
            mocker.patch("builtins.open", side_effect=OSError("Permission denied"))
        else:
            meta_file.write_text(test_case.file_content)

        cfg = Config(data_dir=tmp_path)

        with pytest.raises(test_case.expected_exception):
            cfg.load_homophones()
