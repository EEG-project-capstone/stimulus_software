# tests/test_config.py
"""Tests for config.py - Configuration management."""

import pytest
import sys
import tempfile
import yaml
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.exceptions import ConfigFileError, ConfigValidationError


class TestConfigLoading:
    """Tests for configuration file loading."""

    def test_missing_config_file_raises_error(self, temp_dir):
        """Missing config file should raise ConfigFileError."""
        from lib.config import Config

        non_existent = temp_dir / 'nonexistent.yml'
        with pytest.raises(ConfigFileError):
            Config(str(non_existent))

    def test_invalid_yaml_raises_error(self, temp_dir):
        """Invalid YAML should raise ConfigFileError."""
        from lib.config import Config

        bad_yaml = temp_dir / 'bad.yml'
        with open(bad_yaml, 'w') as f:
            f.write("invalid: yaml: content: [unclosed")

        with pytest.raises(ConfigFileError):
            Config(str(bad_yaml))

    def test_non_dict_yaml_raises_error(self, temp_dir):
        """YAML that doesn't contain a dict should raise ConfigFileError."""
        from lib.config import Config

        list_yaml = temp_dir / 'list.yml'
        with open(list_yaml, 'w') as f:
            f.write("- item1\n- item2\n")

        with pytest.raises(ConfigFileError):
            Config(str(list_yaml))

    def test_missing_required_keys_raises_error(self, temp_dir):
        """Config missing required keys should raise ConfigValidationError."""
        from lib.config import Config

        incomplete = temp_dir / 'incomplete.yml'
        with open(incomplete, 'w') as f:
            yaml.dump({'edf_dir': 'some/path'}, f)

        with pytest.raises(ConfigValidationError):
            Config(str(incomplete))


class TestConfigPaths:
    """Tests for path generation methods."""

    def test_get_results_path_format(self, temp_dir):
        """Results path should follow expected format."""
        from lib.config import Config

        # Create a valid config
        config_data = _create_valid_config(temp_dir)
        config_path = temp_dir / 'config.yml'
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)

        config = Config(str(config_path))
        path = config.get_results_path('PATIENT123')

        assert 'PATIENT123' in str(path)
        assert '_stimulus_results.csv' in str(path)

    def test_get_results_path_with_custom_date(self, temp_dir):
        """Results path should use custom date when provided."""
        from lib.config import Config

        config_data = _create_valid_config(temp_dir)
        config_path = temp_dir / 'config.yml'
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)

        config = Config(str(config_path))
        path = config.get_results_path('PATIENT123', date='2024-01-15')

        assert '2024-01-15' in str(path)

    def test_get_results_path_empty_patient_raises(self, temp_dir):
        """Empty patient ID should raise ValueError."""
        from lib.config import Config

        config_data = _create_valid_config(temp_dir)
        config_path = temp_dir / 'config.yml'
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)

        config = Config(str(config_path))

        with pytest.raises(ValueError):
            config.get_results_path('')

    def test_get_edf_path_format(self, temp_dir):
        """EDF path should follow expected format."""
        from lib.config import Config

        config_data = _create_valid_config(temp_dir)
        config_path = temp_dir / 'config.yml'
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)

        config = Config(str(config_path))
        path = config.get_edf_path('PATIENT123')

        assert 'PATIENT123' in str(path)
        assert '.edf' in str(path)

    def test_get_path_missing_key_raises(self, temp_dir):
        """get_path with non-existent key should raise ConfigValidationError."""
        from lib.config import Config

        config_data = _create_valid_config(temp_dir)
        config_path = temp_dir / 'config.yml'
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)

        config = Config(str(config_path))

        with pytest.raises(ConfigValidationError):
            config.get_path('nonexistent_key')


class TestConfigValidation:
    """Tests for config validation."""

    def test_valid_config_loads_successfully(self, temp_dir):
        """Valid config should load without errors."""
        from lib.config import Config

        config_data = _create_valid_config(temp_dir)
        config_path = temp_dir / 'config.yml'
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)

        # Should not raise
        config = Config(str(config_path))
        assert config is not None

    def test_directories_created_on_load(self, temp_dir):
        """result_dir and edf_dir should be created on config load."""
        from lib.config import Config

        config_data = _create_valid_config(temp_dir)
        result_dir = temp_dir / 'results_new'
        edf_dir = temp_dir / 'edfs_new'
        config_data['result_dir'] = str(result_dir)
        config_data['edf_dir'] = str(edf_dir)

        config_path = temp_dir / 'config.yml'
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)

        Config(str(config_path))

        assert result_dir.exists()
        assert edf_dir.exists()


class TestConfigSummary:
    """Tests for config summary method."""

    def test_get_config_summary(self, temp_dir):
        """Config summary should contain expected keys."""
        from lib.config import Config

        config_data = _create_valid_config(temp_dir)
        config_path = temp_dir / 'config.yml'
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)

        config = Config(str(config_path))
        summary = config.get_config_summary()

        assert 'config_file' in summary
        assert 'current_date' in summary
        assert 'result_dir' in summary
        assert 'edf_dir' in summary


def _create_valid_config(temp_dir: Path) -> dict:
    """Create a minimal valid configuration dictionary.

    Args:
        temp_dir: Temporary directory for test files

    Returns:
        Dictionary with valid config data
    """
    # Create required directories
    sentences_dir = temp_dir / 'audio_data' / 'sentences'
    sentences_dir.mkdir(parents=True, exist_ok=True)
    static_dir = temp_dir / 'audio_data' / 'static'
    static_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = temp_dir / 'audio_data' / 'prompts'
    prompts_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy audio files
    (static_dir / 'right_keep.mp3').touch()
    (static_dir / 'right_stop.mp3').touch()
    (static_dir / 'left_keep.mp3').touch()
    (static_dir / 'left_stop.mp3').touch()
    (static_dir / 'sample_beep.mp3').touch()
    (prompts_dir / 'motorcommandprompt.wav').touch()
    (prompts_dir / 'oddballprompt.wav').touch()

    # Create a test wav file in sentences
    (sentences_dir / 'test_sentence.wav').touch()

    return {
        'edf_dir': str(temp_dir / 'edfs'),
        'result_dir': str(temp_dir / 'results'),
        'sentences_path': str(sentences_dir),
        'right_keep_path': str(static_dir / 'right_keep.mp3'),
        'right_stop_path': str(static_dir / 'right_stop.mp3'),
        'left_keep_path': str(static_dir / 'left_keep.mp3'),
        'left_stop_path': str(static_dir / 'left_stop.mp3'),
        'beep_path': str(static_dir / 'sample_beep.mp3'),
        'motor_prompt_path': str(prompts_dir / 'motorcommandprompt.wav'),
        'oddball_prompt_path': str(prompts_dir / 'oddballprompt.wav')
    }
