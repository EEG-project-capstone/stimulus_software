# tests/conftest.py
"""Pytest fixtures and configuration for EEG Stimulus tests."""

import pytest
import tempfile
import shutil
from pathlib import Path
import logging

# Set up logging for tests
logging.basicConfig(level=logging.DEBUG)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp)


@pytest.fixture
def mock_config_data():
    """Provide mock configuration data."""
    return {
        'edf_dir': 'patient_data/edfs/',
        'result_dir': 'patient_data/results/',
        'sentences_path': 'audio_data/sentences/',
        'right_keep_path': 'audio_data/static/right_keep.mp3',
        'right_stop_path': 'audio_data/static/right_stop.mp3',
        'left_keep_path': 'audio_data/static/left_keep.mp3',
        'left_stop_path': 'audio_data/static/left_stop.mp3',
        'beep_path': 'audio_data/static/sample_beep.mp3',
        'motor_prompt_path': 'audio_data/prompts/motorcommandprompt.wav',
        'oddball_prompt_path': 'audio_data/prompts/oddballprompt.wav'
    }


@pytest.fixture
def temp_config_file(temp_dir, mock_config_data):
    """Create a temporary config file."""
    import yaml

    config_path = temp_dir / 'config.yml'
    with open(config_path, 'w') as f:
        yaml.dump(mock_config_data, f)
    return config_path
