# tests/conftest.py
"""Pytest fixtures and configuration for EEG Stimulus tests."""

import pytest
import tempfile
import shutil
from pathlib import Path
import logging
import numpy as np

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


@pytest.fixture
def mock_audio_samples_mono():
    """Provide mock mono audio samples as int16 numpy array."""
    # 1 second of 440Hz sine wave at 44100Hz sample rate
    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    samples = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    return samples.reshape(-1, 1)


@pytest.fixture
def mock_audio_samples_stereo():
    """Provide mock stereo audio samples as int16 numpy array."""
    # 1 second of 440Hz sine wave at 44100Hz sample rate
    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    left = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    right = (np.sin(2 * np.pi * 880 * t) * 32767).astype(np.int16)
    return np.column_stack([left, right])


@pytest.fixture
def mock_audio_samples_short():
    """Provide short mock audio samples for quick tests."""
    # 100 samples of silence
    return np.zeros((100, 1), dtype=np.int16)


@pytest.fixture
def mock_audio_samples_float():
    """Provide mock float audio samples in range [-1, 1]."""
    sample_rate = 44100
    duration = 0.1
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)
