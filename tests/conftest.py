# tests/conftest.py
"""Pytest fixtures and configuration for EEG Stimulus tests."""

import pytest
import threading
import tempfile
import shutil
import numpy as np
import logging
from pathlib import Path

from lib.state_manager import StateManager
from lib.constants import PlaybackState

logging.basicConfig(level=logging.DEBUG)


# =============================================================================
# Audio sample fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp)


@pytest.fixture
def mock_audio_samples_mono():
    """1 second of 440Hz sine wave at 44100Hz, mono int16."""
    sample_rate = 44100
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    samples = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    return samples.reshape(-1, 1)


@pytest.fixture
def mock_audio_samples_stereo():
    """1 second of 440/880Hz sine wave at 44100Hz, stereo int16."""
    sample_rate = 44100
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    left = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    right = (np.sin(2 * np.pi * 880 * t) * 32767).astype(np.int16)
    return np.column_stack([left, right])


@pytest.fixture
def mock_audio_samples_short():
    """100 samples of silence."""
    return np.zeros((100, 1), dtype=np.int16)


@pytest.fixture
def mock_audio_samples_float():
    """0.1 seconds of 440Hz sine wave as float32."""
    sample_rate = 44100
    t = np.linspace(0, 0.1, int(sample_rate * 0.1), endpoint=False)
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)


# =============================================================================
# Shared mock classes (used by integration tests in test_button_controls.py)
# =============================================================================

class MockRoot:
    """Tkinter-compatible scheduler backed by real threading.Timer."""

    def __init__(self):
        self._lock = threading.Lock()
        self._next_id = 1
        self._timers = {}

    def after(self, delay_ms, callback):
        with self._lock:
            cb_id = self._next_id
            self._next_id += 1
        t = threading.Timer(delay_ms / 1000.0, callback)
        with self._lock:
            self._timers[cb_id] = t
        t.daemon = True
        t.start()
        return cb_id

    def after_cancel(self, cb_id):
        with self._lock:
            t = self._timers.pop(cb_id, None)
        if t:
            t.cancel()


class MockResultsManager:
    def append_result(self, *args, **kwargs):
        pass

    def append_sync_pulse(self, *args, **kwargs):
        pass


class MockStims:
    def __init__(self):
        self.current_stim_index = 0
        self.stim_dictionary = [
            {'type': 'oddball', 'status': 'pending'},
            {'type': 'oddball', 'status': 'pending'},
        ]


class MockGuiCallback:
    def __init__(self):
        self.root = MockRoot()
        self.results_manager = MockResultsManager()
        self.stims = MockStims()
        self.config = type('MockConfig', (), {'current_date': '2024-01-01'})()
        self.state_manager = StateManager(PlaybackState.READY)

    def get_patient_id(self):
        return 'TEST_PATIENT'

    def update_stim_list_status(self):
        pass

    def playback_complete(self):
        pass

    def playback_error(self, msg):
        pass


@pytest.fixture
def mock_gui():
    return MockGuiCallback()
