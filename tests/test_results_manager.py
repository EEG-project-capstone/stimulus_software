# tests/test_results_manager.py
"""Tests for results_manager.py - Results file management."""

import pytest
import sys
import tempfile
import yaml
import time
import threading
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.results_manager import ResultsManager
from lib.exceptions import ResultsFileError


class MockConfig:
    """Mock configuration for testing."""

    def __init__(self, result_dir: Path):
        self.result_dir = result_dir
        self.current_date = time.strftime("%Y-%m-%d")

    def get_results_path(self, patient_id: str) -> Path:
        filename = f"{patient_id}_{self.current_date}_stimulus_results.csv"
        return self.result_dir / filename


@pytest.fixture
def temp_results_dir(temp_dir):
    """Create a temporary results directory."""
    results = temp_dir / 'results'
    results.mkdir()
    return results


@pytest.fixture
def mock_config(temp_results_dir):
    """Create a mock config object."""
    return MockConfig(temp_results_dir)


@pytest.fixture
def results_manager(mock_config):
    """Create a ResultsManager instance."""
    return ResultsManager(mock_config)


class TestResultsManagerInit:
    """Tests for ResultsManager initialization."""

    def test_init_creates_manager(self, mock_config):
        """ResultsManager should initialize successfully."""
        rm = ResultsManager(mock_config)
        assert rm is not None
        assert rm.config == mock_config


class TestAppendResult:
    """Tests for append_result method."""

    def test_append_creates_file(self, results_manager, temp_results_dir):
        """append_result should create the results file if it doesn't exist."""
        data = {
            'start_time': time.time(),
            'end_time': time.time() + 1.0,
            'duration': 1.0
        }

        filepath = results_manager.append_result('PATIENT001', 'language', data)

        assert filepath.exists()

    def test_append_writes_header_on_first_write(self, results_manager):
        """First append should write CSV header."""
        data = {'start_time': 1000.0, 'end_time': 1001.0, 'duration': 1.0}

        filepath = results_manager.append_result('PATIENT001', 'language', data)
        df = pd.read_csv(filepath)

        assert 'patient_id' in df.columns
        assert 'date' in df.columns
        assert 'stim_type' in df.columns
        assert 'start_time' in df.columns
        assert 'end_time' in df.columns
        assert 'duration' in df.columns

    def test_append_includes_correct_data(self, results_manager):
        """append_result should write correct data."""
        start = 1000.5
        end = 1002.5
        duration = 2.0

        data = {'start_time': start, 'end_time': end, 'duration': duration}

        filepath = results_manager.append_result('PATIENT001', 'language', data)
        df = pd.read_csv(filepath)

        assert len(df) == 1
        assert df.iloc[0]['patient_id'] == 'PATIENT001'
        assert df.iloc[0]['stim_type'] == 'language'
        assert df.iloc[0]['start_time'] == start
        assert df.iloc[0]['duration'] == duration

    def test_append_multiple_results(self, results_manager):
        """Multiple appends should accumulate rows."""
        for i in range(5):
            data = {
                'start_time': 1000.0 + i,
                'end_time': 1001.0 + i,
                'duration': 1.0
            }
            filepath = results_manager.append_result('PATIENT001', 'language', data)

        df = pd.read_csv(filepath)
        assert len(df) == 5

    def test_append_with_events(self, results_manager):
        """append_result should handle events array in notes."""
        events = [
            {'type': 'tone', 'freq': 1000, 'time': 0.1},
            {'type': 'tone', 'freq': 2000, 'time': 0.5}
        ]
        data = {
            'start_time': 1000.0,
            'end_time': 1005.0,
            'duration': 5.0,
            'events': events
        }

        filepath = results_manager.append_result('PATIENT001', 'oddball', data)
        df = pd.read_csv(filepath)

        # Events should be JSON-encoded in notes
        notes = df.iloc[0]['notes']
        assert 'tone' in str(notes)
        assert '1000' in str(notes)

    def test_append_with_text_notes(self, results_manager):
        """append_result should handle text notes."""
        data = {
            'start_time': 1000.0,
            'end_time': 1001.0,
            'duration': 1.0,
            'notes': 'Patient showed response'
        }

        filepath = results_manager.append_result('PATIENT001', 'language', data)
        df = pd.read_csv(filepath)

        assert 'Patient showed response' in df.iloc[0]['notes']


class TestAppendNote:
    """Tests for append_note method."""

    def test_append_note(self, results_manager):
        """append_note should create a session_note entry."""
        filepath = results_manager.append_note('PATIENT001', 'Test note content')

        df = pd.read_csv(filepath)
        assert len(df) == 1
        assert df.iloc[0]['stim_type'] == 'session_note'
        assert 'Test note content' in df.iloc[0]['notes']


class TestAppendSyncPulse:
    """Tests for append_sync_pulse method."""

    def test_append_sync_pulse(self, results_manager):
        """append_sync_pulse should create a manual_sync_pulse entry."""
        sync_time = time.time()
        filepath = results_manager.append_sync_pulse('PATIENT001', sync_time)

        df = pd.read_csv(filepath)
        assert len(df) == 1
        assert df.iloc[0]['stim_type'] == 'manual_sync_pulse'
        assert df.iloc[0]['start_time'] == sync_time


class TestReadResults:
    """Tests for read_results method."""

    def test_read_results_returns_dataframe(self, results_manager):
        """read_results should return a DataFrame."""
        data = {'start_time': 1000.0, 'end_time': 1001.0, 'duration': 1.0}
        filepath = results_manager.append_result('PATIENT001', 'language', data)

        df = results_manager.read_results(filepath)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_read_results_nonexistent_file(self, results_manager, temp_dir):
        """read_results should return None for non-existent file."""
        non_existent = temp_dir / 'nonexistent.csv'

        result = results_manager.read_results(non_existent)

        assert result is None


class TestGetStimulusSequence:
    """Tests for get_stimulus_sequence method."""

    def test_get_stimulus_sequence(self, results_manager):
        """get_stimulus_sequence should return list of stimulus dicts."""
        # Add various stimuli
        for stim_type in ['language', 'right_command', 'oddball']:
            data = {'start_time': 1000.0, 'end_time': 1001.0, 'duration': 1.0}
            filepath = results_manager.append_result('PATIENT001', stim_type, data)

        sequence = results_manager.get_stimulus_sequence(filepath)

        assert len(sequence) == 3
        types = [s['type'] for s in sequence]
        assert 'language' in types
        assert 'right_command' in types
        assert 'oddball' in types

    def test_get_stimulus_sequence_excludes_notes(self, results_manager):
        """get_stimulus_sequence should exclude non-stimulus entries."""
        data = {'start_time': 1000.0, 'end_time': 1001.0, 'duration': 1.0}
        filepath = results_manager.append_result('PATIENT001', 'language', data)
        results_manager.append_note('PATIENT001', 'This is a note')
        results_manager.append_sync_pulse('PATIENT001', time.time())

        sequence = results_manager.get_stimulus_sequence(filepath)

        assert len(sequence) == 1
        assert sequence[0]['type'] == 'language'


class TestGetSessionLog:
    """Tests for get_session_log method."""

    def test_get_session_log(self, results_manager):
        """get_session_log should return formatted log entries."""
        filepath = results_manager.append_note('PATIENT001', 'First note')
        results_manager.append_note('PATIENT001', 'Second note')

        logs = results_manager.get_session_log(filepath)

        assert len(logs) >= 2
        assert any('First note' in log for log in logs)
        assert any('Second note' in log for log in logs)


class TestThreadSafety:
    """Tests for thread-safe operations."""

    def test_concurrent_writes(self, results_manager):
        """Concurrent writes should not corrupt file."""
        errors = []
        filepath_result = [None]

        def write_result(i):
            try:
                data = {
                    'start_time': 1000.0 + i,
                    'end_time': 1001.0 + i,
                    'duration': 1.0
                }
                filepath_result[0] = results_manager.append_result(
                    'PATIENT001', f'language_{i}', data
                )
            except Exception as e:
                errors.append(e)

        # Launch concurrent writes
        threads = []
        for i in range(10):
            t = threading.Thread(target=write_result, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Verify no errors
        assert len(errors) == 0

        # Verify all writes succeeded
        if filepath_result[0]:
            df = pd.read_csv(filepath_result[0])
            assert len(df) == 10
