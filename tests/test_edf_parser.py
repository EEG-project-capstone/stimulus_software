# tests/test_edf_parser.py
"""Tests for edf_parser.py - EDF file parsing and validation."""

import pytest
import sys
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.edf_parser import EDFParser
from lib.exceptions import EDFFileError


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp)


@pytest.fixture
def minimal_edf_file(temp_dir):
    """Create a minimal valid EDF file for testing."""
    edf_path = temp_dir / 'test.edf'

    # Create minimal EDF header (256 bytes fixed header + channel headers)
    n_channels = 2

    # Fixed header (256 bytes)
    header = b''
    header += b'0       '  # Version (8 bytes)
    header += b'Test Patient ID'.ljust(80)  # Patient ID (80 bytes)
    header += b'Test Recording'.ljust(80)  # Recording ID (80 bytes)
    header += b'01.01.24'  # Start date (8 bytes)
    header += b'12.00.00'  # Start time (8 bytes)
    header += str(256 + n_channels * 256).ljust(8).encode()  # Header bytes
    header += b'EDF+C'.ljust(44)  # Reserved (44 bytes)
    header += b'10      '  # Number of records (8 bytes)
    header += b'1       '  # Record duration in seconds (8 bytes)
    header += str(n_channels).ljust(4).encode()  # Number of channels (4 bytes)

    # Channel headers (256 bytes per channel)
    for i in range(n_channels):
        header += f'CH{i+1}'.ljust(16).encode()  # Label
    for i in range(n_channels):
        header += b''.ljust(80)  # Transducer type
    for i in range(n_channels):
        header += b'uV'.ljust(8)  # Physical dimension
    for i in range(n_channels):
        header += b'-3200   '  # Physical minimum
    for i in range(n_channels):
        header += b'3200    '  # Physical maximum
    for i in range(n_channels):
        header += b'-32768  '  # Digital minimum
    for i in range(n_channels):
        header += b'32767   '  # Digital maximum
    for i in range(n_channels):
        header += b''.ljust(80)  # Prefiltering
    for i in range(n_channels):
        header += b'256     '  # Samples per record
    for i in range(n_channels):
        header += b''.ljust(32)  # Reserved

    # Data records (10 records, 256 samples per channel per record)
    samples_per_record = 256
    data = np.zeros((10, n_channels * samples_per_record), dtype=np.int16)
    data_bytes = data.tobytes()

    with open(edf_path, 'wb') as f:
        f.write(header)
        f.write(data_bytes)

    return edf_path


class TestEDFParserInit:
    """Tests for EDFParser initialization."""

    def test_initialization(self, temp_dir):
        """EDFParser should initialize with path."""
        parser = EDFParser(str(temp_dir / 'test.edf'))
        assert parser.edf_path == temp_dir / 'test.edf'
        assert parser.raw is None
        assert parser.sync_sample is None
        assert parser.sync_time is None

    def test_accepts_path_object(self, temp_dir):
        """EDFParser should accept Path objects."""
        parser = EDFParser(temp_dir / 'test.edf')
        assert parser.edf_path == temp_dir / 'test.edf'


class TestValidateFile:
    """Tests for file validation."""

    def test_nonexistent_file(self, temp_dir):
        """validate_file should fail for nonexistent file."""
        parser = EDFParser(str(temp_dir / 'nonexistent.edf'))
        result = parser.validate_file()
        assert result['valid'] is False
        assert 'not found' in result['error']

    def test_file_too_small(self, temp_dir):
        """validate_file should fail for file smaller than EDF header."""
        small_file = temp_dir / 'small.edf'
        small_file.write_bytes(b'tiny')
        parser = EDFParser(str(small_file))
        result = parser.validate_file()
        assert result['valid'] is False
        assert 'too small' in result['error']

    def test_wrong_extension_warns(self, temp_dir):
        """validate_file should warn about wrong extension but continue."""
        # Create file with wrong extension and invalid EDF content
        wrong_ext = temp_dir / 'test.txt'
        # Write bytes that will fail numeric field parsing (non-numeric characters)
        # The EDF header has numeric fields that must parse as integers
        wrong_ext.write_bytes(b'INVALID_EDF_HEADER' + b'X' * 300)
        parser = EDFParser(str(wrong_ext))
        result = parser.validate_file()
        # Should fail on header parse, not extension
        assert result['valid'] is False
        assert 'Invalid EDF format' in result['error']

    def test_valid_edf_header(self, minimal_edf_file):
        """validate_file should succeed for valid EDF."""
        parser = EDFParser(str(minimal_edf_file))
        result = parser.validate_file()
        assert result['valid'] is True
        assert result['error'] is None
        assert result['header'] is not None
        assert result['header']['n_channels'] == 2


class TestReadEDFHeader:
    """Tests for _read_edf_header method."""

    def test_header_parsing(self, minimal_edf_file):
        """_read_edf_header should parse header fields correctly."""
        parser = EDFParser(str(minimal_edf_file))
        header = parser._read_edf_header()

        assert header['version'] == '0'
        assert 'Test Patient' in header['patient_id']
        assert 'Test Recording' in header['recording_id']
        assert header['n_channels'] == 2
        assert header['n_records'] == 10
        assert header['record_duration'] == 1.0
        assert len(header['channel_labels']) == 2

    def test_total_duration_calculated(self, minimal_edf_file):
        """_read_edf_header should calculate total duration."""
        parser = EDFParser(str(minimal_edf_file))
        header = parser._read_edf_header()

        # 10 records * 1 second each = 10 seconds
        assert header['total_duration_sec'] == 10.0


class TestLoadEDF:
    """Tests for load_edf method."""

    def test_load_nonexistent_raises(self, temp_dir):
        """load_edf should raise EDFFileError for nonexistent file."""
        parser = EDFParser(str(temp_dir / 'nonexistent.edf'))
        with pytest.raises(EDFFileError, match='not found'):
            parser.load_edf()

    @patch('lib.edf_parser.mne.io.read_raw_edf')
    def test_load_calls_mne(self, mock_read_raw, minimal_edf_file):
        """load_edf should call MNE's read_raw_edf."""
        mock_raw = MagicMock()
        mock_raw.ch_names = ['CH1', 'CH2']
        mock_raw.info = {'sfreq': 256}
        mock_read_raw.return_value = mock_raw

        parser = EDFParser(str(minimal_edf_file))
        parser.load_edf()

        mock_read_raw.assert_called_once()
        assert parser.raw is mock_raw

    @patch('lib.edf_parser.mne.io.read_raw_edf')
    def test_load_mne_error_raises(self, mock_read_raw, minimal_edf_file):
        """load_edf should raise EDFFileError if MNE fails."""
        mock_read_raw.side_effect = Exception("MNE internal error")

        parser = EDFParser(str(minimal_edf_file))
        with pytest.raises(EDFFileError, match='Failed to load EDF with MNE'):
            parser.load_edf()


class TestGetInfoSummary:
    """Tests for get_info_summary method."""

    def test_summary_without_load_raises(self, temp_dir):
        """get_info_summary should raise if EDF not loaded."""
        parser = EDFParser(str(temp_dir / 'test.edf'))
        with pytest.raises(RuntimeError, match='not loaded'):
            parser.get_info_summary()

    @patch('lib.edf_parser.mne.io.read_raw_edf')
    def test_summary_contains_expected_keys(self, mock_read_raw, minimal_edf_file):
        """get_info_summary should return dict with expected keys."""
        mock_raw = MagicMock()
        mock_raw.ch_names = ['CH1', 'CH2']
        mock_raw.times = np.arange(0, 10, 1/256)  # 10 seconds at 256 Hz
        mock_raw.info = {
            'sfreq': 256,
            'meas_date': None,
            'subject_info': None,
            'highpass': 0.1,
            'lowpass': 100,
        }
        mock_raw.get_channel_types.return_value = ['eeg', 'eeg']
        mock_read_raw.return_value = mock_raw

        parser = EDFParser(str(minimal_edf_file))
        parser.load_edf()
        summary = parser.get_info_summary()

        assert 'ch_names' in summary
        assert 'n_channels' in summary
        assert 'sfreq' in summary
        assert 'duration' in summary
        assert 'duration_formatted' in summary
        assert 'subject_info' in summary
        assert 'header_info' in summary
        assert 'channel_types' in summary

    @patch('lib.edf_parser.mne.io.read_raw_edf')
    def test_duration_formatting(self, mock_read_raw, minimal_edf_file):
        """get_info_summary should format duration correctly."""
        mock_raw = MagicMock()
        mock_raw.ch_names = ['CH1']
        mock_raw.times = np.arange(0, 3723, 1/256)  # 1h 2m 3s
        mock_raw.info = {'sfreq': 256, 'meas_date': None, 'subject_info': None}
        mock_raw.get_channel_types.return_value = ['eeg']
        mock_read_raw.return_value = mock_raw

        parser = EDFParser(str(minimal_edf_file))
        parser.load_edf()
        summary = parser.get_info_summary()

        assert '1h' in summary['duration_formatted']
        assert '2m' in summary['duration_formatted']


class TestFormatDuration:
    """Tests for _format_duration method."""

    def test_seconds_only(self, temp_dir):
        """_format_duration should format seconds only."""
        parser = EDFParser(str(temp_dir / 'test.edf'))
        assert parser._format_duration(45.5) == '45.5s'

    def test_minutes_and_seconds(self, temp_dir):
        """_format_duration should format minutes and seconds."""
        parser = EDFParser(str(temp_dir / 'test.edf'))
        assert '5m' in parser._format_duration(305.5)
        assert '5.5s' in parser._format_duration(305.5)

    def test_hours_minutes_seconds(self, temp_dir):
        """_format_duration should format hours, minutes, and seconds."""
        parser = EDFParser(str(temp_dir / 'test.edf'))
        formatted = parser._format_duration(3723.0)  # 1h 2m 3s
        assert '1h' in formatted
        assert '2m' in formatted
        assert '3.0s' in formatted


class TestExtractSubjectInfo:
    """Tests for _extract_subject_info method."""

    @patch('lib.edf_parser.mne.io.read_raw_edf')
    def test_extracts_from_mne_info(self, mock_read_raw, minimal_edf_file):
        """_extract_subject_info should extract from MNE subject_info."""
        mock_raw = MagicMock()
        mock_raw.ch_names = ['CH1']
        mock_raw.times = np.array([0])
        mock_raw.info = {
            'sfreq': 256,
            'meas_date': None,
            'subject_info': {'id': 'SUBJ001', 'sex': 1, 'hand': 1},
        }
        mock_raw.get_channel_types.return_value = ['eeg']
        mock_read_raw.return_value = mock_raw

        parser = EDFParser(str(minimal_edf_file))
        parser.load_edf()
        subject = parser._extract_subject_info()

        assert subject['id'] == 'SUBJ001'

    @patch('lib.edf_parser.mne.io.read_raw_edf')
    def test_falls_back_to_header(self, mock_read_raw, minimal_edf_file):
        """_extract_subject_info should fall back to header info."""
        mock_raw = MagicMock()
        mock_raw.ch_names = ['CH1']
        mock_raw.times = np.array([0])
        mock_raw.info = {'sfreq': 256, 'meas_date': None, 'subject_info': None}
        mock_raw.get_channel_types.return_value = ['eeg']
        mock_read_raw.return_value = mock_raw

        parser = EDFParser(str(minimal_edf_file))
        parser.load_edf()
        subject = parser._extract_subject_info()

        # Should get patient ID from header
        assert 'Test Patient' in subject.get('id', '')


class TestGetChannelTypeSummary:
    """Tests for _get_channel_type_summary method."""

    @patch('lib.edf_parser.mne.io.read_raw_edf')
    def test_summarizes_channel_types(self, mock_read_raw, minimal_edf_file):
        """_get_channel_type_summary should count channel types."""
        mock_raw = MagicMock()
        mock_raw.ch_names = ['CH1', 'CH2', 'CH3']
        mock_raw.times = np.array([0])
        mock_raw.info = {'sfreq': 256, 'meas_date': None, 'subject_info': None}
        mock_raw.get_channel_types.return_value = ['eeg', 'eeg', 'ecg']
        mock_read_raw.return_value = mock_raw

        parser = EDFParser(str(minimal_edf_file))
        parser.load_edf()
        types = parser._get_channel_type_summary()

        assert types.get('eeg') == 2
        assert types.get('ecg') == 1

    @patch('lib.edf_parser.mne.io.read_raw_edf')
    def test_handles_exception(self, mock_read_raw, minimal_edf_file):
        """_get_channel_type_summary should return empty dict on error."""
        mock_raw = MagicMock()
        mock_raw.ch_names = ['CH1']
        mock_raw.times = np.array([0])
        mock_raw.info = {'sfreq': 256, 'meas_date': None, 'subject_info': None}
        mock_raw.get_channel_types.side_effect = Exception("Error")
        mock_read_raw.return_value = mock_raw

        parser = EDFParser(str(minimal_edf_file))
        parser.load_edf()
        types = parser._get_channel_type_summary()

        assert types == {}
