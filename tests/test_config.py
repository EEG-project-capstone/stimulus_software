# tests/test_config.py
"""Tests for config.py - Configuration management."""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.config import Config


class TestConfigPaths:
    """Tests for path generation methods."""

    @patch.object(Config, '_create_directories')
    def test_get_results_path_format(self, _):
        """Results path should contain patient ID and the expected suffix."""
        config = Config()
        path = config.get_results_path('PATIENT123')
        assert 'PATIENT123' in str(path)
        assert '_stimulus_results.csv' in str(path)

    @patch.object(Config, '_create_directories')
    def test_get_results_path_with_custom_date(self, _):
        """Results path should use the supplied date."""
        config = Config()
        path = config.get_results_path('PATIENT123', date='2024-01-15')
        assert '2024-01-15' in str(path)

    @patch.object(Config, '_create_directories')
    def test_get_results_path_empty_patient_raises(self, _):
        """Empty patient ID should raise ValueError."""
        config = Config()
        with pytest.raises(ValueError):
            config.get_results_path('')

    @patch.object(Config, '_create_directories')
    def test_get_edf_path_format(self, _):
        """EDF path should contain patient ID and .edf extension."""
        config = Config()
        path = config.get_edf_path('PATIENT123')
        assert 'PATIENT123' in str(path)
        assert '.edf' in str(path)

    @patch.object(Config, '_create_directories')
    def test_get_edf_path_empty_patient_raises(self, _):
        """Empty patient ID should raise ValueError."""
        config = Config()
        with pytest.raises(ValueError):
            config.get_edf_path('')


class TestConfigSummary:
    """Tests for config summary."""

    @patch.object(Config, '_create_directories')
    def test_get_config_summary_keys(self, _):
        """Config summary should contain expected keys."""
        config = Config()
        summary = config.get_config_summary()
        assert 'current_date' in summary
        assert 'result_dir' in summary
        assert 'edf_dir' in summary
