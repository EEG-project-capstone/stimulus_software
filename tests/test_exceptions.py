# tests/test_exceptions.py
"""Tests for exceptions.py - Custom exception classes."""

import pytest
import sys
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.exceptions import (
    StimulusError,
    StimulusConfigurationError,
    StimulusGenerationError,
    AudioError,
    AudioDeviceError,
    AudioFileError,
    AudioPlaybackError,
    ConfigError,
    ConfigFileError,
    ConfigValidationError,
    FileManagementError,
    ResultsFileError,
    EDFFileError,
    StateError
)


class TestExceptionHierarchy:
    """Tests for exception class hierarchy."""

    def test_stimulus_error_is_base_exception(self):
        """StimulusError should be a base Exception."""
        assert issubclass(StimulusError, Exception)

    def test_stimulus_configuration_error_inherits_stimulus_error(self):
        """StimulusConfigurationError should inherit from StimulusError."""
        assert issubclass(StimulusConfigurationError, StimulusError)

    def test_stimulus_generation_error_inherits_stimulus_error(self):
        """StimulusGenerationError should inherit from StimulusError."""
        assert issubclass(StimulusGenerationError, StimulusError)

    def test_audio_error_is_base_exception(self):
        """AudioError should be a base Exception."""
        assert issubclass(AudioError, Exception)

    def test_audio_device_error_inherits_audio_error(self):
        """AudioDeviceError should inherit from AudioError."""
        assert issubclass(AudioDeviceError, AudioError)

    def test_audio_file_error_inherits_audio_error(self):
        """AudioFileError should inherit from AudioError."""
        assert issubclass(AudioFileError, AudioError)

    def test_audio_playback_error_inherits_audio_error(self):
        """AudioPlaybackError should inherit from AudioError."""
        assert issubclass(AudioPlaybackError, AudioError)

    def test_config_error_is_base_exception(self):
        """ConfigError should be a base Exception."""
        assert issubclass(ConfigError, Exception)

    def test_config_file_error_inherits_config_error(self):
        """ConfigFileError should inherit from ConfigError."""
        assert issubclass(ConfigFileError, ConfigError)

    def test_config_validation_error_inherits_config_error(self):
        """ConfigValidationError should inherit from ConfigError."""
        assert issubclass(ConfigValidationError, ConfigError)

    def test_file_management_error_is_base_exception(self):
        """FileManagementError should be a base Exception."""
        assert issubclass(FileManagementError, Exception)

    def test_results_file_error_inherits_file_management_error(self):
        """ResultsFileError should inherit from FileManagementError."""
        assert issubclass(ResultsFileError, FileManagementError)

    def test_edf_file_error_inherits_file_management_error(self):
        """EDFFileError should inherit from FileManagementError."""
        assert issubclass(EDFFileError, FileManagementError)

    def test_state_error_is_base_exception(self):
        """StateError should be a base Exception."""
        assert issubclass(StateError, Exception)


class TestExceptionRaising:
    """Tests for raising and catching custom exceptions."""

    def test_stimulus_error_can_be_raised(self):
        """StimulusError should be raisable with message."""
        with pytest.raises(StimulusError, match="test stimulus error"):
            raise StimulusError("test stimulus error")

    def test_audio_device_error_can_be_raised(self):
        """AudioDeviceError should be raisable with message."""
        with pytest.raises(AudioDeviceError, match="no audio device found"):
            raise AudioDeviceError("no audio device found")

    def test_config_validation_error_can_be_raised(self):
        """ConfigValidationError should be raisable with message."""
        with pytest.raises(ConfigValidationError, match="missing required key"):
            raise ConfigValidationError("missing required key")

    def test_results_file_error_can_be_raised(self):
        """ResultsFileError should be raisable with message."""
        with pytest.raises(ResultsFileError, match="cannot write to file"):
            raise ResultsFileError("cannot write to file")

    def test_state_error_can_be_raised(self):
        """StateError should be raisable with message."""
        with pytest.raises(StateError, match="invalid state transition"):
            raise StateError("invalid state transition")


class TestExceptionCatching:
    """Tests for catching exceptions in hierarchy."""

    def test_catch_stimulus_subclass_with_base(self):
        """StimulusConfigurationError should be catchable as StimulusError."""
        caught = False
        try:
            raise StimulusConfigurationError("config invalid")
        except StimulusError:
            caught = True
        assert caught is True

    def test_catch_audio_subclass_with_base(self):
        """AudioPlaybackError should be catchable as AudioError."""
        caught = False
        try:
            raise AudioPlaybackError("playback failed")
        except AudioError:
            caught = True
        assert caught is True

    def test_catch_config_subclass_with_base(self):
        """ConfigFileError should be catchable as ConfigError."""
        caught = False
        try:
            raise ConfigFileError("file not found")
        except ConfigError:
            caught = True
        assert caught is True

    def test_catch_file_management_subclass_with_base(self):
        """EDFFileError should be catchable as FileManagementError."""
        caught = False
        try:
            raise EDFFileError("invalid EDF format")
        except FileManagementError:
            caught = True
        assert caught is True


class TestExceptionMessages:
    """Tests for exception message handling."""

    def test_exception_preserves_message(self):
        """Exception should preserve the error message."""
        error = AudioPlaybackError("stream initialization failed")
        assert str(error) == "stream initialization failed"

    def test_exception_with_empty_message(self):
        """Exception with empty message should work."""
        error = ConfigError("")
        assert str(error) == ""

    def test_exception_with_multiline_message(self):
        """Exception with multiline message should preserve newlines."""
        msg = "Error occurred:\n- Item 1\n- Item 2"
        error = StimulusGenerationError(msg)
        assert str(error) == msg
