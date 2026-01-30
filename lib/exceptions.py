# lib/exceptions.py

"""
Custom exceptions for the EEG Stimulus Package.
Provides specific error types for better error handling and user feedback.
"""


class StimulusError(Exception):
    """Base exception for stimulus-related errors."""
    pass


class StimulusConfigurationError(StimulusError):
    """Raised when stimulus configuration is invalid."""
    pass


class StimulusGenerationError(StimulusError):
    """Raised when stimulus generation fails."""
    pass


class AudioError(Exception):
    """Base exception for audio-related errors."""
    pass


class AudioDeviceError(AudioError):
    """Raised when audio device cannot be accessed or configured."""
    pass


class AudioFileError(AudioError):
    """Raised when audio file cannot be loaded or processed."""
    pass


class AudioPlaybackError(AudioError):
    """Raised when audio playback fails."""
    pass


class ConfigError(Exception):
    """Base exception for configuration errors."""
    pass


class ConfigFileError(ConfigError):
    """Raised when configuration file is missing or invalid."""
    pass


class ConfigValidationError(ConfigError):
    """Raised when configuration validation fails."""
    pass


class FileManagementError(Exception):
    """Base exception for file management errors."""
    pass


class ResultsFileError(FileManagementError):
    """Raised when results file operations fail."""
    pass


class EDFFileError(FileManagementError):
    """Raised when EDF file operations fail."""
    pass


class StateError(Exception):
    """Raised when invalid state transitions are attempted."""
    pass
