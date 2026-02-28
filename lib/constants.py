"""
Central constants for the EEG Stimulus Package.
Consolidates all magic numbers and configuration values.
"""

from enum import Enum, auto
from dataclasses import dataclass
from pathlib import Path


# =============================================================================
# STIMULUS PARAMETERS
# Values most likely to be tuned by researchers
# =============================================================================

class CommandStimParams:
    """Parameters for motor command stimuli."""
    TOTAL_CYCLES = 8
    KEEP_PAUSE_MS = 10000  # 10 seconds
    STOP_PAUSE_MS = 10000  # 10 seconds
    PROMPT_DELAY_MS = 2000  # 2 seconds


class OddballStimParams:
    """Parameters for oddball stimuli."""
    INITIAL_TONES = 5
    MAIN_TONES = 20
    TONE_DURATION_MS = 20          # Full-amplitude duration (envelope adds to this)
    STANDARD_FREQ = 1000           # Hz
    RARE_FREQ = 2000               # Hz
    RARE_PROBABILITY = 0.2
    PROMPT_DELAY_MS = 2000
    TONE_AMPLITUDE = 1.0           # 0.0 to 1.0
    TONE_ENVELOPE_MS = 5           # Fade in/out duration to eliminate clicks
    # Padding to absorb stream initialization latency on slower audio backends (e.g., ChromeOS)
    TONE_PADDING_MS = 200          # Must be >= AudioParams.STREAM_LATENCY


class LanguageStimParams:
    """Parameters for language stimuli."""
    SENTENCES_PER_STIMULUS = 12


class TimingParams:
    """Inter-stimulus timing parameters."""
    INTER_STIMULUS_JITTER = False       # If False, use INTER_STIMULUS_FIXED_MS instead
    INTER_STIMULUS_FIXED_MS = 1500      # Used when INTER_STIMULUS_JITTER is False
    INTER_STIMULUS_MIN_MS = 1200        # Minimum delay between stimuli (jitter mode)
    INTER_STIMULUS_MAX_MS = 2200        # Maximum delay between stimuli (jitter mode)
    CALLBACK_RETRY_DELAY_MS = 100       # Delay for retry callbacks


DEFAULT_STIMULUS_COUNTS = {
    'language': 72,
    'command_no_prompt': 3,
    'command_with_prompt': 3,
    'oddball_no_prompt': 4,
    'oddball_with_prompt': 4,
    'familiar_voice': 50
}


class AudioParams:
    """Audio processing parameters."""
    SAMPLE_RATE = 44100
    STREAM_LATENCY = 0.1           # Float value for compatibility
    MAX_AMPLITUDE = 32767          # int16 max
    BUFFER_DTYPE = 'int16'


# =============================================================================
# FILE PATHS
# All paths relative to the project root. Change these if you rename files.
# =============================================================================

class FilePaths:
    """Standard file paths relative to the project root."""
    # Output directories (created automatically on startup)
    RESULTS_DIR = Path("patient_data/results")
    EDFS_DIR    = Path("patient_data/edfs")

    # Audio input directories
    SENTENCES_DIR  = Path("audio_data/sentences")
    FAMILIAR_DIR   = Path("audio_data/static")

    # Command audio files
    RIGHT_KEEP_AUDIO = Path("audio_data/static/right_keep.mp3")
    RIGHT_STOP_AUDIO = Path("audio_data/static/right_stop.mp3")
    LEFT_KEEP_AUDIO  = Path("audio_data/static/left_keep.mp3")
    LEFT_STOP_AUDIO  = Path("audio_data/static/left_stop.mp3")

    # Control / voice audio
    CONTROL_STATEMENTS_DIR = Path("audio_data/control_statements")

    # Prompt audio
    MOTOR_PROMPT  = Path("audio_data/prompts/motorcommandprompt.wav")
    ODDBALL_PROMPT = Path("audio_data/prompts/oddballprompt.wav")


# =============================================================================
# EEG SYNC PARAMETERS
# =============================================================================

class SyncPulseParams:
    """Parameters for the EEG sync pulse.

    The sync pulse is designed to be easily detectable in EEG recordings.
    A longer, lower-frequency square wave is more distinctive and easier
    to detect algorithmically than a short high-frequency pulse.
    """
    FREQUENCY = 100        # Hz - square wave frequency (low for distinctiveness)
    DURATION_MS = 1000     # ms - pulse duration (1 second for clear detection)
    SAMPLE_RATE = 44100    # Hz


# =============================================================================
# STIMULUS TYPE IDENTIFIERS
# =============================================================================


# Names of the unfamiliar control speakers in audio_data/control_statements/
# (files are named <Name>_normalized.wav)
MALE_CONTROL_VOICES = ['Adam', 'Alex', 'Chris', 'Peter']
FEMALE_CONTROL_VOICES = ['Hannah', 'Jennifer', 'Saline', 'Sarah']


STIMULUS_TYPE_DISPLAY_NAMES = {
    "language": "Language",
    "right_command": "Right Command",
    "right_command+p": "Right Command + Prompt",
    "left_command": "Left Command",
    "left_command+p": "Left Command + Prompt",
    "oddball": "Oddball",
    "oddball+p": "Oddball + Prompt",
    "familiar": "Familiar Voice",
    "unfamiliar": "Unfamiliar Voice",
    "session_note": "Session Note",
    "manual_sync_pulse": "Manual Sync Pulse",
    "sync_detection": "Sync Detection"
}


# =============================================================================
# APPLICATION STATE MACHINE
# =============================================================================

class PlaybackState(Enum):
    """Enumeration of possible playback states."""
    EMPTY = auto()          # No patient ID entered
    READY = auto()          # Ready to prepare or play
    PREPARING = auto()      # Generating stimuli
    PLAYING = auto()        # Actively playing stimuli
    PAUSED = auto()         # Playback paused
    STOPPED = auto()        # Playback stopped
    SENDING_SYNC = auto()   # Sending sync pulse


VALID_STATE_TRANSITIONS = {
    PlaybackState.EMPTY:        {PlaybackState.READY},
    PlaybackState.READY:        {PlaybackState.PREPARING, PlaybackState.PLAYING, PlaybackState.EMPTY, PlaybackState.SENDING_SYNC},
    PlaybackState.PREPARING:    {PlaybackState.READY, PlaybackState.STOPPED},
    PlaybackState.PLAYING:      {PlaybackState.PAUSED, PlaybackState.STOPPED, PlaybackState.READY},
    PlaybackState.PAUSED:       {PlaybackState.PLAYING, PlaybackState.STOPPED, PlaybackState.READY},
    PlaybackState.STOPPED:      {PlaybackState.READY, PlaybackState.EMPTY},
    PlaybackState.SENDING_SYNC: {PlaybackState.READY}
}


@dataclass
class StateDisplay:
    """Display properties for a playback state."""
    message: str
    color: str


STATE_DISPLAYS = {
    PlaybackState.EMPTY:        StateDisplay("Please enter a patient ID", "red"),
    PlaybackState.READY:        StateDisplay("Ready to prepare stimulus", "green"),
    PlaybackState.PREPARING:    StateDisplay("Preparing stimulus...", "blue"),
    PlaybackState.PLAYING:      StateDisplay("Playing stimulus...", "blue"),
    PlaybackState.PAUSED:       StateDisplay("Stimulus paused", "orange"),
    PlaybackState.STOPPED:      StateDisplay("Stimulus stopped", "orange"),
    PlaybackState.SENDING_SYNC: StateDisplay("Sending sync pulse...", "blue")
}


# =============================================================================
# UI / LAYOUT
# =============================================================================

class Layout:
    """Layout and spacing constants."""
    WINDOW_SIZE = (1050, 830)
    MAIN_PADDING = 10
    STIMULUS_LIST_HEIGHT = 12
    NOTES_TEXT_HEIGHT = 10
    LOG_TEXT_WIDTH = 40


# =============================================================================
# SYSTEM / INFRASTRUCTURE
# =============================================================================

class LoggingParams:
    """Logging configuration parameters."""
    LOG_FILE_NAME = 'eeg_stimulus_app.log'
    MAX_LOG_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
    BACKUP_COUNT = 5
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


