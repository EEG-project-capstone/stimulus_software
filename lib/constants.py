"""
Central constants for the EEG Stimulus Package.
Consolidates all magic numbers and configuration values.
"""

from enum import Enum, auto
from pathlib import Path
from dataclasses import dataclass

# === PLAYBACK STATE MANAGEMENT ===

class PlaybackState(Enum):
    """Enumeration of possible playback states."""
    EMPTY = auto()      # No patient ID entered
    READY = auto()      # Ready to prepare or play
    PREPARING = auto()  # Generating stimuli
    PLAYING = auto()    # Actively playing stimuli
    PAUSED = auto()     # Playback paused
    STOPPED = auto()    # Playback stopped
    SENDING_SYNC = auto()  # Sending sync pulse


# Valid state transitions
VALID_STATE_TRANSITIONS = {
    PlaybackState.EMPTY: {PlaybackState.READY},
    PlaybackState.READY: {PlaybackState.PREPARING, PlaybackState.PLAYING, PlaybackState.EMPTY, PlaybackState.SENDING_SYNC},
    PlaybackState.PREPARING: {PlaybackState.READY, PlaybackState.STOPPED},
    PlaybackState.PLAYING: {PlaybackState.PAUSED, PlaybackState.STOPPED, PlaybackState.READY},
    PlaybackState.PAUSED: {PlaybackState.PLAYING, PlaybackState.STOPPED, PlaybackState.READY},
    PlaybackState.STOPPED: {PlaybackState.READY, PlaybackState.EMPTY},
    PlaybackState.SENDING_SYNC: {PlaybackState.READY}
}


@dataclass
class StateDisplay:
    """Display properties for each playback state."""
    message: str
    color: str


STATE_DISPLAYS = {
    PlaybackState.EMPTY: StateDisplay("Please enter a patient ID", "red"),
    PlaybackState.READY: StateDisplay("Ready to prepare stimulus", "green"),
    PlaybackState.PREPARING: StateDisplay("Preparing stimulus...", "blue"),
    PlaybackState.PLAYING: StateDisplay("Playing stimulus...", "blue"),
    PlaybackState.PAUSED: StateDisplay("Stimulus paused", "orange"),
    PlaybackState.STOPPED: StateDisplay("Stimulus stopped", "orange"),
    PlaybackState.SENDING_SYNC: StateDisplay("Sending sync pulse...", "blue")
}


# === STIMULUS COUNTS ===

DEFAULT_STIMULUS_COUNTS = {
    'language': 72,
    'command_no_prompt': 3,
    'command_with_prompt': 3,
    'oddball_no_prompt': 4,
    'oddball_with_prompt': 4,
    'loved_one': 50
}


# === STIMULUS TYPE DISPLAY NAMES ===

STIMULUS_TYPE_DISPLAY_NAMES = {
    "language": "Language",
    "right_command": "Right Command",
    "right_command+p": "Right Command + Prompt",
    "left_command": "Left Command",
    "left_command+p": "Left Command + Prompt",
    "oddball": "Oddball",
    "oddball+p": "Oddball + Prompt",
    "loved_one_voice": "Loved One Voice",
    "control": "Control Statement",
    "session_note": "Session Note",
    "manual_sync_pulse": "Manual Sync Pulse",
    "sync_detection": "Sync Detection"
}


# === BUTTON ICONS ===

class ButtonIcons:
    """Button icon sizing constants."""
    PLAY_SUBSAMPLE = (15, 15)
    PAUSE_SUBSAMPLE = (15, 15)
    STOP_SUBSAMPLE = (6, 6)


# === LAYOUT ===

class Layout:
    """Layout and spacing constants."""
    WINDOW_SIZE = (1050, 830)
    MAIN_PADDING = 10
    STIMULUS_LIST_HEIGHT = 12
    NOTES_TEXT_HEIGHT = 10
    LOG_TEXT_WIDTH = 40


# === TREEVIEW TAGS ===

TREEVIEW_TAGS = {
    'COMPLETED': 'completed',
    'IN_PROGRESS': 'inprogress', 
    'PENDING': 'pending'
}


# === COMMAND STIMULUS PARAMETERS ===

class CommandStimParams:
    """Parameters for motor command stimuli."""
    TOTAL_CYCLES = 8
    KEEP_PAUSE_MS = 10000  # 10 seconds
    STOP_PAUSE_MS = 10000  # 10 seconds
    PROMPT_DELAY_MS = 2000  # 2 seconds


# === ODDBALL STIMULUS PARAMETERS ===

class OddballStimParams:
    """Parameters for oddball stimuli."""
    INITIAL_TONES = 5
    MAIN_TONES = 20
    TONE_DURATION_MS = 100
    INTER_TONE_INTERVAL_MS = 900
    STANDARD_FREQ = 1000  # Hz
    RARE_FREQ = 2000      # Hz
    RARE_PROBABILITY = 0.2
    PROMPT_DELAY_MS = 2000


# === SYNC PULSE PARAMETERS ===

class SyncPulseParams:
    """Parameters for EEG sync pulse.

    The sync pulse is designed to be easily detectable in EEG recordings.
    A longer, lower-frequency square wave is more distinctive and easier
    to detect algorithmically than a short high-frequency pulse.
    """
    FREQUENCY = 100        # Hz - square wave frequency (low for distinctiveness)
    DURATION_MS = 1000     # ms - pulse duration (1 second for clear detection)
    SAMPLE_RATE = 44100    # Hz


# === AUDIO PARAMETERS ===

class AudioParams:
    """Audio processing parameters."""
    SAMPLE_RATE = 44100
    STREAM_LATENCY = 'low'
    MAX_AMPLITUDE = 32767  # int16 max
    BUFFER_DTYPE = 'int16'


# === LANGUAGE STIMULUS PARAMETERS ===

class LanguageStimParams:
    """Parameters for language stimuli."""
    SENTENCES_PER_STIMULUS = 12


# === INTER-STIMULUS TIMING ===

class TimingParams:
    """Timing parameters for stimulus presentation."""
    INTER_STIMULUS_MIN_MS = 1200  # Minimum delay between stimuli
    INTER_STIMULUS_MAX_MS = 2200  # Maximum delay between stimuli
    CALLBACK_RETRY_DELAY_MS = 100  # Delay for retry callbacks


# === FILE PATHS ===

class FilePaths:
    """Default file paths and directories."""
    PATIENT_DATA_DIR = Path("patient_data")
    RESULTS_DIR = PATIENT_DATA_DIR / "results"
    EDFS_DIR = PATIENT_DATA_DIR / "edfs"
    AUDIO_DATA_DIR = Path("audio_data")
    SENTENCES_DIR = AUDIO_DATA_DIR / "sentences"
    STATIC_AUDIO_DIR = AUDIO_DATA_DIR / "static"
    PROMPTS_DIR = AUDIO_DATA_DIR / "prompts"


# === LOGGING PARAMETERS ===

class LoggingParams:
    """Logging configuration parameters."""
    LOG_FILE_NAME = 'eeg_stimulus_app.log'
    MAX_LOG_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
    BACKUP_COUNT = 5
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


# === EEG SYNC DETECTION PARAMETERS ===

class SyncDetectionParams:
    """Parameters for EEG sync point detection."""
    SEARCH_DURATION_SEC = 300  # Search first 5 minutes
    THRESHOLD_STD_MULTIPLIER = 3  # Standard deviations above baseline
    BASELINE_WINDOW_FRACTION = 0.1  # Use first 10% for baseline
    PREVIEW_DURATION_SEC = 60  # Show 60s preview


# === CLINICAL ASSESSMENT SCALES ===

CPC_SCALE = [
    "",
    "CPC 1: No neurological deficit",
    "CPC 2: Mild to moderate dysfunction",
    "CPC 3: Severe dysfunction",
    "CPC 4: Coma",
    "CPC 5: Brain death",
]

GOSE_SCALE = [
    "",
    "GOSE 1: Dead",
    "GOSE 2: Vegetative state",
    "GOSE 3: Lower severe disability",
    "GOSE 4: Upper severe disability",
    "GOSE 5: Lower moderate disability",
    "GOSE 6: Upper moderate disability",
    "GOSE 7: Lower good recovery",
    "GOSE 8: Upper good recovery",
]


# === RETRY PARAMETERS ===

class RetryParams:
    """Parameters for retry logic."""
    MAX_RETRIES = 3
    RETRY_DELAY_MS = 100


# === STIMULUS HANDLER IDENTIFIERS ===

class StimHandlerTypes:
    """Identifiers for stimulus handler types."""
    LANGUAGE = 'language'
    COMMAND = 'command'
    ODDBALL = 'oddball'
    VOICE = 'voice'


# === STIMULUS TYPE IDENTIFIERS ===

class StimTypes:
    """Identifiers for stimulus types."""
    LANGUAGE = 'language'
    RIGHT_COMMAND = 'right_command'
    RIGHT_COMMAND_PROMPT = 'right_command+p'
    LEFT_COMMAND = 'left_command'
    LEFT_COMMAND_PROMPT = 'left_command+p'
    ODDBALL = 'oddball'
    ODDBALL_PROMPT = 'oddball+p'
    LOVED_ONE_VOICE = 'loved_one_voice'
    CONTROL = 'control'
    SESSION_NOTE = 'session_note'
    MANUAL_SYNC_PULSE = 'manual_sync_pulse'
    SYNC_DETECTION = 'sync_detection'


# === GENDER OPTIONS ===

class GenderOptions:
    """Gender options for voice stimuli."""
    MALE = 'Male'
    FEMALE = 'Female'


# === CHANNEL NAMES (for EEG analysis) ===

DEFAULT_EEG_CHANNELS = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'FT9', 'FT10', 'T7', 'C3', 'Cz', 'C4', 'T8',
    'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2', 'Fpz'
]