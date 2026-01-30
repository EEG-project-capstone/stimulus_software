import sys
import os

# Add the workspace directory to the Python path
workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, workspace_dir)

import numpy as np
from lib.auditory_stimulator import AuditoryStimulator
from lib.constants import OddballStimParams
from lib.audio_stream_manager import AudioStreamManager

class MockStims:
    def __init__(self):
        self.current_stim_index = 0  # Mock the current_stim_index attribute
        self.stim_dictionary = []  # Mock the stim_dictionary attribute

class MockGuiCallback:
    def __init__(self):
        self.stims = MockStims()  # Provide a mock stims object
        self.config = type('MockConfig', (object,), {})()  # Mock the config attribute
        self.results_manager = type('MockResultsManager', (object,), {})()  # Mock the results_manager attribute

def test_generate_and_play_tone():
    # Initialize the audio stream manager
    stream_manager = AudioStreamManager()

    # Generate a test tone
    mock_gui_callback = MockGuiCallback()
    stimulator = AuditoryStimulator(gui_callback=mock_gui_callback)
    frequency = OddballStimParams.STANDARD_FREQ
    duration_ms = OddballStimParams.TONE_DURATION_MS
    tone_samples = stimulator._generate_tone(frequency, duration_ms)

    # Play the generated tone
    try:
        print("Playing test tone...")
        stream_manager.play(
            samples=tone_samples,
            sample_rate=44100,
            on_finish=lambda: print("Test tone playback finished.")
        )
    except Exception as e:
        print(f"Error during tone playback: {e}")

if __name__ == "__main__":
    test_generate_and_play_tone()