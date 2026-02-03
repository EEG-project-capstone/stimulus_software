import sys
import os
import threading

# Add the workspace directory to the Python path
workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, workspace_dir)

import numpy as np
from lib.auditory_stimulator import AuditoryStimulator
from lib.constants import OddballStimParams


class MockStims:
    """Mock stims object for testing."""
    def __init__(self):
        self.current_stim_index = 0
        self.stim_dictionary = [{'type': 'oddball', 'status': 'pending'}]


class MockResultsManager:
    """Mock results manager for testing."""
    def append_result(self, *args, **kwargs):
        pass

    def append_sync_pulse(self, *args, **kwargs):
        pass


class MockGuiCallback:
    """Mock GUI callback for testing AuditoryStimulator."""
    def __init__(self):
        self.stims = MockStims()
        self.config = type('MockConfig', (), {})()
        self.results_manager = MockResultsManager()


def test_generate_and_play_tone():
    """Test tone generation and playback using AuditoryStimulator."""
    # Create AuditoryStimulator with mocked dependencies
    mock_gui = MockGuiCallback()
    stimulator = AuditoryStimulator(gui_callback=mock_gui)

    # Use an event to wait for playback completion
    done_event = threading.Event()

    def on_finish():
        print("Test tone playback finished.")
        done_event.set()

    # Generate test tone using AuditoryStimulator
    frequency = OddballStimParams.STANDARD_FREQ
    duration_ms = OddballStimParams.TONE_DURATION_MS
    tone_samples = stimulator._generate_tone(frequency, duration_ms)

    print(f"Generated tone: {frequency}Hz, {duration_ms}ms, "
          f"{len(tone_samples)} samples (includes {OddballStimParams.TONE_PADDING_MS}ms padding)")

    # Play the generated tone using AuditoryStimulator's stream manager
    try:
        print("Playing test tone...")
        stimulator.stream_manager.play(
            samples=tone_samples,
            sample_rate=44100,
            on_finish=on_finish
        )
        # Wait for playback to complete (timeout after 5 seconds)
        if not done_event.wait(timeout=5.0):
            print("Warning: Playback did not complete within timeout")
    except Exception as e:
        print(f"Error during tone playback: {e}")


if __name__ == "__main__":
    test_generate_and_play_tone()