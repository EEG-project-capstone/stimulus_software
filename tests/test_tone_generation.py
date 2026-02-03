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


def test_generate_oddball_sequence():
    """Test pre-generated oddball sequence with sample-accurate timing."""
    # Create AuditoryStimulator with mocked dependencies
    mock_gui = MockGuiCallback()
    stimulator = AuditoryStimulator(gui_callback=mock_gui)

    # Use an event to wait for playback completion
    done_event = threading.Event()

    def on_finish():
        print("Oddball sequence playback finished.")
        done_event.set()

    # Generate the complete oddball sequence
    sample_rate = 44100
    audio_samples, tone_events = stimulator._generate_oddball_sequence(sample_rate)

    total_tones = OddballStimParams.INITIAL_TONES + OddballStimParams.MAIN_TONES
    duration_sec = len(audio_samples) / sample_rate

    print(f"\n=== Pre-generated Oddball Sequence ===")
    print(f"Total tones: {total_tones}")
    print(f"Buffer size: {len(audio_samples)} samples ({duration_sec:.2f}s)")
    print(f"Expected interval: 1000ms (44100 samples)")

    # Verify tone timing
    print(f"\nTone events ({len(tone_events)} tones):")
    rare_count = 0
    for i, event in enumerate(tone_events):
        onset_ms = event['onset_sample'] / sample_rate * 1000
        tone_type = event['type']
        if tone_type == 'rare':
            rare_count += 1
        if i < 5 or i >= len(tone_events) - 3:  # Show first 5 and last 3
            print(f"  Tone {i+1}: {tone_type:8s} @ {onset_ms:7.1f}ms (sample {event['onset_sample']})")
        elif i == 5:
            print(f"  ... ({len(tone_events) - 8} more tones) ...")

    print(f"\nRare tones: {rare_count}/{total_tones} ({rare_count/total_tones*100:.1f}%)")

    # Verify sample-accurate intervals
    print(f"\nVerifying onset-to-onset intervals:")
    intervals = []
    for i in range(1, len(tone_events)):
        interval = tone_events[i]['onset_sample'] - tone_events[i-1]['onset_sample']
        intervals.append(interval)

    expected_interval = sample_rate  # 44100 samples = 1 second
    all_correct = all(i == expected_interval for i in intervals)
    print(f"  Expected: {expected_interval} samples (1000ms)")
    print(f"  All intervals correct: {all_correct}")
    if not all_correct:
        print(f"  Actual intervals: {set(intervals)}")

    # Play the sequence
    print(f"\nPlaying oddball sequence...")
    try:
        stimulator.stream_manager.play(
            samples=audio_samples,
            sample_rate=sample_rate,
            on_finish=on_finish
        )
        # Wait for playback to complete
        timeout = duration_sec + 2.0  # Add 2 seconds buffer
        if not done_event.wait(timeout=timeout):
            print(f"Warning: Playback did not complete within {timeout}s timeout")
    except Exception as e:
        print(f"Error during sequence playback: {e}")


if __name__ == "__main__":
    print("Testing single tone generation...")
    test_generate_and_play_tone()
    print("\n" + "="*50 + "\n")
    print("Testing pre-generated oddball sequence...")
    test_generate_oddball_sequence()