"""
Tests for button controls, state management, and playback control.
Tests the play, pause, stop functionality without requiring a GUI.
"""

import sys
import os
import threading
import time

# Add the workspace directory to the Python path
workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, workspace_dir)

import numpy as np
from lib.state_manager import StateManager
from lib.constants import PlaybackState, VALID_STATE_TRANSITIONS, OddballStimParams
from lib.audio_stream_manager import AudioStreamManager
from lib.auditory_stimulator import AuditoryStimulator
from lib.exceptions import StateError


# ==============================================================================
# Mock Classes for Testing
# ==============================================================================

class MockStims:
    """Mock stims object for testing."""
    def __init__(self):
        self.current_stim_index = 0
        self.stim_dictionary = [
            {'type': 'oddball', 'status': 'pending'},
            {'type': 'oddball', 'status': 'pending'},
        ]


class MockResultsManager:
    """Mock results manager for testing."""
    def append_result(self, *args, **kwargs):
        pass

    def append_sync_pulse(self, *args, **kwargs):
        pass


class MockRoot:
    """Mock tkinter root for testing."""
    def __init__(self):
        self._after_callbacks = {}
        self._callback_id = 0
        self._cancelled = set()

    def after(self, delay_ms, callback):
        """Schedule a callback (returns immediately for testing)."""
        self._callback_id += 1
        cb_id = f"after#{self._callback_id}"
        self._after_callbacks[cb_id] = (delay_ms, callback)
        return cb_id

    def after_cancel(self, cb_id):
        """Cancel a scheduled callback."""
        self._cancelled.add(cb_id)
        if cb_id in self._after_callbacks:
            del self._after_callbacks[cb_id]

    def execute_pending(self):
        """Execute all pending callbacks (for testing)."""
        for cb_id, (delay, callback) in list(self._after_callbacks.items()):
            if cb_id not in self._cancelled:
                try:
                    callback()
                except Exception as e:
                    print(f"Callback error: {e}")
        self._after_callbacks.clear()


class MockGuiCallback:
    """Mock GUI callback for testing AuditoryStimulator."""
    def __init__(self):
        self.stims = MockStims()
        self.config = type('MockConfig', (), {'current_date': '2024-01-01'})()
        self.results_manager = MockResultsManager()
        self.state_manager = StateManager(PlaybackState.READY)
        self.root = MockRoot()
        self._status_updates = []
        self._errors = []

    def get_patient_id(self):
        return "TEST_PATIENT"

    def update_stim_list_status(self):
        pass

    def playback_complete(self):
        self._status_updates.append('complete')

    def playback_error(self, msg):
        self._errors.append(msg)


# ==============================================================================
# State Manager Tests
# ==============================================================================

def test_state_manager_initial_state():
    """Test state manager initializes correctly."""
    print("\n=== Test: State Manager Initial State ===")

    sm = StateManager(PlaybackState.EMPTY)
    assert sm.state == PlaybackState.EMPTY, f"Expected EMPTY, got {sm.state}"
    print("✓ Initial state is EMPTY")

    sm2 = StateManager(PlaybackState.READY)
    assert sm2.state == PlaybackState.READY, f"Expected READY, got {sm2.state}"
    print("✓ Can initialize with READY state")


def test_state_manager_valid_transitions():
    """Test valid state transitions."""
    print("\n=== Test: Valid State Transitions ===")

    sm = StateManager(PlaybackState.EMPTY)

    # EMPTY -> READY (valid)
    sm.transition_to(PlaybackState.READY)
    assert sm.state == PlaybackState.READY
    print("✓ EMPTY -> READY")

    # READY -> PLAYING (valid)
    sm.transition_to(PlaybackState.PLAYING)
    assert sm.state == PlaybackState.PLAYING
    print("✓ READY -> PLAYING")

    # PLAYING -> PAUSED (valid)
    sm.transition_to(PlaybackState.PAUSED)
    assert sm.state == PlaybackState.PAUSED
    print("✓ PLAYING -> PAUSED")

    # PAUSED -> PLAYING (valid)
    sm.transition_to(PlaybackState.PLAYING)
    assert sm.state == PlaybackState.PLAYING
    print("✓ PAUSED -> PLAYING")

    # PLAYING -> STOPPED (valid)
    sm.transition_to(PlaybackState.STOPPED)
    assert sm.state == PlaybackState.STOPPED
    print("✓ PLAYING -> STOPPED")

    # STOPPED -> READY (valid)
    sm.transition_to(PlaybackState.READY)
    assert sm.state == PlaybackState.READY
    print("✓ STOPPED -> READY")


def test_state_manager_invalid_transitions():
    """Test invalid state transitions raise errors."""
    print("\n=== Test: Invalid State Transitions ===")

    sm = StateManager(PlaybackState.EMPTY)

    # EMPTY -> PLAYING (invalid - must go through READY)
    try:
        sm.transition_to(PlaybackState.PLAYING)
        print("✗ Should have raised StateError for EMPTY -> PLAYING")
    except StateError:
        print("✓ EMPTY -> PLAYING raises StateError")

    # EMPTY -> PAUSED (invalid)
    try:
        sm.transition_to(PlaybackState.PAUSED)
        print("✗ Should have raised StateError for EMPTY -> PAUSED")
    except StateError:
        print("✓ EMPTY -> PAUSED raises StateError")

    # Setup for more tests
    sm.transition_to(PlaybackState.READY)

    # READY -> PAUSED (invalid - must be playing first)
    try:
        sm.transition_to(PlaybackState.PAUSED)
        print("✗ Should have raised StateError for READY -> PAUSED")
    except StateError:
        print("✓ READY -> PAUSED raises StateError")


def test_state_manager_helper_methods():
    """Test state manager helper methods."""
    print("\n=== Test: State Manager Helper Methods ===")

    sm = StateManager(PlaybackState.EMPTY)

    assert not sm.is_playing()
    assert not sm.is_paused()
    assert not sm.is_ready()
    print("✓ EMPTY: not playing, not paused, not ready")

    sm.transition_to(PlaybackState.READY)
    assert not sm.is_playing()
    assert not sm.is_paused()
    assert sm.is_ready()
    print("✓ READY: not playing, not paused, is ready")

    sm.transition_to(PlaybackState.PLAYING)
    assert sm.is_playing()
    assert not sm.is_paused()
    assert not sm.is_ready()
    assert sm.is_active()
    print("✓ PLAYING: is playing, not paused, not ready, is active")

    sm.transition_to(PlaybackState.PAUSED)
    assert not sm.is_playing()
    assert sm.is_paused()
    assert not sm.is_ready()
    assert sm.is_active()
    print("✓ PAUSED: not playing, is paused, not ready, is active")


def test_state_manager_listeners():
    """Test state change listeners."""
    print("\n=== Test: State Manager Listeners ===")

    sm = StateManager(PlaybackState.EMPTY)
    transitions = []

    def listener(old_state, new_state):
        transitions.append((old_state, new_state))

    sm.add_listener(listener)

    sm.transition_to(PlaybackState.READY)
    sm.transition_to(PlaybackState.PLAYING)
    sm.transition_to(PlaybackState.PAUSED)

    assert len(transitions) == 3
    assert transitions[0] == (PlaybackState.EMPTY, PlaybackState.READY)
    assert transitions[1] == (PlaybackState.READY, PlaybackState.PLAYING)
    assert transitions[2] == (PlaybackState.PLAYING, PlaybackState.PAUSED)
    print(f"✓ Listener received {len(transitions)} transitions")

    sm.remove_listener(listener)
    sm.transition_to(PlaybackState.PLAYING)
    assert len(transitions) == 3  # No new transition recorded
    print("✓ Listener removal works")


# ==============================================================================
# Audio Stream Manager Tests
# ==============================================================================

def test_audio_stream_manager_play_stop():
    """Test audio stream manager play and stop."""
    print("\n=== Test: Audio Stream Manager Play/Stop ===")

    manager = AudioStreamManager()
    done_event = threading.Event()
    callback_called = [False]

    def on_finish():
        callback_called[0] = True
        done_event.set()

    # Generate a short test tone (50ms for faster tests)
    sample_rate = 44100
    duration_sec = 0.05
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), False)
    samples = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16).reshape(-1, 1)

    # Play
    manager.play(samples, sample_rate, on_finish)
    print("✓ Stream started")

    # Wait for natural completion
    completed = done_event.wait(timeout=2.0)
    time.sleep(0.1)  # Small delay for cleanup

    assert completed and callback_called[0]
    print("✓ Callback was called on completion")

    assert not manager.is_playing()
    print("✓ Stream stopped after completion")


def test_audio_stream_manager_stop_during_play():
    """Test stopping audio during playback."""
    print("\n=== Test: Audio Stream Manager Stop During Play ===")

    manager = AudioStreamManager()
    callback_called = [False]

    def on_finish():
        callback_called[0] = True

    # Generate a 500ms tone (enough time to stop mid-playback)
    sample_rate = 44100
    duration_sec = 0.5
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), False)
    samples = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16).reshape(-1, 1)

    # Play
    manager.play(samples, sample_rate, on_finish)
    print("✓ Stream started")

    # Stop after a short delay
    time.sleep(0.05)
    manager.stop()
    time.sleep(0.1)  # Allow cleanup

    assert not manager.is_playing()
    print("✓ Stream stopped")

    # Callback should NOT be called when intentionally stopped
    assert not callback_called[0]
    print("✓ Callback was NOT called during intentional stop (correct behavior)")


def test_audio_stream_manager_rapid_play_stop():
    """Test rapid play/stop cycles don't cause issues."""
    print("\n=== Test: Audio Stream Manager Rapid Play/Stop ===")

    manager = AudioStreamManager()

    # Generate a short test tone (100ms)
    sample_rate = 44100
    duration_sec = 0.1
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), False)
    samples = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16).reshape(-1, 1)

    # Rapid play/stop cycles
    for i in range(5):
        manager.play(samples, sample_rate)
        time.sleep(0.02)
        manager.stop()
        time.sleep(0.02)

    assert not manager.is_playing()
    print(f"✓ Completed 5 rapid play/stop cycles without errors")


# ==============================================================================
# Auditory Stimulator Integration Tests
# ==============================================================================

def test_auditory_stimulator_toggle_pause():
    """Test toggle pause behavior."""
    print("\n=== Test: Auditory Stimulator Toggle Pause ===")

    mock_gui = MockGuiCallback()
    stimulator = AuditoryStimulator(gui_callback=mock_gui)

    # Start in PLAYING state
    mock_gui.state_manager.transition_to(PlaybackState.PLAYING)

    # Simulate starting an oddball handler
    handler = stimulator.handlers['oddball']
    handler.is_active = True

    # Generate and start playing audio
    sample_rate = 44100
    audio_samples, _ = stimulator._generate_oddball_sequence(sample_rate)

    # Play
    stimulator.stream_manager.play(audio_samples, sample_rate)
    assert stimulator.stream_manager.is_playing()
    print("✓ Audio is playing")

    # Transition to PAUSED
    mock_gui.state_manager.transition_to(PlaybackState.PAUSED)

    # Toggle pause (should stop audio and reset handlers)
    stimulator.toggle_pause()
    time.sleep(0.1)

    assert not stimulator.stream_manager.is_playing()
    print("✓ Audio stopped on pause")

    assert not handler.is_active
    print("✓ Handler deactivated on pause")


def test_auditory_stimulator_stop():
    """Test stop stimulus behavior."""
    print("\n=== Test: Auditory Stimulator Stop ===")

    mock_gui = MockGuiCallback()
    stimulator = AuditoryStimulator(gui_callback=mock_gui)

    # Start in PLAYING state
    mock_gui.state_manager.transition_to(PlaybackState.PLAYING)

    # Simulate an active handler
    handler = stimulator.handlers['oddball']
    handler.is_active = True

    # Generate and play audio
    sample_rate = 44100
    audio_samples, _ = stimulator._generate_oddball_sequence(sample_rate)
    stimulator.stream_manager.play(audio_samples, sample_rate)

    # Stop
    stimulator.stop_stimulus()
    time.sleep(0.1)

    assert not stimulator.stream_manager.is_playing()
    print("✓ Audio stopped")

    assert not handler.is_active
    print("✓ Handler stopped")

    assert stimulator.stims.current_stim_index == 0
    print("✓ Stim index reset")


def test_oddball_sequence_generation():
    """Test oddball sequence is generated correctly."""
    print("\n=== Test: Oddball Sequence Generation ===")

    mock_gui = MockGuiCallback()
    stimulator = AuditoryStimulator(gui_callback=mock_gui)

    sample_rate = 44100
    audio_samples, tone_events = stimulator._generate_oddball_sequence(sample_rate)

    total_tones = OddballStimParams.INITIAL_TONES + OddballStimParams.MAIN_TONES

    assert len(tone_events) == total_tones
    print(f"✓ Generated {total_tones} tone events")

    # Verify all intervals are exactly 1 second
    intervals = []
    for i in range(1, len(tone_events)):
        interval = tone_events[i]['onset_sample'] - tone_events[i-1]['onset_sample']
        intervals.append(interval)

    all_correct = all(i == sample_rate for i in intervals)
    assert all_correct
    print(f"✓ All {len(intervals)} intervals are exactly {sample_rate} samples (1000ms)")

    # Verify initial tones are all standard
    for i in range(OddballStimParams.INITIAL_TONES):
        assert tone_events[i]['type'] == 'standard'
    print(f"✓ First {OddballStimParams.INITIAL_TONES} tones are standard")

    # Verify buffer is correct shape
    assert audio_samples.ndim == 2
    assert audio_samples.shape[1] == 1  # Mono
    print(f"✓ Audio buffer shape: {audio_samples.shape}")


def test_state_transitions_full_workflow():
    """Test a complete workflow of state transitions."""
    print("\n=== Test: Full Workflow State Transitions ===")

    mock_gui = MockGuiCallback()
    sm = mock_gui.state_manager
    stimulator = AuditoryStimulator(gui_callback=mock_gui)

    # Start in READY state
    assert sm.state == PlaybackState.READY
    print("✓ Initial state: READY")

    # Simulate pressing Play
    sm.transition_to(PlaybackState.PLAYING)
    assert sm.is_playing()
    print("✓ After Play: PLAYING")

    # Simulate pressing Pause
    sm.transition_to(PlaybackState.PAUSED)
    stimulator.toggle_pause()
    assert sm.is_paused()
    print("✓ After Pause: PAUSED")

    # Simulate pressing Resume (Pause button again)
    sm.transition_to(PlaybackState.PLAYING)
    assert sm.is_playing()
    print("✓ After Resume: PLAYING")

    # Simulate pressing Stop
    sm.transition_to(PlaybackState.STOPPED)
    stimulator.stop_stimulus()
    assert sm.state == PlaybackState.STOPPED
    print("✓ After Stop: STOPPED")

    # Return to READY
    sm.transition_to(PlaybackState.READY)
    assert sm.is_ready()
    print("✓ After reset: READY")


# ==============================================================================
# Main Test Runner
# ==============================================================================

def run_all_tests():
    """Run all tests."""
    import sys
    print("=" * 60)
    print("BUTTON CONTROLS & STATE MANAGEMENT TESTS")
    print("=" * 60)
    sys.stdout.flush()

    tests = [
        # State Manager Tests
        ("State Manager Initial", test_state_manager_initial_state),
        ("State Manager Transitions", test_state_manager_valid_transitions),
        ("State Manager Invalid", test_state_manager_invalid_transitions),
        ("State Manager Helpers", test_state_manager_helper_methods),
        ("State Manager Listeners", test_state_manager_listeners),

        # Audio Stream Manager Tests
        ("Audio Play/Stop", test_audio_stream_manager_play_stop),
        ("Audio Stop During Play", test_audio_stream_manager_stop_during_play),
        ("Audio Rapid Cycles", test_audio_stream_manager_rapid_play_stop),

        # Auditory Stimulator Tests
        ("Stimulator Toggle Pause", test_auditory_stimulator_toggle_pause),
        ("Stimulator Stop", test_auditory_stimulator_stop),
        ("Oddball Generation", test_oddball_sequence_generation),
        ("Full Workflow", test_state_transitions_full_workflow),
    ]

    passed = 0
    failed = 0

    for name, test in tests:
        try:
            test()
            passed += 1
            sys.stdout.flush()
        except Exception as e:
            failed += 1
            print(f"\n✗ FAILED: {name}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    sys.stdout.flush()

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
