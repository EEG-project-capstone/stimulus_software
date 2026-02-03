import sys
import time
import threading
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import numpy as np
import pytest

from lib.auditory_stimulator import AuditoryStimulator
from lib.stim_handlers import OddballStimHandler
from lib.constants import OddballStimParams


class MockRoot:
    """Simple scheduler that mimics tkinter's `after`/`after_cancel`."""
    def __init__(self):
        self._lock = threading.Lock()
        self._next_id = 1
        self._timers = {}

    def after(self, delay_ms, callback):
        with self._lock:
            cb_id = self._next_id
            self._next_id += 1
        t = threading.Timer(delay_ms / 1000.0, callback)
        with self._lock:
            self._timers[cb_id] = t
        t.daemon = True
        t.start()
        return cb_id

    def after_cancel(self, cb_id):
        with self._lock:
            t = self._timers.pop(cb_id, None)
        if t:
            t.cancel()


class MockStateManager:
    def is_playing(self):
        return True

    def is_paused(self):
        return False


class MockResultsManager:
    def append_result(self, *args, **kwargs):
        pass

    def append_sync_pulse(self, *args, **kwargs):
        pass


class MockGuiCallback:
    def __init__(self):
        self.root = MockRoot()
        self.results_manager = MockResultsManager()
        self.stims = type('S', (), {})()
        self.stims.current_stim_index = 0
        self.stims.stim_dictionary = [{'type': 'oddball', 'status': 'pending'}]
        self.config = type('C', (), {})()
        self.state_manager = MockStateManager()

    def get_patient_id(self):
        return 'TEST_PATIENT'

    def update_stim_list_status(self):
        return None

    def playback_error(self, *args, **kwargs):
        return None

    def playback_complete(self):
        return None


def setup_stimulator_and_handler():
    mock_gui = MockGuiCallback()
    stimulator = AuditoryStimulator(gui_callback=mock_gui)

    # Replace actual audio playback with a no-op to avoid hardware use
    stimulator.stream_manager.play = lambda samples, sample_rate, on_finish=None: None

    # Replace tone generation with a short silent buffer
    stimulator._generate_tone = lambda f, d: np.zeros(10, dtype=np.int16)

    handler = OddballStimHandler(stimulator)
    # Ensure handler is active for safe_schedule to enqueue callbacks
    handler.is_active = True

    # Reset global sequence state to make tests independent
    OddballStimHandler._global_last_tone_time = None
    OddballStimHandler._global_tone_sequence = 0

    return stimulator, handler


def test_double_beep_detected(caplog):
    caplog.set_level(logging.WARNING)
    stimulator, handler = setup_stimulator_and_handler()

    # Play first tone immediately
    handler._play_tone(OddballStimParams.STANDARD_FREQ, 'standard_tone')

    # Schedule second tone using the same scheduling paradigm with short delay (50ms)
    handler.safe_schedule(50, lambda: handler._play_tone(OddballStimParams.STANDARD_FREQ, 'standard_tone'))

    # Wait long enough for the scheduled callback to execute
    time.sleep(0.25)

    # Expect a double-beep warning in the logs
    assert any('DOUBLE-BEEP DETECTED' in rec.message for rec in caplog.records)


def test_no_double_beep_with_normal_spacing(caplog):
    caplog.set_level(logging.WARNING)
    stimulator, handler = setup_stimulator_and_handler()

    # Play first tone
    handler._play_tone(OddballStimParams.STANDARD_FREQ, 'standard_tone')

    # Schedule second tone using the normal onset-based spacing (1000ms)
    handler.safe_schedule(1000, lambda: handler._play_tone(OddballStimParams.STANDARD_FREQ, 'standard_tone'))

    # Wait until after the second tone would have played
    time.sleep(1.2)

    # There should be no double-beep warning for 1000ms spacing
    assert not any('DOUBLE-BEEP DETECTED' in rec.message for rec in caplog.records)
