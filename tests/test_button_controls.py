# tests/test_button_controls.py
"""
Integration tests for playback control — covers behaviour that spans
AuditoryStimulator, StateManager and AudioStreamManager together.

Unit-level tests for StateManager and AudioStreamManager live in
test_state_manager.py and test_audio_stream_manager.py respectively.
"""

import time
import numpy as np
import pytest

from lib.auditory_stimulator import AuditoryStimulator
from lib.constants import PlaybackState, OddballStimParams
from tests.conftest import MockGuiCallback


class TestAuditoryStimulatorPause:
    """Toggle-pause integration tests."""

    def test_toggle_pause_stops_audio(self):
        """Transitioning to PAUSED should stop the audio stream."""
        mock_gui = MockGuiCallback()
        stimulator = AuditoryStimulator(gui_callback=mock_gui)

        mock_gui.state_manager.transition_to(PlaybackState.PLAYING)
        stimulator.handlers['oddball'].is_active = True

        audio_samples, _ = stimulator._generate_oddball_sequence(44100)
        stimulator.stream_manager.play(audio_samples, 44100)
        assert stimulator.stream_manager.is_playing()

        mock_gui.state_manager.transition_to(PlaybackState.PAUSED)
        stimulator.toggle_pause()
        time.sleep(0.1)

        assert not stimulator.stream_manager.is_playing()

    def test_toggle_pause_deactivates_handler(self):
        """Transitioning to PAUSED should deactivate the active handler."""
        mock_gui = MockGuiCallback()
        stimulator = AuditoryStimulator(gui_callback=mock_gui)

        mock_gui.state_manager.transition_to(PlaybackState.PLAYING)
        handler = stimulator.handlers['oddball']
        handler.is_active = True

        mock_gui.state_manager.transition_to(PlaybackState.PAUSED)
        stimulator.toggle_pause()
        time.sleep(0.1)

        assert not handler.is_active


class TestAuditoryStimulatorStop:
    """stop_stimulus integration tests."""

    def test_stop_halts_audio(self):
        """stop_stimulus should stop the audio stream."""
        mock_gui = MockGuiCallback()
        stimulator = AuditoryStimulator(gui_callback=mock_gui)

        mock_gui.state_manager.transition_to(PlaybackState.PLAYING)
        stimulator.handlers['oddball'].is_active = True

        audio_samples, _ = stimulator._generate_oddball_sequence(44100)
        stimulator.stream_manager.play(audio_samples, 44100)

        stimulator.stop_stimulus()
        time.sleep(0.1)

        assert not stimulator.stream_manager.is_playing()

    def test_stop_deactivates_handler(self):
        """stop_stimulus should deactivate the active handler."""
        mock_gui = MockGuiCallback()
        stimulator = AuditoryStimulator(gui_callback=mock_gui)

        mock_gui.state_manager.transition_to(PlaybackState.PLAYING)
        handler = stimulator.handlers['oddball']
        handler.is_active = True

        stimulator.stop_stimulus()

        assert not handler.is_active

    def test_stop_resets_stim_index(self):
        """stop_stimulus should reset the stimulus index to 0."""
        mock_gui = MockGuiCallback()
        stimulator = AuditoryStimulator(gui_callback=mock_gui)

        mock_gui.state_manager.transition_to(PlaybackState.PLAYING)
        stimulator.stims.current_stim_index = 1

        stimulator.stop_stimulus()

        assert stimulator.stims.current_stim_index == 0


class TestOddballSequenceGeneration:
    """Tests for _generate_oddball_sequence."""

    def test_correct_tone_count(self):
        """Generated sequence should contain exactly INITIAL_TONES + MAIN_TONES events."""
        stimulator = AuditoryStimulator(gui_callback=MockGuiCallback())
        _, tone_events = stimulator._generate_oddball_sequence(44100)
        expected = OddballStimParams.INITIAL_TONES + OddballStimParams.MAIN_TONES
        assert len(tone_events) == expected

    def test_sample_accurate_intervals(self):
        """Every onset-to-onset interval should be exactly 44100 samples (1 second)."""
        stimulator = AuditoryStimulator(gui_callback=MockGuiCallback())
        _, tone_events = stimulator._generate_oddball_sequence(44100)

        intervals = [
            tone_events[i]['onset_sample'] - tone_events[i - 1]['onset_sample']
            for i in range(1, len(tone_events))
        ]
        assert all(iv == 44100 for iv in intervals)

    def test_initial_tones_are_standard(self):
        """First INITIAL_TONES entries must all be 'standard'."""
        stimulator = AuditoryStimulator(gui_callback=MockGuiCallback())
        _, tone_events = stimulator._generate_oddball_sequence(44100)

        for event in tone_events[:OddballStimParams.INITIAL_TONES]:
            assert event['type'] == 'standard'

    def test_buffer_is_mono(self):
        """Audio buffer should be shaped (n_samples, 1)."""
        stimulator = AuditoryStimulator(gui_callback=MockGuiCallback())
        audio_samples, _ = stimulator._generate_oddball_sequence(44100)

        assert audio_samples.ndim == 2
        assert audio_samples.shape[1] == 1


class TestFullWorkflow:
    """End-to-end state transition workflow."""

    def test_play_pause_resume_stop_cycle(self):
        """Complete READY -> PLAYING -> PAUSED -> PLAYING -> STOPPED cycle."""
        mock_gui = MockGuiCallback()
        sm = mock_gui.state_manager
        stimulator = AuditoryStimulator(gui_callback=mock_gui)

        assert sm.state == PlaybackState.READY

        sm.transition_to(PlaybackState.PLAYING)
        assert sm.is_playing()

        sm.transition_to(PlaybackState.PAUSED)
        stimulator.toggle_pause()
        assert sm.is_paused()

        sm.transition_to(PlaybackState.PLAYING)
        assert sm.is_playing()

        sm.transition_to(PlaybackState.STOPPED)
        stimulator.stop_stimulus()
        assert sm.state == PlaybackState.STOPPED

        sm.transition_to(PlaybackState.READY)
        assert sm.is_ready()
