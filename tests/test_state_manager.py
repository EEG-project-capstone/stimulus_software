# tests/test_state_manager.py
"""Tests for state_manager.py - State machine functionality."""

import pytest
import sys
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.constants import PlaybackState, VALID_STATE_TRANSITIONS
from lib.state_manager import StateManager
from lib.exceptions import StateError


class TestStateManagerInit:
    """Tests for StateManager initialization."""

    def test_default_initial_state(self):
        """StateManager should initialize to EMPTY state by default."""
        sm = StateManager()
        assert sm.state == PlaybackState.EMPTY

    def test_custom_initial_state(self):
        """StateManager should accept custom initial state."""
        sm = StateManager(initial_state=PlaybackState.READY)
        assert sm.state == PlaybackState.READY

    def test_repr(self):
        """StateManager repr should show current state."""
        sm = StateManager()
        assert "EMPTY" in repr(sm)


class TestStateTransitions:
    """Tests for state transitions."""

    def test_valid_transition_empty_to_ready(self):
        """EMPTY -> READY should be valid."""
        sm = StateManager()
        result = sm.transition_to(PlaybackState.READY)
        assert result is True
        assert sm.state == PlaybackState.READY

    def test_valid_transition_ready_to_preparing(self):
        """READY -> PREPARING should be valid."""
        sm = StateManager(initial_state=PlaybackState.READY)
        result = sm.transition_to(PlaybackState.PREPARING)
        assert result is True
        assert sm.state == PlaybackState.PREPARING

    def test_valid_transition_ready_to_playing(self):
        """READY -> PLAYING should be valid."""
        sm = StateManager(initial_state=PlaybackState.READY)
        result = sm.transition_to(PlaybackState.PLAYING)
        assert result is True
        assert sm.state == PlaybackState.PLAYING

    def test_valid_transition_playing_to_paused(self):
        """PLAYING -> PAUSED should be valid."""
        sm = StateManager(initial_state=PlaybackState.PLAYING)
        result = sm.transition_to(PlaybackState.PAUSED)
        assert result is True
        assert sm.state == PlaybackState.PAUSED

    def test_valid_transition_paused_to_playing(self):
        """PAUSED -> PLAYING should be valid (resume)."""
        sm = StateManager(initial_state=PlaybackState.PAUSED)
        result = sm.transition_to(PlaybackState.PLAYING)
        assert result is True
        assert sm.state == PlaybackState.PLAYING

    def test_valid_transition_playing_to_stopped(self):
        """PLAYING -> STOPPED should be valid."""
        sm = StateManager(initial_state=PlaybackState.PLAYING)
        result = sm.transition_to(PlaybackState.STOPPED)
        assert result is True
        assert sm.state == PlaybackState.STOPPED

    def test_valid_transition_stopped_to_ready(self):
        """STOPPED -> READY should be valid."""
        sm = StateManager(initial_state=PlaybackState.STOPPED)
        result = sm.transition_to(PlaybackState.READY)
        assert result is True
        assert sm.state == PlaybackState.READY

    def test_invalid_transition_raises_error(self):
        """Invalid transitions should raise StateError."""
        sm = StateManager()  # EMPTY state
        with pytest.raises(StateError):
            sm.transition_to(PlaybackState.PLAYING)  # Can't go EMPTY -> PLAYING

    def test_invalid_transition_empty_to_paused(self):
        """EMPTY -> PAUSED should be invalid."""
        sm = StateManager()
        with pytest.raises(StateError):
            sm.transition_to(PlaybackState.PAUSED)

    def test_invalid_transition_preparing_to_playing(self):
        """PREPARING -> PLAYING should be invalid (must go through READY)."""
        sm = StateManager(initial_state=PlaybackState.PREPARING)
        with pytest.raises(StateError):
            sm.transition_to(PlaybackState.PLAYING)

    def test_same_state_transition_is_noop(self):
        """Transitioning to same state should be a no-op."""
        sm = StateManager(initial_state=PlaybackState.READY)
        result = sm.transition_to(PlaybackState.READY)
        assert result is True
        assert sm.state == PlaybackState.READY

    def test_force_transition_bypasses_validation(self):
        """Force flag should bypass validation."""
        sm = StateManager()  # EMPTY state
        result = sm.transition_to(PlaybackState.PLAYING, force=True)
        assert result is True
        assert sm.state == PlaybackState.PLAYING


class TestStateValidation:
    """Tests for state validation methods."""

    def test_can_transition_to_valid_target(self):
        """can_transition_to should return True for valid targets."""
        sm = StateManager(initial_state=PlaybackState.READY)
        assert sm.can_transition_to(PlaybackState.PREPARING) is True
        assert sm.can_transition_to(PlaybackState.PLAYING) is True
        assert sm.can_transition_to(PlaybackState.SENDING_SYNC) is True

    def test_can_transition_to_invalid_target(self):
        """can_transition_to should return False for invalid targets."""
        sm = StateManager(initial_state=PlaybackState.EMPTY)
        assert sm.can_transition_to(PlaybackState.PLAYING) is False
        assert sm.can_transition_to(PlaybackState.PAUSED) is False

    def test_get_valid_transitions_from_empty(self):
        """get_valid_transitions from EMPTY should return {READY}."""
        sm = StateManager()
        valid = sm.get_valid_transitions()
        assert valid == {PlaybackState.READY}

    def test_get_valid_transitions_from_ready(self):
        """get_valid_transitions from READY should return correct set."""
        sm = StateManager(initial_state=PlaybackState.READY)
        valid = sm.get_valid_transitions()
        expected = {PlaybackState.PREPARING, PlaybackState.PLAYING,
                   PlaybackState.EMPTY, PlaybackState.SENDING_SYNC}
        assert valid == expected


class TestStateHelpers:
    """Tests for helper methods."""

    def test_is_playing(self):
        """is_playing should return True only when PLAYING."""
        sm = StateManager(initial_state=PlaybackState.PLAYING)
        assert sm.is_playing() is True

        sm2 = StateManager(initial_state=PlaybackState.PAUSED)
        assert sm2.is_playing() is False

    def test_is_paused(self):
        """is_paused should return True only when PAUSED."""
        sm = StateManager(initial_state=PlaybackState.PAUSED)
        assert sm.is_paused() is True

        sm2 = StateManager(initial_state=PlaybackState.PLAYING)
        assert sm2.is_paused() is False

    def test_is_active(self):
        """is_active should return True when PLAYING or PAUSED."""
        sm_playing = StateManager(initial_state=PlaybackState.PLAYING)
        sm_paused = StateManager(initial_state=PlaybackState.PAUSED)
        sm_ready = StateManager(initial_state=PlaybackState.READY)

        assert sm_playing.is_active() is True
        assert sm_paused.is_active() is True
        assert sm_ready.is_active() is False

    def test_is_ready(self):
        """is_ready should return True only when READY."""
        sm = StateManager(initial_state=PlaybackState.READY)
        assert sm.is_ready() is True

        sm2 = StateManager()
        assert sm2.is_ready() is False

    def test_reset(self):
        """reset should return to EMPTY state."""
        sm = StateManager(initial_state=PlaybackState.PLAYING)
        sm.reset()
        assert sm.state == PlaybackState.EMPTY


class TestStateListeners:
    """Tests for state change listeners."""

    def test_add_listener(self):
        """Listeners should be notified on state change."""
        sm = StateManager()
        notifications = []

        def listener(old_state, new_state):
            notifications.append((old_state, new_state))

        sm.add_listener(listener)
        sm.transition_to(PlaybackState.READY)

        assert len(notifications) == 1
        assert notifications[0] == (PlaybackState.EMPTY, PlaybackState.READY)

    def test_multiple_listeners(self):
        """Multiple listeners should all be notified."""
        sm = StateManager()
        count = [0, 0]

        def listener1(old, new):
            count[0] += 1

        def listener2(old, new):
            count[1] += 1

        sm.add_listener(listener1)
        sm.add_listener(listener2)
        sm.transition_to(PlaybackState.READY)

        assert count[0] == 1
        assert count[1] == 1

    def test_remove_listener(self):
        """Removed listeners should not be notified."""
        sm = StateManager()
        notifications = []

        def listener(old_state, new_state):
            notifications.append((old_state, new_state))

        sm.add_listener(listener)
        sm.remove_listener(listener)
        sm.transition_to(PlaybackState.READY)

        assert len(notifications) == 0

    def test_listener_exception_does_not_break_transitions(self):
        """Listener exceptions should be caught and logged."""
        sm = StateManager()

        def bad_listener(old, new):
            raise ValueError("Test error")

        def good_listener(old, new):
            pass  # Does nothing but doesn't error

        sm.add_listener(bad_listener)
        sm.add_listener(good_listener)

        # Should not raise despite bad_listener throwing
        result = sm.transition_to(PlaybackState.READY)
        assert result is True


class TestValidStateTransitionsConsistency:
    """Tests to verify VALID_STATE_TRANSITIONS constant is consistent."""

    def test_all_states_have_transitions(self):
        """All PlaybackStates should have an entry in VALID_STATE_TRANSITIONS."""
        for state in PlaybackState:
            assert state in VALID_STATE_TRANSITIONS, f"Missing transitions for {state}"

    def test_all_targets_are_valid_states(self):
        """All transition targets should be valid PlaybackStates."""
        for source, targets in VALID_STATE_TRANSITIONS.items():
            for target in targets:
                assert isinstance(target, PlaybackState), f"Invalid target {target} from {source}"

    def test_no_self_transitions_defined(self):
        """Self-transitions shouldn't be in VALID_STATE_TRANSITIONS (handled specially)."""
        for source, targets in VALID_STATE_TRANSITIONS.items():
            assert source not in targets, f"Self-transition defined for {source}"
