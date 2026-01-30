# lib/state_manager.py

"""
State management for the EEG Stimulus Package.
Provides a robust state machine for playback control.
"""

import logging
from typing import Callable, Optional, Set
from lib.constants import PlaybackState, VALID_STATE_TRANSITIONS
from lib.exceptions import StateError

logger = logging.getLogger('eeg_stimulus.state_manager')


class StateManager:
    """Manages application state transitions with validation and callbacks."""
    
    def __init__(self, initial_state: PlaybackState = PlaybackState.EMPTY):
        """Initialize state manager.
        
        Args:
            initial_state: Starting state (default: EMPTY)
        """
        self._state = initial_state
        self._listeners = []
        logger.info(f"StateManager initialized with state: {initial_state.name}")
    
    @property
    def state(self) -> PlaybackState:
        """Get current state."""
        return self._state
    
    def transition_to(self, new_state: PlaybackState, force: bool = False) -> bool:
        """Attempt to transition to a new state.
        
        Args:
            new_state: Target state
            force: If True, skip validation (use with caution)
        
        Returns:
            True if transition succeeded, False otherwise
            
        Raises:
            StateError: If transition is invalid and force=False
        """
        if self._state == new_state:
            logger.debug(f"Already in state {new_state.name}, no transition needed")
            return True
        
        # Validate transition
        if not force and not self._is_valid_transition(self._state, new_state):
            error_msg = (f"Invalid state transition: {self._state.name} -> {new_state.name}")
            logger.error(error_msg)
            raise StateError(error_msg)
        
        # Perform transition
        old_state = self._state
        self._state = new_state
        
        logger.info(f"State transition: {old_state.name} -> {new_state.name}")
        self._notify_listeners(old_state, new_state)
        
        return True
    
    def _is_valid_transition(self, from_state: PlaybackState, to_state: PlaybackState) -> bool:
        """Check if transition is valid.
        
        Args:
            from_state: Current state
            to_state: Target state
            
        Returns:
            True if transition is allowed
        """
        valid_targets = VALID_STATE_TRANSITIONS.get(from_state, set())
        return to_state in valid_targets
    
    def get_valid_transitions(self) -> Set[PlaybackState]:
        """Get all valid transitions from current state.
        
        Returns:
            Set of valid target states
        """
        return VALID_STATE_TRANSITIONS.get(self._state, set())
    
    def can_transition_to(self, target_state: PlaybackState) -> bool:
        """Check if can transition to target state.
        
        Args:
            target_state: State to check
            
        Returns:
            True if transition is valid
        """
        return target_state in self.get_valid_transitions()
    
    def add_listener(self, callback: Callable[[PlaybackState, PlaybackState], None]):
        """Add state change listener.
        
        Args:
            callback: Function called with (old_state, new_state)
        """
        if callback not in self._listeners:
            self._listeners.append(callback)
            logger.debug(f"Added state change listener: {callback.__name__}")
    
    def remove_listener(self, callback: Callable[[PlaybackState, PlaybackState], None]):
        """Remove state change listener.
        
        Args:
            callback: Listener to remove
        """
        if callback in self._listeners:
            self._listeners.remove(callback)
            logger.debug(f"Removed state change listener: {callback.__name__}")
    
    def _notify_listeners(self, old_state: PlaybackState, new_state: PlaybackState):
        """Notify all listeners of state change.
        
        Args:
            old_state: Previous state
            new_state: Current state
        """
        for listener in self._listeners:
            try:
                listener(old_state, new_state)
            except Exception as e:
                logger.error(f"Error in state listener {listener.__name__}: {e}", 
                           exc_info=True)
    
    def reset(self):
        """Reset to initial EMPTY state."""
        logger.info("Resetting state manager to EMPTY")
        self.transition_to(PlaybackState.EMPTY, force=True)
    
    def is_playing(self) -> bool:
        """Check if currently playing."""
        return self._state == PlaybackState.PLAYING
    
    def is_paused(self) -> bool:
        """Check if currently paused."""
        return self._state == PlaybackState.PAUSED
    
    def is_active(self) -> bool:
        """Check if playing or paused (active playback session)."""
        return self._state in {PlaybackState.PLAYING, PlaybackState.PAUSED}
    
    def is_ready(self) -> bool:
        """Check if ready for operations."""
        return self._state == PlaybackState.READY
    
    def __repr__(self) -> str:
        return f"StateManager(state={self._state.name})"
