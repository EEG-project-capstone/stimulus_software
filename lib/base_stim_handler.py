# lib/base_stim_handler.py

"""
Base class for stimulus handlers with common functionality.
Reduces code duplication across specific handler implementations.
"""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Optional, Dict, Any

import numpy as np

if TYPE_CHECKING:
    from lib.auditory_stimulator import AuditoryStimulator

logger = logging.getLogger('eeg_stimulus.handlers')


class BaseStimHandler(ABC):
    """Base class for stimulus handlers with shared functionality."""
    
    def __init__(self, auditory_stimulator: 'AuditoryStimulator'):
        """Initialize handler.
        
        Args:
            auditory_stimulator: Reference to AuditoryStimulator instance
        """
        self.audio_stim: 'AuditoryStimulator' = auditory_stimulator
        self.state: Dict[str, Any] = {}
        self.is_active: bool = False
        self._scheduled_callbacks: list = []
        logger.debug(f"{self.__class__.__name__} initialized")
    
    @abstractmethod
    def start(self, stim: dict):
        """Start the stimulus.
        
        Args:
            stim: Stimulus dictionary with configuration
        """
        pass
    
    @abstractmethod
    def continue_stim(self):
        """Continue stimulus execution (for multi-phase stimuli)."""
        pass
    
    def reset(self):
        """Reset handler state to initial conditions."""
        self._cancel_all_callbacks()
        self.state = {}
        self.is_active = False

    def stop(self):
        """Stop handler immediately with full cleanup."""
        self.is_active = False
        self._cancel_all_callbacks()
        self.state = {}
    
    def should_continue(self) -> bool:
        """Check if stimulus should continue execution.

        Returns:
            True if conditions allow continuation
        """
        if not self.is_active:
            return False

        # Check global playback state via state manager
        if self.audio_stim.gui_callback.state_manager.is_paused():
            # Don't schedule anything when paused - resume will restart
            return False

        return True
    
    def safe_schedule(self, delay_ms: int, callback: Callable[[], None]) -> Optional[str]:
        """Schedule a callback with automatic cleanup on stop."""
        if not self.is_active:
            return None

        def wrapped_callback():
            if cb_id in self._scheduled_callbacks:
                self._scheduled_callbacks.remove(cb_id)
            if self.is_active:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Callback error in {self.__class__.__name__}: {e}", exc_info=True)
                    self._handle_error(e)

        cb_id = self.audio_stim._schedule(delay_ms, wrapped_callback)
        self._scheduled_callbacks.append(cb_id)
        return cb_id
    
    def _cancel_all_callbacks(self):
        """Cancel all scheduled callbacks."""
        for cb_id in self._scheduled_callbacks:
            try:
                self.audio_stim.gui_callback.root.after_cancel(cb_id)
            except Exception:
                pass
        self._scheduled_callbacks.clear()
    
    def safe_finish(self):
        """Safely finish stimulus only if still active."""
        if self.is_active:
            self.audio_stim.finish_current_stim()
    
    def _handle_error(self, error: Exception):
        """Handle errors during stimulus execution.
        
        Args:
            error: Exception that occurred
        """
        logger.error(f"Error in {self.__class__.__name__}: {error}", exc_info=True)
        self.stop()
        
        # Notify auditory stimulator of error
        error_msg = f"{self.__class__.__name__} error: {str(error)}"
        self.audio_stim.gui_callback.playback_error(error_msg)
    
    def log_event(self, event_name: str, metadata: Optional[dict] = None):
        """Log an event with metadata.
        
        Args:
            event_name: Name of the event
            metadata: Additional event data
        """
        self.audio_stim._log_event(event_name, metadata)
    
    def play_audio_safe(self,
                       samples,
                       sample_rate: int,
                       on_finish: Optional[Callable[[], None]] = None,
                       log_label: Optional[str] = None,
                       onset_offset_ms: float = 0):
        """Play audio with safe finish callback.

        Args:
            samples: Audio samples
            sample_rate: Sample rate in Hz
            on_finish: Callback after playback
            log_label: Label for logging
            onset_offset_ms: Offset in ms to add to onset time (e.g., for padding)
        """
        def wrapped_finish():
            if on_finish and self.is_active:
                on_finish()

        self.audio_stim.play_audio(
            samples=samples,
            sample_rate=sample_rate,
            callback=wrapped_finish,
            log_label=log_label,
            onset_offset_ms=onset_offset_ms
        )
    
    def reshape_audio_samples(self, audio_segment) -> 'np.ndarray':
        """Reshape audio samples from AudioSegment for playback.

        Args:
            audio_segment: pydub AudioSegment

        Returns:
            Reshaped numpy array (n_samples, channels)
        """
        import numpy as np
        samples = np.array(audio_segment.get_array_of_samples(), dtype=np.int16)
        if audio_segment.channels == 2:
            return samples.reshape(-1, 2)
        return samples.reshape(-1, 1)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(active={self.is_active}, "
                f"callbacks={len(self._scheduled_callbacks)})")