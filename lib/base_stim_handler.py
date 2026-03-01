# lib/base_stim_handler.py

"""
Base class for stimulus handlers with common functionality.
Reduces code duplication across specific handler implementations.
"""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Optional, Dict, Any

import numpy as np
from math import gcd

from scipy.signal import resample_poly

from lib.constants import AudioParams

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
        self.stop()

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
            self.is_active = False  # Prevent double-finish from watchdog + real callback
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
                       log_label: Optional[str] = None):
        """Play audio and guarantee on_finish fires exactly once.

        on_finish is called deterministically by AudioStreamManager.stream_callback
        when the buffer is exhausted.  A watchdog scheduled at audio_duration + 10 s
        provides a fallback in case the stream is interrupted (device sleep, etc.).
        The fired[] guard prevents double-execution regardless of which path fires first.

        Args:
            samples: Audio samples (int16 numpy array)
            sample_rate: Sample rate in Hz
            on_finish: Callback to invoke when playback finishes
            log_label: Label for logging
        """
        fired = [False]  # mutable cell shared by real callback and watchdog

        def safe_on_finish():
            if fired[0]:
                return
            fired[0] = True
            logger.debug(
                f"Playback finish for {self.__class__.__name__} "
                f"(is_active={self.is_active})"
            )
            if on_finish:
                try:
                    on_finish()
                except Exception as e:
                    logger.error(
                        f"Error in on_finish for {self.__class__.__name__}: {e}",
                        exc_info=True
                    )
                    self._handle_error(e)

        self.audio_stim.play_audio(
            samples=samples,
            sample_rate=sample_rate,
            callback=safe_on_finish,
            log_label=log_label,
        )

        # Watchdog fires safe_on_finish if stream_callback exhaustion detection fails
        # (guards against device sleep or other rare stream interruptions).
        # safe_schedule's is_active guard prevents the watchdog from running
        # after a pause/stop resets the handler.
        expected_duration_ms = int(len(samples) / sample_rate * 1000)
        self.safe_schedule(expected_duration_ms + 10000, safe_on_finish)
    
    def reshape_audio_samples(self, audio_segment) -> 'np.ndarray':
        """Reshape audio samples from AudioSegment for playback.

        Resamples to AudioParams.SAMPLE_RATE if needed so audio plays at
        the correct speed on the persistent fixed-rate stream.

        Args:
            audio_segment: pydub AudioSegment

        Returns:
            Reshaped numpy array (n_samples, channels)
        """
        raw = np.array(audio_segment.get_array_of_samples(), dtype=np.int16)

        if audio_segment.frame_rate != AudioParams.SAMPLE_RATE:
            logger.warning(
                f"Resampling audio from {audio_segment.frame_rate}Hz "
                f"to {AudioParams.SAMPLE_RATE}Hz"
            )
            src = audio_segment.frame_rate
            dst = AudioParams.SAMPLE_RATE
            g = gcd(dst, src)
            up, down = dst // g, src // g
            channels = audio_segment.channels
            if channels == 2:
                raw2d = raw.reshape(-1, 2).astype(np.float32)
                resampled = np.stack(
                    [resample_poly(raw2d[:, ch], up, down) for ch in range(2)],
                    axis=1,
                )
            else:
                resampled = resample_poly(raw.astype(np.float32), up, down).reshape(-1, 1)
            return np.clip(resampled, -32768, 32767).astype(np.int16)

        if audio_segment.channels == 2:
            return raw.reshape(-1, 2)
        return raw.reshape(-1, 1)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(active={self.is_active}, "
                f"callbacks={len(self._scheduled_callbacks)})")