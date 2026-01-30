# lib/auditory_stimulator.py

"""
Manages audio playback for stimulus sessions with improved architecture.
Uses AudioStreamManager, ResultsManager, and modular handlers.
"""

import time
import random
import threading
import json
import numpy as np
import logging
from typing import Optional
from functools import partial

from lib.audio_stream_manager import AudioStreamManager
from lib.stim_handlers import (
    LanguageStimHandler,
    CommandStimHandler,
    OddballStimHandler,
    VoiceStimHandler
)
from lib.constants import SyncPulseParams, TimingParams
from lib.exceptions import AudioError
from lib.logging_utils import log_operation

logger = logging.getLogger('eeg_stimulus.auditory_stimulator')

class AuditoryStimulator:
    """Manages audio playback for stimulus sessions with modular handlers."""
    
    def __init__(self, gui_callback):
        """Initialize the auditory stimulator.
        
        Args:
            gui_callback: Reference to main app instance
        """
        self.gui_callback = gui_callback
        self.stims = gui_callback.stims
        self.config = gui_callback.config
        
        # Use improved managers
        self.stream_manager = AudioStreamManager()
        self.results_manager = gui_callback.results_manager

        # Initialize handlers (shared voice handler for control/loved_one)
        voice_handler = VoiceStimHandler(self)
        self.handlers = {
            'language': LanguageStimHandler(self),
            'right_command': CommandStimHandler(self),
            'right_command+p': CommandStimHandler(self),
            'left_command': CommandStimHandler(self),
            'left_command+p': CommandStimHandler(self),
            'oddball': OddballStimHandler(self),
            'oddball+p': OddballStimHandler(self),
            'control': voice_handler,
            'loved_one_voice': voice_handler,
        }

        # Callback scheduling
        self.scheduled_callbacks = []
        self.callback_lock = threading.Lock()
        
        # Current stimulus tracking
        self.current_stim_start_time = 0
        self.current_stim_sentences = []
        
        self.reset_stim_state()
        logger.info("AuditoryStimulator initialized")
    
    def reset_stim_state(self):
        """Reset stimulus state variables."""
        self.stims.current_stim_index = 0
        self.current_stim_start_time = 0
        self.current_stim_sentences = []

        # Reset all handlers
        for handler in self.handlers.values():
            handler.reset()

        # Reset all stimulus statuses to pending
        for stim in self.stims.stim_dictionary:
            stim['status'] = 'pending'

        logger.debug("Stimulus state reset")
    
    def play_stim_sequence(self):
        """Start playing the stimulus sequence."""
        with log_operation("stimulus_sequence_playback"):
            self.stims.current_stim_index = 0
            logger.info(f"Starting stimulus sequence: {len(self.stims.stim_dictionary)} stimuli")
            self.continue_playback()
    
    def continue_playback(self):
        """Continue playback from current stimulus index."""
        # Check if we should continue
        if not self.gui_callback.state_manager.is_playing():
            logger.debug(f"Cannot continue: state={self.gui_callback.state_manager.state.name}")
            return

        # Check if sequence is complete
        if self.stims.current_stim_index >= len(self.stims.stim_dictionary):
            logger.info("Stimulus sequence completed")
            self.gui_callback.playback_complete()
            return

        # Play current stimulus
        logger.debug(f"Playing stimulus {self.stims.current_stim_index + 1}/{len(self.stims.stim_dictionary)}")
        self.play_current_stim()
    
    def play_current_stim(self):
        """Play the current stimulus and schedule the next one."""
        try:
            patient_id = self.gui_callback.get_patient_id()
            stim = self.stims.stim_dictionary[self.stims.current_stim_index]
            stim['status'] = 'in progress'
            
            logger.info(f"Starting stimulus {self.stims.current_stim_index + 1}/"
                       f"{len(self.stims.stim_dictionary)}: "
                       f"type={stim.get('type')}, patient={patient_id}")
            
            # Initialize stimulus result storage
            self.current_stim_start_time = time.time()
            self.current_stim_sentences = []
            
            # Log stimulus start event
            self._log_event('stim_start', {
                'stim_index': self.stims.current_stim_index,
                'stim_type': stim.get('type'),
                'patient_id': patient_id
            })
            
            # Start playing the stimulus using handler registry
            self.start_stim_playback(stim)
            self.gui_callback.update_stim_list_status()
            
        except Exception as e:
            logger.error(f"Error in play_current_stim: {e}", exc_info=True)
            self.gui_callback.playback_error(str(e))
    
    def start_stim_playback(self, stim: dict):
        """Start playback for a specific stimulus type using handlers.
        
        Args:
            stim: Stimulus dictionary
        """
        stim_type = stim.get('type', '')
        logger.debug(f"Starting playback for stimulus type: {stim_type}")
        
        # Route to appropriate handler
        handler = self.handlers.get(stim_type)
        if not handler:
            logger.warning(f"No handler for stimulus type: {stim_type}, skipping")
            self.finish_current_stim()
            return
        try:
            handler.start(stim)
        except Exception as e:
            logger.error(f"Error starting stimulus playback: {e}", exc_info=True)
            self.gui_callback.playback_error(f"Failed to start {stim_type}: {e}")
    
    def _generate_tone(self, frequency: int, duration_ms: int,
                      sample_rate: int = 44100) -> np.ndarray:
        """Generate a pure tone.

        Args:
            frequency: Frequency in Hz
            duration_ms: Duration in milliseconds
            sample_rate: Sample rate in Hz

        Returns:
            Tone samples as int16 array
        """
        if frequency <= 0 or duration_ms <= 0:
            raise ValueError(f"Invalid tone parameters: freq={frequency}, duration={duration_ms}")

        duration_sec = duration_ms / 1000.0
        num_samples = int(sample_rate * duration_sec)
        t = np.linspace(0, duration_sec, num_samples, False)
        tone = np.sin(2 * np.pi * frequency * t)
        return (np.clip(tone, -1.0, 1.0) * 32767).astype(np.int16)
    
    def _generate_square_wave(self, frequency: int, duration_ms: int, 
                             sample_rate: int = 44100) -> np.ndarray:
        """Generate a square wave sync pulse.
        
        Args:
            frequency: Frequency in Hz
            duration_ms: Duration in milliseconds
            sample_rate: Sample rate in Hz
            
        Returns:
            Square wave as int16 array
            
        Raises:
            ValueError: If parameters are invalid
        """
        if frequency <= 0 or duration_ms <= 0:
            raise ValueError(f"Invalid square wave parameters: freq={frequency}, "
                           f"duration={duration_ms}")
        
        duration_sec = duration_ms / 1000.0
        num_samples = int(sample_rate * duration_sec)
        
        t = np.linspace(0, duration_sec, num_samples, False)
        square = np.sign(np.sin(2 * np.pi * frequency * t))
        
        return (square * 32767 * 0.8).astype(np.int16)
    
    def send_sync_pulse(self, patient_id: str):
        """Send a sync pulse to mark the EEG recording.
        
        Args:
            patient_id: Patient identifier for logging
        """
        logger.info(f"Sending sync pulse for patient: {patient_id}")
        
        # Generate square wave sync pulse
        sync_pulse = self._generate_square_wave(
            frequency=SyncPulseParams.FREQUENCY,
            duration_ms=SyncPulseParams.DURATION_MS,
            sample_rate=SyncPulseParams.SAMPLE_RATE
        )
        
        # Reshape for mono playback
        if sync_pulse.ndim == 1:
            sync_pulse = sync_pulse.reshape(-1, 1)
        
        # Record sync time
        sync_time = time.time()
        logger.info(f"Sync pulse generated at time: {sync_time}")
        
        # Play the sync pulse with completion callback
        try:
            callback = partial(self._handle_sync_pulse_complete, patient_id, sync_time)
            
            self.play_audio(
                samples=sync_pulse,
                sample_rate=SyncPulseParams.SAMPLE_RATE,
                callback=callback,
                log_label="sync_pulse"
            )
        except AudioError as e:
            logger.error(f"Failed to play sync pulse: {e}")
            self.gui_callback.playback_error(f"Sync pulse failed: {e}")
    
    def _handle_sync_pulse_complete(self, patient_id: str, sync_time: float):
        """Handle completion of sync pulse playback.
        
        Args:
            patient_id: Patient identifier
            sync_time: Timestamp of sync pulse
        """
        logger.info("Sync pulse playback completed")
        
        try:
            # Save sync event using results manager
            self.results_manager.append_sync_pulse(patient_id, sync_time)
            logger.info("Sync pulse event saved successfully")
        except Exception as e:
            logger.error(f"Error saving sync pulse: {e}", exc_info=True)
    
    def play_audio(self, samples: np.ndarray, sample_rate: int, 
                   callback=None, log_label: Optional[str] = None):
        """Play audio using the stream manager.
        
        Args:
            samples: Audio samples
            sample_rate: Sample rate in Hz
            callback: Optional callback when playback finishes
            log_label: Optional label for logging
        """
        # Log audio event with timestamp
        if log_label is not None:
            onset_time = time.time() + 0.01  # Approximate stream start latency
            self.current_stim_sentences.append({
                'event': log_label,
                'onset_time': onset_time
            })
            logger.debug(f"Audio event: {log_label} at {onset_time:.3f}")
        
        # Create finish handler
        def on_finish():
            # Only run callback if still playing (not paused or stopped)
            if self.gui_callback.state_manager.is_playing() and callback is not None:
                self._schedule(10, callback)
        
        # Play using stream manager
        try:
            self.stream_manager.play(
                samples=samples,
                sample_rate=sample_rate,
                on_finish=on_finish
            )
        except AudioError as e:
            logger.error(f"Audio playback error: {e}", exc_info=True)
            self.gui_callback.playback_error(str(e))
    
    def finish_current_stim(self):
        """Finish the current stimulus and move to next."""
        if not self.gui_callback.state_manager.is_playing():
            return

        patient_id = self.gui_callback.get_patient_id()
        stim = self.stims.stim_dictionary[self.stims.current_stim_index]
        stim_type = stim['type']

        logger.info(f"Stimulus {self.stims.current_stim_index + 1} completed: "
                   f"type={stim_type}, events={len(self.current_stim_sentences)}")

        # Check if this is an oddball stimulus
        is_oddball = stim_type in ('oddball', 'oddball+p')

        try:
            if is_oddball:
                # For oddball stimuli, save each beep as a separate row
                self._save_oddball_results(patient_id, stim_type)
            else:
                # For other stimuli, save as single row with all events
                self._save_standard_result(patient_id, stim_type)
        except Exception as e:
            logger.error(f"Failed to save stimulus result: {e}", exc_info=True)

        # Update stimulus status
        stim['status'] = 'completed'
        self.gui_callback.update_stim_list_status()
        self.stims.current_stim_index += 1

        # Inter-stimulus delay (randomized)
        delay = random.randint(
            TimingParams.INTER_STIMULUS_MIN_MS,
            TimingParams.INTER_STIMULUS_MAX_MS
        )
        logger.debug(f"Inter-stimulus delay: {delay}ms")
        self._schedule(delay, self.continue_playback)

    def _save_oddball_results(self, patient_id: str, stim_type: str):
        """Save oddball stimulus results - one row per beep.

        Args:
            patient_id: Patient identifier
            stim_type: Stimulus type
        """
        # Filter for tone events (standard_tone, rare_tone)
        tone_events = [
            event for event in self.current_stim_sentences
            if event.get('event') in ('standard_tone', 'rare_tone')
        ]

        logger.info(f"Saving {len(tone_events)} oddball beeps to CSV")

        # Save each beep as a separate row
        for event in tone_events:
            onset_time = event.get('onset_time', 0)
            tone_type = event.get('event', 'unknown')
            # Tone duration is 100ms = 0.1s
            tone_duration = 0.1

            beep_data = {
                'notes': tone_type,  # Just the tone type (standard_tone or rare_tone)
                'start_time': onset_time,
                'end_time': onset_time + tone_duration,
                'duration': tone_duration
            }

            self.results_manager.append_result(
                patient_id=patient_id,
                result_type=stim_type,
                data=beep_data
            )

    def _save_standard_result(self, patient_id: str, stim_type: str):
        """Save standard stimulus result - single row with clean notes.

        Args:
            patient_id: Patient identifier
            stim_type: Stimulus type
        """
        end_time = time.time()
        duration = (end_time - self.current_stim_start_time
                   if self.current_stim_start_time else 0)

        # Create clean notes based on stimulus type
        notes = self._format_stimulus_notes(stim_type, self.current_stim_sentences)

        stim_result_data = {
            'notes': notes,
            'start_time': self.current_stim_start_time,
            'end_time': end_time,
            'duration': duration
        }

        self.results_manager.append_result(
            patient_id=patient_id,
            result_type=stim_type,
            data=stim_result_data
        )

    def _format_stimulus_notes(self, stim_type: str, events: list) -> str:
        """Format clean notes for stimulus based on type.

        Args:
            stim_type: Type of stimulus
            events: List of event dictionaries

        Returns:
            Formatted notes string
        """
        if stim_type == 'language':
            # For language, just show the sentence IDs
            for event in events:
                if event.get('event') == 'language_stim_meta':
                    sentence_ids = event.get('sentence_ids', [])
                    if sentence_ids:
                        return f"Sentences: {sentence_ids}"
            return "Language stimulus"

        elif stim_type in ('right_command', 'right_command+p', 'left_command', 'left_command+p'):
            # For command stimuli, extract side and cycles
            side = 'right' if 'right' in stim_type else 'left'
            has_prompt = '+p' in stim_type

            # Look for command_stim_end event to get cycle count
            cycles = None
            for event in events:
                if event.get('event') == 'command_stim_end':
                    cycles = event.get('total_cycles')
                    break

            if cycles:
                prompt_str = " (with prompt)" if has_prompt else ""
                return f"{side.capitalize()} command: {cycles} cycles{prompt_str}"
            return f"{side.capitalize()} command stimulus"

        # For other stimuli, keep the full event log as JSON for now
        return json.dumps(events)
    
    def _log_event(self, event_type: str, metadata: Optional[dict] = None):
        """Log structured event with timestamp.
        
        Args:
            event_type: Type of event
            metadata: Optional metadata dictionary
        """
        event_data = {
            'event': event_type,
            'onset_time': time.time()
        }
        if metadata is not None:
            event_data.update(metadata)
        
        self.current_stim_sentences.append(event_data)
    
    def toggle_pause(self):
        """Toggle pause state. State transition already handled by app.py."""
        # Check current state (already transitioned by app.py)
        if self.gui_callback.state_manager.is_paused():
            # We just transitioned TO paused - stop audio and cancel callbacks
            logger.debug("Pausing playback")
            self.stream_manager.stop()
            self._cancel_scheduled_callbacks()

            # Reset all handler state - stimulus will restart from beginning on resume
            for handler in self.handlers.values():
                handler.reset()

            # Clear current stimulus event data (beeps, etc.)
            self.current_stim_sentences = []
            self.current_stim_start_time = 0

            # Update stimulus status for display
            current_stim = self.stims.stim_dictionary[self.stims.current_stim_index]
            current_stim['status'] = 'pending'
            self.gui_callback.update_stim_list_status()

        else:
            # We just transitioned TO playing - restart stimulus from beginning
            logger.debug("Resuming playback - restarting current stimulus")
            self.play_current_stim()
    
    def reset_current_stim_state(self):
        """Reset state variables for current stimulus."""
        if not self.stims.stim_dictionary:
            return

        self.current_stim_start_time = 0
        self.current_stim_sentences = []

        for handler in self.handlers.values():
            handler.reset()

        stim = self.stims.stim_dictionary[self.stims.current_stim_index]
        stim['status'] = 'pending'
    
    def stop_stimulus(self):
        """Stop all stimulus playback."""

        # Stop all handlers
        for handler in self.handlers.values():
            if handler.is_active:
                handler.stop()

        # Stop audio stream
        self.stream_manager.stop()

        # Cancel callbacks
        self._cancel_scheduled_callbacks()

        # Reset state
        self.reset_stim_state()
    
    def _schedule(self, delay_ms: int, callback):
        """Thread-safe callback scheduling.
        
        Args:
            delay_ms: Delay in milliseconds
            callback: Function to call
            
        Returns:
            Callback ID
        """
        cb_id = None
        
        def wrapped_callback():
            with self.callback_lock:
                if cb_id in self.scheduled_callbacks:
                    self.scheduled_callbacks.remove(cb_id)
            callback()
        
        cb_id = self.gui_callback.root.after(delay_ms, wrapped_callback)
        
        with self.callback_lock:
            self.scheduled_callbacks.append(cb_id)
        
        return cb_id
    
    def _cancel_scheduled_callbacks(self):
        """Cancel all scheduled callbacks."""
        with self.callback_lock:
            cancelled_count = len(self.scheduled_callbacks)
            
            for callback_id in self.scheduled_callbacks:
                try:
                    self.gui_callback.root.after_cancel(callback_id)
                except Exception as e:
                    logger.warning(f"Error canceling callback {callback_id}: {e}")
            
            self.scheduled_callbacks.clear()
            
            if cancelled_count > 0:
                logger.info(f"Cancelled {cancelled_count} scheduled callbacks")