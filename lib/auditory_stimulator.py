# lib/auditory_stimulator.py

import time
import random
import threading
import pandas as pd
import numpy as np
import sounddevice as sd
import logging
import json
from pathlib import Path
from typing import Optional

from lib.stim_handlers import (
    LanguageStimHandler,
    CommandStimHandler,
    OddballStimHandler,
    VoiceStimHandler
)

logger = logging.getLogger('eeg_stimulus.auditory_stimulator')


class AuditoryStimulator:
    """Manages audio playback for stimulus sessions with modular handlers."""
    
    def __init__(self, gui_callback):
        """Initialize the auditory stimulator with configuration"""
        self.gui_callback = gui_callback
        self.stims = gui_callback.stims
        self.config = gui_callback.config

        self.scheduled_callbacks = []
        self.callback_lock = threading.Lock()

        # Audio stream management
        self._active_stream = None
        self._stream_lock = threading.Lock()
        self._current_audio_buffer = None
        self._buffer_position = 0
        
        # File write lock for CSV safety
        self._csv_lock = threading.Lock()
        
        # Track current handler for additional safety
        self._current_handler = None
        
        # Initialize stimulus handlers
        self._init_handlers()
        
        self.reset_stim_state()
        logger.info("AuditoryStimulator initialized")
    
    def _init_handlers(self):
        """Initialize modular stimulus handlers."""
        self.handlers = {
            'language': LanguageStimHandler(self),
            'command': CommandStimHandler(self),
            'oddball': OddballStimHandler(self),
            'voice': VoiceStimHandler(self)
        }
        logger.debug("Stimulus handlers initialized")

    def reset_stim_state(self):
        """Reset stimulus state variables."""
        self.stims.current_stim_index = 0
        self.is_paused = False
        self.current_stim_start_time = 0
        self.current_stim_sentences = []
        
        # Reset all handlers
        for handler in self.handlers.values():
            handler.reset()
        
        # Reset all stimulus statuses to pending
        for stim in self.stims.stim_dictionary:
            stim['status'] = 'pending'
        
        logger.info("Stimulus state reset")

    def play_stim_sequence(self):
        """Start playing the stimulus sequence"""
        self.stims.current_stim_index = 0
        self.is_paused = False
        logger.info(f"Starting stimulus sequence with {len(self.stims.stim_dictionary)} stimuli")
        self.continue_playback()

    def continue_playback(self):
        """Continue playback from current stimulus index"""
        if self.gui_callback.playback_state != "playing" or self.is_paused:
            return
        if self.stims.current_stim_index >= len(self.stims.stim_dictionary):
            # Playback complete
            logger.info("Stimulus sequence completed")
            self.gui_callback.playback_complete()
            return

        # Play current stimulus
        self.play_current_stim()

    def play_current_stim(self):
        """Play the current stimulus and schedule the next one"""
        try:
            patient_id = self.gui_callback.get_patient_id()
            stim = self.stims.stim_dictionary[self.stims.current_stim_index]
            stim['status'] = 'in progress'
            
            logger.info(f"Starting stimulus {self.stims.current_stim_index + 1}/{len(self.stims.stim_dictionary)}: "
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
            
            # Start playing the stimulus using appropriate handler
            self.start_stim_playback(stim)
            self.gui_callback.update_stim_list_status()
            
        except Exception as e:
            logger.error(f"Error in play_current_stim: {e}", exc_info=True)
            self.gui_callback.playback_error(str(e))

    def start_stim_playback(self, stim: dict):
        """Start playback for a specific stimulus type using handlers."""
        stim_type = stim.get('type', '')
        logger.debug(f"Starting playback for stimulus type: {stim_type}")
        
        # Route to appropriate handler and track it
        if stim_type == "language":
            self._current_handler = self.handlers['language']
            self.handlers['language'].start(stim)
        elif "command" in stim_type:
            self._current_handler = self.handlers['command']
            self.handlers['command'].start(stim)
        elif "oddball" in stim_type:
            self._current_handler = self.handlers['oddball']
            self.handlers['oddball'].start(stim)
        elif stim_type in ["control", "loved_one_voice"]:
            self._current_handler = self.handlers['voice']
            self.handlers['voice'].start(stim)
        else:
            logger.warning(f"Unknown stimulus type: {stim_type}, skipping")
            self._current_handler = None
            self.finish_current_stim()

    def _generate_tone(self, frequency: int, duration_ms: int, sample_rate: int = 44100) -> np.ndarray:
        """Generate a pure tone with validation"""
        if frequency <= 0 or duration_ms <= 0:
            raise ValueError(f"Invalid tone parameters: freq={frequency}, duration={duration_ms}")
        
        tone_duration_sec = duration_ms / 1000.0
        tail_duration_sec = 0.1
        total_duration_sec = tone_duration_sec + tail_duration_sec

        num_samples = int(sample_rate * total_duration_sec)
        tone_samples = int(sample_rate * tone_duration_sec)

        t = np.linspace(0, tone_duration_sec, tone_samples, False)
        tone = np.sin(2 * np.pi * frequency * t)
        full = np.zeros(num_samples, dtype=np.float64)
        full[:tone_samples] = tone
        
        # Clip to prevent overflow
        full = np.clip(full, -1.0, 1.0)
        return (full * 32767).astype(np.int16)
    
    def _generate_square_wave(self, frequency: int, duration_ms: int, sample_rate: int = 44100) -> np.ndarray:
        """Generate a square wave sync pulse - highly detectable in EEG.
        
        Args:
            frequency: Frequency in Hz (typically 1000-2000 Hz for sync)
            duration_ms: Duration in milliseconds
            sample_rate: Sample rate in Hz
            
        Returns:
            Square wave as int16 numpy array
        """
        if frequency <= 0 or duration_ms <= 0:
            raise ValueError(f"Invalid square wave parameters: freq={frequency}, duration={duration_ms}")
        
        duration_sec = duration_ms / 1000.0
        num_samples = int(sample_rate * duration_sec)
        
        # Generate time array
        t = np.linspace(0, duration_sec, num_samples, False)
        
        # Square wave using sign of sine wave
        square = np.sign(np.sin(2 * np.pi * frequency * t))
        
        # Convert to int16
        return (square * 32767 * 0.8).astype(np.int16)  # 80% amplitude to avoid clipping
    
    def send_sync_pulse(self, patient_id: str):
        """Send a sync pulse to mark the EEG recording.
        
        This generates a distinctive square wave that creates a sharp artifact
        in the EEG recording, allowing precise synchronization of stimulus
        events with EEG data.
        
        Args:
            patient_id: Patient identifier for logging
        """
        logger.info(f"Sending sync pulse for patient: {patient_id}")
        
        # Get sync pulse parameters from config (with defaults)
        sync_freq = getattr(self.config, 'sync_pulse_frequency', 1000)  # Default 1000 Hz
        sync_duration = getattr(self.config, 'sync_pulse_duration_ms', 200)  # Default 200 ms
        
        logger.debug(f"Sync pulse parameters: {sync_freq}Hz, {sync_duration}ms")
        
        # Generate a square wave at configured frequency and duration
        # This creates a very distinctive artifact in EEG
        sync_pulse = self._generate_square_wave(
            frequency=sync_freq,
            duration_ms=sync_duration,
            sample_rate=44100
        )
        
        # Reshape for mono playback
        if sync_pulse.ndim == 1:
            sync_pulse = sync_pulse.reshape(-1, 1)
        
        # Log the sync pulse event
        sync_time = time.time()
        logger.info(f"Sync pulse generated at time: {sync_time}")
        logger.debug(f"Sync pulse will be saved for patient: {patient_id}")
        
        # Play the sync pulse
        def on_pulse_complete():
            logger.info("Sync pulse playback completed - calling _save_sync_event")
            try:
                # Save sync event to CSV
                self._save_sync_event(patient_id, sync_time)
                logger.info("_save_sync_event completed successfully")
            except Exception as e:
                logger.error(f"Error in _save_sync_event: {e}", exc_info=True)
        
        logger.debug("Starting sync pulse audio playback")
        self.play_audio(
            samples=sync_pulse,
            sample_rate=44100,
            callback=on_pulse_complete,
            log_label="sync_pulse"
        )
    
    def _save_sync_event(self, patient_id: str, sync_time: float):
        """Save sync pulse event to CSV."""
        logger.info(f"_save_sync_event called for patient: {patient_id}, time: {sync_time}")
        
        sync_row = {
            'patient_id': patient_id,
            'date': self.config.current_date,
            'stim_type': 'manual_sync_pulse',
            'sentences': json.dumps([{'event': 'sync_pulse', 'onset_time': sync_time}]),
            'start_time': sync_time,
            'end_time': sync_time + 0.2,  # 200ms duration
            'duration': 0.2,
            'notes': f'Manual sync pulse sent at {time.strftime("%H:%M:%S", time.localtime(sync_time))}'
        }
        
        logger.debug(f"Sync row data: {sync_row}")
        
        try:
            # Use config to get results path (consistent with save_single_stim_result)
            results_path = self.config.get_results_path(patient_id)
            logger.info(f"Results path for sync event: {results_path}")
            
            # Use shared CSV append helper
            self._append_to_results_csv(sync_row, results_path)
            logger.info(f"Sync pulse event saved successfully to {results_path}")
        except Exception as e:
            logger.error(f"Failed to save sync pulse event: {e}", exc_info=True)
            raise  # Re-raise so we see it in outer try-except
    
    def _append_to_results_csv(self, row_dict: dict, results_path: Path):
        """Append a single row to the results CSV file with thread-safe writing.
        
        Args:
            row_dict: Dictionary containing row data
            results_path: Path to the CSV file
        """
        logger.debug(f"_append_to_results_csv called for path: {results_path}")
        
        # Ensure results directory exists
        results_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directory created/verified: {results_path.parent}")
        
        # Thread-safe CSV writing
        with self._csv_lock:
            logger.debug("Acquired CSV lock")
            df = pd.DataFrame([row_dict])
            file_exists = results_path.exists()
            logger.debug(f"File exists: {file_exists}, writing with header: {not file_exists}")
            
            df.to_csv(results_path, mode='a', header=not file_exists, index=False)
            logger.debug(f"CSV write completed to {results_path}")

    def play_audio(self, samples: np.ndarray, sample_rate: int, 
                   callback=None, log_label: Optional[str] = None):
        """Play audio reliably using sounddevice OutputStream with thread safety"""
        # Stop any previous stream
        self._safe_stop_stream()
        
        # Validate and reshape samples
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)
        elif samples.ndim != 2:
            raise ValueError("Samples must be 1D or 2D array")
        
        # LOG: Audio event with precise timestamp
        if log_label is not None:
            onset_time = time.time() + 0.01  # Approximate stream start latency
            self.current_stim_sentences.append({
                'event': log_label,
                'onset_time': onset_time
            })
            logger.debug(f"Audio event: {log_label} at {onset_time:.3f}")
        
        # Thread-safe buffer management
        with self._stream_lock:
            self._current_audio_buffer = samples.copy()
            self._buffer_position = 0
        
        def stream_callback(outdata, frames, time_info, status):
            if status:
                logger.warning(f"Audio stream status: {status}")
            
            with self._stream_lock:
                buffer = self._current_audio_buffer
                position = self._buffer_position
                
                if buffer is None or position >= len(buffer):
                    outdata.fill(0)
                    raise sd.CallbackStop
                
                available = len(buffer) - position
                chunk_size = min(frames, available)
                chunk = buffer[position:position + chunk_size]
                self._buffer_position = position + chunk_size
            
            outdata[:chunk_size] = chunk
            if chunk_size < frames:
                outdata[chunk_size:] = 0

        def on_finish():
            logger.debug("Audio on_finish() callback FIRED")
            logger.debug(f"on_finish() - playback_state={self.gui_callback.playback_state}, is_paused={self.is_paused}")
            
            with self._stream_lock:
                logger.debug("on_finish() - clearing stream buffers...")
                self._current_audio_buffer = None
                self._buffer_position = 0
                self._active_stream = None
                logger.debug("on_finish() - stream buffers cleared")
            
            if log_label:
                logger.debug(f"Audio finished: {log_label}")
            
            # Execute callback if conditions met
            callback_will_run = (self.gui_callback.playback_state == "playing" 
                               and not self.is_paused 
                               and callback is not None)
            
            logger.debug(f"on_finish() - callback_will_run={callback_will_run} (state={self.gui_callback.playback_state}, paused={self.is_paused}, has_callback={callback is not None})")
            
            if callback_will_run:
                # Use try-except to catch any issues with the callback
                try:
                    logger.debug(f"on_finish() - scheduling callback in 10ms...")
                    self._schedule(10, callback)
                    logger.debug(f"on_finish() - callback scheduled successfully")
                except Exception as e:
                    logger.error(f"Error in audio finish callback: {e}", exc_info=True)
            else:
                logger.debug("on_finish() - callback NOT scheduled (conditions not met)")
            
            logger.debug("Audio on_finish() callback COMPLETE")

        try:
            stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=samples.shape[1],
                dtype='int16',
                callback=stream_callback,
                finished_callback=on_finish,
                latency='low'
            )
            
            with self._stream_lock:
                self._active_stream = stream
            
            stream.start()
            logger.debug(f"Audio stream started: {log_label or 'unlabeled'}, "
                        f"{len(samples)} samples, {sample_rate}Hz")
            
        except sd.PortAudioError as e:
            error_msg = f"Audio device error: {e}"
            logger.error(error_msg)
            self.gui_callback.playback_error(error_msg)
            on_finish()
        except Exception as e:
            error_msg = f"Audio playback failed: {e}"
            logger.error(error_msg, exc_info=True)
            self.gui_callback.playback_error(error_msg)
            on_finish()

    def finish_current_stim(self):
        """Finish the current stimulus and move to next"""
        if self.gui_callback.playback_state != "playing":
            return
            
        patient_id = self.gui_callback.get_patient_id()
        stim = self.stims.stim_dictionary[self.stims.current_stim_index]
        end_time = time.time()
        
        duration = end_time - self.current_stim_start_time if self.current_stim_start_time else 0
        logger.info(f"Stimulus {self.stims.current_stim_index + 1} completed: "
                   f"type={stim['type']}, duration={duration:.2f}s, "
                   f"events={len(self.current_stim_sentences)}")
        
        # Log stimulus end event
        self._log_event('stim_end', {
            'stim_index': self.stims.current_stim_index,
            'duration': duration,
            'event_count': len(self.current_stim_sentences)
        })
        
        stim_result = {
            'patient_id': patient_id,
            'date': self.config.current_date,
            'stim_type': stim['type'],
            'sentences': self.current_stim_sentences,
            'start_time': self.current_stim_start_time,
            'end_time': end_time,
            'duration': duration
        }
        
        self.save_single_stim_result(stim_result)
        stim['status'] = 'completed'
        self.gui_callback.update_stim_list_status()
        self.stims.current_stim_index += 1
        
        # Inter-stimulus delay (1.2-2.2 seconds randomized)
        delay = random.randint(1200, 2200)
        logger.debug(f"Inter-stimulus delay: {delay}ms")
        self._schedule(delay, self.continue_playback)

    def _log_event(self, event_type: str, metadata: Optional[dict] = None):
        """Helper to log structured events with timestamps"""
        event_data = {
            'event': event_type,
            'onset_time': time.time()
        }
        if metadata is not None:
            event_data.update(metadata)
        
        self.current_stim_sentences.append(event_data)

    def toggle_pause(self):
        """Toggle pause state"""
        if self.is_paused:
            logger.info("Resuming stimulus playback")
            self.is_paused = False
            self.gui_callback.playback_state = "playing"
            self.gui_callback.update_button_states()
            self.gui_callback.status_label.config(text="Resuming stimulus...", foreground="blue")
            self.reset_current_stim_state()
            self.play_current_stim()
        else:
            logger.info("Pausing stimulus playback - BEGIN PAUSE SEQUENCE")
            logger.debug(f"Current state: playback_state={self.gui_callback.playback_state}, is_paused={self.is_paused}")
            logger.debug(f"Current handler: {self._current_handler.__class__.__name__ if self._current_handler else 'None'}")
            logger.debug(f"Active stream exists: {self._active_stream is not None}")
            
            # Stop stream first
            logger.debug("Calling _safe_stop_stream()...")
            self._safe_stop_stream()
            logger.debug("_safe_stop_stream() completed")
            
            # Cancel callbacks
            logger.debug("Calling _cancel_scheduled_callbacks()...")
            self._cancel_scheduled_callbacks()
            logger.debug("_cancel_scheduled_callbacks() completed")
            
            # Set pause state
            self.is_paused = True
            current_stim = self.stims.stim_dictionary[self.stims.current_stim_index]
            current_stim['status'] = 'pending'
            logger.debug(f"Set current stim status to 'pending': {current_stim['type']}")
            
            self.gui_callback.update_stim_list_status()
            self.gui_callback.playback_state = "paused"
            self.gui_callback.update_button_states()
            self.gui_callback.status_label.config(text="Stimulus paused – will restart", foreground="orange")
            
            logger.info("Pausing stimulus playback - PAUSE SEQUENCE COMPLETE")

    def reset_current_stim_state(self):
        """Reset state variables for current stimulus"""
        if not self.stims.stim_dictionary:
            return

        stim = self.stims.stim_dictionary[self.stims.current_stim_index]
        logger.debug(f"Resetting stimulus state for stimulus {self.stims.current_stim_index}")
        
        self.current_stim_start_time = 0
        self.current_stim_sentences = []
        
        # Only reset the current handler (not all handlers)
        if self._current_handler:
            self._current_handler.reset()
            logger.debug(f"Reset current handler: {self._current_handler.__class__.__name__}")
        
        stim['status'] = 'pending'

    def _safe_stop_stream(self):
        """Safely stop and close the active audio stream"""
        logger.debug("_safe_stop_stream() - acquiring stream lock...")
        with self._stream_lock:
            logger.debug(f"_safe_stop_stream() - lock acquired, stream exists: {self._active_stream is not None}")
            if self._active_stream:
                try:
                    logger.debug(f"_safe_stop_stream() - stream active: {self._active_stream.active}")
                    # Force stop immediately without waiting for buffer to drain
                    if self._active_stream.active:
                        logger.debug("_safe_stop_stream() - calling stream.abort()...")
                        self._active_stream.abort()  # Use abort() instead of stop() for immediate halt
                        logger.debug("_safe_stop_stream() - stream.abort() completed")
                    
                    logger.debug("_safe_stop_stream() - calling stream.close()...")
                    self._active_stream.close()
                    logger.debug("_safe_stop_stream() - stream.close() completed")
                    logger.info("Audio stream stopped and closed")
                except Exception as e:
                    logger.error(f"Error stopping stream: {e}", exc_info=True)
                finally:
                    self._active_stream = None
                    self._current_audio_buffer = None
                    self._buffer_position = 0
                    logger.debug("_safe_stop_stream() - stream references cleared")
            else:
                logger.debug("_safe_stop_stream() - no active stream to stop")

    def stop_stimulus(self):
        """Stop all stimulus playback"""
        logger.info("Stopping all stimulus playback - BEGIN STOP SEQUENCE")
        logger.debug(f"Current state: playback_state={self.gui_callback.playback_state}, is_paused={self.is_paused}")
        logger.debug(f"Current handler: {self._current_handler.__class__.__name__ if self._current_handler else 'None'}")
        logger.debug(f"Active stream exists: {self._active_stream is not None}")
        
        # Clear current handler reference first
        logger.debug("Clearing current handler reference...")
        self._current_handler = None
        logger.debug("Current handler cleared")
        
        # Stop all handlers
        logger.debug("Stopping all handlers...")
        for name, handler in self.handlers.items():
            logger.debug(f"Stopping handler: {name}, active={handler.is_active}")
            handler.stop()
            logger.debug(f"Handler {name} stopped")
        logger.debug("All handlers stopped")
        
        # Stop stream
        logger.debug("Calling _safe_stop_stream()...")
        self._safe_stop_stream()
        logger.debug("_safe_stop_stream() completed")
        
        # Cancel callbacks
        logger.debug("Calling _cancel_scheduled_callbacks()...")
        self._cancel_scheduled_callbacks()
        logger.debug("_cancel_scheduled_callbacks() completed")
        
        # Reset state
        logger.debug("Calling reset_stim_state()...")
        self.reset_stim_state()
        logger.debug("reset_stim_state() completed")
        
        logger.info("Stopping all stimulus playback - STOP SEQUENCE COMPLETE")

    def save_single_stim_result(self, stim_result: dict):
        """Save a single stimulus result with thread-safe file writing"""
        patient_id = self.gui_callback.get_patient_id()
        results_path = self.config.get_results_path(patient_id)
        
        # Ensure consistent schema
        stim_result = {
            'patient_id': patient_id,
            'date': self.config.current_date,
            'stim_type': stim_result.get('stim_type', ''),
            'sentences': json.dumps(stim_result.get('sentences', [])),  # Serialize events as JSON
            'start_time': stim_result.get('start_time', ''),
            'end_time': stim_result.get('end_time', ''),
            'duration': stim_result.get('duration', ''),
            'notes': '' 
        }
        
        # Use shared CSV append helper
        try:
            self._append_to_results_csv(stim_result, results_path)
            logger.info(f"Stimulus result saved to {results_path}")
        except Exception as e:
            logger.error(f"Failed to save stimulus result: {e}", exc_info=True)

    def _schedule(self, delay_ms: int, callback):
        """Thread-safe callback scheduling"""
        # Create the wrapper first, capturing callback_id properly
        cb_id = None  # Pre-declare for closure
        
        def wrapped_callback():
            with self.callback_lock:
                if cb_id in self.scheduled_callbacks:
                    self.scheduled_callbacks.remove(cb_id)
            callback()
        
        # Now assign the actual ID
        cb_id = self.gui_callback.root.after(delay_ms, wrapped_callback)
        
        with self.callback_lock:
            self.scheduled_callbacks.append(cb_id)
        
        return cb_id
    
    def _cancel_scheduled_callbacks(self):
        """Thread-safe callback cancellation"""
        logger.debug("_cancel_scheduled_callbacks() - acquiring callback lock...")
        with self.callback_lock:
            cancelled_count = len(self.scheduled_callbacks)
            logger.debug(f"_cancel_scheduled_callbacks() - found {cancelled_count} scheduled callbacks")
            
            for i, callback_id in enumerate(self.scheduled_callbacks):
                try:
                    logger.debug(f"_cancel_scheduled_callbacks() - cancelling callback {i+1}/{cancelled_count} (id: {callback_id})")
                    self.gui_callback.root.after_cancel(callback_id)
                    logger.debug(f"_cancel_scheduled_callbacks() - cancelled callback {i+1}")
                except Exception as e:
                    logger.warning(f"Error canceling callback {callback_id}: {e}")
            
            self.scheduled_callbacks.clear()
            if cancelled_count > 0:
                logger.info(f"Cancelled {cancelled_count} scheduled callbacks")
            else:
                logger.debug("No scheduled callbacks to cancel")