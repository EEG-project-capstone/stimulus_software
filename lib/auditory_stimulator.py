# lib/auditory_stimulator.py

import os
import time
import random
import threading
import pandas as pd
import numpy as np
import sounddevice as sd
import logging

logger = logging.getLogger('eeg_stimulus.auditory_stimulator')


class AuditoryStimulator:
     
    def __init__(self, gui_callback):
        """Initialize the auditory stimulator with configuration"""
        self.gui_callback = gui_callback
        self.trials = gui_callback.trials
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
        
        self.reset_trial_state()
        logger.info("AuditoryStimulator initialized")

    def reset_trial_state(self):
        self.trials.current_trial_index = 0
        self.is_paused = False
        self.current_trial_start_time = 0
        self.current_trial_sentences = []
        
        # Command trial state
        self.cmd_trial_side = None
        self.cmd_trial_cycle = 0
        self.cmd_trial_phase = None
        
        # Oddball trial state
        self.oddball_tone_count = 0
        self.oddball_phase = None

        # Reset all trial statuses to pending
        for trial in self.trials.trial_dictionary:
            trial['status'] = 'pending'
        
        logger.info("Trial state reset")

    def play_trial_sequence(self):
        """Start playing the trial sequence"""
        self.trials.current_trial_index = 0
        self.is_paused = False
        self.prompt = False
        logger.info(f"Starting trial sequence with {len(self.trials.trial_dictionary)} trials")
        self.continue_playback()

    def continue_playback(self):
        """Continue playback from current trial index"""
        if self.gui_callback.playback_state != "playing" or self.is_paused:
            return
        if self.trials.current_trial_index >= len(self.trials.trial_dictionary):
            # Playback complete
            logger.info("Trial sequence completed")
            self.gui_callback.playback_complete()
            return

        # Play current trial
        self.play_current_trial()

    def play_current_trial(self):
        """Play the current trial and schedule the next one"""
        try:
            patient_id = self.gui_callback.get_patient_id()
            trial = self.trials.trial_dictionary[self.trials.current_trial_index]
            trial['status'] = 'in progress'
            
            # LOG: Trial start
            logger.info(f"Starting trial {self.trials.current_trial_index + 1}/{len(self.trials.trial_dictionary)}: "
                       f"type={trial.get('type')}, patient={patient_id}")
            
            # Initialize trial result storage
            self.current_trial_start_time = time.time()
            self.current_trial_sentences = []
            
            # Log trial start event
            self._log_event('trial_start', {
                'trial_index': self.trials.current_trial_index,
                'trial_type': trial.get('type'),
                'patient_id': patient_id
            })
            
            # Start playing the trial (non-blocking)
            self.start_trial_playback(trial)
            self.gui_callback.update_trial_list_status()
            
        except Exception as e:
            logger.error(f"Error in play_current_trial: {e}", exc_info=True)
            self.gui_callback.playback_error(str(e))

    def start_trial_playback(self, trial):
        """Start playback for a specific trial type"""
        trial_type = trial.get('type', '')
        logger.debug(f"Starting playback for trial type: {trial_type}")

        if trial_type == "language":
            self.start_lang_trial(trial)
        elif trial_type == "right_command":
            self.start_cmd_trial("right")
        elif trial_type == "right_command+p":
            self.start_cmd_trial("right", prompt=True)
        elif trial_type == "left_command":
            self.start_cmd_trial("left")
        elif trial_type == "left_command+p":
            self.start_cmd_trial("left", prompt=True)
        elif trial_type == "oddball":
            self.start_oddball_trial()
        elif trial_type == "oddball+p":
            self.start_oddball_trial(prompt=True)
        elif trial_type == "control":
            self.start_voice_trial("control")
        elif trial_type == "loved_one_voice":
            self.start_voice_trial("loved_one")
        else:
            logger.warning(f"Unknown trial type: {trial_type}, skipping")
            self.finish_current_trial()

    def start_lang_trial(self, trial):
        """Start a language trial"""
        n = trial.get('audio_index', 0)
        logger.debug(f"Starting language trial with audio index {n}")
        
        if 0 <= n < len(self.trials.lang_audio):
            audio_segment = self.trials.lang_audio[n]
            samples = np.array(audio_segment.get_array_of_samples(), dtype=np.int16)
            
            if audio_segment.channels == 2:
                samples = samples.reshape(-1, 2)
            else:
                samples = samples.reshape(-1, 1)
            
            # Log language trial metadata
            self._log_event('language_trial_meta', {
                'audio_index': n,
                'sentence_ids': self.trials.lang_trials_ids[n] if n < len(self.trials.lang_trials_ids) else None,
                'duration_sec': len(samples) / audio_segment.frame_rate
            })
            
            self.play_audio(
                samples=samples,
                sample_rate=audio_segment.frame_rate,
                callback=self.finish_current_trial,
                log_label="language_audio"
            )
        else:
            logger.error(f"Invalid language audio index: {n}")
            self.finish_current_trial()

    def start_cmd_trial(self, side, prompt=False):
        """Start a command trial (right or left)"""
        self.cmd_trial_side = side
        self.cmd_trial_cycle = 0
        self.cmd_trial_phase = "keep"
        
        logger.info(f"Starting {side} command trial (prompt={prompt})")
        self._log_event('command_trial_start', {'side': side, 'prompt': prompt})
        
        if prompt:
            prompt_seg = self.trials.motor_prompt_audio
            samples = np.array(prompt_seg.get_array_of_samples(), dtype=np.int16)
            
            if prompt_seg.channels == 2:
                samples = samples.reshape(-1, 2)
            else:
                samples = samples.reshape(-1, 1)

            self.play_audio(
                samples=samples,
                sample_rate=prompt_seg.frame_rate,
                callback=lambda: self._schedule(2000, self.continue_cmd_trial),
                log_label="motor_prompt"
            )
        else:
            self.continue_cmd_trial()

    def continue_cmd_trial(self):
        """Continue the command trial cycle"""
        if self.gui_callback.playback_state != "playing" or self.is_paused:
            self._schedule(100, self.continue_cmd_trial)
            return
            
        if self.cmd_trial_cycle >= 8:
            logger.debug(f"Command trial completed: {self.cmd_trial_side}, 8 cycles")
            self._log_event('command_trial_end', {
                'side': self.cmd_trial_side,
                'total_cycles': 8
            })
            self.finish_current_trial()
            return
            
        if self.cmd_trial_phase == "keep":
            # Log cycle start
            logger.debug(f"Command trial cycle {self.cmd_trial_cycle + 1}/8: KEEP phase")
            self._log_event('command_cycle', {
                'cycle': self.cmd_trial_cycle + 1,
                'phase': 'keep',
                'side': self.cmd_trial_side
            })
            
            if self.cmd_trial_side == "right":
                audio = self.trials.right_keep_audio
            else: 
                audio = self.trials.left_keep_audio
                
            samples = np.array(audio.get_array_of_samples(), dtype=np.int16).reshape(-1, 1)
            self.play_audio(
                samples=samples,
                sample_rate=audio.frame_rate,
                callback=lambda: self.set_cmd_phase("pause_after_keep"),
                log_label=f"{self.cmd_trial_side}_keep"
            )
            
        elif self.cmd_trial_phase == "pause_after_keep":
            self._log_event('command_pause', {'after': 'keep', 'duration_ms': 10000})
            self._schedule(10000, lambda: self.set_cmd_phase("stop"))
            
        elif self.cmd_trial_phase == "stop":
            logger.debug(f"Command trial cycle {self.cmd_trial_cycle + 1}/8: STOP phase")
            self._log_event('command_cycle', {
                'cycle': self.cmd_trial_cycle + 1,
                'phase': 'stop',
                'side': self.cmd_trial_side
            })
            
            if self.cmd_trial_side == "right":
                audio = self.trials.right_stop_audio
            else:
                audio = self.trials.left_stop_audio
                
            samples = np.array(audio.get_array_of_samples(), dtype=np.int16).reshape(-1, 1)
            self.play_audio(
                samples=samples,
                sample_rate=audio.frame_rate,
                callback=lambda: self.set_cmd_phase("pause_after_stop"),
                log_label=f"{self.cmd_trial_side}_stop"
            )
            
        elif self.cmd_trial_phase == "pause_after_stop":
            self._log_event('command_pause', {'after': 'stop', 'duration_ms': 10000})
            self._schedule(10000, lambda: self.next_cmd_cycle())

    def set_cmd_phase(self, phase):
        """Set the command trial phase"""
        self.cmd_trial_phase = phase
        self.continue_cmd_trial()

    def next_cmd_cycle(self):
        """Move to next command cycle"""
        self.cmd_trial_cycle += 1
        self.cmd_trial_phase = "keep"
        self.continue_cmd_trial()

    def start_oddball_trial(self, prompt=False):
        """Start an oddball trial"""
        self.oddball_tone_count = 0
        self.oddball_phase = "initial_standard"
        self.current_trial_sentences = []
        
        logger.info(f"Starting oddball trial (prompt={prompt})")
        self._log_event('oddball_trial_start', {'prompt': prompt})
        
        if prompt:
            prompt_seg = self.trials.oddball_prompt_audio
            samples = np.array(prompt_seg.get_array_of_samples(), dtype=np.int16)
            
            if prompt_seg.channels == 2:
                samples = samples.reshape(-1, 2)
            else:
                samples = samples.reshape(-1, 1)
                
            self.play_audio(
                samples=samples,
                sample_rate=prompt_seg.frame_rate,
                callback=lambda: self._schedule(2000, self.continue_oddball_trial),
                log_label="oddball_prompt"
            )
        else:
            self.continue_oddball_trial()

    def continue_oddball_trial(self):
        """Continue the oddball trial - FIXED VERSION"""
        if self.gui_callback.playback_state != "playing" or self.is_paused:
            self._schedule(100, self.continue_oddball_trial)
            return
            
        ################# first 5 beeps ##############
        if self.oddball_phase == "initial_standard":
            if self.oddball_tone_count < 5:
                # INCREMENT FIRST to prevent double beeps
                self.oddball_tone_count += 1
                logger.debug(f"Oddball initial tone {self.oddball_tone_count}/5")
                
                tone_samples = self._generate_tone(1000, 100)
                self.play_audio(
                    samples=tone_samples,
                    sample_rate=44100,
                    callback=lambda: self._schedule(900, self.continue_oddball_trial),
                    log_label="standard_tone"
                )
            else:
                logger.debug("Oddball switching to main sequence")
                self._log_event('oddball_phase_change', {
                    'from': 'initial_standard',
                    'to': 'main_sequence'
                })
                self.oddball_phase = "main_sequence"
                self.oddball_tone_count = 0
                self.continue_oddball_trial()
                
        ################## next 20 beeps ###############
        elif self.oddball_phase == "main_sequence":
            if self.oddball_tone_count < 20:
                # INCREMENT FIRST to prevent double beeps
                self.oddball_tone_count += 1
                
                # 20% chance for rare tone
                is_rare = random.random() < 0.2
                frequency = 2000 if is_rare else 1000
                label = "rare_tone" if is_rare else "standard_tone"
                
                logger.debug(f"Oddball main tone {self.oddball_tone_count}/20: {label}")
                
                tone_samples = self._generate_tone(frequency, 100)
                self.play_audio(
                    samples=tone_samples,
                    sample_rate=44100,
                    callback=lambda: self._schedule(900, self.continue_oddball_trial),
                    log_label=label
                )
            else:
                logger.debug("Oddball trial sequence completed")
                self._log_event('oddball_trial_end', {
                    'total_tones': 25  # 5 initial + 20 main
                })
                self.finish_current_trial()

    def start_voice_trial(self, voice_type):
        """Start a voice trial (control or loved_one)"""
        logger.info(f"Starting {voice_type} voice trial")
        self._log_event('voice_trial_start', {'voice_type': voice_type})
        
        audio_data = self.trials.control_voice_audio if voice_type == "control" else self.trials.loved_one_voice_audio
        
        if audio_data is None:
            logger.error(f"{voice_type} audio data is None, skipping trial")
            self.finish_current_trial()
            return
        
        # Ensure int16 format
        if audio_data.dtype != np.int16:
            logger.warning(f"{voice_type} audio has dtype {audio_data.dtype}, converting...")
            if np.issubdtype(audio_data.dtype, np.floating):
                audio_data = (audio_data * 32767).astype(np.int16)
            else:
                audio_data = audio_data.astype(np.int16)
        
        self._log_event('voice_trial_meta', {
            'voice_type': voice_type,
            'duration_sec': len(audio_data) / self.trials.sample_rate,
            'shape': audio_data.shape
        })
        
        self.play_audio(
            samples=audio_data,
            sample_rate=self.trials.sample_rate,
            callback=self.finish_current_trial,
            log_label=f"{voice_type}_voice"
        )

    def _generate_tone(self, frequency, duration_ms, sample_rate=44100):
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
        return (full * 32767).astype(np.int16)

    def play_audio(self, samples, sample_rate, callback=None, log_label=None):
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
            self.current_trial_sentences.append({
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
            with self._stream_lock:
                self._current_audio_buffer = None
                self._buffer_position = 0
                self._active_stream = None
            
            if log_label:
                logger.debug(f"Audio finished: {log_label}")
            
            if (self.gui_callback.playback_state == "playing" 
                and not self.is_paused 
                and callback is not None):
                self._schedule(10, callback)

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

    def finish_current_trial(self):
        """Finish the current trial and move to next"""
        if self.gui_callback.playback_state != "playing":
            return
            
        patient_id = self.gui_callback.get_patient_id()
        trial = self.trials.trial_dictionary[self.trials.current_trial_index]
        end_time = time.time()
        
        # LOG: Trial completion
        duration = end_time - self.current_trial_start_time if self.current_trial_start_time else 0
        logger.info(f"Trial {self.trials.current_trial_index + 1} completed: "
                   f"type={trial['type']}, duration={duration:.2f}s, "
                   f"events={len(self.current_trial_sentences)}")
        
        # Log trial end event
        self._log_event('trial_end', {
            'trial_index': self.trials.current_trial_index,
            'duration': duration,
            'event_count': len(self.current_trial_sentences)
        })
        
        trial_result = {
            'patient_id': patient_id,
            'date': self.config.current_date,
            'trial_type': trial['type'],
            'sentences': self.current_trial_sentences,
            'start_time': self.current_trial_start_time,
            'end_time': end_time,
            'duration': duration
        }
        
        self.save_single_trial_result(trial_result)
        trial['status'] = 'completed'
        self.gui_callback.update_trial_list_status()
        self.trials.current_trial_index += 1
        
        # Inter-trial delay
        delay = random.randint(1200, 2200)
        logger.debug(f"Inter-trial delay: {delay}ms")
        self._schedule(delay, self.continue_playback)

    def _log_event(self, event_type, metadata=None):
        """Helper to log structured events with timestamps"""
        event_data = {
            'event': event_type,
            'onset_time': time.time()
        }
        if metadata:
            event_data.update(metadata)
        
        self.current_trial_sentences.append(event_data)

    def toggle_pause(self):
        """Toggle pause state"""
        if self.is_paused:
            logger.info("Resuming stimulus playback")
            self.is_paused = False
            self.gui_callback.playback_state = "playing"
            self.gui_callback.update_button_states()
            self.gui_callback.status_label.config(text="Resuming stimulus...", foreground="blue")
            self.reset_current_trial_state()
            self.play_current_trial()
        else:
            logger.info("Pausing stimulus playback")
            self._safe_stop_stream()
            self._cancel_scheduled_callbacks()
            self.is_paused = True
            current_trial = self.trials.trial_dictionary[self.trials.current_trial_index]
            current_trial['status'] = 'pending'
            self.gui_callback.update_trial_list_status()
            self.gui_callback.playback_state = "paused"
            self.gui_callback.update_button_states()
            self.gui_callback.status_label.config(text="Stimulus paused – trial will restart", foreground="orange")

    def reset_current_trial_state(self):
        """Reset state variables for current trial"""
        if not self.trials.trial_dictionary:
            return

        trial = self.trials.trial_dictionary[self.trials.current_trial_index]
        logger.debug(f"Resetting trial state for trial {self.trials.current_trial_index}")
        
        self.current_trial_start_time = 0
        self.current_trial_sentences = []
        self.prompt = False
        
        # Reset command-specific state
        self.cmd_trial_side = None
        self.cmd_trial_cycle = 0
        self.cmd_trial_phase = None
        
        # Reset oddball-specific state
        self.oddball_tone_count = 0
        self.oddball_phase = None
        
        trial['status'] = 'pending'

    def _safe_stop_stream(self):
        """Safely stop and close the active audio stream"""
        with self._stream_lock:
            if self._active_stream:
                try:
                    if self._active_stream.active:
                        self._active_stream.stop()
                    self._active_stream.close()
                    logger.debug("Audio stream stopped")
                except Exception as e:
                    logger.warning(f"Error stopping stream: {e}")
                finally:
                    self._active_stream = None
                    self._current_audio_buffer = None
                    self._buffer_position = 0

    def stop_stimulus(self):
        """Stop all stimulus playback"""
        logger.info("Stopping all stimulus playback")
        self._safe_stop_stream()
        self._cancel_scheduled_callbacks()
        self.reset_trial_state()

    def save_single_trial_result(self, trial_result):
        """Save a single trial result with thread-safe file writing"""
        results_dir = self.config.file.get('result_dir', 'patient_data/results')
        patient_id = self.gui_callback.get_patient_id()
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, f"{patient_id}_{self.config.current_date}_stimulus_results.csv")
        
        # Ensure consistent schema
        trial_result = {
            'patient_id': patient_id,
            'date': self.config.current_date,
            'trial_type': trial_result.get('trial_type', ''),
            'sentences': trial_result.get('sentences', ''),
            'start_time': trial_result.get('start_time', ''),
            'end_time': trial_result.get('end_time', ''),
            'duration': trial_result.get('duration', ''),
            'notes': '' 
        }
        
        # Thread-safe CSV writing
        with self._csv_lock:
            try:
                df = pd.DataFrame([trial_result])
                file_exists = os.path.exists(results_path)
                df.to_csv(results_path, mode='a', header=not file_exists, index=False)
                logger.info(f"Trial result saved to {results_path}")
            except Exception as e:
                logger.error(f"Failed to save trial result: {e}", exc_info=True)

    def _schedule(self, delay_ms, callback):
        """Thread-safe callback scheduling"""
        def wrapped_callback():
            with self.callback_lock:
                if callback_id in self.scheduled_callbacks:
                    self.scheduled_callbacks.remove(callback_id)
            callback()
        
        callback_id = self.gui_callback.root.after(delay_ms, wrapped_callback)
        
        with self.callback_lock:
            self.scheduled_callbacks.append(callback_id)
        
        return callback_id
    
    def _cancel_scheduled_callbacks(self):
        """Thread-safe callback cancellation"""
        with self.callback_lock:
            cancelled_count = len(self.scheduled_callbacks)
            for callback_id in self.scheduled_callbacks:
                try:
                    self.gui_callback.root.after_cancel(callback_id)
                except Exception as e:
                    logger.warning(f"Error canceling callback {callback_id}: {e}")
            self.scheduled_callbacks.clear()
            if cancelled_count > 0:
                logger.debug(f"Cancelled {cancelled_count} scheduled callbacks")