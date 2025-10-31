# lib/auditory_stimulator.py

import os
import time
import random
import pandas as pd
import numpy as np
import sounddevice as sd

class AuditoryStimulator:
     
    def __init__(self, gui_callback):
        """Initialize the auditory stimulator with configuration"""

        # Initialize attributes
        self.gui_callback = gui_callback # Callback to TkApp for UI updates and state changes
        self.trials = gui_callback.trials
        self.config = gui_callback.config

        self.scheduled_callbacks = []  # Track Tkinter after IDs

        self.reset_trial_state()


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


    def play_trial_sequence(self):
        self.trials.current_trial_index = 0
        self.is_paused = False
        self.prompt = False
        self.continue_playback()

    def continue_playback(self):
        """Continue playback from current trial index"""
        if self.gui_callback.playback_state != "playing" or self.is_paused:
            return
        if self.trials.current_trial_index >= len(self.trials.trial_dictionary):
            # Playback complete
            self.gui_callback.playback_complete()
            return

        # Play current trial
        self.play_current_trial()

    def play_current_trial(self):
        """Play the current trial and schedule the next one"""
        try:
            patient_id = self.gui_callback.get_patient_id() # Get from GUI
            trial = self.trials.trial_dictionary[self.trials.current_trial_index]
            trial['status'] = 'in progress'
            # Initialize trial result storage
            self.current_trial_start_time = time.time()
            self.current_trial_sentences = []
            # Start playing the trial (non-blocking)
            self.start_trial_playback(trial)
            self.gui_callback.update_trial_list_status()
        except Exception as e:
            self.gui_callback.playback_error(str(e))

    def start_trial_playback(self, trial):
        """Start playback for a specific trial type"""
        trial_type = trial.get('type', '')

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
            # Unknown trial type, skip
            self.finish_current_trial()

    def start_lang_trial(self, trial):
        n = trial.get('audio_index', 0)
        if 0 <= n < len(self.trials.lang_audio):
            audio_segment = self.trials.lang_audio[n]
            samples = np.array(audio_segment.get_array_of_samples(), dtype=np.int16)
            if audio_segment.channels == 2:
                samples = samples.reshape(-1, 2)
            else:
                samples = samples.reshape(-1, 1)
            self.play_audio(
                samples=samples,
                sample_rate=audio_segment.frame_rate,
                callback=self.finish_current_trial,
                log_label="language_audio"
            )
        else:
            self.finish_current_trial()

    def start_cmd_trial(self, side, prompt=False):
        """Start a command trial (right or left)"""
        self.cmd_trial_side = side
        self.cmd_trial_cycle = 0
        self.cmd_trial_phase = "keep"  # "keep" or "stop" or "pause"
        if prompt:  # Play audio prompt
            # Convert prompt audio to numpy
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
            self._schedule(100, self.continue_cmd_trial) # Use callback for Tkinter after
            return
        if self.cmd_trial_cycle >= 8:
            # All 8 cycles complete
            self.finish_current_trial()
            return
        if self.cmd_trial_phase == "keep":
            # Play keep audio
            if self.cmd_trial_side == "right":
                audio = self.trials.right_keep_audio
                samples = np.array(audio.get_array_of_samples(), dtype=np.int16).reshape(-1, 1)
                self.play_audio(
                    samples=samples,
                    sample_rate=audio.frame_rate,
                    callback=lambda: self.set_cmd_phase("pause_after_keep"),
                    log_label="right_keep"
                )
            else: 
                audio = self.trials.left_keep_audio
                samples = np.array(audio.get_array_of_samples(), dtype=np.int16).reshape(-1, 1)
                self.play_audio(
                    samples=samples,
                    sample_rate=audio.frame_rate,
                    callback=lambda: self.set_cmd_phase("pause_after_keep"),
                    log_label="left_keep"
                )
        elif self.cmd_trial_phase == "pause_after_keep":
            # 10 second pause after keep
            self._schedule(10000, lambda: self.set_cmd_phase("stop"))
        elif self.cmd_trial_phase == "stop":
            # Play stop audio

            if self.cmd_trial_side == "right":
                audio = self.trials.right_stop_audio
                samples = np.array(audio.get_array_of_samples(), dtype=np.int16).reshape(-1, 1)
                self.play_audio(
                    samples=samples,
                    sample_rate=audio.frame_rate,
                    callback=lambda: self.set_cmd_phase("pause_after_stop"),
                    log_label="right_stop"
                )       
            else:
                audio = self.trials.left_stop_audio
                samples = np.array(audio.get_array_of_samples(), dtype=np.int16).reshape(-1, 1)
                self.play_audio(
                    samples=samples,
                    sample_rate=audio.frame_rate,
                    callback=lambda: self.set_cmd_phase("pause_after_stop"),
                    log_label="left_stop"
                )
        elif self.cmd_trial_phase == "pause_after_stop":
            # 10 second pause after stop
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
        if prompt:
            # Convert oddball prompt audio to numpy
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
        """Continue the oddball trial"""
        if self.gui_callback.playback_state != "playing" or self.is_paused:
            self._schedule(100, self.continue_oddball_trial) # Use callback
            return
        ################# first 5 beeps ##############
        if self.oddball_phase == "initial_standard":
            if self.oddball_tone_count < 5:
                # Play standard tone
                tone_samples = self._generate_tone(1000, 100)  # standard
                self.play_audio(
                    samples=tone_samples,
                    sample_rate=44100,
                    callback=lambda: self._schedule(1000, self.continue_oddball_trial),
                    log_label="standard_tone"
                )
                self.oddball_tone_count += 1
            else:
                # Switch to main sequence
                self.oddball_phase = "main_sequence"
                self.oddball_tone_count = 0
                self.continue_oddball_trial()
        ################## next 20 beeps ###############
        elif self.oddball_phase == "main_sequence":
            if self.oddball_tone_count < 20:
                # 20% chance for rare tone
                if random.random() < 0.2:
                    tone_samples = self._generate_tone(2000, 100)  # rare
                    self.play_audio(
                        samples=tone_samples,
                        sample_rate=44100,
                        callback=lambda: self._schedule(1000, self.continue_oddball_trial),
                        log_label="rare_tone"
                    )
                else:
                    tone_samples = self._generate_tone(1000, 100)  # standard
                    self.play_audio(
                        samples=tone_samples,
                        sample_rate=44100,
                        callback=lambda: self._schedule(1000, self.continue_oddball_trial),
                        log_label="standard_tone"
                    )
                self.oddball_tone_count += 1
            else:
                # Oddball trial complete
                self.finish_current_trial()

    def start_voice_trial(self, voice_type):
        audio_data = self.trials.control_voice_audio if voice_type == "control" else self.trials.loved_one_voice_audio
        if audio_data is not None:
            # Ensure it's int16
            if audio_data.dtype != np.int16:
                audio_data = audio_data.astype(np.int16)
            self.play_audio(
                samples=audio_data,
                sample_rate=self.trials.sample_rate,
                callback=self.finish_current_trial,
                log_label=f"{voice_type}_voice"
            )
        else:
            self.finish_current_trial()

    def _generate_tone(self, frequency, duration_ms, sample_rate=44100):
        """Generate a pure tone with a short silence tail for reliable playback."""
        # Actual tone duration
        tone_duration_sec = duration_ms / 1000.0
        # Add 100ms silence tail (adjustable)
        tail_duration_sec = 0.1  # 100 ms
        total_duration_sec = tone_duration_sec + tail_duration_sec

        num_samples = int(sample_rate * total_duration_sec)
        tone_samples = int(sample_rate * tone_duration_sec)

        # Create full buffer (tone + silence)
        t = np.linspace(0, tone_duration_sec, tone_samples, False)
        tone = np.sin(2 * np.pi * frequency * t)
        full = np.zeros(num_samples, dtype=np.float64)
        full[:tone_samples] = tone
        return (full * 32767).astype(np.int16)

    def play_audio(self, samples, sample_rate, callback=None, log_label=None):
        """
        Play audio reliably using OutputStream.
        """
        # Cancel any previous playback
        if hasattr(self, '_active_stream'):
            self._safe_stop_stream()

        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)
        elif samples.ndim != 2:
            raise ValueError("Samples must be 1D or 2D array")

        if log_label is not None:
            onset_time = time.time() + 0.01
            self.current_trial_sentences.append({
                'event': log_label,
                'onset_time': onset_time
            })

        # Make a copy to avoid external mutation
        audio_buffer = samples.copy()

        def stream_callback(outdata, frames, time_info, status):
            nonlocal audio_buffer
            if status:
                print(f"Audio stream warning: {status}")
            if len(audio_buffer) == 0:
                outdata.fill(0)
                raise sd.CallbackStop
            chunk = audio_buffer[:frames]
            outdata[:len(chunk)] = chunk
            audio_buffer = audio_buffer[len(chunk):]

        def on_finish():
            if hasattr(self, '_active_stream'):
                del self._active_stream
            # Only invoke callback if playback is still active
            if (self.gui_callback.playback_state == "playing" and not self.is_paused and callback is not None):
                self._schedule(10, callback)

        try:
            stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=samples.shape[1],
                dtype='int16',
                callback=stream_callback,
                finished_callback=on_finish
            )
            stream.start()
            self._active_stream = stream
        except Exception as e:
            self.gui_callback.playback_error(f"Audio playback failed: {e}")
            on_finish()

    def finish_current_trial(self):
        """Finish the current trial and move to next"""
        if self.gui_callback.playback_state != "playing":
            return
        # Record the trial result
        patient_id = self.gui_callback.get_patient_id() # Use callback
        trial = self.trials.trial_dictionary[self.trials.current_trial_index]
        end_time = time.time()
        trial_result = {
            'patient_id': patient_id,
            'date': self.config.current_date,
            'trial_type': trial['type'],
            'sentences': self.current_trial_sentences,
            'start_time': self.current_trial_start_time,
            'end_time': end_time,
            'duration': end_time - self.current_trial_start_time if self.current_trial_start_time is not None else 0
        }
        
        self.save_single_trial_result(trial_result)
        # Mark as completed
        trial['status'] = 'completed'
        # Notify GUI to update trial list
        self.gui_callback.update_trial_list_status()
        # Move to next trial
        self.trials.current_trial_index += 1
        # Add inter-trial delay (1.2-2.2 seconds)
        delay = random.randint(1200, 2200)  # random int in milliseconds
        self._schedule(delay, self.continue_playback)

    def toggle_pause(self):
        if self.is_paused:
            # Resume: restart the current trial from the beginning
            self.is_paused = False
            self.gui_callback.playback_state = "playing"
            self.gui_callback.update_button_states()
            self.gui_callback.status_label.config(text="Resuming stimulus...", foreground="blue")
            # Reset trial state and replay it
            self.reset_current_trial_state()
            self.play_current_trial()  # Start over
        else:
            # Pause: stop everything and mark trial as pending
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
        """Reset state variables specific to the current trial to prepare for a clean restart."""
        if not self.trials.trial_dictionary:
            return

        trial = self.trials.trial_dictionary[self.trials.current_trial_index]

        # Reset common trial state
        self.current_trial_start_time = None
        self.current_trial_sentences = []
        self.prompt = False  # Safe to reset; will be set again based on trial_type

        # Reset command-specific state
        self.cmd_trial_side = None
        self.cmd_trial_cycle = 0
        self.cmd_trial_phase = None

        # Reset oddball-specific state
        self.oddball_tone_count = 0
        self.oddball_phase = None

        # Reset delay-related state (if you keep any simple delays)
        self.delay_callback = None

        # Mark trial as pending (in case it was 'in progress')
        trial['status'] = 'pending'

    def _safe_stop_stream(self):
        """Safely stop and close the active audio stream, if it exists and is alive."""
        if hasattr(self, '_active_stream'):
            try:
                # Check if stream is still active
                if self._active_stream.active or self._active_stream.stopped:
                    self._active_stream.stop()
                self._active_stream.close()
            except Exception as e:
                # Ignore errors — stream may already be closed
                pass
            finally:
                del self._active_stream

    def stop_stimulus(self):
        self._safe_stop_stream()
        self.reset_trial_state()

    def save_single_trial_result(self, trial_result):
        """Save a single trial result to the stimulus results CSV file by appending"""
        results_dir = self.config.file.get('result_dir', 'patient_data/results')
        patient_id = self.gui_callback.get_patient_id()
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, f"{patient_id}_{self.config.current_date}_stimulus_results.csv")
        
        # Ensure consistent schema: add 'notes' field if missing
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
        
        df = pd.DataFrame([trial_result])
        df.to_csv(results_path, mode='a', header=not os.path.exists(results_path), index=False)

    def _schedule(self, delay_ms, callback):
        """Wrapper to track scheduled callbacks"""
        id = self.gui_callback.root.after(delay_ms, callback)
        self.scheduled_callbacks.append(id)
        return id
    
    def _cancel_scheduled_callbacks(self):
        """Cancel all pending Tkinter after callbacks"""
        for id in self.scheduled_callbacks:
            self.gui_callback.root.after_cancel(id)
        self.scheduled_callbacks.clear()
    
