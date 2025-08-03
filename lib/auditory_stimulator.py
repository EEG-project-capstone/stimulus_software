# lib/auditory_stimulator.py

import os
import time
import random
import pandas as pd
import sounddevice as sd
from pydub.generators import Sine

class AuditoryStimulator:
     
    def __init__(self, gui_callback):
        """Initialize the auditory stimulator with configuration"""

        # Initialize attributes
        self.gui_callback = gui_callback # Callback to TkApp for UI updates and state changes
        self.trials = gui_callback.trials
        self.config = gui_callback.config

        self.reset_trial_state()


    def reset_trial_state(self):
        self.trials.current_trial_index = 0
        self.is_paused = False
        self.current_trial_start_time = None
        self.current_trial_sentences = []
        # Command trial state
        self.cmd_trial_side = None
        self.cmd_trial_cycle = 0
        self.cmd_trial_phase = None
        # Oddball trial state
        self.oddball_tone_count = 0
        self.oddball_phase = None
        # Delay state
        self.delay_end_time = None
        self.delay_callback = None

        # Reset all trial statuses to pending
        for trial in self.trials.trial_dictionary:
            trial['status'] = 'pending'


    def play_trial_sequence(self):
        self.trials.current_trial_index = 0
        self.is_paused = False
        self.continue_playback()

    def continue_playback(self):
        """Continue playback from current trial index"""
        if self.gui_callback.get_playback_state() != "playing" or self.is_paused:
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
            self.start_trial_playback(trial, patient_id)
            self.gui_callback.update_trial_list_status()
        except Exception as e:
            self.gui_callback.playback_error(str(e))

    def start_trial_playback(self, trial, patient_id):
        """Start playback for a specific trial type"""

        trial_type = trial.get('type', '')

        if trial_type == "lang":
            self.start_lang_trial(trial)
        elif trial_type == "rcmd":
            self.start_cmd_trial("right")
        elif trial_type == "lcmd":
            self.start_cmd_trial("left")
        elif trial_type == "oddball":
            self.start_oddball_trial()
        elif trial_type == "control":
            self.start_voice_trial("control")
        elif trial_type == "loved_one":
            self.start_voice_trial("loved_one")
        else:
            # Unknown trial type, skip
            self.finish_current_trial()

    def start_lang_trial(self, trial):
        """Start a language trial"""
        n = trial.get('audio_index', 0)
        if 0 <= n < len(self.trials.lang_trials_ids):
            self.current_trial_sentences = self.trials.lang_trials_ids[n]
        if 0 <= n < len(self.trials.lang_audio):
            # Convert pydub AudioSegment to numpy array for sounddevice
            audio_segment = self.trials.lang_audio[n]
            samples = audio_segment.get_array_of_samples()
            if audio_segment.channels == 2:
                samples = samples.reshape((-1, 2))
            # Start non-blocking playback
            sd.play(samples, audio_segment.frame_rate, blocking=False)
            # Monitor playback
            self.monitor_audio_playback()
        else:
            self.finish_current_trial()

    def start_cmd_trial(self, side):
        """Start a command trial (right or left)"""
        self.cmd_trial_side = side
        self.cmd_trial_cycle = 0
        self.cmd_trial_phase = "keep"  # "keep" or "stop" or "pause"
        self.continue_cmd_trial()

    def continue_cmd_trial(self):
        """Continue the command trial cycle"""
        if self.gui_callback.get_playback_state() != "playing" or self.is_paused:
            self.gui_callback.root_after(100, self.continue_cmd_trial) # Use callback for Tkinter after
            return
        if self.cmd_trial_cycle >= 8:
            # All 8 cycles complete
            self.finish_current_trial()
            return
        if self.cmd_trial_phase == "keep":
            # Play keep audio
            audio = (self.trials.right_keep_audio if self.cmd_trial_side == "right" 
                    else self.trials.left_keep_audio)
            self.play_audio_segment_non_blocking(audio, lambda: self.set_cmd_phase("pause_after_keep"))
        elif self.cmd_trial_phase == "pause_after_keep":
            # 10 second pause after keep
            self.start_interruptible_delay(10000, lambda: self.set_cmd_phase("stop"))
        elif self.cmd_trial_phase == "stop":
            # Play stop audio
            audio = (self.trials.right_stop_audio if self.cmd_trial_side == "right" 
                    else self.trials.left_stop_audio)
            self.play_audio_segment_non_blocking(audio, lambda: self.set_cmd_phase("pause_after_stop"))
        elif self.cmd_trial_phase == "pause_after_stop":
            # 10 second pause after stop
            self.start_interruptible_delay(10000, lambda: self.next_cmd_cycle())

    def set_cmd_phase(self, phase):
        """Set the command trial phase"""
        self.cmd_trial_phase = phase
        self.continue_cmd_trial()

    def next_cmd_cycle(self):
        """Move to next command cycle"""
        self.cmd_trial_cycle += 1
        self.cmd_trial_phase = "keep"
        self.continue_cmd_trial()

    def start_oddball_trial(self):
        """Start an oddball trial"""
        self.oddball_tone_count = 0
        self.oddball_phase = "initial_standard"  # "initial_standard" or "main_sequence"
        self.current_trial_sentences = []
        self.continue_oddball_trial()

    def continue_oddball_trial(self):
        """Continue the oddball trial"""
        if self.gui_callback.get_playback_state() != "playing" or self.is_paused:
            self.gui_callback.root_after(100, self.continue_oddball_trial) # Use callback
            return
        if self.oddball_phase == "initial_standard":
            if self.oddball_tone_count < 5:
                # Play standard tone
                self.play_tone_non_blocking(1000, 100, "standard", 
                    lambda: self.start_interruptible_delay(1000, self.continue_oddball_trial))
                self.oddball_tone_count += 1
            else:
                # Switch to main sequence
                self.oddball_phase = "main_sequence"
                self.oddball_tone_count = 0
                self.continue_oddball_trial()
        elif self.oddball_phase == "main_sequence":
            if self.oddball_tone_count < 20:
                # 20% chance for rare tone
                if random.random() < 0.2:
                    self.play_tone_non_blocking(2000, 100, "rare",
                        lambda: self.start_interruptible_delay(1000, self.continue_oddball_trial))
                else:
                    self.play_tone_non_blocking(1000, 100, "standard",
                        lambda: self.start_interruptible_delay(1000, self.continue_oddball_trial))
                self.oddball_tone_count += 1
            else:
                # Oddball trial complete
                self.finish_current_trial()

    def start_voice_trial(self, voice_type):
        """Start a voice trial (control or loved_one)"""
        if voice_type == "control":
            audio_data = self.trials.control_voice_audio
        else:  # loved_one
            audio_data = self.trials.loved_one_voice_audio
        if audio_data is not None:
            # Play the voice audio
            sd.play(audio_data, self.trials.sample_rate, blocking=False)
            self.monitor_audio_playback()
        else:
            self.finish_current_trial()

    def play_audio_segment_non_blocking(self, audio_segment, callback=None):
        """Play a pydub AudioSegment non-blocking"""
        if audio_segment is None:
            if callback:
                callback()
            return
        samples = audio_segment.get_array_of_samples()
        if audio_segment.channels == 2:
            samples = samples.reshape((-1, 2))
        sd.play(samples, audio_segment.frame_rate, blocking=False)
        if callback:
            self.monitor_audio_playback(callback)

    def play_tone_non_blocking(self, frequency, duration_ms, tone_type, callback=None):
        """Play a tone non-blocking"""
        # Generate tone
        audio_segment = Sine(frequency).to_audio_segment(duration=duration_ms)
        samples = audio_segment.get_array_of_samples()
        # Record the tone type
        self.current_trial_sentences.append(tone_type)
        # Play non-blocking
        sd.play(samples, audio_segment.frame_rate, blocking=False)
        if callback:
            self.monitor_audio_playback(callback)

    def monitor_audio_playback(self, callback=None):
        """Monitor audio playback and call callback when done"""
        if self.gui_callback.get_playback_state() != "playing":
            # Stop audio if playback was stopped/paused
            sd.stop()
            return
        if sd.get_stream() is None or not sd.get_stream().active:
            # Audio finished
            if callback:
                callback()
            else:
                self.finish_current_trial()
        else:
            # Still playing, check again in 50ms
            self.gui_callback.root_after(50, lambda: self.monitor_audio_playback(callback)) # Use callback

    def start_interruptible_delay(self, delay_ms, callback):
        """Start an interruptible delay"""
        self.delay_end_time = time.time() + (delay_ms / 1000.0)
        self.delay_callback = callback
        self.continue_interruptible_delay()

    def continue_interruptible_delay(self):
        """Continue the interruptible delay"""
        if self.gui_callback.get_playback_state() != "playing": # Use callback
            return  # Delay interrupted
        if self.is_paused:
            # Extend delay time while paused
            current_time = time.time()
            if self.delay_end_time is not None:
                remaining_time = self.delay_end_time - current_time
                self.delay_end_time = time.time() + remaining_time
            self.gui_callback.root_after(100, self.continue_interruptible_delay) # Use callback
            return
        if self.delay_end_time is not None and time.time() >= self.delay_end_time:
            # Delay complete
            if hasattr(self, 'delay_callback') and self.delay_callback:
                self.delay_callback()
        else:
            # Continue delay
            self.gui_callback.root_after(50, self.continue_interruptible_delay) # Use callback

    def finish_current_trial(self):
        """Finish the current trial and move to next"""
        if self.gui_callback.get_playback_state() != "playing":
            return
        # Stop any remaining audio
        sd.stop()
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
        #
        self.save_single_trial_result(trial_result)
        # Mark as completed
        trial['status'] = 'completed'
        # Notify GUI to update trial list
        self.gui_callback.update_trial_list_status()
        # Move to next trial
        self.trials.current_trial_index += 1
        # Add inter-trial delay (1.2-2.2 seconds)
        delay = random.uniform(1200, 2200)  # milliseconds
        self.start_interruptible_delay(delay, self.continue_playback)

    def toggle_pause(self):
        """Toggle pause state"""
        if self.is_paused:
            # Currently paused, so resume
            self.is_paused = False
            self.gui_callback.root_after(100, self.continue_playback)
        else:
            # Currently playing, so pause
            self.is_paused = True
            sd.stop()
            # check if currently in progress and reset to pending 
            current_trial = self.trials.trial_dictionary[self.trials.current_trial_index]
            if current_trial['status'] == 'in progress':
                current_trial['status'] = 'pending'
                self.gui_callback.update_trial_list_status()


    def stop_stimulus(self):
        """Stop the current stimulus playback"""
        # Stop any playing audio
        sd.stop()
        self.reset_trial_state()

    def save_single_trial_result(self, trial_result):
        """Save a single trial result to the stimulus results CSV file by appending"""
        # Get the results directory and patient ID from the config and GUI
        results_dir = self.config.file.get('result_dir', '.')
        patient_id = self.gui_callback.get_patient_id()
        
        # Create the directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Define the file path
        results_path = os.path.join(results_dir, f"{patient_id}_{self.config.current_date}_stimulus_results.csv")
        
        # Create a DataFrame for the single trial
        df = pd.DataFrame([trial_result])
        
        # Append to the CSV file. 
        # If the file doesn't exist, it will be created. 
        # If it does exist, the new row is appended without a header.
        df.to_csv(results_path, mode='a', header=not os.path.exists(results_path), index=False)