# lib/app.py

import os
import time
import pandas as pd
import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import logging
from lib.config import Config
from lib.trials import Trials
from lib.auditory_stimulator import AuditoryStimulator
from lib.analysis_manager import AnalysisManager

logger = logging.getLogger('eeg_stimulus.app')

# --- Constants ---
NUM_LANGUAGE_TRIALS = 72
NUM_CMD_TRIALS_NO_PROMPT = 3
NUM_CMD_TRIALS_WITH_PROMPT = 3
NUM_ODDBALL_TRIALS_NO_PROMPT = 4
NUM_ODDBALL_TRIALS_WITH_PROMPT = 4
NUM_LOVED_ONE_TRIALS = 50

PATIENT_DATA_DIR = "patient_data"
RESULTS_DIR = os.path.join(PATIENT_DATA_DIR, "results")
EDFS_DIR = os.path.join(PATIENT_DATA_DIR, "edfs")

TRIAL_TYPE_DISPLAY_NAMES = {
    "language": "Language",
    "right_command": "Right Command",
    "right_command+p": "Right Command + Prompt",
    "left_command": "Left Command",
    "left_command+p": "Left Command + Prompt",
    "oddball": "Oddball",
    "oddball+p": "Oddball + Prompt",
    "loved_one_voice": "Loved One Voice",
    "control": "Control Statement",
    "session_note": "Session Note"
}


class TkApp:
    def __init__(self, root: tk.Tk) -> None:
        logger.info("Initializing TkApp")
        self.root = root
        self.root.title("EEG Stimulus Package")
        self.root.geometry("1050x830")

        # Create class instances
        self.config = Config()
        self.trials = Trials(self)
        self.audio_stim = AuditoryStimulator(self)
        self.analysis_manager = AnalysisManager(self)

        self.playback_state = "empty"
        self.current_patient = None

        # UI Variables
        self.language_var = tk.BooleanVar()
        self.right_cmd_var = tk.BooleanVar()
        self.rcmd_prompt_var = tk.BooleanVar()
        self.left_cmd_var = tk.BooleanVar()
        self.lcmd_prompt_var = tk.BooleanVar()
        self.oddball_var = tk.BooleanVar()
        self.oddball_prompt_var = tk.BooleanVar()
        self.loved_one_var = tk.BooleanVar()
        self.gender_var = tk.StringVar(value="Male")

        # file path variables
        self.current_results_path = None
        self.selected_stimulus_file_candidate = None
        self.selected_edf_file_candidate = None

        logger.info("TkApp initialized successfully")
        self.build_main_ui()

    def get_patient_id(self) -> str:
        """Get patient ID from Stimulus tab"""
        patient_id = self.patient_id_entry.get().strip()
        if not patient_id:
            logger.warning("Attempted to get patient ID but field is empty")
            raise ValueError("Patient ID cannot be empty")
        logger.debug(f"Retrieved patient ID: {patient_id}")
        return patient_id

    def playback_complete(self):
        """Thread safe call to playback complete"""
        logger.debug("Playback complete signal received")
        self.root.after(0, self._on_playback_complete)

    def _on_playback_complete(self):
        """Handle completion of stimulus playback"""
        try:
            patient_id = self.patient_id_entry.get().strip()
        except Exception:
            patient_id = "Unknown"
            logger.warning("Could not retrieve patient ID after playback completion")
        
        self.playback_state = "ready"
        self.audio_stim.is_paused = False
        self.pause_button.config(text="Pause", image=self.pause_sym)
        self.update_button_states()
        self.status_label.config(text=f"Stimulus completed for {patient_id}", foreground="green")
        
        logger.info(f"Stimulus playback completed successfully for patient: {patient_id}")
        messagebox.showinfo("Success", f"Stimulus administered successfully to {patient_id}")

    def playback_error(self, error_msg: str) -> None:
        """Thread-safe error handler for playback errors."""
        logger.error(f"Playback error occurred: {error_msg}")
        self.root.after(0, self._on_playback_error, error_msg)

    def _on_playback_error(self, error_msg: str) -> None:
        """Actual GUI update for playback error (runs on main thread)."""
        self.playback_state = "ready"
        self.audio_stim.is_paused = False
        self.pause_button.config(text="Pause", image=self.pause_sym)
        self.update_button_states()
        self.status_label.config(text="Error during playback", foreground="red")
        messagebox.showerror("Playback Error", f"Error during playback: {error_msg}")

    def on_patient_id_change(self, event=None):
        patient_id = self.patient_id_entry.get().strip()
        if patient_id:
            self.current_patient = patient_id
            self.playback_state = "ready"
            self.status_label.config(text="Ready to prepare stimulus", foreground="green")
            logger.info(f"Patient ID set: {patient_id}")
        else:
            self.current_patient = None
            self.playback_state = "empty"
            self.status_label.config(text="Please enter a patient ID", foreground="red")
            logger.debug("Patient ID field cleared")
        self.update_button_states()

    def toggle_loved_one_options(self):
        state = 'normal' if self.loved_one_var.get() else 'disabled'
        for widget in [self.male_radio, self.female_radio, self.file_button]:
            widget.config(state=state)
        logger.debug(f"Loved one options toggled: {state}")

    def upload_voice_file(self):
        logger.debug("Opening voice file dialog")
        file_path = filedialog.askopenfilename(
            title="Select Voice File",
            filetypes=[("Audio files", "*.wav *.mp3"), ("All files", "*.*")]
        )
        if file_path:
            self.audio_stim.trials.loved_one_file = file_path
            self.audio_stim.trials.loved_one_gender = self.gender_var.get()
            self.file_label.config(text=os.path.basename(file_path))
            logger.info(f"Loved one voice file uploaded: {file_path} (Gender: {self.gender_var.get()})")
        else:
            logger.debug("Voice file upload cancelled")

    def toggle_pause(self):
        if self.playback_state == "playing":
            self.playback_state = "paused"
            self.pause_button.config(image=self.play_sym)
            self.pause_button.config(text="Resume")
            self.status_label.config(text="Pausing stimulus...", foreground="red")
            logger.info("User paused stimulus playback")
        elif self.playback_state == "paused":
            self.playback_state = "playing"
            self.pause_button.config(image=self.pause_sym)
            self.pause_button.config(text="Pause")
            self.status_label.config(text="Resuming stimulus...", foreground="blue")
            logger.info("User resumed stimulus playback")

        self.audio_stim.toggle_pause()
        self.update_button_states()
        self.update_trial_list_status()

    def stop_stimulus(self):
        """Stop the current stimulus playback"""
        if self.playback_state in ["playing", "paused"]:
            logger.info("User stopped stimulus playback")
            self.playback_state = "ready"
            self.pause_button.config(text="Pause", image=self.pause_sym)
            self.audio_stim.stop_stimulus()
            self.status_label.config(text="Stimulus stopped", foreground="orange")
            self.update_button_states()
            self.update_trial_list_status()

    def prepare_stimulus(self):
        logger.info("Prepare stimulus button clicked")
        
        if self.playback_state != "ready":
            logger.warning(f"Prepare stimulus called in invalid state: {self.playback_state}")
            return
            
        if not any([self.language_var.get(), self.right_cmd_var.get(), self.left_cmd_var.get(),
                    self.oddball_var.get(), self.loved_one_var.get()]):
            logger.warning("No stimulus type selected")
            messagebox.showwarning("No Stimulus Selected", "Please select at least one stimulus type.")
            return
            
        if self.loved_one_var.get() and not self.audio_stim.trials.loved_one_file:
            logger.warning("Loved one stimulus selected but no voice file uploaded")
            messagebox.showwarning("Missing File", "Please upload a voice file for loved one stimulus.")
            return
        
        try:
            patient_id = self.get_patient_id()
            date_str = time.strftime("%Y-%m-%d")
            self.current_results_path = os.path.join(RESULTS_DIR, f"{patient_id}_{date_str}_stimulus_results.csv")
            logger.info(f"Results will be saved to: {self.current_results_path}")
        except ValueError as e:
            logger.error(f"Invalid patient ID: {e}")
            messagebox.showwarning("Invalid Patient ID", str(e))
            return
        
        self.playback_state = "preparing"
        self.update_button_states()
        self.status_label.config(text="Preparing stimulus...", foreground="blue")
        self.start_preparation()

    def start_preparation(self):
        """Start the preparation process"""
        try:
            # Collect UI selections
            num_of_each_trial = {
                "lang": NUM_LANGUAGE_TRIALS if self.language_var.get() else 0,
                "rcmd": NUM_CMD_TRIALS_NO_PROMPT if self.right_cmd_var.get() and not self.rcmd_prompt_var.get() else 0,
                "rcmd+p": NUM_CMD_TRIALS_WITH_PROMPT if self.right_cmd_var.get() and self.rcmd_prompt_var.get() else 0,
                "lcmd": NUM_CMD_TRIALS_NO_PROMPT if self.left_cmd_var.get() and not self.lcmd_prompt_var.get() else 0,
                "lcmd+p": NUM_CMD_TRIALS_WITH_PROMPT if self.left_cmd_var.get() and self.lcmd_prompt_var.get() else 0,
                "odd": NUM_ODDBALL_TRIALS_NO_PROMPT if self.oddball_var.get() and not self.oddball_prompt_var.get() else 0,
                "odd+p": NUM_ODDBALL_TRIALS_WITH_PROMPT if self.oddball_var.get() and self.oddball_prompt_var.get() else 0,
                "loved": NUM_LOVED_ONE_TRIALS if self.loved_one_var.get() else 0
            }

            logger.info(f"Starting trial preparation with configuration: {num_of_each_trial}")

            # Set gender for loved one in the trials object
            if self.loved_one_var.get():
                self.trials.loved_one_gender = self.gender_var.get()
                logger.debug(f"Loved one gender set to: {self.gender_var.get()}")

            # Pass the configuration to the Trials object
            self.trials.generate_trials(num_of_each_trial)

            # Reset playback state
            self.playback_state = "ready"

            # Update GUI
            self.update_button_states()
            num_trials = len(self.trials.trial_dictionary)
            self.status_label.config(text=f"Stimulus prepared! {num_trials} trials ready.", foreground="green")
            self.populate_trial_list()
            
            logger.info(f"Stimulus preparation completed successfully: {num_trials} trials generated")

        except Exception as e:
            logger.error(f"Error during stimulus preparation: {e}", exc_info=True)
            self.playback_state = "ready"
            self.update_button_states()
            self.status_label.config(text="Error preparing stimulus", foreground="red")
            messagebox.showerror("Error", f"Error preparing stimulus: {e}")

    def play_stimulus(self):
        logger.info("Play stimulus button clicked")
        
        if self.playback_state != "ready" or len(self.trials.trial_dictionary) == 0:
            logger.warning(f"Play stimulus called in invalid state: {self.playback_state}, trials: {len(self.trials.trial_dictionary)}")
            return
            
        try:
            patient_id = self.patient_id_entry.get().strip()
            if not patient_id:
                logger.warning("Play stimulus attempted without patient ID")
                messagebox.showwarning("No Patient ID", "Please enter a patient ID.")
                return
        except Exception as e:
            logger.error(f"Error getting patient ID: {e}")
            return

        self.playback_state = "playing"
        self.update_button_states()
        self.status_label.config(text="Playing stimulus...", foreground="blue")
        self.update_trial_list_status()
        
        logger.info(f"Starting stimulus playback for patient: {patient_id}")
        # Start playback via auditory stimulator
        self.audio_stim.play_trial_sequence()

    def add_note(self):
        logger.debug("Add note button clicked")
        
        if not self.current_results_path:
            logger.warning("Add note attempted before preparing stimulus")
            messagebox.showerror("Error", "Prepare stimulus before adding notes.")
            return
            
        note = self.note_entry.get().strip()
        
        try:
            patient_id = self.get_patient_id()
        except ValueError:
            messagebox.showwarning("No Patient ID", "Please enter a patient ID before adding a note.")
            return
            
        if not note:
            logger.debug("Empty note submission attempted")
            messagebox.showwarning("Empty Note", "Please enter a note before adding.")
            return

        # Create note row
        note_row = {
            'patient_id': patient_id,
            'date': time.strftime("%Y-%m-%d"),
            'trial_type': 'session_note',
            'sentences': '',
            'start_time': '',
            'end_time': '',
            'duration': '',
            'notes': note
        }

        try:
            note_df = pd.DataFrame([note_row])
            note_df.to_csv(
                self.current_results_path,
                mode='a',
                header=not os.path.exists(self.current_results_path),
                index=False
            )
            self.notes_text.insert(tk.END, f"[{note_row['date']}] {note}\n")
            self.notes_text.see(tk.END)
            self.note_entry.delete(0, tk.END)
            logger.info(f"Session note added for patient {patient_id}: {note[:50]}...")
        except Exception as e:
            logger.error(f"Failed to save note: {e}", exc_info=True)
            messagebox.showerror("Note Save Error", f"Failed to save note:\n{str(e)}")

    def on_stimulus_selected(self, event=None):
        filename = self.stimulus_combo.get()
        if filename in ["Select a stimulus file...", "No CSV files found", ""]:
            self.selected_stimulus_file_candidate = None
            logger.debug("Stimulus file selection cleared")
        else:
            self.selected_stimulus_file_candidate = os.path.join(RESULTS_DIR, filename)
            logger.info(f"Stimulus file selected (candidate): {filename}")
        self.update_submit_button_state()

    def on_edf_selected(self, event=None):
        filename = self.edf_combo.get()
        if filename in ["Select an EDF file...", "No EDF files found", ""]:
            self.selected_edf_file_candidate = None
            logger.debug("EDF file selection cleared")
        else:
            self.selected_edf_file_candidate = os.path.join(EDFS_DIR, filename)
            logger.info(f"EDF file selected (candidate): {filename}")
        self.update_submit_button_state()

    def update_results_file_labels(self):
        """Update labels in results tab and the official file paths with the *candidate* files."""
        if self.selected_stimulus_file_candidate and self.selected_edf_file_candidate:
            # Set the *official* file paths to the candidates
            self.stimulus_file_path = self.selected_stimulus_file_candidate
            self.edf_file_path = self.selected_edf_file_candidate

            logger.info(f"Files confirmed for analysis - Stimulus: {os.path.basename(self.stimulus_file_path)}, "
                       f"EDF: {os.path.basename(self.edf_file_path)}")

            # Update the official display label in the Patient Info tab
            stim_file = os.path.basename(self.stimulus_file_path)
            self.official_stimulus_label.config(text=stim_file, foreground="green")
            edf_file = os.path.basename(self.edf_file_path)
            self.official_edf_label.config(text=edf_file, foreground="green")

            # Update the Results tab display
            self.results_stimulus_label.config(text=stim_file)
            self.results_edf_label.config(text=edf_file)

            # Clear the candidates after setting them as official
            self.selected_stimulus_file_candidate = None
            self.selected_edf_file_candidate = None

            # Update the submit button state
            self.update_submit_button_state()
            
            # Enable the sync preview button
            self.sync_preview_btn.config(state='normal')

            # Load session data
            if self.stimulus_file_path:
                self.load_session_data_from_csv(self.stimulus_file_path)

        else:
            logger.warning("'Use Selected Files' pressed, but no candidates were set.")

    def load_session_data_from_csv(self, filepath):
        """Load trial sequence and notes from stimulus CSV for Patient Info tab."""
        logger.info(f"Loading session data from: {filepath}")
        
        # Clear previous data
        for item in self.patient_info_trial_tree.get_children():
            self.patient_info_trial_tree.delete(item)
        self.patient_info_notes_text.config(state="normal")
        self.patient_info_notes_text.delete(1.0, tk.END)

        try:
            df = pd.read_csv(filepath)
            trial_count = 0
            note_count = 0

            for _, row in df.iterrows():
                trial_type = row.get('trial_type', 'unknown')
                display_type = TRIAL_TYPE_DISPLAY_NAMES.get(trial_type, trial_type.replace('_', ' ').title())

                if trial_type == 'session_note':
                    status = "Note"
                    tag = 'session_note'
                    date = row.get('date', 'Unknown')
                    note_text = str(row.get('notes', '')).strip()
                    if note_text:
                        self.patient_info_notes_text.insert(tk.END, f"[{date}] {note_text}\n")
                        note_count += 1
                else:
                    status = "Completed"
                    tag = 'completed'
                    trial_count += 1

                self.patient_info_trial_tree.insert('', 'end', values=(display_type, status), tags=(tag,))

            self.patient_info_notes_text.config(state="disabled")
            if self.patient_info_notes_text.get(1.0, tk.END).strip():
                self.patient_info_notes_text.see(tk.END)

            logger.info(f"Session data loaded: {trial_count} trials, {note_count} notes")

        except Exception as e:
            logger.error(f"Could not load session data from {filepath}: {e}", exc_info=True)
            messagebox.showerror("Load Error", f"Could not load session data:\n{e}")

    def run_selected_analysis(self):
        """Run selected analysis type using the AnalysisManager."""
        logger.info("Run analysis button clicked")
        
        if not self.stimulus_file_path:
            error_msg = "No stimulus CSV file selected. Please select files in the 'Patient Information' tab."
            logger.warning("Analysis attempted without stimulus file")
            self.analysis_results_text.delete(1.0, tk.END)
            self.analysis_results_text.insert(tk.END, error_msg)
            return

        if not self.edf_file_path:
            error_msg = "No EDF file selected. Please select files in the 'Patient Information' tab."
            logger.warning("Analysis attempted without EDF file")
            self.analysis_results_text.delete(1.0, tk.END)
            self.analysis_results_text.insert(tk.END, error_msg)
            return

        analysis_type = self.analysis_type_combo.get()
        bad_channels_str = self.bad_channels_entry.get().strip()
        eog_channels_str = self.eog_channels_entry.get().strip()

        # Parse channel lists
        bad_channels = [ch.strip() for ch in bad_channels_str.split(',') if ch.strip()]
        eog_channels = [ch.strip() for ch in eog_channels_str.split(',') if ch.strip()]

        logger.info(f"Starting {analysis_type} analysis - "
                   f"Stimulus: {os.path.basename(self.stimulus_file_path)}, "
                   f"EDF: {os.path.basename(self.edf_file_path)}, "
                   f"Bad channels: {bad_channels}, EOG channels: {eog_channels}")

        if analysis_type == "CMD Analysis":
            self.analysis_manager.run_cmd_analysis(self.stimulus_file_path, self.edf_file_path, bad_channels, eog_channels)
        elif analysis_type == "Language Tracking":
            logger.info("Language tracking requested (not yet implemented)")
            self.analysis_results_text.delete(1.0, tk.END)
            self.analysis_results_text.insert(tk.END, "Language tracking is not yet implemented.")
        else:
            logger.warning(f"Unknown analysis type requested: {analysis_type}")
            self.analysis_results_text.delete(1.0, tk.END)
            self.analysis_results_text.insert(tk.END, f"Unknown analysis type: {analysis_type}")

    def populate_trial_list(self):
        """Populate the Treeview with all trials from trial_dictionary."""
        logger.debug(f"Populating trial list with {len(self.trials.trial_dictionary)} trials")
        
        # Clear existing items
        for item in self.trial_tree.get_children():
            self.trial_tree.delete(item)

        # Insert each trial
        for idx, trial in enumerate(self.trials.trial_dictionary):
            trial_type = trial['type']
            display_type = TRIAL_TYPE_DISPLAY_NAMES.get(trial_type, trial_type.replace('_', ' ').title())
            status = trial['status'].title()
            self.trial_tree.insert('', 'end', iid=str(idx), values=(display_type, status))

    def update_trial_list_status(self):
        """Update the status column in the trial list from trial_dictionary."""
        for idx, trial in enumerate(self.trials.trial_dictionary):
            if str(idx) in self.trial_tree.get_children():
                display_type = TRIAL_TYPE_DISPLAY_NAMES.get(trial['type'], trial['type'].replace('_', ' ').title())
                status = trial['status'].title()

                # Determine which tag to use
                status_key = trial['status'].lower()
                if 'complete' in status_key:
                    tag = 'completed'
                elif 'in progress' in status_key:
                    tag = 'inprogress'
                else:
                    tag = 'pending'

                # Update row with values AND tag
                self.trial_tree.item(str(idx), values=(display_type, status), tags=(tag,))


    def build_main_ui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        self.tab1 = ttk.Frame(self.notebook)
        self.tab2 = ttk.Frame(self.notebook)
        self.tab3 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1, text="Administer Stimuli")
        self.notebook.add(self.tab2, text="Patient Information")
        self.notebook.add(self.tab3, text="Results")
        self.build_stimulus_tab()
        self.build_patient_info_tab()
        self.build_results_tab()

    def build_stimulus_tab(self):
        """Build the stimulus tab with only the trial list having a scrollbar."""
        main_frame = ttk.Frame(self.tab1)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Patient ID Section
        patient_frame = ttk.LabelFrame(main_frame, text="Patient Information", padding="10")
        patient_frame.pack(fill='x', pady=5)
        ttk.Label(patient_frame, text="Patient/EEG ID:").grid(row=0, column=0, sticky='w', padx=5)
        self.patient_id_entry = ttk.Entry(patient_frame, width=30)
        self.patient_id_entry.grid(row=0, column=1, sticky='ew', padx=5)
        self.patient_id_entry.bind('<KeyRelease>', self.on_patient_id_change)
        self.status_label = ttk.Label(patient_frame, text="Please enter a patient ID", foreground="red")
        self.status_label.grid(row=1, column=0, columnspan=2, pady=5)
        patient_frame.grid_columnconfigure(1, weight=1)

        # Stimulus Selection
        stim_frame = ttk.LabelFrame(main_frame, text="Stimulus Selection", padding="10")
        stim_frame.pack(fill='x', pady=5)
        ttk.Checkbutton(stim_frame, text="Language Stimulus", variable=self.language_var).grid(row=0, column=0, sticky='w')
        # RCMD Section
        rcmd_frame = ttk.Frame(stim_frame)
        rcmd_frame.grid(row=1, column=0, columnspan=2, sticky='ew', pady=5)
        ttk.Checkbutton(rcmd_frame, text="Right Command Stimulus", variable=self.right_cmd_var,
                        command=self.toggle_prompts).pack(side='left')
        self.rcmd_prompt = ttk.Checkbutton(rcmd_frame, text="Include Prompt", variable=self.rcmd_prompt_var, state='disabled')
        self.rcmd_prompt.pack(side='right')
        # LCMD Section
        lcmd_frame = ttk.Frame(stim_frame)
        lcmd_frame.grid(row=2, column=0, columnspan=2, sticky='ew', pady=5)
        ttk.Checkbutton(lcmd_frame, text="Left Command Stimulus", variable=self.left_cmd_var,
                        command=self.toggle_prompts).pack(side='left')
        self.lcmd_prompt = ttk.Checkbutton(lcmd_frame, text="Include Prompt", variable=self.lcmd_prompt_var, state='disabled')
        self.lcmd_prompt.pack(side='right')
        # Oddball Section
        oddball_frame = ttk.Frame(stim_frame)
        oddball_frame.grid(row=4, column=0, columnspan=2, sticky='ew', pady=5)
        ttk.Checkbutton(oddball_frame, text="Oddball Stimulus", variable=self.oddball_var,
                        command=self.toggle_prompts).pack(side='left')
        self.oddball_prompt = ttk.Checkbutton(oddball_frame, text="Include Prompt", variable=self.oddball_prompt_var, state='disabled')
        self.oddball_prompt.pack(side='right')

        # Loved one section
        loved_frame = ttk.Frame(stim_frame)
        loved_frame.grid(row=5, column=0, columnspan=3, sticky='ew', pady=5)
        ttk.Checkbutton(loved_frame, text="Loved One Stimulus", variable=self.loved_one_var,
                        command=self.toggle_loved_one_options).pack(side='left')
        gender_frame = ttk.Frame(loved_frame)
        gender_frame.pack(side='left', padx=20)
        ttk.Label(gender_frame, text="Gender:").pack(side='left')
        self.male_radio = ttk.Radiobutton(gender_frame, text="Male", variable=self.gender_var, value="Male")
        self.male_radio.pack(side='left')
        self.female_radio = ttk.Radiobutton(gender_frame, text="Female", variable=self.gender_var, value="Female")
        self.female_radio.pack(side='left')
        self.file_button = ttk.Button(loved_frame, text="Upload Voice File", command=self.upload_voice_file)
        self.file_button.pack(side='left', padx=20)
        self.file_label = ttk.Label(loved_frame, text="No file selected")
        self.file_label.pack(side='left')
        self.toggle_loved_one_options()

        # Control buttons
        self.button_styles()
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill='x', pady=10)
        self.prepare_button = ttk.Button(control_frame, text="Prepare Stimulus", command=self.prepare_stimulus)
        self.prepare_button.pack(side='left', padx=5, ipady=14)
        self.play_button = ttk.Button(control_frame, text="Play Stimulus", command=self.play_stimulus)
        self.play_button.pack(side='left', padx=5, ipady=14)
        self.pause_button = ttk.Button(control_frame, text="Pause", image=self.pause_sym, compound=tk.LEFT, command=self.toggle_pause)
        self.pause_button.pack(side='left', padx=10, ipady=4)
        self.stop_button = ttk.Button(control_frame, text="", image=self.stop_sym, compound=tk.CENTER, command=self.stop_stimulus)
        self.stop_button.pack(side='right', padx=10, ipady=4)

        # Side-by-side: Trial List + Notes
        side_by_side_frame = ttk.Frame(main_frame)
        side_by_side_frame.pack(fill='both', expand=True, pady=5)
        side_by_side_frame.grid_columnconfigure(0, weight=1)
        side_by_side_frame.grid_columnconfigure(1, weight=1)
        side_by_side_frame.grid_rowconfigure(0, weight=1)

        # Trial List Frame (left)
        trial_list_frame = ttk.LabelFrame(side_by_side_frame, text="Trial Sequence")
        trial_list_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 5))
        trial_list_frame.grid_columnconfigure(0, weight=1)
        trial_list_frame.grid_rowconfigure(0, weight=1)

        self.trial_tree = ttk.Treeview(trial_list_frame, columns=('Type', 'Status'), show='headings', height=12)
        self.trial_tree.heading('Type', text='Trial Type')
        self.trial_tree.heading('Status', text='Status')
        self.trial_tree.column('Type', width=150, minwidth=100)
        self.trial_tree.column('Status', width=100, minwidth=80)
        self.trial_tree.tag_configure('pending', foreground='gray')
        self.trial_tree.tag_configure('inprogress', foreground='blue')
        self.trial_tree.tag_configure('completed', foreground='green')

        tree_scroll = ttk.Scrollbar(trial_list_frame, orient="vertical", command=self.trial_tree.yview)
        tree_scroll.pack(side='right', fill='y')
        self.trial_tree.configure(yscrollcommand=tree_scroll.set)
        self.trial_tree.pack(side='left', fill='both', expand=True, padx=(5, 0), pady=5)

        # Notes Frame (right)
        notes_frame = ttk.LabelFrame(side_by_side_frame, text="Session Notes", padding="10")
        notes_frame.grid(row=0, column=1, sticky='nsew', padx=(5, 0))
        notes_frame.grid_columnconfigure(1, weight=1)
        notes_frame.grid_rowconfigure(1, weight=1)

        ttk.Label(notes_frame, text="Add Note:").grid(row=0, column=0, sticky='w')
        self.note_entry = ttk.Entry(notes_frame, width=40)
        self.note_entry.grid(row=0, column=1, sticky='ew', padx=5)
        ttk.Button(notes_frame, text="Add Note", command=self.add_note).grid(row=0, column=2, padx=5)
        self.notes_text = tk.Text(notes_frame, height=10, width=40)
        self.notes_text.grid(row=1, column=0, columnspan=3, sticky='nsew', pady=5)

        self.update_button_states()

    def build_patient_info_tab(self):
        main_frame = ttk.Frame(self.tab2)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # === File Selection Frame ===
        selection_frame = ttk.LabelFrame(main_frame, text="File Selection", padding="10")
        selection_frame.pack(fill='x', padx=5, pady=5)

        # Stimulus CSV Selection (Dropdown only)
        self.stimulus_combo = ttk.Combobox(selection_frame, state="readonly", width=50)
        self.stimulus_combo.pack(side='top', fill='x', pady=2)
        self.stimulus_combo.bind("<<ComboboxSelected>>", self.on_stimulus_selected)

        # EDF File Selection (Dropdown only)
        self.edf_combo = ttk.Combobox(selection_frame, state="readonly", width=50)
        self.edf_combo.pack(side='top', fill='x', pady=2)
        self.edf_combo.bind("<<ComboboxSelected>>", self.on_edf_selected)

        # --- NEW: Frame for Buttons ---
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x', padx=5, pady=15)

        # Submit Button - Now updates the official selection
        self.submit_btn = ttk.Button(button_frame, text="Use Selected Files",
                                    command=self.update_results_file_labels, state="disabled")
        self.submit_btn.pack(side='left', padx=(0, 5))

        # Sync Detection and Preview Button
        self.sync_preview_btn = ttk.Button(button_frame, text="Detect Sync & Preview",
                                          command=self.detect_and_preview_sync, state="disabled")
        self.sync_preview_btn.pack(side='left')

        # --- Add a frame to display the *officially* selected files ---
        file_display_frame = ttk.LabelFrame(main_frame, text="Currently Confirmed Files for Analysis", padding="10")
        file_display_frame.pack(fill='x', padx=5, pady=5)

        # Official Stimulus Label
        stim_off_frame = ttk.Frame(file_display_frame)
        stim_off_frame.pack(fill='x', pady=2)
        ttk.Label(stim_off_frame, text="Stimulus CSV:").pack(side='left', padx=(0, 5))
        self.official_stimulus_label = ttk.Label(stim_off_frame, text="None", foreground="red")
        self.official_stimulus_label.pack(side='left', padx=(0, 10))

        # Official EDF Label
        edf_off_frame = ttk.Frame(file_display_frame)
        edf_off_frame.pack(fill='x', pady=2)
        ttk.Label(edf_off_frame, text="EDF File:").pack(side='left', padx=(0, 5))
        self.official_edf_label = ttk.Label(edf_off_frame, text="None", foreground="red")
        self.official_edf_label.pack(side='left', padx=(0, 10))

        # === Trial List + Notes Viewer (This part remains unchanged) ===
        viewer_frame = ttk.Frame(main_frame)
        viewer_frame.pack(fill='both', expand=True, pady=10)
        viewer_frame.grid_columnconfigure(0, weight=1)
        viewer_frame.grid_rowconfigure(0, weight=1)

        # Trial List
        trial_frame = ttk.LabelFrame(viewer_frame, text="Trial Sequence")
        trial_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 5))
        trial_frame.grid_columnconfigure(0, weight=1)
        trial_frame.grid_rowconfigure(0, weight=1)

        self.patient_info_trial_tree = ttk.Treeview(trial_frame, columns=('Type', 'Status'), show='headings', height=10)
        self.patient_info_trial_tree.heading('Type', text='Trial Type')
        self.patient_info_trial_tree.heading('Status', text='Status')
        self.patient_info_trial_tree.column('Type', width=150)
        self.patient_info_trial_tree.column('Status', width=100)
        self.patient_info_trial_tree.tag_configure('session_note', foreground='purple')
        self.patient_info_trial_tree.tag_configure('completed', foreground='green')
        self.patient_info_trial_tree.pack(fill='both', expand=True, padx=5, pady=5)

        # Notes Panel
        notes_frame = ttk.LabelFrame(viewer_frame, text="Session Notes")
        notes_frame.grid(row=0, column=1, sticky='nsew', padx=(5, 0))
        notes_frame.grid_columnconfigure(0, weight=1)
        notes_frame.grid_rowconfigure(0, weight=1)

        self.patient_info_notes_text = tk.Text(notes_frame, height=10, state="disabled")
        self.patient_info_notes_text.pack(fill='both', expand=True, padx=5, pady=5)

        # Initialize official file paths as None
        self.stimulus_file_path = None # This is the *official* path used by analysis
        self.edf_file_path = None     # This is the *official* path used by analysis

        # Load file lists into dropdowns
        self.load_file_options()

    def detect_and_preview_sync(self):
        """Command for the new button to run sync detection and show preview."""
        # Check if files are officially selected (paths are set)
        if not self.stimulus_file_path or not self.edf_file_path:
            messagebox.showwarning("No Files Selected", "Please select and confirm files using 'Use Selected Files' first.")
            return

        # Call the AnalysisManager's new method
        self.analysis_manager.run_sync_detection_and_preview()

    # Update the state of the new button along with the submit button
    def update_submit_button_state(self, event=None):
        """Enable 'Use Selected Files' and 'Detect Sync & Preview' buttons only when both candidate files are selected."""
        if self.selected_stimulus_file_candidate and self.selected_edf_file_candidate:
            self.submit_btn.config(state='normal')
            # Do NOT enable the preview button here, it should only be enabled after files are officially selected
        else:
            self.submit_btn.config(state='disabled')
            self.sync_preview_btn.config(state='disabled')

    def load_file_options(self):
        """Populate dropdowns with files from RESULTS_DIR and EDFS_DIR"""
        # Stimulus CSVs
        stim_files = []
        if os.path.exists(RESULTS_DIR):
            stim_files = sorted([f for f in os.listdir(RESULTS_DIR) if f.lower().endswith('.csv')])
        self.stimulus_combo['values'] = stim_files
        if stim_files:
            self.stimulus_combo.set("Select a stimulus file...")
        else:
            self.stimulus_combo.set("No CSV files found")

        # EDF files
        edf_files = []
        if os.path.exists(EDFS_DIR):
            edf_files = sorted([f for f in os.listdir(EDFS_DIR) if f.lower().endswith('.edf')])
        self.edf_combo['values'] = edf_files
        if edf_files:
            self.edf_combo.set("Select an EDF file...")
        else:
            self.edf_combo.set("No EDF files found")

    def build_results_tab(self):
        """Builds the Results tab UI, focusing on analysis configuration and output."""
        results_frame = ttk.LabelFrame(self.tab3, text="EEG Analysis", padding="20")
        results_frame.pack(fill='both', expand=True, pady=10)

        # Information from Patient Info Tab
        info_frame = ttk.LabelFrame(results_frame, text="Selected Files (from Patient Information tab)", padding="10")
        info_frame.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 10))
        results_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(info_frame, text="Stimulus CSV:").grid(row=0, column=0, sticky='w', pady=2)
        self.results_stimulus_label = ttk.Label(info_frame, text="None selected", foreground="gray")
        self.results_stimulus_label.grid(row=0, column=1, sticky='w', padx=(5, 0), pady=2)

        ttk.Label(info_frame, text="EDF File:").grid(row=1, column=0, sticky='w', pady=2)
        self.results_edf_label = ttk.Label(info_frame, text="None selected", foreground="gray")
        self.results_edf_label.grid(row=1, column=1, sticky='w', padx=(5, 0), pady=2)

        # Analysis Configuration
        ttk.Label(results_frame, text="Analysis Type:").grid(row=1, column=0, sticky='w', pady=5)
        self.analysis_type_combo = ttk.Combobox(
            results_frame,
            values=["CMD Analysis", "Language Tracking"],
            state="readonly"
        )
        self.analysis_type_combo.grid(row=1, column=1, sticky='ew', padx=5, pady=5)
        self.analysis_type_combo.set("CMD Analysis")

        ttk.Label(results_frame, text="Bad Channels (comma-separated):").grid(row=2, column=0, sticky='w', pady=5)
        self.bad_channels_entry = ttk.Entry(results_frame, width=40)
        self.bad_channels_entry.grid(row=2, column=1, sticky='ew', padx=5, pady=5)
        self.bad_channels_entry.insert(0, "")

        ttk.Label(results_frame, text="EOG Channels (comma-separated):").grid(row=3, column=0, sticky='w', pady=5)
        self.eog_channels_entry = ttk.Entry(results_frame, width=40)
        self.eog_channels_entry.grid(row=3, column=1, sticky='ew', padx=5, pady=5)
        self.eog_channels_entry.insert(0, "")

        # Run Analysis Button
        self.run_analysis_button = ttk.Button(
            results_frame,
            text="Run Analysis",
            command=self.run_selected_analysis
        )
        self.run_analysis_button.grid(row=4, column=0, columnspan=2, pady=20)

        # Analysis Output Display
        self.analysis_results_text = tk.Text(results_frame, height=12, width=70)
        self.analysis_results_text.grid(row=5, column=0, columnspan=2, sticky='nsew', pady=10)

        # Configure grid weights for dynamic resizing
        results_frame.grid_rowconfigure(5, weight=1)
        results_frame.grid_columnconfigure(1, weight=1)

    def toggle_prompts(self):
        # Right Command Prompt
        if not self.right_cmd_var.get():
            self.rcmd_prompt.config(state='disabled')
            self.rcmd_prompt_var.set(False)
        else:
            self.rcmd_prompt.config(state='normal')
        # Left Command Prompt
        if not self.left_cmd_var.get():
            self.lcmd_prompt.config(state='disabled')
            self.lcmd_prompt_var.set(False)
        else:
            self.lcmd_prompt.config(state='normal')
        # Oddball Prompt
        if not self.oddball_var.get():
            self.oddball_prompt.config(state='disabled')
            self.oddball_prompt_var.set(False)
        else:
            self.oddball_prompt.config(state='normal')


    def update_button_states(self):
        config = {
            "empty":     {'prepare': 'disabled', 'play': 'disabled', 'pause': 'disabled', 'stop': 'disabled'},
            "ready":     {'prepare': 'normal',   'play': 'normal',   'pause': 'disabled', 'stop': 'disabled'},
            "preparing": {'prepare': 'disabled', 'play': 'disabled', 'pause': 'disabled', 'stop': 'disabled'},
            "paused":    {'prepare': 'disabled', 'play': 'disabled', 'pause': 'normal',   'stop': 'normal'},
            "playing":   {'prepare': 'disabled', 'play': 'disabled', 'pause': 'normal',   'stop': 'normal'},
        }
        states = config.get(self.playback_state, config["empty"])
        self.prepare_button.config(state=states['prepare'])
        self.play_button.config(state=states['play'])
        self.pause_button.config(state=states['pause'])
        self.stop_button.config(state=states['stop'])

    def button_styles(self):
        self.play_sym = tk.PhotoImage(file="lib/assets/play_sym.png").subsample(15, 15)
        self.pause_sym = tk.PhotoImage(file="lib/assets/pause_sym.png").subsample(15, 15)
        self.stop_sym = tk.PhotoImage(file="lib/assets/stop_sym3.png").subsample(6, 6)