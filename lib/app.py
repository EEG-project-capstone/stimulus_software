# lib/app.py

"""
Main application for EEG Stimulus Package.
Refactored for simplicity and maintainability.
"""

import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import logging

from lib.config import Config
from lib.stims import Stims
from lib.auditory_stimulator import AuditoryStimulator
from lib.state_manager import StateManager
from lib.results_manager import ResultsManager
from lib.constants import (
    PlaybackState,
    STATE_DISPLAYS,
    DEFAULT_STIMULUS_COUNTS,
    STIMULUS_TYPE_DISPLAY_NAMES,
    FilePaths,
    Layout
)
from lib.logging_utils import log_operation
from lib.edf_parser import EDFParser
from lib.edf_viewer import EDFViewerWindow

logger = logging.getLogger('eeg_stimulus.app')

@dataclass
class AnalysisFiles:
    """Tracks files selected for analysis."""
    stimulus_path: Optional[Path] = None
    edf_path: Optional[Path] = None
    
    @property
    def ready(self) -> bool:
        """Check if both files are selected."""
        return self.stimulus_path is not None and self.edf_path is not None
    
    def clear(self):
        """Clear file selections."""
        self.stimulus_path = None
        self.edf_path = None


class TkApp:
    """Main application class for EEG Stimulus Package."""

    # UI widget attributes set during build
    rcmd_prompt: ttk.Checkbutton
    lcmd_prompt: ttk.Checkbutton
    oddball_prompt: ttk.Checkbutton

    def __init__(self, root: tk.Tk) -> None:
        """Initialize the application.
        
        Args:
            root: Tkinter root window
        """
        logger.info("Initializing TkApp")
        self.root = root
        self.root.title("EEG Stimulus Package")
        width, height = Layout.WINDOW_SIZE
        self.root.geometry(f"{width}x{height}")
        
        # Initialize core components
        self._initialize_components()
        
        # Initialize state management
        self.state_manager = StateManager(PlaybackState.EMPTY)
        self.state_manager.add_listener(self._on_state_change)
        
        # Track current patient and analysis files
        self.current_patient = None
        self.analysis_files = AnalysisFiles()
        self.current_results_path = None
        
        # Initialize Tkinter variables
        self._initialize_variables()
        
        # Build UI
        self.build_main_ui()
        
        logger.info("TkApp initialized successfully")
    
    def _initialize_components(self):
        """Initialize core application components."""
        with log_operation("component_initialization"):
            self.config = Config()
            self.stims = Stims()
            self.results_manager = ResultsManager(self.config)
            self.audio_stim = AuditoryStimulator(self)
    
    def _initialize_variables(self):
        """Initialize Tkinter variables for UI."""
        self.language_var = tk.BooleanVar()
        self.right_cmd_var = tk.BooleanVar()
        self.rcmd_prompt_var = tk.BooleanVar()
        self.left_cmd_var = tk.BooleanVar()
        self.lcmd_prompt_var = tk.BooleanVar()
        self.oddball_var = tk.BooleanVar()
        self.oddball_prompt_var = tk.BooleanVar()
        self.familiar_var = tk.BooleanVar()
        self.gender_var = tk.StringVar(value="Male")
    
    def get_patient_id(self) -> str:
        """Get patient ID from entry field.
        
        Returns:
            Patient ID string (may be empty)
        """
        return self.patient_id_entry.get().strip()
    
    def require_patient_id(self) -> str:
        """Get patient ID, showing warning if empty.
        
        Returns:
            Patient ID string, or empty string if validation fails
        """
        patient_id = self.get_patient_id()
        
        if not patient_id:
            messagebox.showwarning("No Patient ID", 
                                 "Please enter a patient ID.")
        
        return patient_id
    
    def _on_state_change(self, old_state: PlaybackState, new_state: PlaybackState):
        """Handle state changes from state manager.
        
        Args:
            old_state: Previous state
            new_state: New current state
        """
        logger.info(f"State: {old_state.name} -> {new_state.name}")
        
        # Update UI on main thread
        self.root.after(0, self.update_button_states)
        
        # Update status label
        display = STATE_DISPLAYS.get(new_state)
        if display:
            self.status_label.config(text=display.message, foreground=display.color)
    
    def playback_complete(self):
        """Handle completion of stimulus playback (called from audio thread)."""
        logger.debug("Playback complete signal received")
        self.root.after(0, self._on_playback_complete)
    
    def _on_playback_complete(self):
        """Handle completion of stimulus playback (runs on main thread)."""
        patient_id = self.get_patient_id() or "Unknown"
        
        # Transition back to ready state
        self.state_manager.transition_to(PlaybackState.READY)
        
        # Reset pause button
        self.pause_button.config(text="Pause")

        # Reload file lists in Patient Info tab
        self.load_file_options()
        
        logger.info(f"Stimulus playback completed for patient: {patient_id}")
        messagebox.showinfo("Success", 
                          f"Stimulus administered successfully to {patient_id}")
    
    def playback_error(self, error_msg: str) -> None:
        """Handle playback error (thread-safe).
        
        Args:
            error_msg: Error message to display
        """
        logger.error(f"Playback error occurred: {error_msg}")
        self.root.after(0, self._on_playback_error, error_msg)
    
    def _on_playback_error(self, error_msg: str) -> None:
        """Handle playback error on main thread.
        
        Args:
            error_msg: Error message to display
        """
        self.state_manager.transition_to(PlaybackState.READY)
        self.pause_button.config(text="Pause")
        messagebox.showerror("Playback Error", f"Error during playback:\n{error_msg}")
    
    def send_sync_pulse(self):
        """Send a sync pulse to the EEG recording."""
        logger.info("Send sync pulse button clicked")

        patient_id = self.require_patient_id()
        if not patient_id:
            return

        # Check current state
        if not self.state_manager.is_ready():
            messagebox.showwarning("Invalid State",
                                 "Cannot send sync pulse while stimulus is active.")
            return

        # Ensure results path exists and CSV is initialized
        if not self.current_results_path:
            self.current_results_path = self.config.get_results_path(patient_id)
            logger.info(f"Created results path for sync pulse: {self.current_results_path}")
        self.results_manager.initialize_csv(patient_id)

        # Transition to SENDING_SYNC to disable pause/stop during pulse
        self.state_manager.transition_to(PlaybackState.SENDING_SYNC)

        # Send the pulse via audio stimulator
        self.audio_stim.send_sync_pulse(patient_id)

        # Reset state after pulse completes (pulse is 1000ms, wait 1500ms to be safe)
        def reset_after_pulse():
            if self.state_manager.is_sending_sync():
                self.state_manager.transition_to(PlaybackState.READY)
                self.pause_button.config(text="Pause")

        self.root.after(1500, reset_after_pulse)
        logger.info(f"Sync pulse sent for patient: {patient_id}")
    
    def on_patient_id_change(self, event=None):
        """Handle patient ID entry changes.

        Args:
            event: Tkinter event (unused)
        """
        patient_id = self.get_patient_id()

        # Don't disrupt an active playback session — editing the ID field
        # during PLAYING or PAUSED would cause an invalid state transition.
        if self.state_manager.is_active():
            self.current_patient = patient_id or None
            return

        if patient_id:
            self.current_patient = patient_id
            self.state_manager.transition_to(PlaybackState.READY)
            logger.info(f"Patient ID set: {patient_id}")
        else:
            self.current_patient = None
            self.state_manager.transition_to(PlaybackState.EMPTY)
            logger.debug("Patient ID field cleared")
    
    def toggle_familiar_options(self):
        """Toggle familiar voice options based on checkbox."""
        state = 'normal' if self.familiar_var.get() else 'disabled'
        self.male_radio.config(state=state)
        self.female_radio.config(state=state)
        self.file_button.config(state=state)
        logger.debug(f"Familiar voice options toggled: {state}")
    
    def upload_voice_file(self):
        """Handle familiar voice file upload."""
        logger.debug("Opening voice file dialog")
        
        audio_data_dir = Path(__file__).resolve().parent.parent / "audio_data"
        file_path = filedialog.askopenfilename(
            title="Select Voice File",
            initialdir=str(audio_data_dir),
            filetypes=[("Audio files", "*.wav *.mp3"), ("All files", "*.*")]
        )
        
        if file_path:
            self.stims.familiar_file = file_path
            self.stims.familiar_gender = self.gender_var.get()
            self.file_label.config(text=Path(file_path).name)
            logger.info(f"Familiar voice file uploaded: {file_path} "
                       f"(Gender: {self.gender_var.get()})")
        else:
            logger.debug("Voice file upload cancelled")
    
    def toggle_pause(self):
        """Toggle pause/resume of stimulus playback."""
        if self.state_manager.is_playing():
            logger.debug("Pausing stimulus")
            self.state_manager.transition_to(PlaybackState.PAUSED)
            self.pause_button.config(text="Resume")
            self.audio_stim.toggle_pause()

        elif self.state_manager.is_paused():
            logger.debug("Resuming stimulus")
            self.state_manager.transition_to(PlaybackState.PLAYING)
            self.pause_button.config(text="Pause")
            self.audio_stim.toggle_pause()

        self.update_stim_list_status()
    
    def stop_stimulus(self):
        """Stop the current stimulus playback."""
        if not self.state_manager.is_active():
            return

        logger.info("Stop stimulus button pressed")
        try:
            logger.debug("Transitioning state to READY")
            self.state_manager.transition_to(PlaybackState.READY)
            self.pause_button.config(text="Pause")

            logger.debug("Calling audio_stim.stop_stimulus()")
            self.audio_stim.stop_stimulus()

            logger.debug("Updating stim list status")
            self.update_stim_list_status()

            logger.info("Stop stimulus completed successfully")
        except Exception as e:
            logger.error(f"Error in stop_stimulus: {e}", exc_info=True)
    
    def prepare_stimulus(self):
        """Prepare stimulus sequence based on user selections."""
        logger.info("Prepare stimulus button clicked")
        
        if not self.state_manager.is_ready():
            logger.warning(f"Prepare stimulus called in invalid state: "
                         f"{self.state_manager.state.name}")
            return
        
        # Validate selections
        if not self._validate_stimulus_selections():
            return
        
        patient_id = self.require_patient_id()
        if not patient_id:
            return
        
        self.current_results_path = self.config.get_results_path(patient_id)
        logger.info(f"Results will be saved to: {self.current_results_path}")
        
        # Transition to preparing state
        self.state_manager.transition_to(PlaybackState.PREPARING)
        self.start_preparation()
    
    def _validate_stimulus_selections(self) -> bool:
        """Validate stimulus type selections.
        
        Returns:
            True if valid, False otherwise
        """
        if not any([self.language_var.get(), self.right_cmd_var.get(),
                   self.left_cmd_var.get(), self.oddball_var.get(),
                   self.familiar_var.get()]):
            logger.warning("No stimulus type selected")
            messagebox.showwarning("No Stimulus Selected", 
                                 "Please select at least one stimulus type.")
            return False
        
        if self.familiar_var.get() and not self.stims.familiar_file:
            logger.warning("Familiar voice stimulus selected but no voice file uploaded")
            messagebox.showwarning("Missing File",
                                 "Please upload a voice file for familiar voice stimulus.")
            return False
        
        return True
    
    def start_preparation(self):
        """Start the stimulus preparation process."""
        try:
            with log_operation("stimulus_preparation"):
                # Collect UI selections
                num_of_each_stim = self._collect_stimulus_configuration()
                
                logger.info(f"Starting stimulus preparation with configuration: "
                          f"{num_of_each_stim}")
                
                # Set gender for familiar voice if applicable
                if self.familiar_var.get():
                    self.stims.familiar_gender = self.gender_var.get()
                    logger.debug(f"Familiar voice gender set to: {self.gender_var.get()}")
                
                # Generate stimuli
                self.stims.generate_stims(num_of_each_stim)

                # Initialize CSV now that stims are prepared
                patient_id = self.get_patient_id()
                self.results_manager.initialize_csv(patient_id)

            # Transition back to ready
            self.state_manager.transition_to(PlaybackState.READY)
            
            # Update UI
            num_stims = len(self.stims.stim_dictionary)
            self.status_label.config(
                text=f"Stimulus prepared! {num_stims} stimuli ready.", 
                foreground="green"
            )
            self.populate_stim_list()
            
            logger.info(f"Stimulus preparation completed: {num_stims} stimuli generated")
            
        except Exception as e:
            logger.error(f"Error during stimulus preparation: {e}", exc_info=True)
            self.state_manager.transition_to(PlaybackState.READY)
            messagebox.showerror("Error", f"Error preparing stimulus:\n{e}")
    
    def _collect_stimulus_configuration(self) -> dict:
        """Collect stimulus configuration from UI.
        
        Returns:
            Dictionary of stimulus counts by type
        """
        return {
            "lang": (DEFAULT_STIMULUS_COUNTS['language'] 
                    if self.language_var.get() else 0),
            "rcmd": (DEFAULT_STIMULUS_COUNTS['command_no_prompt'] 
                    if self.right_cmd_var.get() and not self.rcmd_prompt_var.get() 
                    else 0),
            "rcmd+p": (DEFAULT_STIMULUS_COUNTS['command_with_prompt'] 
                      if self.right_cmd_var.get() and self.rcmd_prompt_var.get() 
                      else 0),
            "lcmd": (DEFAULT_STIMULUS_COUNTS['command_no_prompt'] 
                    if self.left_cmd_var.get() and not self.lcmd_prompt_var.get() 
                    else 0),
            "lcmd+p": (DEFAULT_STIMULUS_COUNTS['command_with_prompt'] 
                      if self.left_cmd_var.get() and self.lcmd_prompt_var.get() 
                      else 0),
            "odd": (DEFAULT_STIMULUS_COUNTS['oddball_no_prompt'] 
                   if self.oddball_var.get() and not self.oddball_prompt_var.get() 
                   else 0),
            "odd+p": (DEFAULT_STIMULUS_COUNTS['oddball_with_prompt'] 
                     if self.oddball_var.get() and self.oddball_prompt_var.get() 
                     else 0),
            "familiar_voice": (DEFAULT_STIMULUS_COUNTS['familiar_voice']
                              if self.familiar_var.get() else 0)
        }
    
    def play_stimulus(self):
        """Start stimulus playback."""
        logger.info("Play stimulus button clicked")

        if not self.state_manager.is_ready() or len(self.stims.stim_dictionary) == 0:
            logger.warning(f"Play stimulus called in invalid state: "
                         f"{self.state_manager.state.name}, "
                         f"stims: {len(self.stims.stim_dictionary)}")
            return

        patient_id = self.require_patient_id()
        if not patient_id:
            return

        # Transition to playing
        self.state_manager.transition_to(PlaybackState.PLAYING)
        self.update_stim_list_status()

        logger.info(f"Starting stimulus playback for patient: {patient_id}")
        self.audio_stim.play_stim_sequence()
    
    def add_note(self):
        """Add a session note."""
        logger.debug("Add note button clicked")
        
        if not self.current_results_path:
            logger.warning("Add note attempted before preparing stimulus")
            messagebox.showerror("Error", "Prepare stimulus before adding notes.")
            return
        
        note = self.note_entry.get().strip()
        
        patient_id = self.require_patient_id()
        if not patient_id:
            return
        
        if not note:
            logger.debug("Empty note submission attempted")
            messagebox.showwarning("Empty Note", "Please enter a note before adding.")
            return
        
        try:
            # Use results manager to save note
            self.results_manager.append_note(patient_id, note)
            
            # Update UI
            self.notes_text.insert(tk.END, f"[{self.config.current_date}] {note}\n")
            self.notes_text.see(tk.END)
            self.note_entry.delete(0, tk.END)
            
            logger.info(f"Session note added for patient {patient_id}: {note[:50]}...")
            
        except Exception as e:
            logger.error(f"Failed to save note: {e}", exc_info=True)
            messagebox.showerror("Note Save Error", f"Failed to save note:\n{e}")
    
    def _is_valid_file_selection(self, filename: str) -> bool:
        """Check if combobox selection is a valid file (not a placeholder).
        
        Args:
            filename: Selected filename from combobox
            
        Returns:
            True if valid file selection
        """
        placeholders = [
            "Select a stimulus file...", 
            "Select an EDF file...", 
            "No CSV files found", 
            "No EDF files found", 
            ""
        ]
        return filename not in placeholders
    
    def on_stimulus_selected(self, event=None):
        """Handle stimulus file selection.
        
        Args:
            event: Tkinter event (unused)
        """
        filename = self.stimulus_combo.get()
        
        if self._is_valid_file_selection(filename):
            self.analysis_files.stimulus_path = FilePaths.RESULTS_DIR / filename
            logger.info(f"Stimulus file selected: {filename}")
        else:
            self.analysis_files.stimulus_path = None
            logger.debug("Stimulus file selection cleared")
        
        self._update_analysis_ui()
    
    def on_edf_selected(self, event=None):
        """Handle EDF file selection.
        
        Args:
            event: Tkinter event (unused)
        """
        filename = self.edf_combo.get()
        
        if self._is_valid_file_selection(filename):
            self.analysis_files.edf_path = FilePaths.EDFS_DIR / filename
            logger.info(f"EDF file selected: {filename}")
        else:
            self.analysis_files.edf_path = None
            logger.debug("EDF file selection cleared")
        
        self._update_analysis_ui()
    
    def _update_analysis_ui(self):
        """Update analysis UI based on file selections."""
        both_ready = self.analysis_files.ready
        if both_ready:
            self.submit_btn.config(state='normal')
            self.view_edf_btn.config(state='normal')
        else:
            self.submit_btn.config(state='disabled')
            self.view_edf_btn.config(state='disabled')
    
    def confirm_file_selection(self):
        """Confirm selected files for analysis, auto-parse EDF, and populate info panels."""
        if not self.analysis_files.ready:
            logger.warning("Confirm files pressed but files not ready")
            return

        if self.analysis_files.stimulus_path is None or self.analysis_files.edf_path is None:
            logger.error("Files were not ready despite ready check - this should not happen")
            return

        stim_path = self.analysis_files.stimulus_path
        edf_path  = self.analysis_files.edf_path
        logger.info(f"Files confirmed — Stimulus: {stim_path.name}, EDF: {edf_path.name}")

        self.official_stimulus_label.config(text=stim_path.name, foreground="green")
        self.official_edf_label.config(text=edf_path.name, foreground="green")

        self._show_csv_info(stim_path)
        self.parse_edf()

    def _show_csv_info(self, filepath: Path):
        """Populate the CSV info panel with stats and session notes."""
        import pandas as pd
        lines = []
        try:
            df = pd.read_csv(filepath)
            patient_id = (df['patient_id'].dropna().iloc[0]
                          if 'patient_id' in df.columns and not df.empty else 'unknown')

            log_types  = {'session_note', 'manual_sync_pulse', 'sync_detection'}
            stim_rows  = df[~df['stim_type'].isin(log_types)] if 'stim_type' in df.columns else df
            note_rows  = df[df['stim_type'].isin(log_types)]  if 'stim_type' in df.columns else pd.DataFrame()

            lines = [
                f"Patient:    {patient_id}",
                f"Stimuli:    {len(stim_rows)}",
                f"Total rows: {len(df)}",
            ]
            if 'date' in df.columns:
                dates = df['date'].dropna().unique()
                lines.append(f"Date(s):    {', '.join(str(d) for d in dates[:3])}")

            if 'stim_type' in df.columns:
                lines.append("")
                lines.append("── Stimulus counts ──────────────")
                for t, c in stim_rows['stim_type'].value_counts().items():
                    lines.append(f"  {t}: {c}")

            if not note_rows.empty:
                lines.append("")
                lines.append("── Session notes ────────────────")
                for _, row in note_rows.iterrows():
                    stype = row.get('stim_type', '')
                    note  = row.get('notes', '')
                    t     = row.get('start_time', '')
                    lines.append(f"  [{stype}]  {note}  {t}")

        except Exception as e:
            lines = [f"Could not read CSV:\n{e}"]

        self.csv_info_text.config(state='normal')
        self.csv_info_text.delete('1.0', tk.END)
        self.csv_info_text.insert(tk.END, '\n'.join(lines))
        self.csv_info_text.config(state='disabled')
    
    
    def populate_stim_list(self):
        """Populate the stimulus list Treeview."""
        logger.debug(f"Populating stimulus list with "
                    f"{len(self.stims.stim_dictionary)} stimuli")
        
        # Clear existing items
        for item in self.stim_tree.get_children():
            self.stim_tree.delete(item)
        
        # Insert each stimulus
        for idx, stim in enumerate(self.stims.stim_dictionary):
            stim_type = stim['type']
            display_type = STIMULUS_TYPE_DISPLAY_NAMES.get(
                stim_type, 
                stim_type.replace('_', ' ').title()
            )
            status = stim['status'].title()
            self.stim_tree.insert('', 'end', iid=str(idx), 
                                 values=(display_type, status))
    
    def update_stim_list_status(self):
        """Update the status column in the stimulus list."""
        for idx, stim in enumerate(self.stims.stim_dictionary):
            if str(idx) in self.stim_tree.get_children():
                display_type = STIMULUS_TYPE_DISPLAY_NAMES.get(
                    stim['type'], 
                    stim['type'].replace('_', ' ').title()
                )
                status = stim['status'].title()
                
                # Determine tag
                status_key = stim['status'].lower()
                if 'complete' in status_key:
                    tag = 'completed'
                elif 'in progress' in status_key:
                    tag = 'inprogress'
                else:
                    tag = 'pending'
                
                self.stim_tree.item(str(idx), values=(display_type, status), 
                                   tags=(tag,))
    
    def toggle_prompts(self):
        """Toggle prompt checkboxes based on main stimulus selection."""
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
    
    def _apply_ctrl_btn(self, btn: tk.Button, enabled: bool) -> None:
        """Apply enabled/disabled state with corresponding shading to a control button."""
        if enabled:
            btn.config(state='normal', bg='#e2e2e2', fg='#111111',
                       activebackground='#c8c8c8', activeforeground='#111111',
                       relief='raised')
        else:
            btn.config(state='disabled', bg='#c0c0c0', fg='#888888',
                       activebackground='#c0c0c0', activeforeground='#888888',
                       relief='flat')

    def update_button_states(self):
        """Update button enabled/disabled states based on current state."""
        # Define button states for each playback state
        state_configs = {
            PlaybackState.EMPTY: {
                'prepare': 'disabled', 'play': 'disabled',
                'pause': 'disabled', 'stop': 'disabled', 'sync': 'disabled'
            },
            PlaybackState.READY: {
                'prepare': 'normal', 'play': 'normal',
                'pause': 'disabled', 'stop': 'disabled', 'sync': 'normal'
            },
            PlaybackState.PREPARING: {
                'prepare': 'disabled', 'play': 'disabled',
                'pause': 'disabled', 'stop': 'disabled', 'sync': 'disabled'
            },
            PlaybackState.PAUSED: {
                'prepare': 'disabled', 'play': 'disabled',
                'pause': 'normal', 'stop': 'normal', 'sync': 'disabled'
            },
            PlaybackState.PLAYING: {
                'prepare': 'disabled', 'play': 'disabled',
                'pause': 'normal', 'stop': 'normal', 'sync': 'disabled'
            },
            PlaybackState.STOPPED: {
                'prepare': 'normal', 'play': 'normal',
                'pause': 'disabled', 'stop': 'disabled', 'sync': 'normal'
            },
            PlaybackState.SENDING_SYNC: {
                'prepare': 'disabled', 'play': 'disabled',
                'pause': 'disabled', 'stop': 'disabled', 'sync': 'disabled'
            },
        }

        states = state_configs.get(
            self.state_manager.state,
            state_configs[PlaybackState.EMPTY]
        )

        self._apply_ctrl_btn(self.prepare_button,   states['prepare'] == 'normal')
        self._apply_ctrl_btn(self.play_button,       states['play']    == 'normal')
        self._apply_ctrl_btn(self.pause_button,      states['pause']   == 'normal')
        self._apply_ctrl_btn(self.stop_button,       states['stop']    == 'normal')
        self._apply_ctrl_btn(self.sync_pulse_button, states['sync']    == 'normal')
    
    def load_file_options(self):
        """Populate file selection dropdowns."""
        # Stimulus CSVs
        stim_files = []
        if FilePaths.RESULTS_DIR.exists():
            stim_files = sorted([f.name for f in FilePaths.RESULTS_DIR.glob('*.csv')])

        self.stimulus_combo['values'] = stim_files
        if stim_files:
            self.stimulus_combo.set("Select a stimulus file...")
        else:
            self.stimulus_combo.set("No CSV files found")

        # EDF files (case-insensitive glob for .edf/.EDF)
        edf_files = []
        if FilePaths.EDFS_DIR.exists():
            edf_files = sorted([f.name for f in FilePaths.EDFS_DIR.iterdir()
                               if f.suffix.lower() == '.edf'])
        
        self.edf_combo['values'] = edf_files
        if edf_files:
            self.edf_combo.set("Select an EDF file...")
        else:
            self.edf_combo.set("No EDF files found")

    def _refresh_edf_options(self):
        """Refresh EDF file list when the dropdown is opened."""
        edf_files = []
        if FilePaths.EDFS_DIR.exists():
            edf_files = sorted([f.name for f in FilePaths.EDFS_DIR.iterdir()
                                if f.suffix.lower() == '.edf'])
        current = self.edf_combo.get()
        self.edf_combo['values'] = edf_files
        if current not in edf_files:
            self.edf_combo.set("Select an EDF file..." if edf_files else "No EDF files found")

    def _refresh_stimulus_options(self):
        """Refresh stimulus CSV list when the dropdown is opened."""
        stim_files = []
        if FilePaths.RESULTS_DIR.exists():
            stim_files = sorted([f.name for f in FilePaths.RESULTS_DIR.glob('*.csv')])
        current = self.stimulus_combo.get()
        self.stimulus_combo['values'] = stim_files
        if current not in stim_files:
            self.stimulus_combo.set("Select a stimulus file..." if stim_files else "No CSV files found")

    def parse_edf(self):
        """Parse the confirmed EDF file and display info in the EDF info panel."""
        if self.analysis_files.edf_path is None:
            return

        edf_path = self.analysis_files.edf_path
        logger.info(f"Parsing EDF file: {edf_path}")

        def _set(text):
            self.edf_info_text.config(state='normal')
            self.edf_info_text.delete('1.0', tk.END)
            self.edf_info_text.insert(tk.END, text)
            self.edf_info_text.config(state='disabled')

        try:
            parser = EDFParser(str(edf_path))
            parser.load_edf()
            info = parser.get_info_summary()

            n_channels = len(info['ch_names'])
            duration_sec = info['duration']
            hours   = int(duration_sec // 3600)
            minutes = int((duration_sec % 3600) // 60)
            seconds = duration_sec % 60

            lines = [
                f"Duration:  {hours}h {minutes}m {seconds:.1f}s",
                f"           ({duration_sec:.1f} s total)",
                f"Sfreq:     {info['sfreq']} Hz",
                f"Samples:   {info['n_times']:,}",
            ]
            if info.get('meas_date'):
                lines.append(f"Recorded:  {info['meas_date']}")
            lines.append(f"Channels ({n_channels}):")
            ch_names = info['ch_names']
            for i in range(0, len(ch_names), 4):
                lines.append("  " + "  ".join(f"{c:<10}" for c in ch_names[i:i+4]))

            _set('\n'.join(lines))
            logger.info(f"EDF parsed: {n_channels} channels, {duration_sec:.1f}s")

        except Exception as e:
            logger.error(f"Failed to parse EDF file: {e}", exc_info=True)
            _set(f"Parse error:\n{e}")
            messagebox.showerror("EDF Parse Error", f"Failed to parse EDF file:\n{e}")

    def view_edf(self):
        """Open interactive EDF viewer window."""
        if self.analysis_files.edf_path is None:
            messagebox.showwarning("No EDF Selected",
                                   "Please select an EDF file first.")
            return

        edf_path = self.analysis_files.edf_path
        logger.info(f"Opening EDF viewer for: {edf_path}")

        try:
            parser = EDFParser(str(edf_path))
            parser.load_edf()
            EDFViewerWindow(self.root, parser,
                            stimulus_path=self.analysis_files.stimulus_path)
        except Exception as e:
            logger.error(f"Failed to open EDF viewer: {e}", exc_info=True)
            messagebox.showerror("EDF Viewer Error",
                                 f"Failed to open EDF viewer:\n{e}")

    # ========================================================================
    # UI BUILDING METHODS
    # ========================================================================
    
    def build_main_ui(self):
        """Build the main UI with tabs."""
        # Create notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, 
                          padx=Layout.MAIN_PADDING, pady=Layout.MAIN_PADDING)
        
        # Create tabs
        tab1 = ttk.Frame(self.notebook)
        tab2 = ttk.Frame(self.notebook)
        
        self.notebook.add(tab1, text="Administer Stimuli")
        self.notebook.add(tab2, text="Patient Information")

        # Build tabs
        self._build_stimulus_tab(tab1)
        self._build_patient_info_tab(tab2)
        
        # Initialize file options
        self.load_file_options()
        
        # Initialize button states
        self.update_button_states()

    def _build_stimulus_tab(self, parent):
        """Build the Administer Stimuli tab."""
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill='both', expand=True, padx=Layout.MAIN_PADDING, 
                       pady=Layout.MAIN_PADDING)
        
        self._build_patient_section(main_frame)
        self._build_stimulus_checkboxes(main_frame)
        self._build_control_panel(main_frame)
        self._build_sequence_and_notes(main_frame)
    
    def _build_patient_section(self, parent):
        """Build patient information section."""
        patient_frame = ttk.LabelFrame(parent, text="Patient Information", padding="10")
        patient_frame.pack(fill='x', pady=5)
        
        ttk.Label(patient_frame, text="Patient/EEG ID:").grid(
            row=0, column=0, sticky='w', padx=5
        )
        
        self.patient_id_entry = ttk.Entry(patient_frame, width=30)
        self.patient_id_entry.grid(row=0, column=1, sticky='ew', padx=5)
        self.patient_id_entry.bind('<KeyRelease>', self.on_patient_id_change)
        
        self.status_label = ttk.Label(
            patient_frame, 
            text="Please enter a patient ID", 
            foreground="red"
        )
        self.status_label.grid(row=1, column=0, columnspan=2, pady=5)
        
        patient_frame.grid_columnconfigure(1, weight=1)
    
    def _build_stimulus_checkboxes(self, parent):
        """Build all stimulus selection checkboxes."""
        stim_frame = ttk.LabelFrame(parent, text="Stimulus Selection", padding="10")
        stim_frame.pack(fill='x', pady=5)
        
        # Language
        ttk.Checkbutton(
            stim_frame, 
            text="Language Stimulus", 
            variable=self.language_var
        ).grid(row=0, column=0, sticky='w', pady=2)
        
        # Right Command
        self._build_command_row(stim_frame, row=1, side="Right",
                               cmd_var=self.right_cmd_var,
                               prompt_var=self.rcmd_prompt_var,
                               prompt_attr='rcmd_prompt')
        
        # Left Command
        self._build_command_row(stim_frame, row=2, side="Left",
                               cmd_var=self.left_cmd_var,
                               prompt_var=self.lcmd_prompt_var,
                               prompt_attr='lcmd_prompt')
        
        # Oddball
        self._build_oddball_row(stim_frame, row=3)
        
        # Familiar Voice
        self._build_familiar_row(stim_frame, row=4)
    
    def _build_command_row(self, parent, row: int, side: str,
                          cmd_var, prompt_var, prompt_attr: str):
        """Build a command stimulus row with prompt checkbox."""
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, columnspan=2, sticky='ew', pady=2)
        
        ttk.Checkbutton(
            frame,
            text=f"{side} Command Stimulus",
            variable=cmd_var,
            command=self.toggle_prompts
        ).pack(side='left')
        
        prompt = ttk.Checkbutton(
            frame,
            text="Include Prompt",
            variable=prompt_var,
            state='disabled'
        )
        prompt.pack(side='right')
        
        setattr(self, prompt_attr, prompt)
    
    def _build_oddball_row(self, parent, row: int):
        """Build oddball stimulus row."""
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, columnspan=2, sticky='ew', pady=2)
        
        ttk.Checkbutton(
            frame,
            text="Oddball Stimulus",
            variable=self.oddball_var,
            command=self.toggle_prompts
        ).pack(side='left')
        
        self.oddball_prompt = ttk.Checkbutton(
            frame,
            text="Include Prompt",
            variable=self.oddball_prompt_var,
            state='disabled'
        )
        self.oddball_prompt.pack(side='right')
    
    def _build_familiar_row(self, parent, row: int):
        """Build familiar voice stimulus row with gender and file selection."""
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, columnspan=3, sticky='ew', pady=2)

        ttk.Checkbutton(
            frame,
            text="Familiar Voice Stimulus",
            variable=self.familiar_var,
            command=self.toggle_familiar_options
        ).pack(side='left')
        
        # Gender selection
        gender_frame = ttk.Frame(frame)
        gender_frame.pack(side='left', padx=20)
        
        ttk.Label(gender_frame, text="Gender:").pack(side='left')
        
        self.male_radio = ttk.Radiobutton(
            gender_frame,
            text="Male",
            variable=self.gender_var,
            value="Male"
        )
        self.male_radio.pack(side='left')
        
        self.female_radio = ttk.Radiobutton(
            gender_frame,
            text="Female",
            variable=self.gender_var,
            value="Female"
        )
        self.female_radio.pack(side='left')
        
        # File upload
        self.file_button = ttk.Button(
            frame,
            text="Upload Voice File",
            command=self.upload_voice_file
        )
        self.file_button.pack(side='left', padx=20)
        
        self.file_label = ttk.Label(frame, text="No file selected")
        self.file_label.pack(side='left')
    
    def _build_control_panel(self, parent):
        """Build control buttons panel."""
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill='x', pady=10)

        self.prepare_button = tk.Button(
            control_frame,
            text="Prepare Stimulus",
            command=self.prepare_stimulus
        )
        self.prepare_button.pack(side='left', padx=5, ipady=4)

        self.sync_pulse_button = tk.Button(
            control_frame,
            text="Send Sync Pulse",
            command=self.send_sync_pulse,
        )
        self.sync_pulse_button.pack(side='left', padx=5, ipady=4)

        self.play_button = tk.Button(
            control_frame,
            text="Play Stimulus",
            command=self.play_stimulus
        )
        self.play_button.pack(side='left', padx=5, ipady=4)

        self.stop_button = tk.Button(
            control_frame,
            text="Stop",
            command=self.stop_stimulus
        )
        self.stop_button.pack(side='left', padx=5, ipady=4)

        self.pause_button = tk.Button(
            control_frame,
            text="Pause",
            command=self.toggle_pause
        )
        self.pause_button.pack(side='left', padx=5, ipady=4)
    
    def _build_sequence_and_notes(self, parent):
        """Build side-by-side stimulus sequence list and session notes."""
        container = ttk.Frame(parent)
        container.pack(fill='both', expand=True, pady=5)
        container.grid_columnconfigure(0, weight=1)
        container.grid_columnconfigure(1, weight=1)
        container.grid_rowconfigure(0, weight=1)
        
        # Stimulus list (left)
        self._build_stimulus_list(container)
        
        # Notes (right)
        self._build_notes_section(container)
    
    def _build_stimulus_list(self, parent):
        """Build stimulus sequence treeview."""
        frame = ttk.LabelFrame(parent, text="Stimulus Sequence")
        frame.grid(row=0, column=0, sticky='nsew', padx=(0, 5))
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(0, weight=1)
        
        self.stim_tree = ttk.Treeview(
            frame,
            columns=('Type', 'Status'),
            show='headings',
            height=Layout.STIMULUS_LIST_HEIGHT
        )
        
        self.stim_tree.heading('Type', text='Stimulus Type')
        self.stim_tree.heading('Status', text='Status')
        self.stim_tree.column('Type', width=150, minwidth=100)
        self.stim_tree.column('Status', width=100, minwidth=80)
        
        # Configure tags for status colors
        self.stim_tree.tag_configure('pending', foreground='gray')
        self.stim_tree.tag_configure('inprogress', foreground='blue')
        self.stim_tree.tag_configure('completed', foreground='green')
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(
            frame,
            orient="vertical",
            command=self.stim_tree.yview
        )
        scrollbar.pack(side='right', fill='y')
        self.stim_tree.configure(yscrollcommand=scrollbar.set)
        
        self.stim_tree.pack(side='left', fill='both', expand=True, 
                           padx=(5, 0), pady=5)
    
    def _build_notes_section(self, parent):
        """Build session log/notes section."""
        frame = ttk.LabelFrame(parent, text="Session Log", padding="10")
        frame.grid(row=0, column=1, sticky='nsew', padx=(5, 0))
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_rowconfigure(1, weight=1)
        
        # Add note entry
        ttk.Label(frame, text="Add Note:").grid(row=0, column=0, sticky='w')
        
        self.note_entry = ttk.Entry(frame, width=Layout.LOG_TEXT_WIDTH)
        self.note_entry.grid(row=0, column=1, sticky='ew', padx=5)
        
        ttk.Button(frame, text="Add Note", command=self.add_note).grid(
            row=0, column=2, padx=5
        )
        
        # Notes display
        self.notes_text = tk.Text(
            frame,
            height=Layout.NOTES_TEXT_HEIGHT,
            width=Layout.LOG_TEXT_WIDTH
        )
        self.notes_text.grid(row=1, column=0, columnspan=3, sticky='nsew', pady=5)
    
    def _build_patient_info_tab(self, parent):
        """Build the Patient Information tab."""
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill='both', expand=True, padx=Layout.MAIN_PADDING, 
                       pady=Layout.MAIN_PADDING)
        
        self._build_file_selection(main_frame)
        self._build_action_buttons(main_frame)
        self._build_confirmed_files(main_frame)
    
    def _build_file_selection(self, parent):
        """Build file selection dropdowns."""
        frame = ttk.LabelFrame(parent, text="File Selection", padding="10")
        frame.pack(fill='x', padx=5, pady=5)
        
        # Stimulus CSV dropdown
        self.stimulus_combo = ttk.Combobox(
            frame, state="readonly", width=50,
            postcommand=self._refresh_stimulus_options)
        self.stimulus_combo.pack(side='top', fill='x', pady=2)
        self.stimulus_combo.bind("<<ComboboxSelected>>", self.on_stimulus_selected)

        # EDF dropdown
        self.edf_combo = ttk.Combobox(
            frame, state="readonly", width=50,
            postcommand=self._refresh_edf_options)
        self.edf_combo.pack(side='top', fill='x', pady=2)
        self.edf_combo.bind("<<ComboboxSelected>>", self.on_edf_selected)
    
    def _build_action_buttons(self, parent):
        """Build action buttons."""
        frame = ttk.Frame(parent)
        frame.pack(fill='x', padx=5, pady=15)

        self.submit_btn = ttk.Button(
            frame,
            text="Use Selected Files",
            command=self.confirm_file_selection,
            state="disabled"
        )
        self.submit_btn.pack(side='left', padx=(0, 5))

        self.view_edf_btn = ttk.Button(
            frame,
            text="View EDF",
            command=self.view_edf,
            state="disabled"
        )
        self.view_edf_btn.pack(side='left', padx=(5, 0))
    
    def _build_confirmed_files(self, parent):
        """Build confirmed files display with side-by-side info panels."""
        outer = ttk.LabelFrame(
            parent,
            text="Confirmed Files for Analysis",
            padding="8"
        )
        outer.pack(fill='both', expand=True, padx=5, pady=5)
        outer.grid_columnconfigure(0, weight=1)
        outer.grid_columnconfigure(1, weight=1)
        outer.grid_rowconfigure(0, weight=1)

        # ── Stimulus CSV panel ─────────────────────────
        csv_frame = ttk.LabelFrame(outer, text="Stimulus CSV", padding="6")
        csv_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 4))
        csv_frame.grid_columnconfigure(0, weight=1)
        csv_frame.grid_rowconfigure(1, weight=1)

        self.official_stimulus_label = ttk.Label(
            csv_frame, text="None selected", foreground="red", anchor='w')
        self.official_stimulus_label.grid(row=0, column=0, sticky='ew', pady=(0, 4))

        self.csv_info_text = tk.Text(
            csv_frame, height=16, state='disabled',
            font=('TkFixedFont', 9), wrap='word')
        self.csv_info_text.grid(row=1, column=0, sticky='nsew')

        # ── EDF panel ──────────────────────────────────
        edf_frame = ttk.LabelFrame(outer, text="EDF File", padding="6")
        edf_frame.grid(row=0, column=1, sticky='nsew', padx=(4, 0))
        edf_frame.grid_columnconfigure(0, weight=1)
        edf_frame.grid_rowconfigure(1, weight=1)

        self.official_edf_label = ttk.Label(
            edf_frame, text="None selected", foreground="red", anchor='w')
        self.official_edf_label.grid(row=0, column=0, sticky='ew', pady=(0, 4))

        self.edf_info_text = tk.Text(
            edf_frame, height=16, state='disabled',
            font=('TkFixedFont', 9), wrap='word')
        self.edf_info_text.grid(row=1, column=0, sticky='nsew')
    

    def cleanup(self):
        """Clean up resources before closing the application."""
        logger.info("Cleaning up application resources")
        try:
            if hasattr(self, 'audio_stim') and self.audio_stim:
                logger.debug("Stopping audio stimulator")
                self.audio_stim.stop_stimulus()
                self.audio_stim._cancel_scheduled_callbacks()
                self.audio_stim.stream_manager.shutdown()

            logger.info("Cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True)