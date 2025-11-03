# lib/app.py

import os
import time
import pandas as pd
import tkinter as tk
from tkinter import messagebox, ttk, filedialog, DISABLED, NORMAL
from lib.config import Config
from lib.trials import Trials
from lib.auditory_stimulator import AuditoryStimulator
from lib.cmd_analysis import CMDAnalyzer
from lib.edf_parser import EDFParser

class TkApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EEG Stimulus Package")
        self.root.geometry("1050x830")
        
        # create class instances 
        self.config = Config()
        self.trials = Trials(self)
        self.audio_stim = AuditoryStimulator(self)

        self.playback_state = "empty"
        self.current_patient = None
        self.language_var = tk.BooleanVar()
        self.right_cmd_var = tk.BooleanVar()
        self.rcmd_prompt_var = tk.BooleanVar()
        self.left_cmd_var = tk.BooleanVar()
        self.lcmd_prompt_var = tk.BooleanVar()
        self.oddball_var = tk.BooleanVar()
        self.oddball_prompt_var = tk.BooleanVar()
        self.loved_one_var = tk.BooleanVar()
        self.gender_var = tk.StringVar(value="Male")

        self.build_main_ui()
    
    def get_patient_id(self):
        """Get patient ID from Stimulus tab"""
        patient_id = self.patient_id_entry.get().strip()
        if not patient_id:
            raise ValueError("Patient ID cannot be empty")
        # Add any format validation here
        return patient_id

    def playback_complete(self):
        """Handle completion of stimulus playback"""
        patient_id = self.patient_id_entry.get().strip()
        self.playback_state = "ready"
        self.audio_stim.is_paused = False # Ensure sync
        self.pause_button.config(text="Pause", image=self.pause_sym)
        self.update_button_states()
        self.status_label.config(text=f"Stimulus completed for {patient_id}", foreground="green")
        messagebox.showinfo("Success", f"Stimulus administered successfully to {patient_id}")

    def playback_error(self, error_msg):
        """Handle playback errors"""
        self.playback_state = "ready"
        self.audio_stim.is_paused = False # Ensure sync
        self.pause_button.config(text="Pause", image=self.pause_sym)
        self.update_button_states()
        messagebox.showerror("Playback Error", f"Error during playback: {error_msg}")

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
        # Main frame (no canvas or outer scrollbar)
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

        # === SIDE-BY-SIDE: Trial List + Notes ===
        side_by_side_frame = ttk.Frame(main_frame)
        side_by_side_frame.pack(fill='both', expand=True, pady=5)
        side_by_side_frame.grid_columnconfigure(0, weight=1)  # Trial list gets more space
        side_by_side_frame.grid_columnconfigure(1, weight=1)  # Notes gets equal space
        side_by_side_frame.grid_rowconfigure(0, weight=1)

        # Trial List Frame (left)
        trial_list_frame = ttk.LabelFrame(side_by_side_frame, text="Trial Sequence")
        trial_list_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 5))
        trial_list_frame.grid_columnconfigure(0, weight=1)
        trial_list_frame.grid_rowconfigure(0, weight=1)

        # Treeview with scrollbar
        self.trial_tree = ttk.Treeview(trial_list_frame, columns=('Type', 'Status'), show='headings', height=12)
        self.trial_tree.heading('Type', text='Trial Type')
        self.trial_tree.heading('Status', text='Status')
        self.trial_tree.column('Type', width=150, minwidth=100)
        self.trial_tree.column('Status', width=100, minwidth=80)
        self.trial_tree.tag_configure('pending', foreground='gray')
        self.trial_tree.tag_configure('inprogress', foreground='blue')
        self.trial_tree.tag_configure('completed', foreground='green')

        # Scrollbar for trial list only
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

        # Update button states
        self.update_button_states()

    def build_patient_info_tab(self):

        main_frame = ttk.Frame(self.tab2)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # === Stimulus CSV Dropdown ===
        stim_frame = ttk.LabelFrame(main_frame, text="Stimulus CSV File")
        stim_frame.pack(fill='x', padx=5, pady=5)
        
        self.stimulus_combo = ttk.Combobox(stim_frame, state="readonly", width=50)
        self.stimulus_combo.pack(side='left', padx=5, fill='x', expand=True)
        self.stimulus_combo.bind("<<ComboboxSelected>>", self.on_stimulus_selected)
        
        # === EDF Dropdown ===
        edf_frame = ttk.LabelFrame(main_frame, text="EDF File")
        edf_frame.pack(fill='x', padx=5, pady=5)
        
        self.edf_combo = ttk.Combobox(edf_frame, state="readonly", width=50)
        self.edf_combo.pack(side='left', padx=5, fill='x', expand=True)
        self.edf_combo.bind("<<ComboboxSelected>>", self.on_edf_selected)

        # Submit Button
        self.submit_btn = ttk.Button(main_frame, text="Use Selected Files", 
                                    command=self.update_results_file_labels, state=DISABLED)
        self.submit_btn.pack(pady=15)

        # === Trial List + Notes Viewer ===
        viewer_frame = ttk.Frame(main_frame)
        viewer_frame.pack(fill='both', expand=True, pady=10)
        viewer_frame.grid_columnconfigure(0, weight=1)
        viewer_frame.grid_columnconfigure(1, weight=1)
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

        # Initialize file paths as None
        self.stimulus_file_path = None
        self.edf_file_path = None

        # Load file lists into dropdowns
        self.load_file_options()

    def load_file_options(self):
        """Populate dropdowns with files from patient_data/results and patient_data/edfs"""
        # Stimulus CSVs
        results_dir = "patient_data/results"
        stim_files = []
        if os.path.exists(results_dir):
            stim_files = sorted([
                f for f in os.listdir(results_dir) 
                if f.lower().endswith('.csv')
            ])
        self.stimulus_combo['values'] = stim_files
        if stim_files:
            self.stimulus_combo.set("Select a stimulus file...")
        else:
            self.stimulus_combo.set("No CSV files found")

        # EDF files
        edf_dir = "patient_data/edfs"
        edf_files = []
        if os.path.exists(edf_dir):
            edf_files = sorted([
                f for f in os.listdir(edf_dir) 
                if f.lower().endswith('.edf')
            ])
        self.edf_combo['values'] = edf_files
        if edf_files:
            self.edf_combo.set("Select an EDF file...")
        else:
            self.edf_combo.set("No EDF files found")

    def on_stimulus_selected(self, event=None):
        filename = self.stimulus_combo.get()
        # Ignore placeholder text
        if filename in ["Select a stimulus file...", "No CSV files found", ""]:
            self.stimulus_file_path = None
        else:
            self.stimulus_file_path = os.path.join("patient_data/results", filename)
            # Load session data into Patient Info tab
            self.load_session_data_from_csv(self.stimulus_file_path)
        self.update_submit_button_state()

    def on_edf_selected(self, event=None):
        filename = self.edf_combo.get()
        if filename in ["Select an EDF file...", "No EDF files found", ""]:
            self.edf_file_path = None
        else:
            self.edf_file_path = os.path.join("patient_data/edfs", filename)
        self.update_submit_button_state()

    def load_session_data_from_csv(self, filepath):
        """Load trial sequence and notes from stimulus CSV for Patient Info tab."""
        # Clear previous data
        for item in self.patient_info_trial_tree.get_children():
            self.patient_info_trial_tree.delete(item)
        self.patient_info_notes_text.config(state="normal")
        self.patient_info_notes_text.delete(1.0, tk.END)
        
        try:
            df = pd.read_csv(filepath)
            
            type_display_names = {
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
            
            for _, row in df.iterrows():
                trial_type = row.get('trial_type', 'unknown')
                display_type = type_display_names.get(trial_type, trial_type.replace('_', ' ').title())
                
                if trial_type == 'session_note':
                    status = "Note"
                    tag = 'session_note'
                    # Add to notes panel
                    date = row.get('date', 'Unknown')
                    note_text = str(row.get('notes', '')).strip()
                    if note_text:
                        self.patient_info_notes_text.insert(tk.END, f"[{date}] {note_text}\n")
                else:
                    status = "Completed"
                    tag = 'completed'
                
                self.patient_info_trial_tree.insert('', 'end', values=(display_type, status), tags=(tag,))
            
            self.patient_info_notes_text.config(state="disabled")
            if self.patient_info_notes_text.get(1.0, tk.END).strip():
                self.patient_info_notes_text.see(tk.END)
                        
        except Exception as e:
            messagebox.showerror("Load Error", f"Could not load session data:\n{e}")

    def update_submit_button_state(self, event=None):
        """Enable submit button only when both files are selected"""
        if self.stimulus_file_path and self.edf_file_path:
            self.submit_btn.config(state=NORMAL)
        else:
            self.submit_btn.config(state=DISABLED)

    def build_results_tab(self):
        """Builds the Results tab UI, focusing on analysis configuration and output."""
        results_frame = ttk.LabelFrame(self.tab3, text="EEG Analysis", padding="20")
        results_frame.pack(fill='both', expand=True, pady=10)

        # --- Information from Patient Info Tab ---
        # Displays the files currently selected in the Patient Info tab.
        info_frame = ttk.LabelFrame(results_frame, text="Selected Files (from Patient Info Tab)", padding="10")
        info_frame.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 10))
        results_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(info_frame, text="Stimulus CSV:").grid(row=0, column=0, sticky='w', pady=2)
        self.results_stimulus_label = ttk.Label(info_frame, text="None selected", foreground="gray")
        self.results_stimulus_label.grid(row=0, column=1, sticky='w', padx=(5, 0), pady=2)

        ttk.Label(info_frame, text="EDF File:").grid(row=1, column=0, sticky='w', pady=2)
        self.results_edf_label = ttk.Label(info_frame, text="None selected", foreground="gray")
        self.results_edf_label.grid(row=1, column=1, sticky='w', padx=(5, 0), pady=2)

        # --- Analysis Configuration ---
        ttk.Label(results_frame, text="Analysis Type:").grid(row=1, column=0, sticky='w', pady=5)
        self.analysis_type_combo = ttk.Combobox(
            results_frame,
            values=["EDF Parse Only", "CMD Analysis", "Language Tracking"],
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

        # --- Run Analysis Button ---
        self.run_analysis_button = ttk.Button(
            results_frame,
            text="Run Analysis",
            command=self.run_selected_analysis
        )
        self.run_analysis_button.grid(row=4, column=0, columnspan=2, pady=20)

        # --- Analysis Output Display ---
        self.analysis_results_text = tk.Text(results_frame, height=12, width=70) # Increased height for better visibility
        self.analysis_results_text.grid(row=5, column=0, columnspan=2, sticky='nsew', pady=10)

        # Configure grid weights for dynamic resizing
        results_frame.grid_rowconfigure(5, weight=1)
        results_frame.grid_columnconfigure(1, weight=1)

    def update_results_file_labels(self):
        """Update labels in results tab with selected files"""
        if hasattr(self, 'results_stimulus_label') and self.stimulus_file_path:
            stim_file = os.path.basename(self.stimulus_file_path)
            self.results_stimulus_label.config(text=stim_file)
        
        if hasattr(self, 'results_edf_label') and self.edf_file_path:
            edf_file = os.path.basename(self.edf_file_path)
            self.results_edf_label.config(text=edf_file)

    def run_selected_analysis(self):
        """Run selected analysis type with files selected in Patient Info tab."""
        if not self.stimulus_file_path or not self.edf_file_path:
            messagebox.showwarning("Missing Files", "Please select both a stimulus file and an EDF file in the 'Patient Information' tab.")
            return

        analysis_type = self.analysis_type_combo.get()

        if analysis_type == "EDF Parse Only":  # 🔥 Add this new case
            self.run_edf_parse_only()
        elif analysis_type == "CMD Analysis":
            self.run_cmd_analysis()
        elif analysis_type == "Language Tracking":
            messagebox.showinfo("Not Implemented", "Language tracking is not yet implemented.")

    def run_edf_parse_only(self):
        """Parse the selected EDF file and display basic information."""
        try:
            # Create parser instance
            parser = EDFParser(self.edf_file_path)
            parser.load_edf()  # Load the file first
            
            # Get the summary info
            info = parser.get_info_summary()
            
            # Get channel types
            channel_types = parser.get_channel_types()
            
            # Format and display in the results text box
            result_text = "=== EDF FILE PARSED SUCCESSFULLY ===\n\n"
            result_text += f"Duration: {info['duration']:.2f} seconds\n"
            result_text += f"Sampling Rate: {info['sfreq']} Hz\n"
            result_text += f"Number of Channels: {info['n_channels']}\n"
            result_text += f"Number of Time Points: {info['n_times']:,}\n"
            result_text += f"Measurement Date: {info['meas_date']}\n"
            result_text += f"Number of Annotations: {info['annotations_count']}\n\n"
            
            result_text += "CHANNEL INFORMATION:\n"
            for i, (ch_name, ch_type) in enumerate(zip(info['ch_names'], channel_types)):
                result_text += f"  {i+1:2d}. {ch_name:<10} | Type: {ch_type}\n"
            
            # Clear and update display
            self.analysis_results_text.delete(1.0, tk.END)
            self.analysis_results_text.insert(tk.END, result_text)
            
        except Exception as e:
            error_msg = f"EDF Parse Error:\n{str(e)}"
            self.analysis_results_text.delete(1.0, tk.END)
            self.analysis_results_text.insert(tk.END, error_msg)

    def run_cmd_analysis(self):
        """Executes the CMD analysis workflow."""
        print("nothing to see here")

    def on_patient_id_change(self, event=None):
        patient_id = self.patient_id_entry.get().strip()
        if patient_id:
            self.current_patient = patient_id
            self.playback_state = "ready"
            self.status_label.config(text="Ready to prepare stimulus", foreground="green")
        else:
            self.current_patient = None
            self.playback_state = "empty"
            self.status_label.config(text="Please enter a patient ID", foreground="red")
        self.update_button_states()

    def toggle_loved_one_options(self):
        state = 'normal' if self.loved_one_var.get() else 'disabled'
        for widget in [self.male_radio, self.female_radio, self.file_button]:
            widget.config(state=state)

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

    def upload_voice_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Voice File",
            filetypes=[("Audio files", "*.wav *.mp3"), ("All files", "*.*")]
        )
        if file_path:
            self.audio_stim.trials.loved_one_file = file_path
            self.audio_stim.trials.loved_one_gender = self.gender_var.get()
            self.file_label.config(text=os.path.basename(file_path))

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

    def toggle_pause(self):
        if self.playback_state == "playing":
            self.playback_state = "paused"
            self.pause_button.config(image=self.play_sym)
            self.pause_button.config(text="Resume")
            self.status_label.config(text="Pausing stimulus...", foreground="red")
        elif self.playback_state == "paused":
            self.playback_state = "playing"
            self.pause_button.config(image=self.pause_sym)
            self.pause_button.config(text="Pause")
            self.status_label.config(text="Resuming stimulus...", foreground="blue")

        self.audio_stim.toggle_pause()
        self.update_button_states()
        self.update_trial_list_status()

    def stop_stimulus(self):
        """Stop the current stimulus playback"""
        if self.playback_state in ["playing", "paused"]:
            self.playback_state = "ready"
            self.pause_button.config(text="Pause", image=self.pause_sym)
            self.audio_stim.stop_stimulus()
            self.status_label.config(text="Stimulus stopped", foreground="orange")
            self.update_button_states()
            self.update_trial_list_status()

    def prepare_stimulus(self):
        if self.playback_state != "ready":
            return
        if not any([self.language_var.get(), self.right_cmd_var.get(), self.left_cmd_var.get(), 
                    self.oddball_var.get(), self.loved_one_var.get()]):
            messagebox.showwarning("No Stimulus Selected", "Please select at least one stimulus type.")
            return
        if self.loved_one_var.get() and not self.audio_stim.trials.loved_one_file:
            messagebox.showwarning("Missing File", "Please upload a voice file for loved one stimulus.")
            return
        self.playback_state = "preparing"
        self.update_button_states()
        self.status_label.config(text="Preparing stimulus...", foreground="blue")
        self.start_preparation()

    def start_preparation(self):
        """Start the preparation process"""
        try:
            num_of_each_trial = {
                "lang": 72 if self.language_var.get() else 0,
                "rcmd": 3 if self.right_cmd_var.get() and not self.rcmd_prompt_var.get() else 0,
                "rcmd+p": 3 if self.right_cmd_var.get() and self.rcmd_prompt_var.get() else 0,
                "lcmd": 3 if self.left_cmd_var.get() and not self.lcmd_prompt_var.get() else 0,
                "lcmd+p": 3 if self.left_cmd_var.get() and self.lcmd_prompt_var.get() else 0,
                "odd": 4 if self.oddball_var.get() and not self.oddball_prompt_var.get() else 0,
                "odd+p": 4 if self.oddball_var.get() and self.oddball_prompt_var.get() else 0,
                "loved": 50 if self.loved_one_var.get() else 0
            }
        
            # set gender for loved one
            if self.loved_one_var.get():
                self.audio_stim.trials.loved_one_gender = self.gender_var.get()         
            
            self.audio_stim.trials.generate_trials(num_of_each_trial)

            # Reset playback state
            self.playback_state = "ready"

            # update GUI
            self.update_button_states()
            self.status_label.config(text=f"Stimulus prepared! {len(self.trials.trial_dictionary)} trials ready.", foreground="green")
            self.populate_trial_list()
        
        except Exception as e:
            self.playback_state = "ready"
            self.update_button_states()
            self.status_label.config(text="Error preparing stimulus", foreground="red")
            messagebox.showerror("Error", f"Error preparing stimulus: {e}")

    def play_stimulus(self):
        if self.playback_state != "ready" or len(self.trials.trial_dictionary) == 0:
            return
        patient_id = self.patient_id_entry.get().strip()
        if not patient_id:
            messagebox.showwarning("No Patient ID", "Please enter a patient ID.")
            return
        self.playback_state = "playing"
        self.update_button_states()
        self.status_label.config(text="Playing stimulus...", foreground="blue")
        self.update_trial_list_status()
        # Start playback via auditory stimulator 
        self.audio_stim.play_trial_sequence()

    def add_note(self):
        """Add a note by appending a row to the stimulus results CSV in real time."""
        note = self.note_entry.get().strip()
        patient_id = self.get_patient_id()
        
        if not patient_id:
            messagebox.showwarning("No Patient ID", "Please enter a patient ID before adding a note.")
            return
        if not note:
            messagebox.showwarning("Empty Note", "Please enter a note before adding.")
            return

        # Use the same result directory and filename pattern as AuditoryStimulator
        results_dir = self.config.file.get('result_dir', 'patient_data/results')
        date_str = time.strftime("%Y-%m-%d")
        results_path = os.path.join(results_dir, f"{patient_id}_{date_str}_stimulus_results.csv")
        
        # Ensure directory exists
        os.makedirs(results_dir, exist_ok=True)

        # Create note row with same structure as trial rows
        note_row = {
            'patient_id': patient_id,
            'date': date_str,
            'trial_type': 'session_note',
            'sentences': '',
            'start_time': '',
            'end_time': '',
            'duration': '',
            'notes': note
        }

        try:
            # Append to CSV (header only if file doesn't exist)
            note_df = pd.DataFrame([note_row])
            note_df.to_csv(
                results_path,
                mode='a',
                header=not os.path.exists(results_path),
                index=False
            )
            
            # Display in UI
            self.notes_text.insert(tk.END, f"[{date_str}] {note}\n")
            self.notes_text.see(tk.END)
            self.note_entry.delete(0, tk.END)
            
        except Exception as e:
            messagebox.showerror("Note Save Error", f"Failed to save note:\n{str(e)}")
            
    # Placeholder methods for Results tab (keep as is or implement)
    def run_lang_analysis(self):
        """Run language analysis for the selected patient and date."""
        messagebox.showinfo("Language Analysis", "Language analysis is not yet implemented.")

    def populate_trial_list(self):
        """Populate the Treeview with all trials from trial_dictionary."""
        # Clear existing items
        for item in self.trial_tree.get_children():
            self.trial_tree.delete(item)

        # Map internal type to display name
        type_display_names = {
            "language": "Language",
            "right_command": "Right Command",
            "left_command": "Left Command",
            "oddball": "Oddball",
            "loved_one_voice": "Loved One Voice"
        }

        # Insert each trial
        for idx, trial in enumerate(self.trials.trial_dictionary):
            trial_type = trial['type']
            display_type = type_display_names.get(trial_type, trial_type.replace('_', ' ').title())
            status = trial['status'].title()  # "pending" → "Pending"
            self.trial_tree.insert('', 'end', iid=str(idx), values=(display_type, status))

    def update_trial_list_status(self):
        """Update the status column in the trial list from trial_dictionary."""
        type_display_names = {
            "language": "Language",
            "right_command": "Right Command",
            "left_command": "Left Command",
            "oddball": "Oddball",
            "loved_one_voice": "Loved One Voice"
        }

        for idx, trial in enumerate(self.trials.trial_dictionary):
            if str(idx) in self.trial_tree.get_children():
                display_type = type_display_names.get(trial['type'], trial['type'].replace('_', ' ').title())
                status = trial['status'].title()  # "pending" → "Pending"
                self.trial_tree.item(str(idx), values=(display_type, status))

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

    def button_styles(self):
        self.play_sym = tk.PhotoImage(file="lib/assets/play_sym.png").subsample(15, 15)
        self.pause_sym = tk.PhotoImage(file="lib/assets/pause_sym.png").subsample(15, 15)
        self.stop_sym = tk.PhotoImage(file="lib/assets/stop_sym3.png").subsample(6, 6)

