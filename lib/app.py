# gui/app.py
import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import datetime
import os
import pandas as pd
import time
import sys
from lib.config import Config
from lib.auditory_stimulator import AuditoryStimulator
from lib.trial_manager import TrialManager
from lib.stimulus_package_notes import add_notes, add_history

class TkApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EEG Stimulus Package")
        self.root.geometry("800x600")
        self.config = Config()
        self.audio_stim = AuditoryStimulator()
        # Instantiate TrialManager
        self.trial_manager = TrialManager(self.audio_stim, self) # Pass self as callback
        self.playback_state = "empty"
        # self.is_paused = False # Moved to TrialManager
        self.trial_types = []
        self.current_patient = None
        self.language_var = tk.BooleanVar()
        self.right_cmd_var = tk.BooleanVar()
        self.left_cmd_var = tk.BooleanVar()
        self.oddball_var = tk.BooleanVar()
        self.loved_one_var = tk.BooleanVar()
        self.gender_var = tk.StringVar(value="Male")
        # Variables for non-blocking operations
        # self.current_trial_index = 0 # Moved to TrialManager
        # self.administered_stimuli = [] # Moved to TrialManager
        # self.preparation_progress = 0 # Kept locally for preparation
        # self.preparation_total = 0 # Kept locally for preparation
        self.build_main_ui()

    # --- Callback methods for TrialManager ---
    def get_playback_state(self):
        return self.playback_state

    def get_patient_id(self):
        return self.patient_id_entry.get().strip()

    def root_after(self, ms, func):
        self.root.after(ms, func)

    def update_progress(self, progress):
        self.progress_var.set(progress)

    def playback_complete(self):
        """Handle completion of stimulus playback"""
        patient_id = self.patient_id_entry.get().strip()
        self.playback_state = "ready"
        # self.is_paused = False # Managed by TrialManager
        self.trial_manager.is_paused = False # Ensure sync
        self.pause_button.config(text="Pause")
        self.update_button_states()
        self.progress_var.set(100)
        # Save results using data from TrialManager
        self.save_results(patient_id, self.trial_manager.administered_stimuli)
        self.status_label.config(text=f"Stimulus completed for {patient_id}", foreground="green")
        messagebox.showinfo("Success", f"Stimulus administered successfully to {patient_id}")
        # Reset progress
        self.root.after(2000, lambda: self.progress_var.set(0))

    def playback_error(self, error_msg):
        """Handle playback errors"""
        self.playback_state = "ready"
        # self.is_paused = False # Managed by TrialManager
        self.trial_manager.is_paused = False # Ensure sync
        # Add more error handling UI updates if needed
        self.update_button_states()
        messagebox.showerror("Playback Error", f"Error during playback: {error_msg}")

    def browse_edf_file(self):
        """Open a file dialog to select an EDF file and update the label."""
        file_path = filedialog.askopenfilename(
            title="Select EDF File",
            filetypes=[("EDF files", "*.edf"), ("All files", "*.*")]
        )
        if file_path:
            self.edf_file_label.config(text=os.path.basename(file_path))
            self.selected_edf_file = file_path
        else:
            self.edf_file_label.config(text="No file selected")
            self.selected_edf_file = None

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
        canvas = tk.Canvas(self.tab1)
        scrollbar = ttk.Scrollbar(self.tab1, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        # Patient ID Section
        patient_frame = ttk.LabelFrame(scrollable_frame, text="Patient Information", padding="10")
        patient_frame.pack(fill='x', pady=5)
        ttk.Label(patient_frame, text="Patient/EEG ID:").grid(row=0, column=0, sticky='w', padx=5)
        self.patient_id_entry = ttk.Entry(patient_frame, width=30)
        self.patient_id_entry.grid(row=0, column=1, sticky='ew', padx=5)
        self.patient_id_entry.bind('<KeyRelease>', self.on_patient_id_change)
        self.status_label = ttk.Label(patient_frame, text="Please enter a patient ID", foreground="red")
        self.status_label.grid(row=1, column=0, columnspan=2, pady=5)
        # Stimulus Selection
        stim_frame = ttk.LabelFrame(scrollable_frame, text="Stimulus Selection", padding="10")
        stim_frame.pack(fill='x', pady=5)
        ttk.Checkbutton(stim_frame, text="Language Stimulus", variable=self.language_var).grid(row=0, column=0, sticky='w')
        ttk.Checkbutton(stim_frame, text="Right Command Stimulus", variable=self.right_cmd_var).grid(row=1, column=0, sticky='w')
        ttk.Checkbutton(stim_frame, text="Left Command Stimulus", variable=self.left_cmd_var).grid(row=2, column=0, sticky='w')
        ttk.Checkbutton(stim_frame, text="Oddball Stimulus", variable=self.oddball_var).grid(row=4, column=0, sticky='w')
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
        control_frame = ttk.Frame(scrollable_frame)
        control_frame.pack(fill='x', pady=10)
        self.prepare_button = ttk.Button(control_frame, text="Prepare Stimulus", command=self.prepare_stimulus)
        self.prepare_button.pack(side='left', padx=5)
        self.play_button = ttk.Button(control_frame, text="Play Stimulus", command=self.play_stimulus)
        self.play_button.pack(side='left', padx=5)
        self.pause_button = ttk.Button(control_frame, text="Pause", command=self.toggle_pause)
        self.pause_button.pack(side='left', padx=5)
        self.stop_button = ttk.Button(control_frame, text="Stop", command=self.stop_stimulus)
        self.stop_button.pack(side='left', padx=5)
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(scrollable_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill='x', pady=5)
        # Notes section
        notes_frame = ttk.LabelFrame(scrollable_frame, text="Notes", padding="10")
        notes_frame.pack(fill='both', expand=True, pady=5)
        ttk.Label(notes_frame, text="Add Note:").grid(row=0, column=0, sticky='w')
        self.note_entry = ttk.Entry(notes_frame, width=50)
        self.note_entry.grid(row=0, column=1, sticky='ew', padx=5)
        ttk.Button(notes_frame, text="Add Note", command=self.add_note).grid(row=0, column=2, padx=5)
        ttk.Label(notes_frame, text="Find Notes:").grid(row=1, column=0, sticky='w', pady=(10,0))
        self.notes_text = tk.Text(notes_frame, height=5, width=60)
        self.notes_text.grid(row=2, column=0, columnspan=3, sticky='ew', pady=5)
        notes_frame.grid_columnconfigure(1, weight=1)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.update_button_states()

    def build_patient_info_tab(self):
        info_frame = ttk.LabelFrame(self.tab2, text="Upload EEG Files", padding="20")
        info_frame.pack(fill='x', pady=10)
        ttk.Label(info_frame, text="Patient ID:").grid(row=0, column=0, sticky='w', pady=5)
        self.info_patient_id = ttk.Entry(info_frame, width=30)
        self.info_patient_id.grid(row=0, column=1, sticky='ew', padx=5, pady=5)
        ttk.Label(info_frame, text="Recording Date:").grid(row=1, column=0, sticky='w', pady=5)
        self.date_entry = ttk.Entry(info_frame, width=30)
        self.date_entry.grid(row=1, column=1, sticky='ew', padx=5, pady=5)
        self.date_entry.insert(0, datetime.date.today().strftime("%Y-%m-%d"))
        ttk.Label(info_frame, text="EDF File:").grid(row=2, column=0, sticky='w', pady=5)
        file_frame = ttk.Frame(info_frame)
        file_frame.grid(row=2, column=1, sticky='ew', padx=5, pady=5)
        self.edf_file_label = ttk.Label(file_frame, text="No file selected")
        self.edf_file_label.pack(side='left')
        ttk.Button(file_frame, text="Browse", command=self.browse_edf_file).pack(side='right')
        ttk.Label(info_frame, text="CPC Score:").grid(row=3, column=0, sticky='w', pady=5)
        self.cpc_combo = ttk.Combobox(info_frame, values=self.config.cpc_scale, state="readonly")
        self.cpc_combo.grid(row=3, column=1, sticky='ew', padx=5, pady=5)
        ttk.Label(info_frame, text="GOSE Score:").grid(row=4, column=0, sticky='w', pady=5)
        self.gose_combo = ttk.Combobox(info_frame, values=self.config.gose_scale, state="readonly")
        self.gose_combo.grid(row=4, column=1, sticky='ew', padx=5, pady=5)
        ttk.Button(info_frame, text="Submit", command=self.submit_patient_info).grid(row=5, column=0, columnspan=2, pady=20)
        info_frame.grid_columnconfigure(1, weight=1)

    def build_results_tab(self):
        results_frame = ttk.LabelFrame(self.tab3, text="EEG Analysis", padding="20")
        results_frame.pack(fill='both', expand=True, pady=10)
        ttk.Label(results_frame, text="Select Patient:").grid(row=0, column=0, sticky='w', pady=5)
        self.results_patient_combo = ttk.Combobox(results_frame, state="readonly")
        self.results_patient_combo.grid(row=0, column=1, sticky='ew', padx=5, pady=5)
        self.results_patient_combo.bind('<<ComboboxSelected>>', self.update_dates_combo)
        ttk.Label(results_frame, text="Select Date:").grid(row=1, column=0, sticky='w', pady=5)
        self.results_date_combo = ttk.Combobox(results_frame, state="readonly")
        self.results_date_combo.grid(row=1, column=1, sticky='ew', padx=5, pady=5)
        ttk.Label(results_frame, text="Graph Type:").grid(row=2, column=0, sticky='w', pady=5)
        self.graph_type_combo = ttk.Combobox(results_frame, values=self.config.graphs, state="readonly")
        self.graph_type_combo.grid(row=2, column=1, sticky='ew', padx=5, pady=5)
        self.graph_type_combo.bind('<<ComboboxSelected>>', self.on_graph_type_change)
        self.lang_options_frame = ttk.LabelFrame(results_frame, text="Language Tracking Options", padding="10")
        ttk.Label(self.lang_options_frame, text="Bad Channels:").grid(row=0, column=0, sticky='w', pady=5)
        self.bad_channels_entry = ttk.Entry(self.lang_options_frame, width=40)
        self.bad_channels_entry.grid(row=0, column=1, sticky='ew', padx=5, pady=5)
        self.bad_channels_entry.insert(0, "T7,Fp1,Fp2")
        ttk.Label(self.lang_options_frame, text="EOG Channels:").grid(row=1, column=0, sticky='w', pady=5)
        self.eog_channels_entry = ttk.Entry(self.lang_options_frame, width=40)
        self.eog_channels_entry.grid(row=1, column=1, sticky='ew', padx=5, pady=5)
        self.eog_channels_entry.insert(0, "Fp1,Fp2,T7")
        ttk.Label(self.lang_options_frame, text="Display:").grid(row=2, column=0, sticky='w', pady=5)
        self.display_option_combo = ttk.Combobox(self.lang_options_frame, values=["Average ITPC", "Individual Channel"], state="readonly")
        self.display_option_combo.grid(row=2, column=1, sticky='ew', padx=5, pady=5)
        self.display_option_combo.bind('<<ComboboxSelected>>', self.on_display_option_change)
        self.channel_label = ttk.Label(self.lang_options_frame, text="Channel:")
        self.channel_combo = ttk.Combobox(self.lang_options_frame, values=['C3','C4','O1','O2','FT9','FT10','Cz','F3','F4','F7','F8','Fz','Fp1','Fp2','Fpz','P3','P4','Pz','T7','T8','P7','P8'], state="readonly")
        button_frame = ttk.Frame(results_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=20)
        self.cmd_button = ttk.Button(button_frame, text="Run CMD Analysis", command=self.run_cmd_analysis)
        self.cmd_button.pack(side='left', padx=5)
        self.lang_button = ttk.Button(button_frame, text="Run Language Analysis", command=self.run_lang_analysis)
        self.lang_button.pack(side='left', padx=5)
        self.image_frame = ttk.Frame(results_frame)
        self.image_frame.grid(row=5, column=0, columnspan=2, sticky='nsew', pady=10)
        self.image_canvas = tk.Canvas(self.image_frame, bg='white')
        self.image_v_scrollbar = ttk.Scrollbar(self.image_frame, orient="vertical", command=self.image_canvas.yview)
        self.image_h_scrollbar = ttk.Scrollbar(self.image_frame, orient="horizontal", command=self.image_canvas.xview)
        self.image_canvas.configure(yscrollcommand=self.image_v_scrollbar.set, xscrollcommand=self.image_h_scrollbar.set)
        self.image_canvas.grid(row=0, column=0, sticky='nsew')
        self.image_v_scrollbar.grid(row=0, column=1, sticky='ns')
        self.image_h_scrollbar.grid(row=1, column=0, sticky='ew')
        self.image_frame.grid_rowconfigure(0, weight=1)
        self.image_frame.grid_columnconfigure(0, weight=1)
        results_frame.grid_columnconfigure(1, weight=1)
        results_frame.grid_rowconfigure(5, weight=1)
        self.update_patients_combo()

    def update_patients_combo(self):
        """Update the patient combo box with available patient IDs."""
        # Load patient IDs from the patient label CSV file
        label_path = self.config.file.get('patient_label_path', 'patient_labels.csv')
        if os.path.exists(label_path):
            df = pd.read_csv(label_path)
            patient_ids = df['patient_id'].unique().tolist()
        else:
            patient_ids = []
        self.results_patient_combo['values'] = patient_ids
        if patient_ids:
            self.results_patient_combo.set(patient_ids[0])
        else:
            self.results_patient_combo.set('')

    def update_dates_combo(self, event=None):
        """Update the date combo box based on selected patient."""
        selected_patient = self.results_patient_combo.get()
        label_path = self.config.file.get('patient_label_path', 'patient_labels.csv')
        if os.path.exists(label_path) and selected_patient:
            df = pd.read_csv(label_path)
            dates = df[df['patient_id'] == selected_patient]['date'].unique().tolist()
        else:
            dates = []
        self.results_date_combo['values'] = dates
        if dates:
            self.results_date_combo.set(dates[0])
        else:
            self.results_date_combo.set('')

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

    def upload_voice_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Voice File",
            filetypes=[("Audio files", "*.wav *.mp3"), ("All files", "*.*")]
        )
        if file_path:
            self.audio_stim.loved_one_file = file_path
            self.audio_stim.loved_one_gender = self.gender_var.get()
            self.file_label.config(text=os.path.basename(file_path))

    def update_button_states(self):
        if self.playback_state == "empty":
            self.prepare_button.config(state='disabled')
            self.play_button.config(state='disabled')
            self.pause_button.config(state='disabled')
            self.stop_button.config(state='disabled')
        elif self.playback_state == "ready":
            self.prepare_button.config(state='normal')
            self.play_button.config(state='normal' if self.trial_types else 'disabled')
            self.pause_button.config(state='disabled')
            self.stop_button.config(state='disabled')
        elif self.playback_state == "preparing":
            self.prepare_button.config(state='disabled')
            self.play_button.config(state='disabled')
            self.pause_button.config(state='disabled')
            self.stop_button.config(state='disabled')
        elif self.playback_state == "paused":
            self.prepare_button.config(state='disabled')
            self.play_button.config(state='disabled')
            self.pause_button.config(state='normal')
            self.stop_button.config(state='normal')
        elif self.playback_state == "playing":
            self.prepare_button.config(state='disabled')
            self.play_button.config(state='disabled')
            self.pause_button.config(state='normal')
            self.stop_button.config(state='normal')

    def toggle_pause(self):
        if self.playback_state == "playing":
            self.playback_state = "paused"
            # self.is_paused = True # Managed by TrialManager
        elif self.playback_state == "paused":
            self.playback_state = "playing"
            # self.is_paused = False # Managed by TrialManager
            self.pause_button.config(text="Pause")
        # Delegate pause logic to TrialManager
        self.trial_manager.toggle_pause()
        self.update_button_states()

    def stop_stimulus(self):
        """Stop the current stimulus playback"""
        if self.playback_state in ["playing", "paused"]:
            self.playback_state = "ready"
            # self.is_paused = False # Managed by TrialManager
            self.pause_button.config(text="Pause")
            self.progress_var.set(0)
            # self.current_trial_index = 0 # Managed by TrialManager
            # self.administered_stimuli = [] # Managed by TrialManager
            # Stop any playing audio (TrialManager also does this)
            import sounddevice as sd
            sd.stop()
            # Delegate stop logic to TrialManager
            self.trial_manager.stop_stimulus()
            self.status_label.config(text="Stimulus stopped", foreground="orange")
            self.update_button_states()

    def prepare_stimulus(self):
        if self.playback_state != "ready":
            return
        if not any([self.language_var.get(), self.right_cmd_var.get(), self.left_cmd_var.get(), 
                    self.oddball_var.get(), self.loved_one_var.get()]):
            messagebox.showwarning("No Stimulus Selected", "Please select at least one stimulus type.")
            return
        if self.loved_one_var.get() and not hasattr(self.audio_stim, 'loved_one_file'):
            messagebox.showwarning("Missing File", "Please upload a voice file for loved one stimulus.")
            return
        self.playback_state = "preparing"
        self.update_button_states()
        self.status_label.config(text="Preparing stimulus...", foreground="blue")
        # Start preparation process
        self.root.after(10, self.start_preparation)

    def start_preparation(self):
        """Start the non-blocking preparation process"""
        try:
            num_of_each_trial = {
                "lang": 72 if self.language_var.get() else 0,
                "rcmd": 3 if self.right_cmd_var.get() else 0,
                "lcmd": 3 if self.left_cmd_var.get() else 0,
                "odd": 4 if self.oddball_var.get() else 0,
                "loved": 50 if self.loved_one_var.get() else 0
            }
            if self.loved_one_var.get():
                self.audio_stim.loved_one_gender = self.gender_var.get()
            # Calculate total preparation steps for progress
            self.preparation_total = num_of_each_trial["lang"]
            self.preparation_progress = 0
            # Start the preparation - this will be done synchronously but with progress updates
            self.trial_types = self.audio_stim.generate_trials(num_of_each_trial)
            # Preparation complete
            self.preparation_complete()
        except Exception as e:
            self.preparation_error(str(e))

    def preparation_complete(self):
        self.playback_state = "ready"
        self.update_button_states()
        self.progress_var.set(0)  # Reset progress bar
        self.status_label.config(text=f"Stimulus prepared! {len(self.trial_types)} trials ready.", foreground="green")
        messagebox.showinfo("Success", f"Stimulus prepared successfully!\n{len(self.trial_types)} trials ready.")

    def preparation_error(self, error_msg):
        self.playback_state = "ready"
        self.update_button_states()
        self.progress_var.set(0)
        self.status_label.config(text="Error preparing stimulus", foreground="red")
        messagebox.showerror("Error", f"Error preparing stimulus: {error_msg}")

    def play_stimulus(self):
        if self.playback_state != "ready" or not self.trial_types:
            return
        patient_id = self.patient_id_entry.get().strip()
        if not patient_id:
            messagebox.showwarning("No Patient ID", "Please enter a patient ID.")
            return
        # Initialize playback variables via TrialManager
        self.playback_state = "playing"
        # self.current_trial_index = 0 # Managed by TrialManager
        # self.administered_stimuli = [] # Managed by TrialManager
        # self.is_paused = False # Managed by TrialManager
        self.trial_manager.reset_trial_state() # Reset TrialManager state
        self.config.current_date = time.strftime("%Y-%m-%d")
        self.update_button_states()
        self.status_label.config(text="Playing stimulus...", foreground="blue")
        # Start playback via TrialManager
        self.trial_manager.play_trial_sequence(self.trial_types)

    def save_results(self, patient_id, administered_stimuli):
        """Save administered stimuli results to a CSV file."""
        # import pandas as pd # Assume imported globally or handled
        # import os # Assume imported globally or handled
        results_dir = self.config.file.get('result_dir', '.')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        results_path = os.path.join(results_dir, f"{patient_id}_{self.config.current_date}_stimulus_results.csv")
        df = pd.DataFrame(administered_stimuli)
        df.to_csv(results_path, index=False)

    def add_note(self):
        """Add a note to the notes section."""
        note = self.note_entry.get().strip()
        if note:
            self.notes_text.insert(tk.END, f"{note}\n")
            self.note_entry.delete(0, tk.END)
        else:
            messagebox.showwarning("Empty Note", "Please enter a note before adding.")

    # Placeholder methods for Results tab (keep as is or implement)
    def run_lang_analysis(self):
        """Run language analysis for the selected patient and date."""
        messagebox.showinfo("Language Analysis", "Language analysis is not yet implemented.")

    def run_cmd_analysis(self):
        """Run CMD analysis for the selected patient and date."""
        messagebox.showinfo("CMD Analysis", "CMD analysis is not yet implemented.")

    def on_display_option_change(self, event=None):
        """Handle changes in display option selection for Language Tracking."""
        selected_option = self.display_option_combo.get()
        if selected_option == "Individual Channel":
            self.channel_label.grid(row=3, column=0, sticky='w', pady=5)
            self.channel_combo.grid(row=3, column=1, sticky='ew', padx=5, pady=5)
        else:
            self.channel_label.grid_remove()
            self.channel_combo.grid_remove()

    def on_graph_type_change(self, event=None):
        """Handle changes in graph type selection."""
        selected_graph = self.graph_type_combo.get()
        if selected_graph == "Language Tracking":
            self.lang_options_frame.grid(row=3, column=0, columnspan=2, sticky='ew', pady=10)
        else:
            self.lang_options_frame.grid_remove()

    def submit_patient_info(self):
        """Handle submission of patient information from the Patient Information tab."""
        patient_id = self.info_patient_id.get().strip()
        recording_date = self.date_entry.get().strip()
        edf_file = getattr(self, 'selected_edf_file', None)
        cpc_score = self.cpc_combo.get()
        gose_score = self.gose_combo.get()
        if not patient_id:
            messagebox.showwarning("Missing Patient ID", "Please enter a patient ID.")
            return
        if not recording_date:
            messagebox.showwarning("Missing Date", "Please enter a recording date.")
            return
        if not edf_file:
            messagebox.showwarning("Missing EDF File", "Please select an EDF file.")
            return
        if not cpc_score or cpc_score == "":
            messagebox.showwarning("Missing CPC Score", "Please select a CPC score.")
            return
        if not gose_score or gose_score == "":
            messagebox.showwarning("Missing GOSE Score", "Please select a GOSE score.")
            return
        # Save patient info to CSV
        label_path = self.config.file.get('patient_label_path', 'patient_labels.csv')
        # import pandas as pd # Assume imported globally or handled
        # import os # Assume imported globally or handled
        if os.path.exists(label_path):
            df = pd.read_csv(label_path)
        else:
            df = pd.DataFrame(columns=['patient_id', 'date', 'cpc', 'gose'])
        new_row = {
            'patient_id': patient_id,
            'date': recording_date,
            'cpc': cpc_score,
            'gose': gose_score
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(label_path, index=False)
        messagebox.showinfo("Success", "Patient information submitted successfully.")
