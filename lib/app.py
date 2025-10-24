# lib/app.py

import datetime
import os
import time
import pandas as pd
import tkinter as tk
import sounddevice as sd
from tkinter import messagebox, ttk, filedialog
from lib.config import Config
from lib.trials import Trials
from lib.auditory_stimulator import AuditoryStimulator
from lib.notes import add_notes

class TkApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EEG Stimulus Package")
        self.root.geometry("775x830")
        
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
        return self.patient_id_entry.get().strip()

    def playback_complete(self):
        """Handle completion of stimulus playback"""
        patient_id = self.patient_id_entry.get().strip()
        self.playback_state = "ready"
        self.audio_stim.is_paused = False # Ensure sync
        self.pause_button.config(text="Pause")
        self.update_button_states()
        self.status_label.config(text=f"Stimulus completed for {patient_id}", foreground="green")
        messagebox.showinfo("Success", f"Stimulus administered successfully to {patient_id}")

    def playback_error(self, error_msg):
        """Handle playback errors"""
        self.playback_state = "ready"
        self.audio_stim.is_paused = False # Ensure sync
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

        # Trial List Frame (with scrollbar)
        trial_list_frame = ttk.LabelFrame(main_frame, text="Trial Sequence")
        trial_list_frame.pack(fill='x', pady=5)

        # Treeview with scrollbar
        self.trial_tree = ttk.Treeview(trial_list_frame, columns=('Type', 'Status'), show='headings', height=8)
        self.trial_tree.heading('Type', text='Trial Type')
        self.trial_tree.heading('Status', text='Status')
        self.trial_tree.column('Type', width=150)
        self.trial_tree.column('Status', width=100)
        self.trial_tree.tag_configure('pending', foreground='gray')
        self.trial_tree.tag_configure('inprogress', foreground='blue')
        self.trial_tree.tag_configure('completed', foreground='green')

        # Scrollbar for trial list only
        tree_scroll = ttk.Scrollbar(trial_list_frame, orient="vertical", command=self.trial_tree.yview)
        tree_scroll.pack(side='right', fill='y')
        self.trial_tree.configure(yscrollcommand=tree_scroll.set)
        self.trial_tree.pack(side='left', fill='x', expand=True, padx=(5, 0), pady=5)

        # Notes section
        notes_frame = ttk.LabelFrame(main_frame, text="Notes", padding="10")
        notes_frame.pack(fill='both', expand=True, pady=5)
        ttk.Label(notes_frame, text="Add Note:").grid(row=0, column=0, sticky='w')
        self.note_entry = ttk.Entry(notes_frame, width=50)
        self.note_entry.grid(row=0, column=1, sticky='ew', padx=5)
        ttk.Button(notes_frame, text="Add Note", command=self.add_note).grid(row=0, column=2, padx=5)
        ttk.Label(notes_frame, text="Find Notes:").grid(row=1, column=0, sticky='w', pady=(10,0))
        self.notes_text = tk.Text(notes_frame, height=12, width=60)  # Increased height from 5 to 8
        self.notes_text.grid(row=2, column=0, columnspan=3, sticky='ew', pady=5)
        notes_frame.grid_columnconfigure(1, weight=1)
        notes_frame.grid_rowconfigure(2, weight=1)  # Allow the text widget to expand vertically

        # Update button states
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
            self.load_patient_notes(patient_id)
        else:
            self.current_patient = None
            self.playback_state = "empty"
            self.status_label.config(text="Please enter a patient ID", foreground="red")
        self.update_button_states()

    def load_patient_notes(self, patient_id):
        """Load and display saved notes for the patient."""
        self.notes_text.delete(1.0, tk.END)
        try:
            from lib.notes import load_notes
            notes = load_notes(patient_id)
            for note in notes:
                self.notes_text.insert(tk.END, note + "\n")
            if notes:
                self.notes_text.see(tk.END)
        except Exception as e:
            print(f"Could not load notes: {e}")

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
            self.pause_button.config(text="Pause")
            sd.stop()
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
        self.config.current_date = time.strftime("%Y-%m-%d")
        self.update_button_states()
        self.status_label.config(text="Playing stimulus...", foreground="blue")
        self.update_trial_list_status()
        # Start playback via auditory stimulator 
        self.audio_stim.play_trial_sequence()

    def add_note(self):
        """Add a note to the notes section."""
        note = self.note_entry.get().strip()
        patient_id = self.get_patient_id()
        if not patient_id:
            messagebox.showwarning("No Patient ID", "Please enter a patient ID before adding a note.")
            return
        if not note:
            messagebox.showwarning("Empty Note", "Please enter a note before adding.")
            return
        
        try:
            # Save the note persistently
            current_date = time.strftime("%Y-%m-%d")
            add_notes(patient_id=patient_id, note=note, recorded_date=current_date)
            
            # Display in UI
            self.notes_text.insert(tk.END, f"[{current_date}] {note}\n")
            self.notes_text.see(tk.END)  # Scroll to bottom
            self.note_entry.delete(0, tk.END)
            
        except Exception as e:
            messagebox.showerror("Note Error", f"Failed to save note: {e}")

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