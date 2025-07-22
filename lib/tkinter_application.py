import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import threading
import datetime
import os
import yaml
import pandas as pd
import time
import sys

from lib.auditory_stimulator import AuditoryStimulator
from lib.stimulus_package_notes import add_notes, add_history

class Config:
    def __init__(self):
        self.test_run = '--test' in sys.argv
        print(f"Test run: {self.test_run}")  
        with open('config.yml', 'r') as f:
            self.file = yaml.safe_load(f)
        self._initialize_data_structures()
        self._upload_data()
        self._ouput_data()
    
    def _initialize_data_structures(self):
        if not os.path.exists(self.file['patient_output_dir']):
            os.makedirs(self.file['patient_output_dir'])
        if not os.path.exists(self.file['stimuli_dir']):
            os.makedirs(self.file['stimuli_dir'])
        if not os.path.exists(self.file['sentences_path']):
            raise FileNotFoundError(f"Sentences directory not found at {self.file['sentences_path']}")
        if os.path.exists(self.file['patient_df_path']):
            self.patient_df = pd.read_csv(self.file['patient_df_path'])
        else:
            self.patient_df = pd.DataFrame(columns=['patient_id', 'date', 'trial_type',
                                        'sentences', 'start_time', 'end_time', 'duration'])
            self.patient_df.to_csv(self.file['patient_df_path'])
        self.current_date = time.strftime("%Y-%m-%d")

    def _upload_data(self):
        if not os.path.exists(self.file['edf_dir']):
            os.makedirs(self.file['edf_dir'])
        if os.path.exists(self.file['patient_label_path']):
            patient_label_df = pd.read_csv(self.file['patient_label_path'])
        else:
            patient_label_df = pd.DataFrame(columns=['patient_id', 'date', 'cpc', 'gose'])
            patient_label_df.to_csv(self.file['patient_label_path'], index=False)
        self.cpc_scale = [
            "",
            "CPC 1: No neurological deficit",
            "CPC 2: Mild to moderate dysfunction",
            "CPC 3: Severe dysfunction",
            "CPC 4: Coma",
            "CPC 5: Brain death",
        ]
        self.cpc_options = list(range(len(self.cpc_scale)))
        self.gose_scale = [
            "",
            "GOSE 1: Dead",
            "GOSE 2: Vegetative state",
            "GOSE 3: Lower severe disability",
            "GOSE 4: Upper severe disability",
            "GOSE 5: Lower moderate disability",
            "GOSE 6: Upper moderate disability",
            "GOSE 7: Lower good recovery",
            "GOSE 8: Upper good recovery",
        ]
        self.gose_options = list(range(len(self.gose_scale)))

    def _ouput_data(self):
        if not os.path.exists(self.file['result_dir']):
            os.makedirs(self.file['result_dir'])
        if not os.path.exists(self.file['cmd_result_dir']):
            os.makedirs(self.file['cmd_result_dir'])
        if not os.path.exists(self.file['lang_tracking_dir']):
            os.makedirs(self.file['lang_tracking_dir'])
        self.graphs = ["", "CMD", "Language Tracking"]
        self.graph_options = list(range(len(self.graphs)))
        self.patient_ids = ["CON001a", "CON001b", "CON002", "CON003", "CON004", "CON005"]

class TkApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EEG Stimulus Package")
        self.root.geometry("800x600")
        self.config = Config()
        self.audio_stim = AuditoryStimulator()
        self.playback_state = "empty"
        self.is_paused = False
        self.trial_types = []
        self.current_patient = None
        self.language_var = tk.BooleanVar()
        self.right_cmd_var = tk.BooleanVar()
        self.left_cmd_var = tk.BooleanVar()
        self.beep_var = tk.BooleanVar()
        self.oddball_var = tk.BooleanVar()
        self.loved_one_var = tk.BooleanVar()
        self.gender_var = tk.StringVar(value="Male")
        self.build_main_ui()

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
        ttk.Checkbutton(stim_frame, text="Beep Stimulus", variable=self.beep_var).grid(row=3, column=0, sticky='w')
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
        elif self.playback_state == "ready":
            self.prepare_button.config(state='normal')
            self.play_button.config(state='normal' if self.trial_types else 'disabled')
        elif self.playback_state == "preparing":
            self.prepare_button.config(state='disabled')
            self.play_button.config(state='disabled')
        elif self.playback_state == "playing":
            self.prepare_button.config(state='disabled')
            self.play_button.config(state='disabled')

    def prepare_stimulus(self):
        if self.playback_state != "ready":
            return
        if not any([self.language_var.get(), self.right_cmd_var.get(), self.left_cmd_var.get(), 
                   self.beep_var.get(), self.oddball_var.get(), self.loved_one_var.get()]):
            messagebox.showwarning("No Stimulus Selected", "Please select at least one stimulus type.")
            return
        if self.loved_one_var.get() and not hasattr(self.audio_stim, 'loved_one_file'):
            messagebox.showwarning("Missing File", "Please upload a voice file for loved one stimulus.")
            return
        self.playback_state = "preparing"
        self.update_button_states()
        threading.Thread(target=self._prepare_stimulus_thread, daemon=True).start()

    def _prepare_stimulus_thread(self):
        try:
            num_of_each_trial = {
                "lang": 72 if self.language_var.get() else 0,
                "rcmd": 3 if self.right_cmd_var.get() else 0,
                "lcmd": 3 if self.left_cmd_var.get() else 0,
                "beep": 6 if self.beep_var.get() else 0,
                "odd": 4 if self.oddball_var.get() else 0,
                "loved": 50 if self.loved_one_var.get() else 0
            }
            if self.loved_one_var.get():
                self.audio_stim.loved_one_gender = self.gender_var.get()
            self.trial_types = self.audio_stim.generate_trials(num_of_each_trial)
            self.root.after(0, self._preparation_complete)
        except Exception as e:
            self.root.after(0, lambda: self._preparation_error(str(e)))

    def _preparation_complete(self):
        self.playback_state = "ready"
        self.update_button_states()
        self.status_label.config(text=f"Stimulus prepared! {len(self.trial_types)} trials ready.", foreground="green")
        messagebox.showinfo("Success", f"Stimulus prepared successfully!\n{len(self.trial_types)} trials ready.")

    def _preparation_error(self, error_msg):
        self.playback_state = "ready"
        self.update_button_states()
        self.status_label.config(text="Error preparing stimulus", foreground="red")
        messagebox.showerror("Error", f"Error preparing stimulus: {error_msg}")

    def play_stimulus(self):
        if self.playback_state != "ready" or not self.trial_types:
            return
        patient_id = self.patient_id_entry.get().strip()
        if not patient_id:
            messagebox.showwarning("No Patient ID", "Please enter a patient ID.")
            return
        self.playback_state = "playing"
        self.update_button_states()
        threading.Thread(target=self._play_stimulus_thread, args=(patient_id,), daemon=True).start()

    def _play_stimulus_thread(self, patient_id):
        try:
            self.config.current_date = time.strftime("%Y-%m-%d")
            administered_stimuli = []
            for i, trial in enumerate(self.trial_types):
                progress = int((i / len(self.trial_types)) * 100)
                self.root.after(0, lambda p=progress: self.progress_var.set(p))
                start_time, end_time, sentences = self.audio_stim.play_stimuli(trial)
                administered_stimuli.append({
                    'patient_id': patient_id,
                    'date': self.config.current_date,
                    'trial_type': trial[:4] if trial[:4] == "lang" else trial,
                    'sentences': sentences,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time
                })
            self.root.after(0, lambda: self.progress_var.set(100))
            self.root.after(0, lambda: self._playback_complete(patient_id, administered_stimuli))
        except Exception as e:
            self.root.after(0, lambda: self._playback_error(str(e)))

    def _playback_complete(self, patient_id, administered_stimuli):
        self.playback_state = "ready"
        self.update_button_states()
        self.progress_var.set(0)
        self.save_results(patient_id, administered_stimuli)
        self.status_label.config(text=f"Stimulus completed for {patient_id}", foreground="green")
        messagebox.showinfo("Success", f"Stimulus administered successfully to {patient_id}")

    def _playback_error(self, error_msg):
        self.playback_state = "ready"
        self.update_button_states()
        self.progress_var.set(0)
        self.status_label.config(text="Error during playback", foreground="red")
        messagebox.showerror("Error", f"Error during playback: {error_msg}")

    def save_results(self, patient_id, administered_stimuli):
        new_df = pd.DataFrame(administered_stimuli)
        updated_df = pd.concat([self.config.patient_df, new_df], ignore_index=True)
        updated_df.to_csv(self.config.file['patient_df_path'], index=False)
        output_dir = self.config.file['patient_output_dir']
        os.makedirs(output_dir, exist_ok=True)
        formatted_date = self.config.current_date.replace("-", "")
        output_file = f"{patient_id}_{formatted_date}.csv"
        output_path = os.path.join(output_dir, output_file)
        new_df.to_csv(output_path, index=False)
        add_history(patient_id, self.config.current_date)
        self.config.patient_df = updated_df

    def add_note(self):
        patient_id = self.patient_id_entry.get().strip()
        note = self.note_entry.get().strip()
        if not patient_id or not note:
            messagebox.showwarning("Missing Information", "Please enter a patient ID and note content.")
            return
        try:
            add_notes(patient_id, note, self.config.current_date)
            messagebox.showinfo("Success", "Note added successfully!")
            self.note_entry.delete(0, tk.END)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add note: {str(e)}")

    def browse_edf_file(self):
        file_path = filedialog.askopenfilename(
            title="Select EDF File",
            filetypes=[("EDF files", "*.edf"), ("All files", "*.*")]
        )
        if file_path:
            self.edf_file_label.config(text=os.path.basename(file_path))

    def submit_patient_info(self):
        patient_id = self.info_patient_id.get().strip()
        date = self.date_entry.get().strip()
        edf_file = self.edf_file_label.cget("text")
        cpc_score = self.cpc_combo.current()
        gose_score = self.gose_combo.current()
        
        if not patient_id or not date or edf_file == "No file selected":
            messagebox.showwarning("Missing Information", "Please fill all required fields.")
            return
            
        # Implementation for saving patient info would go here
        messagebox.showinfo("Success", "Patient information submitted successfully!")

    def update_patients_combo(self):
        self.results_patient_combo['values'] = self.config.patient_ids

    def update_dates_combo(self, event=None):
        # This would be implemented to show dates for selected patient
        self.results_date_combo['values'] = ["2023-01-01", "2023-01-02"]  # Example data

    def on_graph_type_change(self, event=None):
        # Show/hide language options based on selection
        pass

    def on_display_option_change(self, event=None):
        # Show/hide channel selection based on display option
        pass

    def run_cmd_analysis(self):
        messagebox.showinfo("Info", "CMD analysis would run here")

    def run_lang_analysis(self):
        messagebox.showinfo("Info", "Language analysis would run here")