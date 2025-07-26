# lib/config.py
import os
import yaml
import pandas as pd
import time

class Config:
    def __init__(self):
        with open('config.yml', 'r') as f:
            self.file = yaml.safe_load(f)
        self._initialize_data_structures()
        self._upload_data()
        self._output_data()

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

    def _output_data(self):
        if not os.path.exists(self.file['result_dir']):
            os.makedirs(self.file['result_dir'])
        if not os.path.exists(self.file['cmd_result_dir']):
            os.makedirs(self.file['cmd_result_dir'])
        if not os.path.exists(self.file['lang_tracking_dir']):
            os.makedirs(self.file['lang_tracking_dir'])
        self.graphs = ["", "CMD", "Language Tracking"]
        self.graph_options = list(range(len(self.graphs)))
        self.patient_ids = ["CON001a", "CON001b", "CON002", "CON003", "CON004", "CON005"]
