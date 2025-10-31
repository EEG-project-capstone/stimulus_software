# lib/config.py

import os
import yaml
import time

class Config:
    def __init__(self):
        with open('config.yml', 'r') as f:
            self.file = yaml.safe_load(f)
        
        # Only create directories you actually use
        os.makedirs(self.file.get('result_dir', 'patient_data/results'), exist_ok=True)
        os.makedirs(self.file.get('edf_dir', 'patient_data/edfs'), exist_ok=True)
        
        # Set up UI options (no legacy data loading)
        self.cpc_scale = [
            "",
            "CPC 1: No neurological deficit",
            "CPC 2: Mild to moderate dysfunction",
            "CPC 3: Severe dysfunction",
            "CPC 4: Coma",
            "CPC 5: Brain death",
        ]
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
        self.graphs = ["", "CMD", "Language Tracking"]
        self.current_date = time.strftime("%Y-%m-%d")