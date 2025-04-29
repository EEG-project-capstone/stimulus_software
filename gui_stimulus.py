# NEW FULLY CORRECTED gui_stimulus.py

import pandas as pd
import os
import time
import yaml
import sys
import streamlit as st
import random
from PIL import Image

from auditory_stim.stimulus_package_notes import add_notes, add_history
from auditory_stim.auditory_stim import (
    randomize_trials, generate_stimuli, play_stimuli,
    randomize_loved_one_trials, generate_loved_one_stimuli
)
from eeg_auditory_stimulus import rodika_modularized
from eeg_auditory_stimulus import claassen_analysis

# --- Config loading ---
test_run = '--test' in sys.argv
try:
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
except Exception as e:
    st.error(f"Config file error: {e}")
    st.stop()

required_dirs = ['patient_output_dir', 'stimuli_dir', 'edf_dir', 'result_dir', 'cmd_result_dir', 'lang_tracking_dir']
for dir_key in required_dirs:
    os.makedirs(config.get(dir_key, f'data/{dir_key}/'), exist_ok=True)

# --- Load or initialize patient dataframe ---
patient_df_path = config.get('patient_df_path', 'data/patient_df.csv')
if os.path.exists(patient_df_path):
    patient_df = pd.read_csv(patient_df_path)
else:
    patient_df = pd.DataFrame(columns=['patient_id', 'date', 'trial_index', 'paradigm', 'trial_type', 'stimulus_details', 'start_time', 'end_time', 'duration'])
    patient_df.to_csv(patient_df_path, index=False)

current_date = time.strftime("%Y-%m-%d")

# --- Streamlit Interface ---
tab1, tab2, tab3 = st.tabs(["Administer Stimuli", "Patient Information", "Results"])

with tab1:
    st.title("EEG Stimulus Package")
    patient_id = st.text_input("Enter Patient/EEG ID")
    st.subheader("Select Stimulus Paradigms to Prepare")

    loved_one_stim_selected = st.checkbox("Loved One / Control Stimulus Paradigm")
    uploaded_loved_one_file = st.file_uploader("Upload Loved One's Voice Recording", type=['wav', 'mp3'], disabled=not loved_one_stim_selected)
    family_member_gender = st.radio("Select Family Member Gender", ('Male', 'Female'), horizontal=True)

    language_stim_selected = st.checkbox("language_stim")
    right_cmd_stim_selected = st.checkbox("right_cmd_stim")
    left_cmd_stim_selected = st.checkbox("left_cmd_stim")
    beep_stim_selected = st.checkbox("beep_stim")
    oddball_stim_selected = st.checkbox("oddball_stim")

    if st.button("Prepare Stimulus"):
        st.session_state['stimulus_details'] = []
        all_prepared_details = []

        # --- Loved One/Control preparation ---
        if loved_one_stim_selected:
            if not family_member_gender or uploaded_loved_one_file is None:
                st.error("Upload loved one file and select gender.")
                st.stop()

            temp_dir = os.path.join(config['stimuli_dir'], "temp_upload")
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, uploaded_loved_one_file.name)
            with open(temp_path, 'wb') as f:
                f.write(uploaded_loved_one_file.getvalue())

            trial_labels = randomize_loved_one_trials()
            loved_one_details = generate_loved_one_stimuli(trial_labels, family_member_gender, config, temp_path)
            all_prepared_details.extend(loved_one_details)

        # --- Standard paradigms preparation ---
        if any([language_stim_selected, right_cmd_stim_selected, left_cmd_stim_selected, beep_stim_selected, oddball_stim_selected]):
            num_of_each = {
                "lang": 72 if language_stim_selected else 0,
                "rcmd": 3 if right_cmd_stim_selected else 0,
                "lcmd": 3 if left_cmd_stim_selected else 0,
                "beep": 6 if beep_stim_selected else 0,
                "odd": 40 if oddball_stim_selected else 0
            }
            trial_labels = randomize_trials(num_of_each)
            standard_details = generate_stimuli(trial_labels, config)
            all_prepared_details.extend(standard_details)

        # --- Finalize stimuli ---
        random.shuffle(all_prepared_details)
        for i, d in enumerate(all_prepared_details):
            d['index'] = i
        st.session_state['stimulus_details'] = all_prepared_details
        st.success(f"Prepared {len(all_prepared_details)} trials.")

    if st.button("Play Stimulus"):
        if 'stimulus_details' not in st.session_state or not st.session_state['stimulus_details']:
            st.error("No prepared stimuli. Prepare first.")
            st.stop()
        if patient_id.strip() == "":
            st.error("Enter Patient ID.")
            st.stop()

        progress_bar = st.progress(0)
        log_entries = []
        for i, stim in enumerate(st.session_state['stimulus_details']):
            trial_label = stim.get('type', 'unknown')
            start_time, end_time = play_stimuli(stim, test_run)
            log_entries.append({
                'patient_id': patient_id,
                'date': current_date,
                'trial_index': i,
                'paradigm': 'mixed',
                'trial_type': trial_label,
                'stimulus_details': stim.get('source_info', 'N/A'),
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time
            })
            progress_bar.progress((i+1)/len(st.session_state['stimulus_details']))

        new_df = pd.DataFrame(log_entries)
        patient_df_updated = pd.concat([patient_df, new_df], ignore_index=True)
        patient_df_updated.to_csv(patient_df_path, index=False)
        st.success("All stimuli played and logged.")

with tab2:
    st.header("Upload EEG Files")
    with st.form("upload_form"):
        patient_id = st.text_input("Patient ID")
        date = st.date_input("Recording Date")
        edf_file = st.file_uploader("Upload EDF", type=["edf"])
        cpc = st.selectbox("CPC Score", ["CPC 1", "CPC 2", "CPC 3", "CPC 4", "CPC 5"])
        gose = st.selectbox("GOSE Score", ["GOSE 1", "GOSE 2", "GOSE 3", "GOSE 4", "GOSE 5", "GOSE 6", "GOSE 7", "GOSE 8"])
        submitted = st.form_submit_button("Submit")
        if submitted and edf_file:
            save_dir = config.get('edf_dir', 'data/edfs/')
            save_path = os.path.join(save_dir, f"{patient_id}_{date.strftime('%Y%m%d')}.edf")
            with open(save_path, 'wb') as f:
                f.write(edf_file.getvalue())
            st.success(f"Saved {save_path}")

with tab3:
    st.header("Results Viewer")
    if not patient_df.empty:
        selected_patient = st.selectbox("Select Patient", patient_df['patient_id'].dropna().unique())
        dates = patient_df[patient_df['patient_id']==selected_patient]['date'].dropna().unique()
        selected_date = st.selectbox("Select Date", dates)
        filtered = patient_df[(patient_df['patient_id']==selected_patient) & (patient_df['date']==selected_date)]
        st.dataframe(filtered)
        if 'trial_type' in filtered.columns:
            counts = filtered['trial_type'].value_counts()
            st.bar_chart(counts)
            st.line_chart(filtered['duration'].fillna(0))
