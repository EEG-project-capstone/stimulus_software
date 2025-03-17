"""
GUI Stimulus Package

This script provides a graphical user interface (GUI) for administering auditory stimuli to patients, managing patient records, and adding notes. It utilizes the Streamlit library to create a web-based interface. The main functionalities include:

1. **Administer Auditory Stimuli:** Users can input patient/EEG ID and start administering auditory stimuli. If a patient has already been administered the stimulus protocol on the current date, an error message is displayed.

2. **Search Patients Already Administered Stimuli:** Enables users to search for patients who have already been administered stimuli. Users can select a patient ID and date to view the administered stimuli and their order.

3. **Add Notes to Your Selected Patient and Date:** Provides a text input field for users to add notes to a selected patient and date. Users can click the "Add Note" button to append the note to the patient's record.

4. **Find Patient Notes:** Allows users to find notes written for a selected patient and date. Users can select a patient ID and date to view the notes.

Output:
The script generates and saves data to 'patient_df.csv' and 'patient_notes.csv' files.
"""

import pandas as pd
import os
import time
import yaml
import sys
import streamlit as st
from PIL import Image
from auditory_stim.stimulus_package_notes import add_notes, add_history
from auditory_stim.auditory_stim import randomize_trials, generate_stimuli, play_stimuli
from eeg_auditory_stimulus import rodika_modularized
from eeg_auditory_stimulus import claassen_analysis

# Check for test flag
test_run = '--test' in sys.argv
print(f"Test run: {test_run}")  

# Load configuration
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

# Check if the directory exists
if not os.path.exists(config['patient_output_dir']):
    os.makedirs(config['patient_output_dir'])
if not os.path.exists(config['stimuli_dir']):
    os.makedirs(config['stimuli_dir'])
if not os.path.exists(config['sentences_path']):
    raise FileNotFoundError(f"Sentences directory not found at {config['sentences_path']}")

# Load patient data
if os.path.exists(config['patient_df_path']):
    patient_df = pd.read_csv(config['patient_df_path'])
else:
    patient_df = pd.DataFrame(columns=['patient_id', 'date', 'trial_type',
                                'sentences', 'start_time', 'end_time', 'duration'])
    patient_df.to_csv(config['patient_df_path'])
current_date = time.strftime("%Y-%m-%d")

### Display plots & log function ###
def display_all_plots(patient_folder):
    log_file_path = os.path.join(patient_folder, 'log.txt')

    # Get log details
    auc_score, permutation_results = read_log_file(log_file_path)

    # Define the order of plots
    plot_order = [
        'epochs_during_instructions.png',
        'cross_validation_performance.png',
        'average_predicted_probability.png',
        'EEG_spatial_patterns.png',
        'permutation_test_distribution.png',
        'permutation_results.png'
    ]

    # Filter only existing plots in the folder
    plot_files = [f for f in plot_order if f in os.listdir(patient_folder)]

    # Display each plot
    for plot_file in plot_files:
        st.subheader(f"{plot_file.replace('_', ' ').replace('.png', '').capitalize()}")
        plot_path = os.path.join(patient_folder, plot_file)
        image = Image.open(plot_path)
        st.image(image, use_container_width=True)

        # Show AUC score below average_predicted_probability.png
        if plot_file == 'average_predicted_probability.png' and auc_score:
            st.text(auc_score.strip())

        # Show permutation results below permutation_distribution.png
        if plot_file == 'permutation_distribution.png' and permutation_results:
            for line in permutation_results:
                st.text(line.strip())

def read_log_file(log_file_path):
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r') as f:
            logs = f.readlines()

        # Extract relevant logs
        auc_score = next((line for line in logs if 'Mean scores across split' in line), None)
        permutation_results = [line for line in logs if 'Permutation' in line or 'AUC' in line]

        return auc_score, permutation_results
    return None, None

### Streamlit Interface ###

# Streamlit app title
tab1, tab2, tab3 = st.tabs(["Administer Stimuli", "Patient Information", "Results"])

with tab1:
    st.title("EEG Stimulus Package")

    st.header("Administer Auditory Stimuli", divider='rainbow')

    # Patient ID input
    patient_id = st.text_input("Enter Patient/EEG ID")
    trial_types = []
    lang_trials_ids = []

    if st.button("Prepare Stimulus"):
        trial_types = randomize_trials()
        st.session_state['trial_types'] = trial_types
        lang_trials_ids = []
        lang_trials_ids = generate_stimuli(trial_types)
        st.session_state['lang_trials_ids'] = lang_trials_ids

    if st.button("Play Stimulus"):
        print(f"patient_id: {patient_id}")
        current_date = time.strftime("%Y-%m-%d")

        if os.listdir(config['stimuli_dir']) == []:
            st.error("Please prepare stimuli first.")
        else:
            if patient_id.strip() == "":
                st.error("Please enter a patient ID.")
            elif ((patient_df['patient_id'] == patient_id) & (patient_df['date'] == current_date)).any():
                st.error("Patient has already been administered stimulus protocol today")
            else:
                progress_bar = st.progress(0, text="0")
                if not trial_types and 'trial_types' in st.session_state:
                    trial_types = st.session_state['trial_types']
                else:
                    st.error("Please prepare stimuli first.")
                if not lang_trials_ids and 'lang_trials_ids' in st.session_state:
                    lang_trials_ids = st.session_state['lang_trials_ids']
                else:
                    st.error("Please prepare stimuli first.")

                print(f"trial_types: {trial_types}")
                n = len(trial_types)
                administered_stimuli = []
                for i in range(n):
                    trial = trial_types[i]
                    print(f"Trial {i}: {trial}")
                    start_time, end_time = play_stimuli(trial, test_run)
                    administered_stimuli.append({
                                'patient_id': patient_id,
                                'date': current_date,
                                'trial_type': trial[:4] if trial[:4] == "lang" else trial,
                                'sentences': lang_trials_ids[i],
                                'start_time': start_time,
                                'end_time': end_time,
                                'duration': end_time - start_time
                            })
                    percent = int(i/n*100)
                    progress_bar.progress(percent, text=f"{percent}%")
                progress_bar.progress(100, text=f"Done")

                pd.DataFrame(administered_stimuli)
                administered_stimuli_df = pd.concat([patient_df, pd.DataFrame(administered_stimuli)], ignore_index=True)
                administered_stimuli_df.to_csv(config['patient_df_path'], index=False)
                print(f"administered_stimuli_df: {administered_stimuli_df}")

                # Save each patient output into seaprated csv files with 'patientId_currentDate'
                output_dir = config['patient_output_dir']
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                formatted_date = current_date.replace("-", "")
                output_file = f"{patient_id}_{formatted_date}.csv"
                output_path = os.path.join(output_dir, output_file)
                pd.DataFrame(administered_stimuli).to_csv(output_path, index=False)
                print(f"Data saved to {output_path}")

                # Add history after saving csv output files
                add_history(patient_id, current_date)
                st.success(f"Stimuli has been administered to patient {patient_id} on {current_date}.")

    st.header("Add Notes to your Selected Patient and Date", divider='rainbow')
    your_note = st.text_input("Write your note here")

    # Add Note button
    if st.button("Add Note"):
        selected_date = time.strftime("%Y-%m-%d")
        add_notes(patient_id, your_note, selected_date)
        st.success("Your note was successfully added to patient_notes.csv")

    st.header("Find Patient Notes", divider='rainbow')
    st.subheader("The following notes have been written for the selected patient and date:")

    selected_patient_find_notes = None
    selected_date_find_notes = None
    if not os.path.exists(config["patient_note_path"]):
        st.error("You haven't added any notes yet, add a note first.")
    else:
        patient_notes = pd.read_csv(config["patient_note_path"])
        selected_patient_find_notes = st.selectbox(
            "Select Patient ID", 
            patient_notes.patient_id.value_counts().index.sort_values(), 
            key="widget_key_for_find_patient_notes"
        )
        selected_date_find_notes = st.selectbox(
            "Select Administered Date", 
            patient_notes[patient_notes.patient_id == selected_patient_find_notes].date.value_counts().index.sort_values(), 
            key="widget_key_for_find_date_notes"
        )
        for note in patient_notes[(patient_notes['patient_id'] == selected_patient_find_notes) & (patient_notes['date'] == selected_date_find_notes)]['notes'].tolist():
            st.write(note)

# Tab 2: Upload Data
# Check if the directory exists
if not os.path.exists(config['edf_dir']):
    os.makedirs(config['edf_dir'])
if os.path.exists(config['patient_label_path']):
    patient_label_df = pd.read_csv(config['patient_label_path'])
else:
    patient_label_df = pd.DataFrame(columns=['patient_id', 'date', 'cpc', 'gose'])
    patient_label_df.to_csv(config['patient_label_path'], index=False)

# CPC Scale
cpc_scale = [
    "",
    "CPC 1: No neurological deficit",
    "CPC 2: Mild to moderate dysfunction",
    "CPC 3: Severe dysfunction",
    "CPC 4: Coma",
    "CPC 5: Brain death",
]
cpc_options = list(range(len(cpc_scale)))

# GOSE Scale
gose_scale = [
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
gose_options = list(range(len(gose_scale)))
    
with tab2:
    st.header("Upload EEG Files")

    with st.form("my_form"):
        patient_id = st.text_input("Patient ID", placeholder="Enter Patient ID")
        date = st.date_input("Recording Date")
        date_str = date.strftime("%Y%m%d")
        edf_file = st.file_uploader("Upload EDF File", type=["edf"])
        cpc_input = st.selectbox("Select CPC Score", cpc_options, format_func=lambda x: cpc_scale[x])
        gose_input = st.selectbox("Select GOSE Score", gose_options, format_func=lambda x: gose_scale[x])
        submitted = st.form_submit_button("Submit")
        if submitted:
            # Check no duplicates      
            full_path = os.path.join(config['edf_dir'], f"{patient_id}_{date_str}.edf")
            if os.path.exists(full_path):
                st.error(f"File already exists for {patient_id} on {date_str}")

            elif not os.path.exists(full_path) and edf_file:
                # Save the file
                with open(full_path, 'wb') as f: 
                    f.write(edf_file.getvalue())
                st.success(f"Uploaded {edf_file.name} for {patient_id}")

                # Save the label
                patient_label_row = pd.DataFrame([{
                    'patient_id': patient_id,
                    'date': date_str,
                    'cpc': cpc_input if cpc_input > 0 else None,
                    'gose': gose_input if gose_input > 0 else None
                }])
                patient_label_row.to_csv(config['patient_label_path'], mode='a', header=False, index=False)
                patient_label_df = pd.read_csv(config['patient_label_path'])

# Tab 3: EEG Graphs
# Check if the directory exists
if not os.path.exists(config['result_dir']):
    os.makedirs(config['result_dir'])
if not os.path.exists(config['cmd_result_dir']):
    os.makedirs(config['cmd_result_dir'])
if not os.path.exists(config['lang_tracking_dir']):
    os.makedirs(config['lang_tracking_dir'])

graphs = ["", "CMD", "Language Tracking"]
graph_options = list(range(len(graphs)))
patient_ids = ["CON001a", "CON001b", "CON002", "CON003", "CON004", "CON005"]

with tab3:
    st.header("EEG Graphs")
    selected_patient = st.selectbox(
        "Select Patient ID", 
        patient_df['patient_id'].sort_values().unique())
    selected_date_find_patient = st.selectbox(
        "Select Administered Date", 
        patient_df[patient_df['patient_id'] == selected_patient]['date'].unique())
    date_str = pd.to_datetime(selected_date_find_patient).strftime("%Y%m%d")
    fname = f"{selected_patient}_{date_str}"
    selected_graph = st.selectbox("Choose Graph Type", graph_options, format_func=lambda x: graphs[x])
    
    st.subheader("Graph Display")

    if selected_graph==1: # CMD
        fig_full_path = os.path.join(config['cmd_result_dir'], f"{fname}.png")
        patient_folder = os.path.join(config['cmd_result_dir'], f"{selected_patient}_{date_str}")

        if os.path.exists(patient_folder): # create folder under selected_patient, under date 
            display_all_plots(patient_folder)
        else:
            # Create the directory if it doesn't exist
            os.makedirs(patient_folder, exist_ok=True)
        
        if st.button("Run Analysis"):
            claassen_analysis.run_analysis(selected_patient, config['cmd_result_dir'], config['edf_dir'], config['patient_df_path'], date_str)
        display_all_plots(patient_folder)

    elif selected_graph==2:
        # TODO: Add comments for analysis results
        fig_full_path = os.path.join(config['lang_tracking_dir'], f"{fname}.png")
        if os.path.exists(fig_full_path):
            st.image(fig_full_path)
        else:
            st.warning(f"No ITPC graph found for {selected_patient} on {date_str}. Run the analysis first.")

            