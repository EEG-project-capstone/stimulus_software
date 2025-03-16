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
from auditory_stim.stimulus_package_notes import add_notes, add_history
from auditory_stim.auditory_stim import randomize_trials, generate_stimuli, play_stimuli

from eeg_auditory_stimulus import rodika_modularized

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

graphs = ["", "Language Tracking", "CMD"]
graph_options = list(range(len(graphs)))

with tab3:
    st.header("EEG Graphs")

    patient_ids = patient_label_df['patient_id'].unique()
    selected_patient = st.selectbox("Choose Patient", patient_ids, index=None)
    date = st.date_input("Recording Date")
    date_str = date.strftime("%Y%m%d")
    fname = f"{selected_patient}_{date_str}"
    selected_graph = st.selectbox("Choose Graph Type", graph_options, format_func=lambda x: graphs[x])
    
    st.subheader("Graph Display")
    
    if selected_graph==2:
        # TODO: Add comments for analysis results
        fig_full_path = os.path.join(config['cmd_result_dir'], f"{fname}.png")
        if os.path.exists(fig_full_path):
            st.image(fig_full_path)
        else:
            pass
    elif selected_graph==1:
        expected_filename = "avg_itpc_plot.png"
        patient_folder = os.path.join(config['lang_tracking_dir'], selected_patient)
        image_path = os.path.join(config['lang_tracking_dir'], expected_filename)
        
        # Button to run analysis and generate plots
        if st.button("Run Language Tracking Analysis"):
            # You can call your main() function from rodika_modularized here.
            # It should process the data and save the plots in the appropriate folder.
            # Make sure to pass in the necessary parameters.
            eeg_file_path = os.path.join(config['edf_dir'], f"{selected_patient}_{date_str}.edf")
            stimulus_csv_path = config['patient_df_path']
            use_channels = ['C3','C4','O1','O2','FT9','FT10','Cz','F3','F4','F7','F8',
                    'Fz','Fp1','Fp2','Fpz','P3','P4','Pz','T7','T8','P7','P8']
            bad_channels = ['T7', 'Fp1', 'Fp2']
            eog_chs = ['Fp1', 'Fp2', 'T7']
            
            rodika_modularized.main(eeg_file_path, stimulus_csv_path, selected_patient, use_channels, bad_channels, eog_chs)
            st.success("Analysis complete! The ITPC graphs have been generated.")

        st.subheader("Graph Display")
        if os.path.exists(image_path):
            st.image(image_path, caption=f"Language Tracking ITPC for {selected_patient} on {date_str}")
        else:
            st.warning(f"No ITPC graph found for {selected_patient} on {date_str}. Run the analysis first.")

            