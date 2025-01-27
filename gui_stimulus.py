import streamlit as st
from auditory_stim.stimulus_package_notes import add_notes, add_history
from auditory_stim.auditory_stim import randomize_trials, generate_stimuli, play_stimuli
import pandas as pd
import os
import time
import yaml
import sys

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

# Check for test flag
test_run = '--test' in sys.argv
print(f"Test run: {test_run}")  

# Load configuration
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)


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
st.title("EEG Stimulus Package")

st.header("Administer Auditory Stimuli", divider='rainbow')

# Patient ID input
patient_id = st.text_input("Enter Patient/EEG ID")
trial_types = []
lang_trials_ids = []

selected_patient = st.selectbox("Select Patient ID", patient_df.patient_id.value_counts().index.sort_values())
selected_date = st.selectbox("Select Administered Date", patient_df[patient_df.patient_id == selected_patient].date.value_counts().index.sort_values())

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
            st.success("Your note was successfully added to patient_notes.csv and patient_history.csv")

st.header("Add Notes to your Selected Patient and Date", divider='rainbow')
your_note = st.text_input("Write your note here")

# Add Note button
if st.button("Add Note"):
    add_notes(selected_patient, your_note, selected_date)
    st.success("Your note was successfully added to patient_notes.csv")

st.header("Find Patient Notes", divider='rainbow')
st.subheader("The following notes have been written for the selected patient and date:")

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

st.header("Search Patients Already Administered Stimuli", divider='rainbow')

st.subheader("The following auditory stimuli were administered:")
for stimulus in patient_df[(patient_df.patient_id == selected_patient) & (patient_df.date == selected_date)].sentences.tolist():
    st.write(stimulus)

# st.subheader("Stimuli were administered in the following order:")
# for order in patient_df[(patient_df.patient_id == selected_patient) & (patient_df.date == selected_date)].order.value_counts().index.tolist():
#     st.write(order)

