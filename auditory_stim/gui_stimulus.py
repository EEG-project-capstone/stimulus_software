import streamlit as st
from auditory_stim import generate_and_play_stimuli
from stimulus_package_notes import add_notes
import pandas as pd
import os
import time

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


# Load patient data
if os.path.exists('data/patient_df.csv'):
    patient_df = pd.read_csv('data/patient_df.csv')
else:
    patient_df = pd.DataFrame(columns=['patient_id', 'date', 'trial_type',
                                'sentences', 'start_time', 'duration', 'order'])
    patient_df.to_csv("data/patient_df.csv")
current_date = time.strftime("%Y-%m-%d")

### Streamlit Interface ###

# Streamlit app title
st.title("EEG Stimulus Package")

st.header("Administer Auditory Stimuli", divider='rainbow')

# Patient ID input
patient_id = st.text_input("Enter Patient/EEG ID")

def start_stimulus(input_patient_id):
    """
    Administers the auditory stimuli to the patient.
    
    Parameters:
    input_patient_id (str): The ID of the patient.
    """
    if patient_id.strip() == "":
        st.error("Please enter a patient ID.")
    elif ((patient_df['patient_id'] == patient_id) & (patient_df['date'] == current_date)).any():
        st.error("Patient has already been administered stimulus protocol today")
    else:
        # Create placeholders for the messages
        #administering_placeholder = st.empty()
        running_placeholder = st.empty()

        # Change the screen to "Administering Stimulus"
        #administering_placeholder.write("Administering Stimulus...")
        running_placeholder.write("Stimulus is running...")  # Placeholder for actual stimulus running

        # Generate and play sentences
        generate_and_play_stimuli(input_patient_id)

        # Clear the previous messages
        #administering_placeholder.empty()
        running_placeholder.empty()

        st.experimental_rerun()
        # Show success message
        st.success("Stimulus protocol successfully administered and data saved to patient_df.csv.")


# Start button
if st.button("Start Stimulus"):
    start_stimulus(patient_id)


st.header("Search Patients Already Administered Stimuli", divider='rainbow')

# Add searchable dropdown menu of patient IDs
selected_patient = st.selectbox("Select Patient ID", patient_df.patient_id.value_counts().index.sort_values())
selected_date = st.selectbox("Select Administered Date", patient_df[patient_df.patient_id == selected_patient].date.value_counts().index.sort_values())

st.subheader("The following auditory stimuli were administered:")
for stimulus in patient_df[(patient_df.patient_id == selected_patient) & (patient_df.date == selected_date)].sentences.tolist():
    st.write(stimulus)

st.subheader("Stimuli were administered in the following order:")
for order in patient_df[(patient_df.patient_id == selected_patient) & (patient_df.date == selected_date)].order.value_counts().index.tolist():
    st.write(order)

st.header("Add Notes to your Selected Patient and Date", divider='rainbow')
your_note = st.text_input("Write your note here")

# Add Note button
if st.button("Add Note"):
    add_notes(selected_patient, your_note, selected_date)
    st.success("Your note was successfully added to patient_notes.csv")

st.header("Find Patient Notes", divider='rainbow')
st.subheader("The following notes have been written for the selected patient and date:")

if not os.path.exists("patient_notes.csv"):
    st.error("You haven't added any notes yet, add a note first.")
else:
    patient_notes = pd.read_csv("patient_notes.csv")
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
    for note in patient_notes[(patient_notes.patient_id == selected_patient_find_notes) & (patient_notes.date == selected_date_find_notes)].notes.tolist():
        st.write(note)
