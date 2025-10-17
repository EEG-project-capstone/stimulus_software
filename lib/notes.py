# lib/notes.py

import os
import pandas as pd
import yaml
import os
import pandas as pd
import yaml
from tkinter import messagebox

def add_notes(patient_id="patient0", note="blank test note", recorded_date="00/00/0000"):
    """
    Adds a note for a patient to a CSV file if the
    patient has already been administered stimulus.

    Parameters:
    - patient_id (str): The identifier for the
    patient, such as a patient ID or EEG ID.
    - note (str): The note to be added for the patient.

    Returns:
    - None, but overwrites/updates the patient_notes.csv

    Note:
    The function checks if the patient has been administered stimulus by
    verifying if there exists a record for the patient in 'patient_df.csv'.
    If such a record exists, the note is appended to 'patient_notes.csv'. 
    If 'patient_df.csv' does not exist, it prints a message indicating
    that stimulus package hasn't been run for the patient yet.
    """

    # Load configuration
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)

    # Load or create notes DataFrame
    notes_path = config["patient_note_path"]
    if os.path.exists(notes_path):
        patient_notes = pd.read_csv(notes_path)
    else:
        patient_notes = pd.DataFrame(columns=['patient_id', 'notes', 'date'])

    # Check if patient_df exists — if not, abort
    df_path = config['patient_df_path']
    if not os.path.exists(df_path):
        print('patient_df.csv missing.')
        return  # Early exit
    
    # Now safe to load
    patient_df = pd.read_csv(df_path)

    # Create a DataFrame with the new note
    new_note = pd.DataFrame([{'patient_id': patient_id, 'date': recorded_date, 'notes': note}])
    # Concatenate the new note with the existing patient notes DataFrame
    patient_notes = pd.concat([patient_notes, new_note], ignore_index=True)
    # Save the updated DataFrame to CSV
    patient_notes.to_csv(notes_path, index=False)    



def load_notes(patient_id):
    """
    Load all notes for a given patient_id from patient_notes.csv
    
    Returns:
        List[str]: List of formatted note strings like "[YYYY-MM-DD] Note text"
    """

    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)

    notes_file = config["patient_note_path"]
    if not os.path.exists(notes_file):
        return []

    try:
        df = pd.read_csv(notes_file)
        patient_notes = df[df['patient_id'] == patient_id]
        result = []
        for _, row in patient_notes.iterrows():
            date = row.get('date', 'Unknown')
            note = row.get('notes', '')
            result.append(f"[{date}] {note}")
        return result
    except Exception as e:
        print(f"Error loading notes: {e}")
        return []