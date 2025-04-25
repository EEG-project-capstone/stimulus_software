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
from auditory_stim.auditory_stim import send_trigger, randomize_trials, generate_stimuli, play_stimuli
from eeg_auditory_stimulus import rodika_modularized
from eeg_auditory_stimulus import claassen_analysis

def main():
    """Main function to run the EEG Stimulus Package GUI."""
    # Initialize Streamlit tabs
    tab1, tab2, tab3 = st.tabs(["Administer Stimuli", "Patient Information", "Results"])
    
    with tab1:
        handle_stimulus_administration()
    
    with tab2:
        handle_patient_information()
    
    with tab3:
        handle_eeg_results() 

class Config:

    test_run: bool

    file: any

    current_date: any
    patient_df: any

    cpc_scale: any
    cpc_options: list[int]
    gose_scale: any
    gose_options: list[int]

    graph_options: list[int]
    patient_ids: list[str]

    def __init__(self):

        # Check for test flag
        self.test_run = '--test' in sys.argv
        print(f"Test run: {self.test_run}")  

        # Load configuration first
        with open('config.yml', 'r') as f:
            self.file = yaml.safe_load(f)
        
        # Initialize data structures
        self._initialize_data_structures()

        # Upload EEG Data
        self._upload_data()

        # prepare output files
        self._ouput_data()
    

    def _initialize_data_structures(self):
        # Check if the directory exists
        if not os.path.exists(self.file['patient_output_dir']):
            os.makedirs(self.file['patient_output_dir'])
        if not os.path.exists(self.file['stimuli_dir']):
            os.makedirs(self.file['stimuli_dir'])
        if not os.path.exists(self.file['sentences_path']):
            raise FileNotFoundError(f"Sentences directory not found at {self.file['sentences_path']}")

        # Load patient data
        if os.path.exists(self.file['patient_df_path']):
            self.patient_df = pd.read_csv(self.file['patient_df_path'])
        else:
            self.patient_df = pd.DataFrame(columns=['patient_id', 'date', 'trial_type',
                                        'sentences', 'start_time', 'end_time', 'duration'])
            self.patient_df.to_csv(self.file['patient_df_path'])
        self.current_date = time.strftime("%Y-%m-%d")

    def _upload_data(self):
        # for Tab 2: Upload Data
        # Check if the directory exists
        if not os.path.exists(self.file['edf_dir']):
            os.makedirs(self.file['edf_dir'])
        if os.path.exists(self.file['patient_label_path']):
            patient_label_df = pd.read_csv(self.file['patient_label_path'])
        else:
            patient_label_df = pd.DataFrame(columns=['patient_id', 'date', 'cpc', 'gose'])
            patient_label_df.to_csv(self.file['patient_label_path'], index=False)

        # CPC Scale
        self.cpc_scale = [
            "",
            "CPC 1: No neurological deficit",
            "CPC 2: Mild to moderate dysfunction",
            "CPC 3: Severe dysfunction",
            "CPC 4: Coma",
            "CPC 5: Brain death",
        ]

        self.cpc_options = list(range(len(self.cpc_scale)))

        # GOSE Scale
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
        # for Tab 3: EEG Graphs
        # Check if the directory exists
        if not os.path.exists(self.file['result_dir']):
            os.makedirs(self.file['result_dir'])
        if not os.path.exists(self.file['cmd_result_dir']):
            os.makedirs(self.file['cmd_result_dir'])
        if not os.path.exists(self.file['lang_tracking_dir']):
            os.makedirs(self.file['lang_tracking_dir'])

        graphs = ["", "CMD", "Language Tracking"]
        self.graph_options = list(range(len(graphs)))
        self.patient_ids = ["CON001a", "CON001b", "CON002", "CON003", "CON004", "CON005"]

### Streamlit Interface ###
def handle_stimulus_administration():
    st.title("EEG Stimulus Package")
    st.header("Administer Auditory Stimuli", divider='rainbow')

    # Patient ID input
    patient_id = st.text_input("Enter Patient/EEG ID")
    trial_types = []
    lang_trials_ids = []
 
    language_stim_selcted = st.checkbox("language_stim", key="language_checkbox")
    right_cmd_stim_selcted = st.checkbox("right_cmd_stim", key="right_cmd_checkbox")
    left_cmd_stim_selcted = st.checkbox("left_cmd_stim", key="rleft_cmd_checkbox")
    beep_stim_selcted = st.checkbox("beep_stim", key="beep_checkbox")
    oddball_stim_selected = st.checkbox("oddball_stim", key="oddball_checkbox")

    num_of_each_trial = {
        "lang": 0,
        "rcmd": 0,
        "lcmd": 0,
        "beep": 0,
        "odd": 0
    }


    # Initialize stop flag in session state
    if 'stop_playback' not in st.session_state:
        st.session_state.stop_playback = False

    # Create Prepare Stimulus button
    if st.button("Prepare Stimulus"):

        # Validate selection (only one can be selected)
        if language_stim_selcted:
            num_of_each_trial["lang"] = 72
        if right_cmd_stim_selcted:
            num_of_each_trial["rcmd"] = 3 
        if left_cmd_stim_selcted:
            num_of_each_trial["lcmd"] = 3 
        if beep_stim_selcted:
            num_of_each_trial["beep"] = 6    
        if oddball_stim_selected:
            num_of_each_trial["odd"] = 4

        # Original random stimulus
        trial_types = randomize_trials(num_of_each_trial)
        st.session_state['trial_types'] = trial_types
        lang_trials_ids = generate_stimuli(trial_types)
        st.session_state['lang_trials_ids'] = lang_trials_ids

    # Create columns for Play and Stop buttons
    col1, col2 = st.columns(2)

    with col1:
        play_btn = st.button("Play Stimulus")

    with col2:
        stop_btn = st.button("Stop Playback", type="primary")
    
    # play
    if play_btn:

        # send_trigger()

        print(f"patient_id: {patient_id}")
        config.current_date = time.strftime("%Y-%m-%d")

        if os.listdir(config.file['stimuli_dir']) == []:
            st.error("Please prepare stimuli first.")
        else:
            if patient_id.strip() == "":
                st.error("Please enter a patient ID.")
            elif ((config.patient_df['patient_id'] == patient_id) & (config.patient_df['date'] == config.current_date)).any():
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
                    start_time, end_time = play_stimuli(trial, config.test_run)
                    administered_stimuli.append({
                                'patient_id': patient_id,
                                'date': config.current_date,
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
                administered_stimuli_df = pd.concat([config.patient_df, pd.DataFrame(administered_stimuli)], ignore_index=True)
                administered_stimuli_df.to_csv(config.file['patient_df_path'], index=False)
                print(f"administered_stimuli_df: {administered_stimuli_df}")

                # Save each patient output into seaprated csv files with 'patientId_currentDate'
                output_dir = config.file['patient_output_dir']
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                formatted_date = config.current_date.replace("-", "")
                output_file = f"{patient_id}_{formatted_date}.csv"
                output_path = os.path.join(output_dir, output_file)
                pd.DataFrame(administered_stimuli).to_csv(output_path, index=False)
                print(f"Data saved to {output_path}")

                # Add history after saving csv output files
                add_history(patient_id, config.current_date)
                st.success(f"Stimuli has been administered to patient {patient_id} on {config.current_date}.")

    # stop
    if stop_btn:
        st.session_state.stop_playback = True
        st.warning("Stopping playback after current stimulus...")

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
    if not os.path.exists(config.file["patient_note_path"]):
        st.error("You haven't added any notes yet, add a note first.")
    else:
        patient_notes = pd.read_csv(config.file["patient_note_path"])
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

def handle_patient_information():
    st.header("Upload EEG Files")

    with st.form("my_form"):
        patient_id = st.text_input("Patient ID", placeholder="Enter Patient ID")
        date = st.date_input("Recording Date")
        date_str = date.strftime("%Y%m%d")
        edf_file = st.file_uploader("Upload EDF File", type=["edf"])
        cpc_input = st.selectbox("Select CPC Score", config.cpc_options, format_func=lambda x: config.cpc_scale[x])
        gose_input = st.selectbox("Select GOSE Score", config.gose_options, format_func=lambda x: config.gose_scale[x])
        submitted = st.form_submit_button("Submit")
        if submitted:
            # Check no duplicates      
            full_path = os.path.join(config.file['edf_dir'], f"{patient_id}_{date_str}.edf")
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
                patient_label_row.to_csv(config.file['patient_label_path'], mode='a', header=False, index=False)
                patient_label_df = pd.read_csv(config.file['patient_label_path'])

def handle_eeg_results():
    st.header("EEG Graphs")

    # Get unique patient IDs (excluding 'joobee' and 'khanh')
    valid_patient_ids = config.patient_df.loc[~config.patient_df['patient_id'].isin(['joobee', 'khanh']), 'patient_id'].sort_values().unique()
    
    if len(valid_patient_ids) == 0:
        st.warning("No patient data available")
        return
    
    selected_patient = st.selectbox("Select Patient ID", valid_patient_ids)

    # Get dates for selected patient
    patient_dates = config.patient_df[config.patient_df['patient_id'] == selected_patient]['date'].unique()
    
    if len(patient_dates) == 0:
        st.warning(f"No dates available for patient {selected_patient}")
        return
    
    selected_date_find_patient = st.selectbox("Select Administered Date", patient_dates)
        
    try:
        date_str = pd.to_datetime(selected_date_find_patient).strftime("%Y%m%d")
    except Exception as e:
        st.error(f"Error processing date: {e}")
        return

    # date_str = pd.to_datetime(selected_date_find_patient).strftime("%Y%m%d")

    fname = f"{selected_patient}_{date_str}"
    selected_graph = st.selectbox("Choose Graph Type", config.graph_options, format_func=lambda x: config.graphs[x])
    
    st.subheader("Graph Display")

    if selected_graph==1: # CMD
        fig_full_path = os.path.join(config.file['cmd_result_dir'], f"{fname}.png")
        patient_folder = os.path.join(config.file['cmd_result_dir'], f"{selected_patient}_{date_str}")
        eeg_file_path = os.path.join(config.file['edf_dir'], f"{selected_patient}_{date_str}.edf")

        if os.path.exists(patient_folder): # create folder under selected_patient, under date 
            display_all_plots(patient_folder)
        else:
            # Create the directory if it doesn't exist
            os.makedirs(patient_folder, exist_ok=True)
        if st.button("Run CMD Analysis"):
            claassen_analysis.run_analysis(selected_patient, config.file['cmd_result_dir'], eeg_file_path, config.file['patient_df_path'], date_str)
            display_all_plots(patient_folder)

    elif selected_graph==2:
        expected_filename = "avg_itpc_plot.png"
        patient_folder = os.path.join(config.file['lang_tracking_dir'], selected_patient)
        image_path = os.path.join(config.file['lang_tracking_dir'], expected_filename)
        
        st.subheader("Language Tracking Options")
        # Let user pick channels for bad and EOG
        available_channels = [
            'C3','C4','O1','O2','FT9','FT10','Cz','F3','F4','F7','F8',
            'Fz','Fp1','Fp2','Fpz','P3','P4','Pz','T7','T8','P7','P8'
        ]
        selected_bad_channels = st.multiselect(
            "Select Bad Channels", available_channels,
            default=['T7','Fp1','Fp2']
        )
        selected_eog_chs = st.multiselect(
            "Select EOG Channels", available_channels,
            default=['Fp1','Fp2','T7']
        )

        # Let user pick which graph to display: average or individual channel
        display_option = st.selectbox("Select Graph to Display", ["Average ITPC", "Individual Channel"])
        if display_option == "Individual Channel":
            chosen_channel = st.selectbox("Choose a Channel", available_channels)

        # Button to run analysis
        if st.button("Run Language Tracking Analysis"):
            eeg_file_path = os.path.join(config.file['edf_dir'], f"{selected_patient}_{date_str}.edf")

            # relative_path = config.file.get(f"{selected_patient}_path", "")
            # # Combine with edf_dir
            # eeg_file_path = os.path.join(config.file['edf_dir'], os.path.basename(relative_path))
            
            stimulus_csv_path = config.file['patient_df_path']
            use_channels = available_channels
            bad_channels = selected_bad_channels
            eog_chs = selected_eog_chs

            # Call your rodika_modularized.main
            rodika_modularized.main(eeg_file_path, stimulus_csv_path,
                                    selected_patient, use_channels, bad_channels, eog_chs)
            st.success("Analysis complete! The ITPC graphs have been generated.")

        # Once the analysis is done, images are in data/results/lang_tracking/<patient_id>/...
        patient_folder = os.path.join("data", "results", "lang_tracking", selected_patient)

        if display_option == "Average ITPC":
            expected_filename = "avg_itpc_plot.png"
        else:
            expected_filename = f"ITPC_{chosen_channel}.png"

        image_path = os.path.join(patient_folder, expected_filename)

        st.subheader("Graph Display")
        if os.path.exists(image_path):
            st.image(image_path, caption=f"ITPC for {selected_patient}")
        else:
            st.warning(f"No ITPC graph found at {image_path}. Please run the analysis first or check your folder structure.")

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
        if plot_file == 'permutation_results.png' and permutation_results:
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


if __name__ == "__main__":

    # set up configuration settings
    config = Config()

    # Run the main application
    main()