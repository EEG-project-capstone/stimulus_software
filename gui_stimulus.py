# gui_stimulus.py

import pandas as pd
import os
import time
import yaml
import sys
import streamlit as st
from PIL import Image
import random
# Assuming these imports remain relevant for notes/history/results
from auditory_stim.stimulus_package_notes import add_notes, add_history
# Import ALL potentially needed backend functions
from auditory_stim.auditory_stim import (
    randomize_trials, generate_stimuli, play_stimuli, # Old/Modified
    randomize_loved_one_trials, generate_loved_one_stimuli # New (will be created in backend)
)
# Assuming these are still needed for Tab 3
from eeg_auditory_stimulus import rodika_modularized
from eeg_auditory_stimulus import claassen_analysis

# --- Config loading and other setup code ---
# Check for test flag
test_run = '--test' in sys.argv
print(f"Test run: {test_run}")

# Load configuration
try:
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    st.error("Config file 'config.yml' not found. Please ensure it exists.")
    st.stop()
except yaml.YAMLError as e:
    st.error(f"Error parsing config.yml: {e}")
    st.stop()


# Check and create necessary directories from config
required_dirs = ['patient_output_dir', 'stimuli_dir', 'edf_dir', 'result_dir', 'cmd_result_dir', 'lang_tracking_dir']
for dir_key in required_dirs:
    if dir_key in config:
        os.makedirs(config[dir_key], exist_ok=True)
    else:
        st.warning(f"Config key '{dir_key}' not found. Directory checks skipped.")

# Check for sentences path (needed for old paradigm)
if 'sentences_path' not in config or not os.path.exists(config['sentences_path']):
    st.warning(f"Sentences directory path 'sentences_path' missing in config or directory not found. 'language_stim' might fail.")
# Check for control statement paths (needed for new paradigm)
if 'control_statement_male' not in config or 'control_statement_female' not in config:
    st.warning(f"Paths for 'control_statement_male' or 'control_statement_female' missing in config. 'Loved One/Control' paradigm might fail.")


# Load patient data DataFrame
patient_df_path = config.get('patient_df_path', 'data/patient_df.csv') # Default path
try:
    if os.path.exists(patient_df_path):
        patient_df = pd.read_csv(patient_df_path)
        # Ensure required columns exist - add if missing (important for concat later)
        base_cols = ['patient_id', 'date'] # Minimum required
        for col in base_cols:
             if col not in patient_df.columns: patient_df[col] = pd.NA
    else:
        # Define columns for a new DataFrame - include ALL expected columns from logging
        log_columns = ['patient_id', 'date', 'trial_index', 'paradigm', 'trial_type', 'stimulus_details', 'start_time', 'end_time', 'duration']
        patient_df = pd.DataFrame(columns=log_columns)
        # Save the empty df with headers immediately
        patient_df.to_csv(patient_df_path, index=False)
except Exception as e:
    st.error(f"Error loading or initializing patient data file '{patient_df_path}': {e}")
    st.stop()


current_date = time.strftime("%Y-%m-%d")
# ---------------------------------------------

### Display plots & log function ###
# (Keep display_all_plots and read_log_file functions as they were)
# ...
def display_all_plots(patient_folder):
    # ... (implementation from previous version)
    pass
def read_log_file(log_file_path):
    # ... (implementation from previous version)
    pass
# ----------------------------------


### Streamlit Interface ###
tab1, tab2, tab3 = st.tabs(["Administer Stimuli", "Patient Information", "Results"])

with tab1:
    st.title("EEG Stimulus Package")
    st.header("Administer Auditory Stimuli", divider='rainbow')

    # --- Patient ID ---
    patient_id = st.text_input("Enter Patient/EEG ID")

    st.subheader("Select Stimulus Paradigms to Prepare")

    # --- NEW Paradigm Selection ---
    loved_one_stim_selected = st.checkbox("Loved One / Control Stimulus Paradigm")

    # --- Loved One File Uploader (only relevant if loved_one_stim_selected) ---
    uploaded_loved_one_file = st.file_uploader(
        "Upload Loved One's Voice Recording (.wav or .mp3)",
        type=['wav', 'mp3'],
        disabled=(not loved_one_stim_selected)
    )
    # Store uploaded file info in session state to persist across reruns
    if uploaded_loved_one_file is not None:
        # IMPORTANT: Handling uploaded files requires care. For passing to backend,
        # saving temporarily and passing path might be safer than passing BytesIO.
        # For now, just store the fact that a file was uploaded.
        st.session_state['uploaded_loved_one_file_info'] = {
            'name': uploaded_loved_one_file.name,
            'type': uploaded_loved_one_file.type,
            # 'bytes': uploaded_loved_one_file.getvalue() # Avoid storing large data in state if possible
        }
        st.success(f"Uploaded: {uploaded_loved_one_file.name}")
    elif loved_one_stim_selected:
         # Clear state if checkbox is selected but file is removed/not uploaded
         if 'uploaded_loved_one_file_info' in st.session_state:
              del st.session_state['uploaded_loved_one_file_info']


    # --- Gender Selection (Always visible, relevant for Loved One paradigm) ---
    family_member_gender = st.radio(
        "Select Recorded Family Member Gender (for Loved One Paradigm):",
        ('Male', 'Female'),
        index=None, # Default to no selection unless loaded from state?
        key='gender_selection',
        horizontal=True,
        # disabled=(not loved_one_stim_selected) # Keep enabled based on user request
    )

    # --- OLD Paradigm Checkboxes (kept as requested, NOT disabled) ---
    st.markdown("---") # Separator
    st.write("**AND/OR** Select from Standard Paradigms:")
    language_stim_selected = st.checkbox("language_stim", key="language_checkbox")
    right_cmd_stim_selected = st.checkbox("right_cmd_stim", key="right_cmd_checkbox")
    left_cmd_stim_selected = st.checkbox("left_cmd_stim", key="left_cmd_checkbox")
    beep_stim_selected = st.checkbox("beep_stim", key="beep_checkbox")
    oddball_stim_selected = st.checkbox("oddball_stim", key="oddball_checkbox") # Assuming oddball exists
    st.markdown("---") # Separator

    # --- Session State Initialization ---
    stimulus_info_key = 'stimulus_details' # Combined details list
    paradigm_key = 'prepared_paradigm' # Tracks what was prepared ('loved_one', 'standard', 'mixed', None)
    # Initialize keys if they don't exist
    if stimulus_info_key not in st.session_state:
        st.session_state[stimulus_info_key] = []
    if paradigm_key not in st.session_state:
        st.session_state[paradigm_key] = None
    if 'trial_labels_combined' not in st.session_state: # For combined labels before generation if needed
         st.session_state['trial_labels_combined'] = []


    # --- Prepare Stimulus Button ---
    if st.button("Prepare Stimulus"):
        # Reset state before preparing
        st.session_state[stimulus_info_key] = []
        st.session_state[paradigm_key] = None
        st.session_state['trial_labels_combined'] = []

        # Lists to hold results from potentially multiple paradigms
        all_prepared_details = []
        prepared_paradigm_types = []

        # --- 1. Prepare NEW Loved One / Control Paradigm (if selected) ---
        if loved_one_stim_selected:
            st.write("Preparing Loved One / Control Stimulus...")
            # Validation
            if not family_member_gender:
                st.error("Please select the family member's gender for the Loved One paradigm.")
                st.stop()
            if 'uploaded_loved_one_file_info' not in st.session_state:
                 st.error("Please upload the Loved One's voice recording file.")
                 st.stop()

            # --- Handle uploaded file ---
            # Best practice: Save the uploaded file temporarily and pass the path.
            # Create a temporary directory within stimuli_dir maybe?
            temp_dir = os.path.join(config.get('stimuli_dir', 'data/stimuli'), "temp_upload")
            os.makedirs(temp_dir, exist_ok=True)
            # WARNING: Simple approach, potential filename collisions if not handled carefully
            uploaded_file_info = st.session_state['uploaded_loved_one_file_info']
            # Construct a safe temporary path
            temp_file_path = os.path.join(temp_dir, f"temp_loved_one_{uploaded_file_info['name']}")

            # Need to access the actual file object again if it wasn't stored
            # This requires the file_uploader widget to be present *before* the button
            # If using file_uploader value directly:
            if uploaded_loved_one_file:
                 try:
                      with open(temp_file_path, "wb") as f:
                           f.write(uploaded_loved_one_file.getvalue())
                      print(f"Saved uploaded file temporarily to: {temp_file_path}") # Debug
                 except Exception as e:
                      st.error(f"Error saving uploaded file: {e}")
                      st.stop()
            else:
                 # This case shouldn't happen if validation above worked, but as safety
                 st.error("Uploaded file not found during prepare step.")
                 st.stop()
            # --- End Handle uploaded file ---

            try:
                # 1a. Randomize 'control' and 'loved_one' labels
                num_trials_each = 50
                loved_one_trial_labels = randomize_loved_one_trials(num_control=num_trials_each, num_loved_one=num_trials_each)
                # Don't store in session state yet, add to combined list later

                # 1b. Generate the 100 audio files
                # Assumes generate_loved_one_stimuli handles loading, silence, export, and returns details list
                loved_one_details = generate_loved_one_stimuli(
                    trial_types=loved_one_trial_labels, # Pass the generated labels
                    gender=family_member_gender,
                    config=config,
                    loved_one_voice_path=temp_file_path # Pass path to temp saved file
                )
                all_prepared_details.extend(loved_one_details)
                prepared_paradigm_types.append("loved_one")

            except Exception as e:
                st.error(f"Error preparing Loved One / Control stimulus: {e}")
                # Clean up temp file?
                if os.path.exists(temp_file_path): os.remove(temp_file_path) # Basic cleanup
                st.stop() # Stop processing if this part fails
            finally:
                 # Clean up temp file after generation attempt
                 if os.path.exists(temp_file_path):
                      try:
                           os.remove(temp_file_path)
                           print(f"Cleaned up temp file: {temp_file_path}") # Debug
                      except Exception as e_clean:
                           st.warning(f"Could not remove temp file {temp_file_path}: {e_clean}")

        # --- 2. Prepare OLD Standard Paradigms (if selected) ---
        standard_paradigms_selected = any([language_stim_selected, right_cmd_stim_selected, left_cmd_stim_selected, beep_stim_selected, oddball_stim_selected])
        if standard_paradigms_selected:
            st.write("Preparing Standard Stimulus Paradigms...")
            num_of_each_trial = {
                "lang": 72 if language_stim_selected else 0,
                "rcmd": 3 if right_cmd_stim_selected else 0,
                "lcmd": 3 if left_cmd_stim_selected else 0,
                "beep": 6 if beep_stim_selected else 0,
                "odd": 40 if oddball_stim_selected else 0 # Needs 'odd' handling in backend
            }
            num_of_each_trial = {k: v for k, v in num_of_each_trial.items() if v > 0}

            if not num_of_each_trial:
                # This condition shouldn't be met if standard_paradigms_selected is true
                 st.warning("Internal state error: Standard paradigms selected but counts are zero.")
                 st.stop()

            try:
                # 2a. Use existing randomize_trials for standard types
                standard_trial_labels = randomize_trials(num_of_each_trial) # Pass dict

                # 2b. Use existing generate_stimuli
                # IMPORTANT: Existing generate_stimuli likely needs modification
                # to return a details list compatible with the modified play_stimuli.
                # It currently returns lang_trials_ids.
                # Placeholder: Assume it's modified to return details list
                # standard_details = generate_stimuli_modified(standard_trial_labels, config) # Need to create this backend function
                # For now, simulate details based on labels:
                standard_details = []
                for i, label in enumerate(standard_trial_labels):
                     # Simulate output path - generate_stimuli needs to create these
                     sim_path = os.path.join(config.get('stimuli_dir', 'data/stimuli'), f"trial_std_{i}_{label}.mp3")
                     standard_details.append({'type': label, 'output_path': sim_path, 'source_info': label})
                # --- End Placeholder ---

                all_prepared_details.extend(standard_details)
                prepared_paradigm_types.append("standard")

            except Exception as e:
                 st.error(f"Error preparing Standard stimulus: {e}")
                 st.stop() # Stop processing if this part fails

        # --- 3. Finalize Preparation ---
        if not all_prepared_details:
            st.warning("No stimulus paradigms were selected or prepared.")
        else:
            # Combine and Shuffle ALL prepared trials together
            random.shuffle(all_prepared_details)
            # Re-assign index after shuffling if needed by playback/logging
            for i, details in enumerate(all_prepared_details):
                 details['index'] = i

            # Determine final paradigm type
            if "loved_one" in prepared_paradigm_types and "standard" in prepared_paradigm_types:
                final_paradigm_type = "mixed"
            elif "loved_one" in prepared_paradigm_types:
                final_paradigm_type = "loved_one"
            elif "standard" in prepared_paradigm_types:
                final_paradigm_type = "standard"
            else: # Should not happen if all_prepared_details is not empty
                final_paradigm_type = None

            st.session_state[stimulus_info_key] = all_prepared_details
            st.session_state[paradigm_key] = final_paradigm_type
            st.success(f"Successfully prepared {len(all_prepared_details)} stimuli (Paradigm: {final_paradigm_type}). Ready to Play.")


    # --- Play Stimulus Button ---
    # (This section remains largely the same as the previous corrected version,
    #  as it relies on the unified 'stimulus_details' structure prepared above
    #  and the modified play_stimuli backend function)
    if st.button("Play Stimulus"):
        print(f"Play button clicked for patient_id: {patient_id}") # Debug
        current_date = time.strftime("%Y-%m-%d")
        prepared_paradigm = st.session_state.get(paradigm_key)
        stimulus_info_key = 'stimulus_details' # Use consistent key

        # --- Prerequisites Check ---
        # (Checks for patient_id, previous run, prepared state)
        # ... (Keep the checks from the previous response) ...
        if patient_id.strip() == "":
            st.error("Please enter a patient ID.")
        elif ((patient_df['patient_id'] == patient_id) & (patient_df['date'] == current_date)).any():
            st.error(f"Patient '{patient_id}' has already been administered stimulus protocol today ({current_date}).")
        elif not prepared_paradigm:
            st.error("Please prepare stimuli first using the 'Prepare Stimulus' button.")
        elif not st.session_state.get(stimulus_info_key):
            st.error("Stimulus details not found or empty. Please Prepare Stimuli again.")
        else:
            # Retrieve prepared data from session state
            stimulus_details = st.session_state[stimulus_info_key] # List of dicts
            n = len(stimulus_details)

            if n == 0:
                 st.error("No stimuli prepared. Please Prepare Stimuli again.")
                 st.stop()

            progress_bar = st.progress(0, text="0%")
            administered_stimuli_log = []

            for i in range(n):
                current_stim_details = stimulus_details[i] # Details dict for this trial
                # Get actual type from details dict, don't rely on separate trial_types list anymore
                trial_label = current_stim_details.get('type', 'unknown')
                # Ensure index is correctly set if needed
                current_stim_details['index'] = i

                print(f"Playing Trial {i+1}/{n}: Type='{trial_label}'") # Debug

                # --- Call to MODIFIED play_stimuli ---
                # Assumes play_stimuli now just needs the details dict
                try:
                    start_time, end_time = play_stimuli(current_stim_details, test_run=test_run)
                except Exception as e:
                    st.error(f"Error during playback of trial {i+1} ({trial_label}): {e}")
                    st.stop()

                # --- Unified Logging ---
                log_entry = {
                            'patient_id': patient_id,
                            'date': current_date,
                            'trial_index': i,
                            'paradigm': prepared_paradigm, # Log which paradigm(s) were run
                            'trial_type': trial_label,
                            'stimulus_details': current_stim_details.get('source_info', 'N/A'), # Log source info
                            'start_time': start_time,
                            'end_time': end_time,
                            'duration': end_time - start_time if start_time and end_time else None
                        }
                administered_stimuli_log.append(log_entry)

                percent = int((i + 1) / n * 100)
                progress_bar.progress(percent, text=f"{percent}% Complete (Trial {i+1}/{n})")

            progress_bar.progress(100, text="Stimulus presentation complete.")

            # --- DataFrame Handling (Should be okay now) ---
            # (Keep the DataFrame logic from the previous response)
            # ... (Create new_stimuli_df, define expected_cols, ensure patient_df cols, convert types, concat, save) ...
            new_stimuli_df = pd.DataFrame(administered_stimuli_log)
            expected_columns = list(log_entry.keys())

            for col in expected_columns:
                if col not in patient_df.columns:
                    print(f"Adding missing column to patient_df: {col}")
                    patient_df[col] = pd.NA

            # Convert types before concat (Add specific conversions as needed)
            for col in ['patient_id', 'date', 'paradigm', 'trial_type', 'stimulus_details']:
                 if col in patient_df.columns: patient_df[col] = patient_df[col].astype(object)
                 if col in new_stimuli_df.columns: new_stimuli_df[col] = new_stimuli_df[col].astype(object)
            for col in ['start_time', 'end_time', 'duration']:
                 if col in patient_df.columns: patient_df[col] = pd.to_numeric(patient_df[col], errors='coerce')
                 if col in new_stimuli_df.columns: new_stimuli_df[col] = pd.to_numeric(new_stimuli_df[col], errors='coerce')
            if 'trial_index' in patient_df.columns: patient_df['trial_index'] = patient_df['trial_index'].astype('Int64')
            if 'trial_index' in new_stimuli_df.columns: new_stimuli_df['trial_index'] = new_stimuli_df['trial_index'].astype('Int64')

            try:
                 updated_patient_df = pd.concat([patient_df, new_stimuli_df[expected_columns]], ignore_index=True)
                 # Update the global or accessible patient_df variable after concat
                 patient_df = updated_patient_df
            except Exception as e:
                 st.error(f"Error during DataFrame concatenation: {e}")
                 st.stop()

            # Save updated DataFrame
            try:
                 patient_df.to_csv(patient_df_path, index=False)
            except Exception as e:
                 st.error(f"Failed to save patient data: {e}")

            # Save individual patient output file
            output_dir = config['patient_output_dir']
            # os.makedirs(output_dir, exist_ok=True) # Already ensured above
            formatted_date = current_date.replace("-", "")
            output_file = f"{patient_id}_{formatted_date}.csv"
            output_path = os.path.join(output_dir, output_file)
            try:
                new_stimuli_df.to_csv(output_path, index=False)
            except Exception as e:
                st.error(f"Failed to save individual patient file: {e}")


            # Add history and reset state
            add_history(patient_id, current_date)
            st.session_state[paradigm_key] = None # Reset prepared state
            st.session_state[stimulus_info_key] = []
            st.success(f"Stimuli administered to patient {patient_id} on {current_date}. Data logged.")


    # --- Add Notes / Find Notes Sections ---
    # (Keep existing code from previous version)
    st.header("Add Notes to your Selected Patient and Date", divider='rainbow')
    # ... (rest of notes implementation) ...

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


# --- Code for tab2 and tab3 ---
# (Keep existing code - assumes no direct dependency changes needed here)
with tab2:
    st.header("Upload EEG Files")
    with st.form("upload_form"):
        patient_id = st.text_input("Patient ID", placeholder="Enter Patient ID")
        date = st.date_input("Recording Date")
        edf_file = st.file_uploader("Upload EDF File", type=["edf"])

        # Replace config options with hardcoded lists or safe fallback
        cpc_choices = ["CPC 1", "CPC 2", "CPC 3", "CPC 4", "CPC 5"]
        gose_choices = ["GOSE 1", "GOSE 2", "GOSE 3", "GOSE 4", "GOSE 5", "GOSE 6", "GOSE 7", "GOSE 8"]

        cpc_input = st.selectbox("Select CPC Score", cpc_choices)
        gose_input = st.selectbox("Select GOSE Score", gose_choices)

        submitted = st.form_submit_button("Submit")
        if submitted:
            date_str = date.strftime("%Y%m%d")
            edf_dir = config.get('edf_dir', 'data/edfs')
            label_path = config.get('patient_label_path', 'data/patient_records.csv')

            file_path = os.path.join(edf_dir, f"{patient_id}_{date_str}.edf")
            if os.path.exists(file_path):
                st.error(f"File already exists for {patient_id} on {date_str}")
            elif edf_file:
                with open(file_path, 'wb') as f:
                    f.write(edf_file.getvalue())
                st.success(f"Uploaded {edf_file.name} for {patient_id}")

                # Append label
                label_entry = pd.DataFrame([{
                    'patient_id': patient_id,
                    'date': date_str,
                    'cpc': cpc_input,
                    'gose': gose_input
                }])
                if not os.path.exists(label_path):
                    label_entry.to_csv(label_path, index=False)
                else:
                    label_entry.to_csv(label_path, mode='a', header=False, index=False)

with tab3:
    st.header("EEG Results Viewer")
    if patient_df.empty:
        st.warning("No patient data available.")
    else:
        selected_patient = st.selectbox("Select Patient", patient_df['patient_id'].dropna().unique())
        patient_dates = patient_df[patient_df['patient_id'] == selected_patient]['date'].dropna().unique()
        selected_date = st.selectbox("Select Date", patient_dates)

        df_filtered = patient_df[(patient_df['patient_id'] == selected_patient) & (patient_df['date'] == selected_date)]

        st.subheader("Trial Log")
        st.dataframe(df_filtered)

        st.subheader("Trial Type Distribution")
        if 'trial_type' in df_filtered.columns:
            chart_data = df_filtered['trial_type'].value_counts().reset_index()
            chart_data.columns = ['Trial Type', 'Count']
            st.bar_chart(chart_data.set_index('Trial Type'))

        st.subheader("Trial Durations (sec)")
        if 'duration' in df_filtered.columns:
            st.line_chart(df_filtered['duration'].fillna(0))


# --- Main execution / Config instantiation ---
# If using the class structure from one of the uploaded files:
# class Config: ...
# if __name__ == "__main__":
#     config_obj = Config()
#     # Need to pass config_obj or make it accessible to functions if using class
#     # e.g., main(config_obj) or make config_obj global (not recommended)

# If config is loaded globally at the start (as in the current script):
# No extra instantiation needed here. Code above uses global 'config' and 'patient_df'.