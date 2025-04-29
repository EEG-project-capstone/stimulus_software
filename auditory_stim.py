# auditory_stim.py

import time
import os
import random
import yaml
import shutil # Added for potential file operations
import streamlit as st # Keep for st.progress if needed
import pydub
from pydub import AudioSegment
from pydub.generators import Sine
from pydub.playback import play

# Import specific playback function if needed (e.g., playsound)
try:
    from playsound import playsound
except ImportError:
    print("playsound library not found, playback might rely on pydub's default (requires ffmpeg/simpleaudio).")
    playsound = None # Set to None if not available

# --- Config Loading ---
# Load config globally or pass it around as needed
try:
    config_file_path = 'config.yml'
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)
except FileNotFoundError:
    print(f"ERROR: Configuration file '{config_file_path}' not found.")
    config = {} # Default empty config
except yaml.YAMLError as e:
    print(f"ERROR: Failed to parse configuration file '{config_file_path}': {e}")
    config = {}
# -----------------------------

# --- Pydub Setup (Optional) ---
# Explicitly setting paths can help on Windows
# Adjust these paths based on your ffmpeg installation
# ffmpeg_path = os.path.join(os.getcwd(), 'ffmpeg', 'bin') # Example
# if config.get('os', '').lower() == 'windows' and os.path.exists(ffmpeg_path):
#     try:
#         pydub.AudioSegment.converter = os.path.join(ffmpeg_path, "ffmpeg.exe")
#         pydub.AudioSegment.ffprobe   = os.path.join(ffmpeg_path, "ffprobe.exe")
#         print(f"Pydub using converter: {pydub.AudioSegment.converter}") # Debug
#     except Exception as e:
#          print(f"Warning: Could not set pydub ffmpeg paths: {e}")
# -----------------------------

# --- Playback Helper (Simplified) ---
def play_mp3(mp3_path, verbose=True):
    """Plays the specified MP3 file using playsound or pydub."""
    if not os.path.exists(mp3_path):
        print(f"ERROR: Audio file not found: {mp3_path}")
        st.warning(f"Playback Error: File not found {os.path.basename(mp3_path)}")
        return

    if verbose:
        print(f"Playing {os.path.basename(mp3_path)}...")

    try:
        audio = AudioSegment.from_mp3(mp3_path)
        play(audio)

    except Exception as e:
        print(f"Error during playback of {mp3_path}: {e}")
        st.warning(f"Playback error for {os.path.basename(mp3_path)}: {e}. Check audio setup.")

# === NEW Functions for Loved One / Control Paradigm ===

def randomize_loved_one_trials(num_control=50, num_loved_one=50):
    """Generates a shuffled list of 'control' and 'loved_one' trial labels."""
    if not isinstance(num_control, int) or not isinstance(num_loved_one, int) or num_control < 0 or num_loved_one < 0:
         raise ValueError("Number of trials must be non-negative integers.")

    control_labels = ['control'] * num_control
    loved_one_labels = ['loved_one'] * num_loved_one # Changed label for clarity
    trial_types = control_labels + loved_one_labels
    random.shuffle(trial_types)
    print(f"Generated {len(trial_types)} loved one paradigm trials: {trial_types.count('control')} control, {trial_types.count('loved_one')} loved_one.") # Debug
    return trial_types

def generate_loved_one_stimuli(trial_types, gender, config, loved_one_voice_path):
    """
    Generates audio stimuli for the Loved One/Control paradigm.

    Loads the specified loved one voice file and the gender-matched control file,
    adds silence to each, exports 100 unique .mp3 files based on trial_types list,
    and returns details for each trial.

    Args:
        trial_types (list): Shuffled list of 'control' and 'loved_one' labels.
        gender (str): 'Male' or 'Female'.
        config (dict): Configuration dictionary.
        loved_one_voice_path (str): Path to the uploaded loved one voice file.

    Returns:
        list: List of dictionaries, each like:
              {'type': 'loved_one'/'control',
               'output_path': 'path/to/trial_N.mp3',
               'source_info': 'filename.wav'}
    """
    print(f"Generating Loved One/Control stimuli. Gender: {gender}, Loved one file: {loved_one_voice_path}")
    prepared_stimuli_details = []
    stimuli_dir = config.get('stimuli_dir', 'data/stimuli')
    silence_duration_ms = config.get('silence_duration_ms', 5000)

    # --- Load Source Audio and Add Silence ---
    try:
        # Load Loved One voice
        loved_one_audio = AudioSegment.from_file(loved_one_voice_path) # Handles wav/mp3
        loved_one_source_info = os.path.basename(loved_one_voice_path)

        # Load Control voice
        control_path = config.get('control_statement_male') if gender == 'Male' else config.get('control_statement_female')
        if not control_path or not os.path.exists(control_path):
             raise FileNotFoundError(f"Control statement file path missing or invalid in config for gender '{gender}'.")
        control_audio = AudioSegment.from_wav(control_path) # Assuming control is WAV
        control_source_info = os.path.basename(control_path)

        # Create silence segment
        silence_segment = AudioSegment.silent(duration=silence_duration_ms)

        # Pre-process: Add silence to both source audios
        loved_one_processed = loved_one_audio + silence_segment
        control_processed = control_audio + silence_segment

    except FileNotFoundError as e:
        st.error(f"File not found during stimulus generation: {e}")
        raise
    except Exception as e:
        st.error(f"Error loading or processing source audio: {e}")
        raise
    # --- End Loading ---

    # Ensure output directory exists
    os.makedirs(stimuli_dir, exist_ok=True)

    # Setup progress bar
    gen_bar = st.progress(0, text="Generating Loved One Stimuli: 0%")
    n = len(trial_types)

    for i, trial_label in enumerate(trial_types):
        output_filename = f"trial_{i}_{trial_label}.mp3"
        output_path = os.path.join(stimuli_dir, output_filename)
        trial_details = {'type': trial_label, 'index': i, 'output_path': output_path}

        try:
            # Select the correct pre-processed audio segment
            if trial_label == "loved_one":
                audio_to_export = loved_one_processed
                trial_details['source_info'] = loved_one_source_info
            elif trial_label == "control":
                audio_to_export = control_processed
                trial_details['source_info'] = control_source_info
            else:
                print(f"Warning: Unknown trial type '{trial_label}' in loved one generation. Skipping.")
                continue

            # Export the segment to its unique file
            audio_to_export.export(output_path, format="mp3")
            prepared_stimuli_details.append(trial_details)

            # Update progress bar
            percent_complete = int(((i + 1) / n) * 100)
            gen_bar.progress(percent_complete, text=f"Generating Loved One Stimuli: {percent_complete}%")

        except Exception as e:
             st.error(f"Error exporting trial {i} ({trial_label}) to {output_path}: {e}")
             raise # Re-raise error to stop generation

    gen_bar.progress(100, text="Loved One Stimulus Generation Complete.")
    print(f"Finished generating loved one stimuli details: {len(prepared_stimuli_details)} trials.") # Debug
    return prepared_stimuli_details

# ======================================================

# === Existing Functions for Standard Paradigms (Kept) ===

# --- Randomization (Existing - takes dict) ---
def randomize_trials(num_of_each_trial):
    """Generates shuffled list of labels for standard paradigms."""
    trial_types = []
    print(f"Randomizing standard trials based on: {num_of_each_trial}") # Debug
    for key in num_of_each_trial:
        # Ensure key is a valid label prefix (lang, rcmd, lcmd, beep, odd)
        valid_keys = ["lang", "rcmd", "lcmd", "beep", "odd"]
        if key not in valid_keys:
             print(f"Warning: Skipping unknown trial type key '{key}' in randomize_trials.")
             continue

        for i in range(num_of_each_trial[key]):
            if key == "lang":
                # Creates labels like lang_0, lang_1,... for unique lang stimuli files
                trial = f"lang_{i}"
            else:
                # For others (rcmd, lcmd, beep, odd), use the key directly as the label
                # The count determines repetitions
                trial = key
            trial_types.append(trial)

    random.shuffle(trial_types)
    print(f"Generated {len(trial_types)} standard trial labels.") # Debug
    return trial_types
# --------------------------------------------

# --- Stimulus Generation (Existing - MODIFIED to return details) ---
def generate_stimuli(trial_types, config):
    """
    Generates stimuli for STANDARD paradigms (lang, rcmd, lcmd, beep, odd).
    MODIFIED to return a list of details dictionaries for compatibility
    with the updated play_stimuli function.
    """
    print(f"Generating standard stimuli for labels: {trial_types}") # Debug
    prepared_stimuli_details = []
    stimuli_dir = config.get('stimuli_dir', 'data/stimuli')
    os.makedirs(stimuli_dir, exist_ok=True)

    gen_bar = st.progress(0, text="Generating Standard Stimuli: 0%")
    n = len(trial_types)
    lang_counter = 0 # To handle lang_0, lang_1, etc.

    for i, trial_label in enumerate(trial_types):
        output_path = None
        source_info = trial_label # Default source info is the label itself
        trial_details = {'type': trial_label, 'index': i}

        try:
            # --- Language Trial Generation ---
            if trial_label.startswith("lang_"):
                 # Assumes random_lang_stim creates ONE file named lang_N.mp3
                 output_filename = f"{trial_label}.mp3"
                 output_path = os.path.join(stimuli_dir, output_filename)
                 # random_lang_stim needs number of sentences - how is this determined?
                 # Using a default from config for now.
                 num_sentences = config.get('lang_num_sentences', 12)
                 # random_lang_stim originally returned sample_ids, now returns None (just creates file)
                 random_lang_stim(output_path, num_sentence=num_sentences, config=config)
                 source_info = f"Generated Lang {lang_counter} ({num_sentences} sentences)"
                 lang_counter += 1

            # --- Placeholder Generation for Other Standard Types ---
            # These currently don't generate files in the backend.
            # To make play_stimuli work uniformly, we SHOULD generate files here.
            # For now, we'll just create placeholder details and rely on play_stimuli's old logic.
            # If play_stimuli is fully simplified, file generation MUST happen here.
            elif trial_label in ["rcmd", "lcmd", "beep", "odd"]:
                 # Placeholder: No file generated here yet by default.
                 # Play_stimuli will handle these using old administer_* functions.
                 # Store the label itself, play_stimuli will use it.
                 trial_details['output_path'] = None # Indicate no pre-generated file
                 trial_details['source_info'] = f"Direct Playback ({trial_label})"

            else:
                 print(f"Warning: Unknown standard trial type '{trial_label}'. Skipping.")
                 continue

            trial_details['output_path'] = output_path # Store path if generated, else None
            prepared_stimuli_details.append(trial_details)

            percent = int(((i + 1) / n) * 100)
            gen_bar.progress(percent, text=f"Generating Standard Stimuli: {percent}%")

        except Exception as e:
            st.error(f"Error processing standard trial {i} ({trial_label}): {e}")
            raise

    gen_bar.progress(100, text=f"Standard Stimulus Generation Complete.")
    print(f"Finished generating standard stimuli details: {len(prepared_stimuli_details)} trials.") # Debug
    # Returns list of details dicts (some paths might be None if not generated)
    return prepared_stimuli_details
# ----------------------------------------------------------------

# --- Helper: Generate concatenated lang stimulus (Keep for generate_stimuli) ---
def random_lang_stim(output_path, num_sentence=12, config=None):
    """Selects random sentences, concatenates, exports ONE mp3 file."""
    if config is None: config = {} # Safety check
    sentences_path = config.get('sentences_path', 'data/sentences')
    if not os.path.exists(sentences_path):
         raise FileNotFoundError(f"Sentences directory not found: {sentences_path}")

    sentence_files = os.listdir(sentences_path)
    wav_files = [f for f in sentence_files if f.lower().endswith('.wav')]

    if not wav_files:
         raise ValueError(f"No .wav files found in {sentences_path}")
    if num_sentence > len(wav_files):
        print(f"Warning: Requested {num_sentence} sentences, only {len(wav_files)} available. Using all.")
        num_sentence = len(wav_files)

    # Select unique files
    selected_files = random.sample(wav_files, num_sentence)

    combined = AudioSegment.empty()
    for file_name in selected_files:
        file_path = os.path.join(sentences_path, file_name)
        try:
            audio = AudioSegment.from_wav(file_path)
            combined += audio
        except Exception as e:
            print(f"Warning: Error loading language file {file_path}: {e}. Skipping.")

    if len(combined) > 0:
         combined.export(output_path, format="mp3")
         print(f"Generated language stimulus: {output_path}") # Debug
    else:
         raise ValueError("Failed to combine any language audio files.")
    # Removed return sample_ids as it wasn't used effectively
# -----------------------------------------------------------------------

# --- Playback Functions for Standard Stimuli (Kept) ---
def administer_lang(input_path, test_run=False):
    """Plays the pre-generated concatenated language file."""
    start_time = time.time()
    if not test_run:
        play_mp3(input_path)
        # Removed time.sleep(10) - timing should be based on file duration + desired gap
    end_time = time.time()
    return start_time, end_time

def administer_right_cmd(test_run=False, config=None):
    if config is None: config = {}
    start_time = time.time()
    if not test_run:
        if 'right_keep_path' not in config or 'right_stop_path' not in config:
             print("Error: Missing right_keep_path or right_stop_path in config.")
             return start_time, start_time # Return immediately
        # Simpler loop for testing, adjust timing as needed
        play_mp3(config['right_keep_path'])
        time.sleep(config.get('cmd_gap_s', 5)) # Use config for gap, default 5s
        play_mp3(config['right_stop_path'])
    end_time = time.time()
    return start_time, end_time

def administer_left_cmd(test_run=False, config=None):
    # Similar to right_cmd
    if config is None: config = {}
    start_time = time.time()
    if not test_run:
        if 'left_keep_path' not in config or 'left_stop_path' not in config:
             print("Error: Missing left_keep_path or left_stop_path in config.")
             return start_time, start_time
        play_mp3(config['left_keep_path'])
        time.sleep(config.get('cmd_gap_s', 5))
        play_mp3(config['left_stop_path'])
    end_time = time.time()
    return start_time, end_time

def administer_beep(test_run=False, config=None):
    if config is None: config = {}
    start_time = time.time()
    if not test_run:
        if 'beep_path' not in config:
             print("Error: Missing beep_path in config.")
             return start_time, start_time
        play_mp3(config['beep_path'])
    end_time = time.time()
    return start_time, end_time

def play_oddball_stimulus(n_tones=20, freq_standard=1000, freq_rare=2000, prob_rare=0.2, tone_ms=100, gap_ms=1000):
    """Generates and plays oddball sequence directly."""
    tones_sequence = []
    silent_gap = AudioSegment.silent(duration=gap_ms)
    for _ in range(n_tones):
        freq = freq_rare if random.random() < prob_rare else freq_standard
        tone = Sine(freq).to_audio_segment(duration=tone_ms)
        tones_sequence.append(tone)
        tones_sequence.append(silent_gap) # Add gap after each tone

    if tones_sequence:
        combined_audio = sum(tones_sequence)
        play(combined_audio)
    else:
        print("Warning: No tones generated for oddball.")

def administer_oddball(test_run=False, config=None):
    # Uses parameters from config if available
    if config is None: config = {}
    start_time = time.time()
    if not test_run:
        play_oddball_stimulus(
            n_tones=config.get('oddball_n_tones', 20),
            freq_standard=config.get('oddball_freq_standard', 1000),
            freq_rare=config.get('oddball_freq_rare', 2000),
            prob_rare=config.get('oddball_prob_rare', 0.2),
            tone_ms=config.get('oddball_tone_ms', 100),
            gap_ms=config.get('oddball_gap_ms', 1000)
        )
    end_time = time.time()
    return start_time, end_time

# --- Main Playback Function (MODIFIED) ---
def play_stimuli(stimulus_info, test_run=False):
    start_time, end_time = time.time(), time.time()

    if isinstance(stimulus_info, dict):
        trial_type = stimulus_info.get('type', 'unknown')
        output_path = stimulus_info.get('output_path')

        # --- Handle built-in (non-file) paradigms first ---
        if trial_type == "rcmd":
            return administer_right_cmd(test_run, config)
        elif trial_type == "lcmd":
            return administer_left_cmd(test_run, config)
        elif trial_type == "beep":
            return administer_beep(test_run, config)
        elif trial_type == "odd":
            return administer_oddball(test_run, config)

        # --- Otherwise: use file path ---
        if output_path and os.path.exists(output_path):
            start_time = time.time()
            if not test_run:
                play_mp3(output_path)
            else:
                audio = AudioSegment.from_mp3(output_path)
                time.sleep(len(audio) / 1000.0)
            end_time = time.time()
            return start_time, end_time
        else:
            st.error(f"Playback Error: File missing for trial type '{trial_type}'")
            return time.time(), time.time()

    else:
        st.error("Invalid stimulus_info passed to play_stimuli()")
        return time.time(), time.time()
# --------------------------------------

# --- Deprecated / Unused Placeholders ---
# Keep jittered_delay only if explicitly needed for old paradigm timing
def jittered_delay():
     """Provides a random delay between 1.2 and 2.2 seconds."""
     # Generally REMOVED from play_stimuli logic now. Keep function if needed elsewhere.
     # time.sleep(random.uniform(1.2, 2.2))
     pass # Do nothing by default now

# Placeholder kept from original code, seems unused.
def administer_loved_ones(test_run=False):
    # This function is NOT used by the new paradigm logic.
    # generate_loved_one_stimuli creates the files, play_stimuli plays them.
    print("Warning: administer_loved_ones function called - this should not happen in the new flow.")
    start_time = time.time()
    # Placeholder action
    if not test_run: time.sleep(1)
    end_time = time.time()
    return start_time, end_time
# ---------------------------------------
