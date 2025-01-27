# Brain Waves Stimulus Package 2.0

## Introduction
The EEG Stimulus Package is a comprehensive tool designed to assist researchers and clinicians in administering and managing auditory stimuli during EEG (electroencephalogram) sessions. This software, built using Python and Streamlit, provides a user-friendly graphical interface that facilitates the seamless integration of stimulus protocols and patient data management. It is especially useful for experiments involving precise auditory stimulus timing, randomized trial administration, and detailed patient record-keeping.
## Features
### Graphical User Interface (GUI):
A web-based interface powered by Streamlit, allowing for easy interaction with the software without requiring deep technical knowledge.

### Stimulus Administration:

**Randomized Trials:** Automatically randomizes and prepares trial sequences, including language-based stimuli, command prompts, and auditory beeps.  
**Jittered Delay:** Introduces random delays between stimuli to reduce predictability and improve experiment robustness.  
**Multiple Stimulus Types:** Supports language stimuli, right/left command prompts, and beep stimuli, with configurable playback options.

### Patient Data Management:

**Patient/EEG ID Input:** Allows users to input and track patient IDs during stimulus administration.  
**Trial Tracking:** Records detailed trial information, including the type of trial, start and end times, and duration, which are stored in a CSV format for easy access and analysis.  
**Prevent Redundant Administration:** Automatically checks if a patient has already received the stimulus protocol on the current date to avoid redundancy.

### Search and Retrieval:

**Search Administered Stimuli:** Users can search for and review stimuli that have been administered to patients on specific dates.  
**View Stimuli Details:** Displays the specific stimuli administered to a patient, helping researchers track the protocol's progress and effectiveness.

### Notes Management:

**Add Notes:** Enables users to append notes to patient records, ensuring that all observations and important details are documented.  
**Retrieve Notes:** Allows for easy retrieval of previously added notes, providing a comprehensive overview of patient interactions and observations.
### Cross-Platform Support:
Compatible with *Windows/Linux/MacOS* operating systems.

## Installation
### *Prerequisites*
This software requires the installation of:
* [Anaconda](https://docs.anaconda.com/anaconda/install/)/[Miniconda](https://docs.anaconda.com/miniconda/)
* Python
### *Setup Environment*
#### 1. Create CONDA env
```bash
conda create -n "eeg"
conda activate eeg
conda install pip
pip install -r requirements.txt
```

#### 2. Install mpv package (*for Windows*)
```bash
pip install mpv
```
Go to this [repo](https://github.com/jaseg/python-mpv), download `mpv.py` to /Script folder.    
To check where Script folder is, use command `conda env list`.    
Go to this [link](https://sourceforge.net/projects/mpv-player-windows/) to download dev package.
Choose the one with x84_64 if your computer is 64-bit. If 32 bit, choose the one with 32 bit.
After downloading, extract libmpv-2.dll (or a combination of mpv.exe + mpv.com + mpv-1.dll) to /Script folder. Refer to this [issue](https://github.com/jaseg/python-mpv/issues/60#issuecomment-352719773).    
Then run
```bash
pip install python-mpv
```
#### 3. Install ffmpeg (*for Windows or Mac if an error occurs*)
##### For Windows
Install [ffmpeg](https://github.com/BtbN/FFmpeg-Builds/releases) to current directory.    
Extract and rename to `ffmpeg`.    
Make sure to have these 2 files:    
`ffmpeg\bin\ffmpeg.exe`    
`ffmpeg\bin\ffprobe.exe`

##### For Mac
`brew install ffmpeg`.

#### 4. Install audio folder
Download the audio folder from this [link](https://drive.google.com/drive/folders/1VktnddvsY1kFihuCpRO4GKf7Z4wXVKIa) 
to the `data` folder. Make sure the folder name matches the `sentences_path` field in `config.yml`.    
If the `data/static/` is not already available, run ` python auditort_stim/static_sound.py`
to create the static audio files. 

## Usage
### Steps to Run Stimuli App
1. Open `config.yml` and make changes according to instructions.
2. Open Terminal & relocate to `brain-waves-2.0` directory
3. Activate Environment
```bash
conda activate eeg
```
4. Run Streamlit App:    
*Normal Mode*

```bash
python -m streamlit run gui_stimulus.py
```    

  *Test Mode*  
To enable test mode, use the `--test` flag:    

```bash
python -m streamlit run gui_stimulus.py -- --test
```

## Configuration

To set up the project, you need to modify specific parts of the `config.yml` file:

### Required Changes

In the `config.yml` file, locate the section marked with `# INPUT CHANGES HERE` and update the following fields:

- **`os`**: Set this to your operating system, either `'os'` for macOS, `'windows'` for Windows, or `'linux'` for Linux.
  
  ```bash
  os: 'os' # 'windows' or 'linux' or 'os'
  ```


## Contributing
- Nguyen Ha (Summer 2024)
- Khanh Ha (Summer 2024)

## License
- The license under which the project is distributed.
- Link to the full license text.

## Contact
- Dr. Peter Schwab, MD