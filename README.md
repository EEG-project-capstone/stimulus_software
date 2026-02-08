# Stimulus Software

A clinical-grade auditory stimulus administration platform for EEG research.

Built with Python and Tkinter, this software provides a cross-platform graphical interface for delivering precisely timed auditory stimuli during EEG recording sessions. Designed for TBI and disorders-of-consciousness research, it supports multiple experimental paradigms, real-time session management, and structured data export for downstream analysis.

<!-- markdownlint-disable MD033 -->
<img width="1000" alt="Stimulus Software GUI" src="https://github.com/user-attachments/assets/eeb55158-1a26-47cf-b6da-322d35dd44e1" />
<!-- markdownlint-enable MD033 -->

---

## Table of Contents

- [Stimulus Paradigms](#stimulus-paradigms)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Organization](#data-organization)
- [Testing](#testing)
- [Configuration](#configuration)
- [Analysis Tools](#analysis-tools)
- [References](#references)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Stimulus Paradigms

The software supports five auditory stimulus types, each targeting a different aspect of neural responsiveness. The paradigms are based on published methods for assessing covert cognition in patients with disorders of consciousness (see [References](#references)).

| Paradigm              | Description                                                            | Trials       | Key Parameters                     |
| --------------------- | ---------------------------------------------------------------------- | ------------ | ---------------------------------- |
| **Language**          | Randomized sentence playback for speech-tracking assessment            | 72           | 1.2-2.2s randomized ISI            |
| **Motor Command**     | "Keep"/"Stop" instructions for left or right hand movement             | 8 cycles     | 10s keep / 10s stop phases         |
| **Oddball**           | Standard (1000 Hz) and deviant (2000 Hz) tones for MMN/P300 detection  | 25 tones     | 80/20 ratio, 1000ms onset-to-onset |
| **Loved One Voice**   | Patient's loved one's recorded voice for emotional response assessment | 50           | Gender-matched control comparison  |
| **Control Statement** | Neutral statement baseline (male/female)                               | configurable | Comparison condition               |

Motor command and oddball paradigms each have prompt and no-prompt variants, allowing researchers to assess instruction comprehension alongside the primary stimulus.

---

## Features

### Stimulus Administration

- **Randomized trial sequencing** with configurable stimulus combinations per session
- **Precise audio timing** with latency compensation and callback-based playback
- **Real-time session display** showing stimulus queue, progress, and completion status
- **Pause/resume/stop controls** for flexible session management
- **Manual sync pulse** (1s, 100 Hz square wave) for EEG time-alignment

### Patient Data Management

- Patient/EEG ID tracking with per-session results logging
- Automatic duplicate-session detection to prevent redundant administration
- Searchable history of previously administered stimuli
- Session notes with persistent storage

### Analysis Integration

- CSV export of trial-level data (stimulus type, timing, duration) compatible with MATLAB, Python, and R
- EDF file import and sync-point detection via MNE-Python
- Built-in EDF viewer for quick signal inspection
- Structured output directories for command-following and language-tracking analyses

### Cross-Platform

Compatible with **Windows**, **macOS**, and **Linux** (including ChromeOS with Linux enabled).

---

## Installation

### Prerequisites

- [Anaconda](https://docs.anaconda.com/anaconda/install/) or [Miniconda](https://docs.anaconda.com/miniconda/)
- Python 3.12+

### Setup

```bash
./setup.sh
```

This creates a conda environment (`eeg`), installs Tkinter, and installs all Python dependencies. Safe to re-run.

### Dependencies

| Package       | Purpose                                |
| ------------- | -------------------------------------- |
| `pandas`      | Results data management                |
| `pyyaml`      | Configuration file parsing             |
| `pydub`       | Audio format handling (MP3/WAV)        |
| `sounddevice` | Low-latency audio playback             |
| `numpy`       | Tone generation and signal processing  |
| `mne`         | EDF file reading and EEG processing    |

---

## Usage

### Running the Application

```bash
./run.sh
```

### Typical Session Workflow

1. **Enter the patient/EEG ID** in the Administer Stimuli tab
2. **Select stimulus types** using the checkboxes (language, motor commands, oddball, loved one voice)
3. **Click "Prepare Stimulus"** to randomize and load the trial sequence
4. **Review the stimulus queue** displayed in the session table
5. **Click "Play Stimulus"** to begin administration
6. **Use Pause/Resume/Stop** as needed during the session
7. **Send a sync pulse** before or after stimulus delivery for EEG alignment
8. **Add session notes** in the text area for any observations
9. Results are automatically saved to `patient_data/results/`

### Viewing Previous Data

Switch to the **Patient Information** tab to:

- Search for previously administered stimuli by patient ID
- Load and preview EDF recordings
- Detect sync points in EEG data
- View analysis results

---

## Project Structure

```text
stimulus_software/
├── main.py                        # Application entry point
├── setup.sh                       # One-time environment setup
├── run.sh                         # Launch the application
├── config.yml                     # File paths and settings
├── requirements.txt               # Python dependencies
│
├── lib/                           # Core application code
│   ├── app.py                     # Main Tkinter GUI (tabbed interface)
│   ├── config.py                  # YAML config loading and validation
│   ├── constants.py               # Enums, timing parameters, clinical scales
│   ├── exceptions.py              # Custom exception hierarchy
│   ├── state_manager.py           # Playback state machine
│   ├── stims.py                   # Stimulus randomization and sequencing
│   ├── auditory_stimulator.py     # Playback orchestration
│   ├── audio_stream_manager.py    # Low-level audio device control
│   ├── stim_handlers.py           # Per-paradigm handler implementations
│   ├── base_stim_handler.py       # Abstract handler base class
│   ├── results_manager.py         # Thread-safe CSV result writing
│   ├── analysis_manager.py        # EDF sync detection and analysis
│   ├── edf_parser.py              # EDF file reading
│   ├── edf_viewer.py              # GUI EDF signal viewer
│   ├── logging_utils.py           # Logging context managers
│   └── assets/                    # UI icons (play, pause, stop)
│
├── tests/                         # Pytest test suite
│
├── audio_data/                    # Stimulus audio files
│   ├── sentences/                 # Language trial recordings
│   ├── prompts/                   # Instruction audio (motor, oddball)
│   ├── static/                    # Command audio, control statements, beeps
│   ├── lang_trials/               # Language trial metadata
│   └── Trimmed/                   # Trimmed audio files
│
├── doc/                           # Protocol documentation and guides
│
├── patient_data/                  # Session output (gitignored)
│   ├── edfs/                      # EEG recordings (.edf)
│   └── results/                   # Stimulus results (.csv) and analysis output
│
└── logs/                          # Rotating application logs
```

---

## Data Organization

Session results and analysis outputs are stored under `patient_data/`:

```text
patient_data/
├── edfs/
│   └── [PatientID]_[Date].edf           # Raw EEG recordings
│
└── results/
    ├── [PatientID]_[Date]_stimulus_results.csv   # Trial-level timing data
    │
    ├── cmd/                              # Motor command analysis
    │   └── [PatientID]_[Date]/
    │       ├── log.txt                   # AUC scores and analysis log
    │       ├── cross_validation_performance.png
    │       ├── EEG_spatial_patterns.png
    │       ├── average_predicted_probability.png
    │       ├── epochs_during_instructions.png
    │       └── permutation_*.png
    │
    └── lang_tracking/                    # Language tracking analysis
        └── [PatientID]/
            ├── ITPC_[Channel].png        # Per-channel ITPC plots (21 channels)
            └── avg_itpc_plot.png         # Averaged ITPC summary
```

> **Re-running analysis:** Delete the corresponding `[PatientID]_[Date]/` folder under `cmd/` to regenerate results.

---

## Testing

The project includes a pytest-based test suite covering state management, configuration, audio playback, stimulus generation, EDF parsing, and UI controls.

```bash
conda activate eeg
pytest tests/
```

---

## Configuration

All file paths are defined in [config.yml](config.yml):

```yaml
# Output paths
edf_dir: 'patient_data/edfs/'
result_dir: 'patient_data/results/'

# Audio input paths
sentences_path: 'audio_data/sentences/'
right_keep_path: 'audio_data/static/right_keep.mp3'
# ... (see config.yml for full listing)
```

Key timing parameters are defined in `lib/constants.py` and include inter-stimulus intervals, tone durations, command cycle timing, and oddball probabilities.

---

## Analysis Tools

Post-session EEG analysis is handled by the companion [`eeg-auditory-stimulus`](https://github.com/EEG-project-capstone/eeg-auditory-stimulus) package, which can be installed separately:

```bash
pip install git+https://github.com/EEG-project-capstone/eeg-auditory-stimulus.git
```

This package provides two main analysis modules:

- **`claassen_analysis`** — Cognitive-motor dissociation (CMD) detection using SVM classification of EEG responses to motor commands. Produces cross-validation performance plots, spatial pattern maps, and permutation test results. Based on the approach described in Claassen et al. (2019).
- **`rodika_modularized`** — Language tracking analysis via inter-trial phase coherence (ITPC) of EEG responses to isochronous speech streams. Generates per-channel and averaged ITPC plots at the word, phrase, and sentence frequencies. Based on the approach described in Sokoliuk et al. (2021).

Analysis results are saved to `patient_data/results/cmd/` and `patient_data/results/lang_tracking/` respectively.

---

## References

1. Claassen, J., Doyle, K., Matory, A., Couch, C., Burger, K. M., Velazquez, A., Rohaut, B., et al. (2019). Detection of brain activation in unresponsive patients with acute brain injury. *New England Journal of Medicine*, 380(26), 2497–2505. [doi:10.1056/NEJMoa1812757](https://doi.org/10.1056/NEJMoa1812757)

2. Sokoliuk, R., Degano, G., Banellis, L., Melloni, L., Hayton, T., Sturman, S., Veenith, T., Yakoub, K. M., Belli, A., Noppeney, U., & Cruse, D. (2021). Covert speech comprehension predicts recovery from acute unresponsive states. *Annals of Neurology*, 89(4), 646–656. [doi:10.1002/ana.25995](https://doi.org/10.1002/ana.25995)

3. Bekinschtein, T. A., Dehaene, S., Rohaut, B., Tadel, F., Cohen, L., & Naccache, L. (2009). Neural signature of the conscious processing of auditory regularities. *Proceedings of the National Academy of Sciences*, 106(5), 1672–1677. [doi:10.1073/pnas.0809667106](https://doi.org/10.1073/pnas.0809667106)

---

## Contributing

| Contributor     | Period                |
| --------------- | --------------------- |
| Nguyen Ha       | Summer 2024           |
| Khanh Ha        | Summer 2024           |
| Joobee Jung     | Fall 2024             |
| Trisha Prasant  | Fall 2024             |
| Joe Moore       | Spring 2025 – Present |

---

## License

See [LICENSE](LICENSE) for details.

---

## Contact

Dr. Peter Schwab, MD

For more details on the auditory stimulus protocol, see the [stimulus documentation](https://github.com/EEG-project-capstone/brain-waves-2.0/blob/main/doc/EEG-TBI_AuditoryStimulusDetails.pdf).
