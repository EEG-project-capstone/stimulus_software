# Brain Waves Stimulus Package 2.0

## Introduction

## Features


## Installation
### *Prerequisites*
- Download `conda`, `python`
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
#### 3. Instal ffmpeg (*for Windows*)
Install [ffmpeg](https://github.com/BtbN/FFmpeg-Builds/releases) to current directory.    
Extract and rename to `ffmpeg`.    
Make sure to have these 2 files:    
`ffmpeg\bin\ffmpeg.exe`    
`ffmpeg\bin\ffprobe.exe`

## Usage
### Steps to Run Stimuli App
1. Open `config.yml` and make changes according to instructions.
2. Open Terminal & relocate to `brain-waves-2.0` directory
3. Activate Environment
```bash
conda activate eeg
```
4. Run Streamlit App
```bash
python -m streamlit run gui_stimulus.py
```

## Configuration

To set up the project, you need to modify specific parts of the `config.yml` file:

### Required Changes

In the `config.yml` file, locate the section marked with `# INPUT CHANGES HERE` and update the following fields:

- **`os`**: Set this to your operating system, either `'os'` for macOS/Linux or `'windows'` for Windows.
  
  ```bash
  os: 'os'  # Change to 'os' or 'windows' depending on your system
  ```

## Examples

## Contributing
- Nguyen Ha (Summer 2024)
- Khanh Ha (Summer 2024)

## License
- The license under which the project is distributed.
- Link to the full license text.

## Contact
- Dr. Peter Schwab, MD