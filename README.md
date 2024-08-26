# Brain Waves Stimulus Package 2.0
## A. Setup Environment
### 1. Create CONDA env
```bash
conda create -n "eeg"
conda activate eeg
conda install pip
pip install -r requirements.txt
```

### 2. Install mpv package (*for Windows*)
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
### 3. Instal ffmpeg (*for Windows*)
Install [ffmpeg](https://github.com/BtbN/FFmpeg-Builds/releases) to current directory.    
Extract and rename to `ffmpeg`.    
Make sure to have these 2 files:    
`ffmpeg\bin\ffmpeg.exe`    
`ffmpeg\bin\ffprobe.exe`

## B. Streamlit App Usage
### *Steps to run Stimuli App:*
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

