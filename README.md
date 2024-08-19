Brain Waves Stimulus Package 2.0

### Create CONDA env
```bash
conda create -n "cheme"
conda activate cheme
conda install pip
pip install -r requirements.txt
```

### To install mpv package
#### For Windows
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

#### For MacOS
```bash
brew install mpv
```

### For Linux
```bash
sudo apt-get mpv
```

### Run streamlit app
```bash
python -m streamlit run gui_stimulus.py
```

