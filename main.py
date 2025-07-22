"""
GUI Stimulus Package
This script provides a graphical user interface (GUI) for administering auditory stimuli to patients, managing patient records, and adding notes. 
It utilizes the Tkinter library to create a GUI interface.

Output:
    The script generates and saves data to 'patient_df.csv' and 'patient_notes.csv' files.
"""

import tkinter as tk
from lib.stimulus_package_notes import add_notes, add_history
from lib.tkinter_application import TkApp

# from eeg_auditory_stimulus import rodika_modularized
# from eeg_auditory_stimulus import claassen_analysis

if __name__ == "__main__":

    # Run the tkinter application
    root = tk.Tk()
    app = TkApp(root)
    root.mainloop()