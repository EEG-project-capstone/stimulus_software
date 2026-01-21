import os
import tkinter as tk
import pandas as pd
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import logging
import time
from lib.edf_parser import EDFParser

logger = logging.getLogger('eeg_stimulus')

class AnalysisManager:
    """
    Manages only stimulus-EEG sync detection and preview.
    All analysis (CMD, language) is handled offline in notebooks.
    """
    def __init__(self, app_instance):
        self.app = app_instance

    def run_sync_detection_and_preview(self):
        """Detect stimulus sync point in EDF and display a preview plot.
        Called from the Patient Info tab to verify alignment between
        stimulus events (CSV) and EEG data (EDF).
        Stores sync_time in app instance and logs it to the stimulus CSV.
        """
        edf_file_path = self.app.edf_file_path
        stimulus_csv_path = self.app.stimulus_file_path

        if not edf_file_path or not stimulus_csv_path:
            error_msg = "EDF file or Stimulus CSV not selected. Please use 'Use Selected Files' first."
            self.app.root.after(0, lambda: messagebox.showerror("File Error", error_msg))
            return

        # Create parser and load EDF
        parser = EDFParser(edf_file_path)
        try:
            parser.load_edf()
        except Exception as e:
            error_msg = f"Failed to load EDF file:\n{e}"
            self.app.root.after(0, lambda: messagebox.showerror("EDF Load Error", error_msg))
            return

        # Detect sync point using the stimulus CSV
        try:
            parser.find_sync_point(stimulus_csv_path)
        except Exception as e:
            logger.error(f"Sync detection failed: {e}", exc_info=True)
            error_msg = f"Sync detection failed:\n{e}"
            self.app.root.after(0, lambda: messagebox.showerror("Sync Error", error_msg))
            return

        # Store sync time in app and log to CSV
        if parser.sync_time is not None:
            self.app.sync_time = parser.sync_time
            # Extract patient ID from CSV filename (robust to underscores in ID)
            csv_basename = os.path.basename(stimulus_csv_path)
            # Expected format: {PatientID}_{YYYY-MM-DD}_stimulus_results.csv
            parts = csv_basename.split('_')
            if len(parts) >= 3:
                patient_id = '_'.join(parts[:-2])  # Handles IDs like "P101_subj"
            else:
                patient_id = "Unknown"

            sync_row = {
                'patient_id': patient_id,
                'date': time.strftime("%Y-%m-%d"),
                'trial_type': 'sync_detection',
                'sentences': '',
                'start_time': '',
                'end_time': '',
                'duration': '',
                'notes': f"SYNC_TIME_EDF_SEC={parser.sync_time:.6f}"
            }
            sync_df = pd.DataFrame([sync_row])
            sync_df.to_csv(stimulus_csv_path, mode='a', header=False, index=False)
            logger.info(f"Sync time {parser.sync_time:.3f}s saved to {stimulus_csv_path}")
        else:
            self.app.sync_time = None
            logger.warning("No sync artifact detected.")

        # Prepare 60s preview segment starting at sync_time (or 0 if not found)
        info = parser.get_info_summary()
        plot_start_time = parser.sync_time if parser.sync_time is not None else 0.0
        duration = min(60, info['duration'] - plot_start_time)
        if duration <= 0:
            duration = min(60, info['duration'])
            plot_start_time = max(0, info['duration'] - duration)

        channels_to_plot = info['ch_names']
        try:
            data, times = parser.get_data_segment(
                start_sec=plot_start_time,
                duration_sec=duration,
                ch_names=channels_to_plot
            )
        except Exception as e:
            error_msg = f"Failed to extract EEG segment for preview:\n{e}"
            self.app.root.after(0, lambda: messagebox.showerror("Preview Error", error_msg))
            return

        plot_data = {
            'data': data,
            'times': times,
            'channels': channels_to_plot,
            'file_path': edf_file_path,
            'sync_time': parser.sync_time,
            'plot_start_time': plot_start_time
        }

        self.app.root.after(0, self._show_sync_preview_plot, plot_data)

    def _show_sync_preview_plot(self, plot_data):
        """Display a preview plot of EEG with sync point marked."""
        data = plot_data['data']
        times = plot_data['times']
        channels_to_plot = plot_data['channels']
        edf_file_path = plot_data['file_path']
        sync_time = plot_data.get('sync_time')

        fig, ax = plt.subplots(figsize=(15, 8))
        cmap = plt.cm.get_cmap('hsv', len(channels_to_plot))

        for i, (ch_name, ch_data) in enumerate(zip(channels_to_plot, data)):
            color = cmap(i / len(channels_to_plot))
            ax.plot(times, ch_data, linewidth=1.0, label=ch_name, color=color)

        title_suffix = f" (Sync at {sync_time:.3f}s)" if sync_time is not None else " (Sync Not Found)"
        ax.set_title(f'EEG Preview: {times[0]:.1f}s to {times[-1]:.1f}s{title_suffix}')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude ($\mu V$)')

        if sync_time is not None:
            ax.axvline(x=sync_time, color='red', linestyle='--', linewidth=2, label=f'Sync Point ({sync_time:.3f}s)')

        ax.legend(loc='upper right', ncol=4, fontsize=8)
        ax.set_xlim(times[0], times[-1])
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_window = tk.Toplevel(self.app.root)
        plot_window.title(f"Sync Preview: {os.path.basename(edf_file_path)}")
        plot_window.geometry("1200x800")

        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        def on_closing():
            plt.close(fig)
            plot_window.destroy()
        plot_window.protocol("WM_DELETE_WINDOW", on_closing)