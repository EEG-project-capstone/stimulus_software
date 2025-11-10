# lib/analysis_manager.py

import os
import threading
import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import logging

# Assume these modules contain the core analysis logic
from lib.cmd_analysis import CMDAnalyzer
from lib.edf_parser import EDFParser

# --- Configure logging ---
# You can configure this globally in your main app if desired
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnalysisManager:
    """
    Manages EEG analysis tasks, running them in background threads
    and updating the UI through callbacks.
    """
    def __init__(self, app_instance):
        """
        Initialize the analysis manager.

        Parameters:
        - app_instance: The TkApp instance, used to access file paths and GUI elements.
        """
        self.app = app_instance  # Store the reference to the TkApp instance

    def run_cmd_analysis(self, stimulus_file_path, edf_file_path, bad_channels, eog_channels, n_permutations=100):
        """
        Starts the CMD analysis in a separate thread to keep the GUI responsive.
        """
        # Clear and inform the user
        self.app.analysis_results_text.delete(1.0, tk.END)
        self.app.analysis_results_text.insert(tk.END, "Running CMD Analysis... Please wait (This may take a minute).")

        # Start the heavy computation in a new thread
        thread = threading.Thread(
            target=self._run_cmd_analysis_worker,
            args=(stimulus_file_path, edf_file_path, bad_channels, eog_channels, n_permutations)
        )
        thread.daemon = True  # Allows the application to exit even if the thread is still running
        thread.start()

    def _run_cmd_analysis_worker(self, stimulus_file_path, edf_file_path, bad_channels, eog_channels, n_permutations):
        """Worker function for running CMD analysis."""
        try:
            # --- File Existence Check ---
            if not os.path.exists(stimulus_file_path):
                error_msg = f"Stimulus CSV file not found: {stimulus_file_path}"
                self._update_results_text(error_msg)
                return
            if not os.path.exists(edf_file_path):
                error_msg = f"EDF file not found: {edf_file_path}"
                self._update_results_text(error_msg)
                return

            print(f"DEBUG: Parsed bad channels (strings): {bad_channels}")
            print(f"DEBUG: Parsed EOG channels (strings): {eog_channels}")

            # Create analyzer instance
            analyzer = CMDAnalyzer(
                eeg_path=edf_file_path,
                stimulus_csv_path=stimulus_file_path,
                bad_channels=bad_channels,  # List of strings
                eog_channels=eog_channels   # List of strings
            )

            # --- Load and preprocess EEG data ---
            try:
                analyzer.load_and_preprocess_eeg()
            except Exception as e:
                error_msg = f"Failed to load EEG data: {str(e)}"
                self._update_results_text(error_msg)
                return

            # Check if raw data loaded successfully
            if analyzer.raw is None:
                error_msg = "EEG data failed to load (raw is None)"
                print(f"ERROR: {error_msg}")
                self._update_results_text(error_msg)
                return

            # --- GUARDED DEBUG PRINTS (raw) ---
            if analyzer.raw is not None:
                print(f"DEBUG: Loaded {len(analyzer.raw.ch_names)} EEG channels")
                print(f"DEBUG: EEG sampling rate: {analyzer.raw.info['sfreq']} Hz")
                print(f"DEBUG: EEG duration: {len(analyzer.raw.times) / analyzer.raw.info['sfreq']:.2f} seconds")

            # --- Load stimulus events ---
            try:
                analyzer.load_stimulus_events()
            except Exception as e:
                error_msg = f"Failed to load stimulus events: {str(e)}"
                self._update_results_text(error_msg)
                return

            # Check if events loaded successfully
            if analyzer.metadata is None or len(analyzer.metadata) == 0:
                error_msg = "No stimulus events found in CSV file"
                print(f"ERROR: {error_msg}")
                self._update_results_text(error_msg)
                return

            # --- GUARDED DEBUG PRINTS (metadata) ---
            if analyzer.metadata is not None:
                print(f"DEBUG: Found {len(analyzer.events)} events")
                print(f"DEBUG: Metadata shape: {analyzer.metadata.shape}")
                print(f"DEBUG: Unique move values: {analyzer.metadata['move'].unique()}")
                print(f"DEBUG: Unique instruction types: {analyzer.metadata['instruction_type'].unique()}")

            # --- Create epochs and compute features ---
            try:
                analyzer.create_epochs()
                analyzer.compute_psd_features()
            except Exception as e:
                error_msg = f"Analysis failed during epoching/feature computation: {str(e)}"
                self._update_results_text(error_msg)
                return

            # --- Run analysis (The permutation heavy part) ---
            print("DEBUG: Running analysis...")
            try:
                results = analyzer.run_analysis(n_permutations=n_permutations)
            except Exception as e:
                error_msg = f"Analysis failed during run: {str(e)}"
                self._update_results_text(error_msg)
                return

            # --- Plotting Section ---
            print("DEBUG: Generating CMD analysis plots...")
            try:
                # Create plots in the main thread context (using Tkinter's after)
                # We pass the data needed by the plot function
                plot_args = (results, analyzer.raw.ch_names if analyzer.raw else [], bad_channels)
                self.app.root.after(0, self._show_cmd_plots, *plot_args)
            except Exception as plot_e:
                print(f"ERROR generating plots: {plot_e}")
                # Log traceback but continue to display results text

            # --- Display results in the GUI ---
            result_text = "=== CMD ANALYSIS RESULTS ===\n\n"
            result_text += f"AUC: {results['auc']:.3f} ± {results['auc_std']:.3f}\n"
            result_text += f"P-value: {results['p_value']:.3f}\n"
            result_text += f"Is CMD: {'Yes' if results['is_cmd'] else 'No'}\n"
            if analyzer.metadata is not None:
                result_text += f"Total epochs: {analyzer.metadata.shape[0]}\n"
            if analyzer.raw is not None:
                result_text += f"EEG channels used: {len(analyzer.raw.ch_names)}\n"
            result_text += f"Bad channels excluded: {bad_channels}\n"

            self._update_results_text(result_text)
            print("DEBUG: Analysis completed successfully!")

        except Exception as e:
            error_msg = f"An unexpected error occurred during analysis: {str(e)}"
            self._update_results_text(error_msg)

    def _show_cmd_plots(self, results, ch_names, bad_channels_list):
        """Handles plot creation and display in the main UI thread."""
        # --- Create Figure ---
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'CMD Analysis Results\nAUC: {results["auc"]:.3f}, P-value: {results["p_value"]:.3f}', fontsize=14)

        # 1. Distribution of CV AUCs
        scores = results.get('scores', [])
        if scores:
            axes[0, 0].hist(scores, bins=10, edgecolor='black')
            axes[0, 0].axvline(x=0.5, color='red', linestyle='--', label='Chance (0.5)')
            axes[0, 0].set_xlabel('Cross-Validation AUC')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Distribution of CV AUCs')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, 'CV scores not available', horizontalalignment='center', verticalalignment='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Distribution of CV AUCs (N/A)')

        # 2. Permutation Distribution
        perm_scores = results.get('permutation_scores', [])
        if perm_scores:
            axes[0, 1].hist(perm_scores, bins=50, edgecolor='black', alpha=0.7, label='Permuted AUCs')
            axes[0, 1].axvline(x=results['auc'], color='red', linestyle='-', linewidth=2, label=f'Observed AUC ({results["auc"]:.3f})')
            axes[0, 1].set_xlabel('AUC')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Permutation Distribution')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'Permutation scores not available', horizontalalignment='center', verticalalignment='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Permutation Distribution (N/A)')

        # 3. Classification Map Placeholder
        axes[1, 0].text(0.5, 0.5, 'Classification Map\n(Requires spatial info & weights)', horizontalalignment='center', verticalalignment='center', transform=axes[1, 0].transAxes, fontsize=12)
        axes[1, 0].set_title('Classification Map (TBD)')
        axes[1, 0].axis('off')

        # 4. Feature Importance Placeholder
        axes[1, 1].text(0.5, 0.5, 'Feature Importance\n(e.g., PSD band power)', horizontalalignment='center', verticalalignment='center', transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('Feature Importance (TBD)')
        axes[1, 1].axis('off')

        # FIX: tuple for the 'rect' argument
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))

        # We must use Toplevel for plot display in Tkinter
        plot_window = tk.Toplevel(self.app.root)
        plot_window.title("CMD Analysis Plots")
        plot_window.geometry("1200x800")

        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        def on_closing():
            plt.close(fig)  # Close the specific figure
            plot_window.destroy()
        plot_window.protocol("WM_DELETE_WINDOW", on_closing)


    def run_edf_parser(self, edf_file_path, bad_channels, eog_channels):
        """
        Starts the EDF parsing and plotting in a separate thread.
        """
        self.app.analysis_results_text.delete(1.0, tk.END)
        self.app.analysis_results_text.insert(tk.END, "Loading EDF data and searching for first stimulus artifact... Please wait.")

        # Start the heavy computation in a new thread
        thread = threading.Thread(
            target=self._run_edf_parser_worker,
            args=(edf_file_path, bad_channels, eog_channels)
        )
        thread.daemon = True
        thread.start()

    def _run_edf_parser_worker(self, edf_file_path, bad_channels, eog_channels):
        """Worker function for EDF parsing and butterfly plotting."""
        try:
            # Check if file exists
            if not os.path.exists(edf_file_path):
                error_msg = f"EDF file not found: {edf_file_path}"
                self._update_results_text(error_msg)
                return

            # Create parser instance and load
            parser = EDFParser(edf_file_path)
            parser.load_edf()

            # --- NEW: Detect sync point ---
            stimulus_csv_path = self.app.stimulus_file_path # Use the official path from the app
            if stimulus_csv_path and os.path.exists(stimulus_csv_path):
                 parser.find_sync_point(stimulus_csv_path)
            else:
                 logger.warning("Stimulus CSV path not available or not found. Cannot detect sync point.")
                 parser.sync_time = 0.0 # Default to start if sync not found

            # Get info
            info = parser.get_info_summary()
            # channel_types = parser.get_channel_types() # Example if needed

            # Format and display in the results text box
            info_text = "=== EDF FILE INFORMATION ===\n\n"
            for key, value in dict(info).items():
                info_text += f"{key.replace('_', ' ').title()}: {value}\n"

            # --- Add Sync Point Info ---
            if parser.sync_time is not None:
                info_text += f"\nDetected Sync Time (Start of first command trial): {parser.sync_time:.3f} seconds\n"
            else:
                info_text += f"\n[WARNING] Could not detect stimulus onset artifact. Plotting from EDF start.\n"
                info_text += "Please ensure EDF file is aligned with stimulus presentation or use a trigger.\n"
            # info_text += "\nChannel Types:\n"
            # for ch_type, count in dict(channel_types).items():
            #     info_text += f"  {ch_type.title()}: {count}\n"
            self._update_results_text(info_text)

            # Use ALL channels
            channels_to_plot = info['ch_names']

            # --- Use Sync Time for Plotting ---
            plot_start_time = parser.sync_time if parser.sync_time is not None else 0.0
            duration = min(60, info['duration'] - plot_start_time) # Plot 60s from sync point, or remaining duration
            if duration <= 0:
                 duration = min(60, info['duration']) # Fallback if sync time is at or past end
                 plot_start_time = max(0, info['duration'] - duration)
                 logger.warning(f"Sync time ({parser.sync_time}) is at or past EDF end ({info['duration']}). Plotting last {duration}s.")

            # Note: The EDFParser.get_data_segment call is computationally heavy
            data, times = parser.get_data_segment(start_sec=plot_start_time, duration_sec=duration, ch_names=channels_to_plot)

            # Prepare data for plotting in the main thread
            plot_data = {
                'data': data,
                'times': times, # This will now be relative to plot_start_time
                'channels': channels_to_plot,
                'file_path': edf_file_path,
                'sync_time': parser.sync_time, # Pass sync time for potential plotting
                'plot_start_time': plot_start_time # Pass the actual start time of the plot segment
            }
            # Use Tkinter's after to schedule the plotting function on the main thread
            self.app.root.after(0, self._show_edf_plot, plot_data)

        except Exception as e:
            error_msg = f"EDF Parse Error:\n{str(e)}"
            self._update_results_text(error_msg)

    def _show_edf_plot(self, plot_data):
        """Handles EDF plot creation and display in the main UI thread."""
        data = plot_data['data']
        times = plot_data['times']
        channels_to_plot = plot_data['channels']
        edf_file_path = plot_data['file_path']
        sync_time = plot_data.get('sync_time', None)
        plot_start_time = plot_data.get('plot_start_time', 0.0)

        # --- Create COLOR-CODED OVERLAID Plot ---
        print("DEBUG: Creating color-coded overlaid plot (User preference)...")
        fig, ax = plt.subplots(figsize=(15, 8))

        # Define colors using Matplotlib's colormaps for distinct colors
        cmap = plt.cm.get_cmap('hsv', len(channels_to_plot))

        # Plot each channel overlaid
        for i, (ch_name, ch_data) in enumerate(zip(channels_to_plot, data)):
            color = cmap(i / len(channels_to_plot))
            ax.plot(times, ch_data, linewidth=1.0, label=ch_name, color=color)

        # Customize the plot
        ax.set_title(f'Raw EEG Data - {times[0]:.1f}s to {times[-1]:.1f}s (from sync time {sync_time:.3f}s)')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude ($\mu V$)')

        # Add a vertical line at the sync time (relative to the plot's x-axis, which starts at plot_start_time)
        if sync_time is not None:
             ax.axvline(x=sync_time, color='red', linestyle='--', linewidth=2, label=f'Sync Point ({sync_time:.3f}s)')

        # Add a legend to show which color belongs to which channel
        ax.legend(loc='upper right', ncol=4, fontsize=8)

        # Set limits back to just the data range
        ax.set_xlim(times[0], times[-1])

        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        # We must use Toplevel for plot display in Tkinter
        plot_window = tk.Toplevel(self.app.root)
        plot_window.title(f"EEG Overlaid Plot: {os.path.basename(edf_file_path)}")
        plot_window.geometry("1200x800")

        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        def on_closing():
            plt.close(fig)
            plot_window.destroy()
        plot_window.protocol("WM_DELETE_WINDOW", on_closing)
    
    def run_language_tracking(self):
        """Placeholder for language tracking."""
        self._update_results_text("Language tracking not yet implemented in AnalysisManager.")

    def _update_results_text(self, text):
        """
        Safely updates the Tkinter text widget from a worker thread.
        Since Tkinter is not thread-safe, this uses the main thread's after method.
        """
        # Schedule the update on the main thread
        self.app.root.after(0, self.__do_update_text, text)

    def __do_update_text(self, text):
        """Actual function to update the Tkinter text widget on the main thread."""
        self.app.analysis_results_text.delete(1.0, tk.END)
        self.app.analysis_results_text.insert(tk.END, text)

    def run_sync_detection_and_preview(self, progress_callback=None):
        """
        Runs the EDF loading, sync point detection, and a preview plot.
        Intended to be called from the Patient Info tab.
        """
        # Use the official paths stored in the app instance
        edf_file_path = self.app.edf_file_path
        stimulus_csv_path = self.app.stimulus_file_path

        if not edf_file_path or not stimulus_csv_path:
            error_msg = "EDF file or Stimulus CSV not selected. Please use 'Select Files' in Patient Information tab first."
            self.app.root.after(0, lambda: messagebox.showerror("File Error", error_msg))
            return

        # Create parser instance and load EDF
        parser = EDFParser(edf_file_path)
        try:
            parser.load_edf()
        except Exception as e:
            error_msg = f"Failed to load EDF file: {e}"
            self.app.root.after(0, lambda: messagebox.showerror("EDF Load Error", error_msg))
            return

        # Detect sync point using the stimulus CSV
        parser.find_sync_point(stimulus_csv_path)

        # Prepare data for preview plot based on sync point
        info = parser.get_info_summary()
        plot_start_time = parser.sync_time if parser.sync_time is not None else 0.0
        duration = min(60, info['duration'] - plot_start_time) # Plot 60s from sync point, or remaining duration
        if duration <= 0:
             duration = min(60, info['duration']) # Fallback if sync time is at or past end
             plot_start_time = max(0, info['duration'] - duration)
             print(f"Warning: Sync time ({parser.sync_time}) is at or past EDF end ({info['duration']}). Plotting last {duration}s.")

        channels_to_plot = info['ch_names']
        data, times = parser.get_data_segment(start_sec=plot_start_time, duration_sec=duration, ch_names=channels_to_plot)

        # Prepare plot data
        plot_data = {
            'data': data,
            'times': times,
            'channels': channels_to_plot,
            'file_path': edf_file_path,
            'sync_time': parser.sync_time,
            'plot_start_time': plot_start_time
        }

        # Schedule plot creation on the main thread
        self.app.root.after(0, self._show_sync_preview_plot, plot_data)

    def _show_sync_preview_plot(self, plot_data):
        """Handles the preview plot creation and display in the main UI thread."""
        data = plot_data['data']
        times = plot_data['times']
        channels_to_plot = plot_data['channels']
        edf_file_path = plot_data['file_path']
        sync_time = plot_data.get('sync_time', None)
        plot_start_time = plot_data.get('plot_start_time', 0.0)

        print("DEBUG: Creating sync preview plot...")
        fig, ax = plt.subplots(figsize=(15, 8))

        # Define colors
        cmap = plt.cm.get_cmap('hsv', len(channels_to_plot))
        for i, (ch_name, ch_data) in enumerate(zip(channels_to_plot, data)):
            color = cmap(i / len(channels_to_plot))
            ax.plot(times, ch_data, linewidth=1.0, label=ch_name, color=color)

        title_sync_str = f" (Sync at {sync_time:.3f}s)" if sync_time is not None else " (Sync Not Found)"
        ax.set_title(f'EEG Preview: {times[0]:.1f}s to {times[-1]:.1f}s{title_sync_str}')
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
