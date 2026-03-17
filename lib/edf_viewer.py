# lib/edf_viewer.py

"""Interactive EDF signal viewer with channel selection."""

import datetime
import math
import tkinter as tk
from tkinter import ttk, messagebox
import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
from scipy.signal import iirnotch, filtfilt

from lib.edf_parser import EDFParser
from lib.constants import STIMULUS_TYPE_DISPLAY_NAMES

logger = logging.getLogger('eeg_stimulus.edf_viewer')

# Colors per stimulus type for event annotations
STIM_COLORS = {
    'language':       '#2196F3',   # blue
    'right_command':  '#4CAF50',   # green
    'right_command+p':'#4CAF50',
    'left_command':   '#9C27B0',   # purple
    'left_command+p': '#9C27B0',
    'oddball':        '#FF9800',   # orange
    'oddball+p':      '#FF9800',
    'familiar':       '#E91E63',   # magenta
    'unfamiliar':     '#795548',   # brown
}

DEFAULT_WINDOW_SEC = 300
MIN_WINDOW_SEC = 1
MAX_WINDOW_SEC = 3600
DEFAULT_SCALE_UV = 50000  # microvolts per channel spacing

# Stepped presets for window (s) and scale (µV) — arrow buttons walk up/down these lists
WINDOW_STEPS = [1, 2, 5, 10, 20, 30, 60, 120, 180, 300, 600, 1800, 3600]
SCALE_STEPS  = [10, 25, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 25000, 50000, 100000, 200000]


class EDFViewerWindow:
    """Toplevel window for interactive EDF viewing with channel selection."""

    def __init__(self, parent, parser: EDFParser,
                 stimulus_path: Optional[Path] = None):
        self.parent = parent
        self.parser = parser
        self.stimulus_path = stimulus_path
        self.info = parser.get_info_summary()

        self.ch_names = self.info['ch_names']
        self.sfreq = self.info['sfreq']
        self.total_duration = self.info['duration']

        self.window_sec = DEFAULT_WINDOW_SEC
        self.t0 = 0.0
        self.scale_uv = DEFAULT_SCALE_UV

        # Timestamp from EDF header (may be None)
        self.meas_date = self.info.get('meas_date')

        # Channel visibility: {name: BooleanVar}
        self.ch_vars = {}

        # Notch filter toggles (Hz -> BooleanVar)
        self.notch_vars = {}

        # Clock time vs relative time toggle
        self.use_clock_time = tk.BooleanVar(value=True)

        # Detected sync pulse time (EDF-relative seconds), None until detected
        self.sync_time_sec: Optional[float] = None
        self.sync_end_sec: Optional[float] = None

        # CSV stimulus events aligned to EDF time (populated after sync is known)
        self.stim_events: list = []

        self._build_ui()
        self._load_existing_sync()
        self._update_plot()

    def _build_ui(self):
        self.win = tk.Toplevel(self.parent)
        self.win.title(f"EDF Viewer: {self.parser.edf_path.name}")
        self.win.geometry("1400x850")
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        main = ttk.PanedWindow(self.win, orient=tk.HORIZONTAL)
        main.pack(fill='both', expand=True)

        # Left: channel panel
        self._build_channel_panel(main)

        # Right: controls pinned to bottom, then plot fills remaining space
        right = ttk.Frame(main)
        main.add(right, weight=4)

        # Build controls first so pack reserves space at the bottom before
        # the canvas claims everything with expand=True.
        self._build_controls(right)
        self._build_plot_area(right)

    # ── Channel panel ──────────────────────────────────────────────

    def _build_channel_panel(self, parent):
        frame = ttk.Frame(parent, width=180)
        parent.add(frame, weight=0)

        ttk.Label(frame, text="Channels", font=('TkDefaultFont', 11, 'bold')).pack(
            pady=(8, 4))

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill='x', padx=5)
        ttk.Button(btn_frame, text="All", width=6,
                   command=self._select_all).pack(side='left', padx=2)
        ttk.Button(btn_frame, text="None", width=6,
                   command=self._deselect_all).pack(side='left', padx=2)

        # Scrollable checkbox list
        self._ch_canvas = tk.Canvas(frame, width=160, highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient='vertical',
                                  command=self._ch_canvas.yview)
        self._ch_inner = ttk.Frame(self._ch_canvas)

        self._ch_inner.bind(
            '<Configure>',
            lambda e: self._ch_canvas.configure(
                scrollregion=self._ch_canvas.bbox('all')))
        self._ch_canvas.create_window((0, 0), window=self._ch_inner, anchor='nw')
        self._ch_canvas.configure(yscrollcommand=scrollbar.set)

        self._ch_canvas.pack(side='left', fill='both', expand=True,
                              padx=(5, 0), pady=5)
        scrollbar.pack(side='right', fill='y', pady=5)

        # Mouse-wheel scrolling (macOS / Windows use delta; Linux uses Button-4/5)
        def _on_ch_scroll(event):
            if event.num == 4:
                self._ch_canvas.yview_scroll(-1, 'units')
            elif event.num == 5:
                self._ch_canvas.yview_scroll(1, 'units')
            elif event.delta:
                self._ch_canvas.yview_scroll(
                    -1 if event.delta > 0 else 1, 'units')

        for widget in (self._ch_canvas, self._ch_inner):
            widget.bind('<MouseWheel>', _on_ch_scroll)
            widget.bind('<Button-4>', _on_ch_scroll)
            widget.bind('<Button-5>', _on_ch_scroll)

        for name in self.ch_names:
            var = tk.BooleanVar(value=False)
            self.ch_vars[name] = var
            cb = ttk.Checkbutton(
                self._ch_inner, text=name, variable=var,
                command=self._on_channel_toggle)
            cb.pack(anchor='w')
            cb.bind('<MouseWheel>', _on_ch_scroll)
            cb.bind('<Button-4>', _on_ch_scroll)
            cb.bind('<Button-5>', _on_ch_scroll)

    def _select_all(self):
        for var in self.ch_vars.values():
            var.set(True)
        self._update_plot()

    def _deselect_all(self):
        for var in self.ch_vars.values():
            var.set(False)
        self._update_plot()

    # ── Plot area ──────────────────────────────────────────────────

    def _build_plot_area(self, parent):
        self.fig, self.ax = plt.subplots(figsize=(12, 7))
        self.fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.08)

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        toolbar = NavigationToolbar2Tk(self.canvas, parent)
        toolbar.update()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        # Mouse-wheel scrolls through time on the plot
        def _on_plot_scroll(event):
            if event.num == 4 or (event.delta and event.delta > 0):
                direction = -1  # scroll back in time
            else:
                direction = 1   # scroll forward in time
            step = max(1, self.window_sec // 4)
            new_t0 = max(0.0, min(self.t0 + direction * step,
                                  self.total_duration - self.window_sec))
            if new_t0 != self.t0:
                self.t0 = new_t0
                self.time_slider.set(self.t0)
                self._update_plot()

        self.canvas.get_tk_widget().bind('<MouseWheel>', _on_plot_scroll)
        self.canvas.get_tk_widget().bind('<Button-4>', _on_plot_scroll)
        self.canvas.get_tk_widget().bind('<Button-5>', _on_plot_scroll)

    # ── Controls ───────────────────────────────────────────────────

    def _build_controls(self, parent):
        # Row 2: notch filters + clock time — pinned to bottom first
        ctrl2 = ttk.Frame(parent)
        ctrl2.pack(side='bottom', fill='x', padx=10, pady=(0, 4))

        ttk.Label(ctrl2, text="Notch:").pack(side='left')
        for freq in (60, 120, 180):
            var = tk.BooleanVar(value=False)
            self.notch_vars[freq] = var
            ttk.Checkbutton(
                ctrl2, text=f"{freq} Hz", variable=var,
                command=self._update_plot
            ).pack(side='left', padx=4)

        ttk.Separator(ctrl2, orient='vertical').pack(side='left', fill='y', padx=8)

        clock_cb = ttk.Checkbutton(
            ctrl2, text="Clock Time", variable=self.use_clock_time,
            command=self._update_plot)
        clock_cb.pack(side='left', padx=4)
        if self.meas_date is None:
            clock_cb.config(state='disabled')

        ttk.Separator(ctrl2, orient='vertical').pack(side='left', fill='y', padx=8)

        self.sync_btn = ttk.Button(ctrl2, text="Detect Sync",
                                   command=self._detect_sync)
        self.sync_btn.pack(side='left', padx=4)

        self.sync_status_label = ttk.Label(ctrl2, text="", foreground='green')
        self.sync_status_label.pack(side='left', padx=4)

        # Row 1: time slider, window, scale — above row 2
        ctrl = ttk.Frame(parent)
        ctrl.pack(side='bottom', fill='x', padx=10, pady=(4, 0))

        # Time slider
        ttk.Label(ctrl, text="Time (s):").pack(side='left')
        max_t = max(0, self.total_duration - self.window_sec)
        self.time_slider = tk.Scale(
            ctrl, from_=0, to=max_t, orient=tk.HORIZONTAL,
            resolution=1, command=self._on_scroll, length=400)
        self.time_slider.pack(side='left', fill='x', expand=True, padx=(4, 10))

        # Window size: [-] [entry] [+]  — steps through WINDOW_STEPS
        ttk.Label(ctrl, text="Window (s):").pack(side='left')
        ttk.Button(ctrl, text="−", width=2,
                   command=self._window_decrease).pack(side='left', padx=2)
        self.window_entry = ttk.Entry(ctrl, width=6)
        self.window_entry.insert(0, str(self.window_sec))
        self.window_entry.bind('<Return>', lambda e: self._on_window_entry())
        self.window_entry.bind('<FocusOut>', lambda e: self._on_window_entry())
        self.window_entry.pack(side='left', padx=2)
        ttk.Button(ctrl, text="+", width=2,
                   command=self._window_increase).pack(side='left', padx=(2, 10))

        # Amplitude scale: [-] [entry] [+]  — steps through SCALE_STEPS
        ttk.Label(ctrl, text="Scale (µV):").pack(side='left')
        ttk.Button(ctrl, text="−", width=2,
                   command=self._scale_decrease).pack(side='left', padx=2)
        self.scale_entry = ttk.Entry(ctrl, width=8)
        self.scale_entry.insert(0, str(self.scale_uv))
        self.scale_entry.bind('<Return>', lambda e: self._on_scale_entry())
        self.scale_entry.bind('<FocusOut>', lambda e: self._on_scale_entry())
        self.scale_entry.pack(side='left', padx=2)
        ttk.Button(ctrl, text="+", width=2,
                   command=self._scale_increase).pack(side='left', padx=2)

    def _on_scroll(self, value):
        self.t0 = float(value)
        self._update_plot()

    # ── Window helpers ─────────────────────────────────────────────

    def _on_window_entry(self):
        try:
            val = int(float(self.window_entry.get()))
            val = max(MIN_WINDOW_SEC, min(int(self.total_duration), val))
        except ValueError:
            val = self.window_sec
        self._apply_window(val)

    def _window_decrease(self):
        steps = [s for s in WINDOW_STEPS if s < self.window_sec]
        self._apply_window(steps[-1] if steps else MIN_WINDOW_SEC)

    def _window_increase(self):
        max_win = max(MIN_WINDOW_SEC, int(self.total_duration))
        steps = [s for s in WINDOW_STEPS if s > self.window_sec and s <= max_win]
        self._apply_window(steps[0] if steps else max_win)

    def _apply_window(self, val: int):
        self.window_sec = val
        self.window_entry.delete(0, 'end')
        self.window_entry.insert(0, str(self.window_sec))
        max_t = max(0, self.total_duration - self.window_sec)
        self.time_slider.configure(to=max_t)
        if self.t0 > max_t:
            self.t0 = max_t
            self.time_slider.set(self.t0)
        self._update_plot()

    # ── Scale helpers ──────────────────────────────────────────────

    def _on_scale_entry(self):
        try:
            val = int(float(self.scale_entry.get()))
            val = max(1, val)
        except ValueError:
            val = self.scale_uv
        self._apply_scale(val)

    def _scale_decrease(self):
        steps = [s for s in SCALE_STEPS if s < self.scale_uv]
        self._apply_scale(steps[-1] if steps else SCALE_STEPS[0])

    def _scale_increase(self):
        steps = [s for s in SCALE_STEPS if s > self.scale_uv]
        self._apply_scale(steps[0] if steps else SCALE_STEPS[-1])

    def _apply_scale(self, val: int):
        self.scale_uv = val
        self.scale_entry.delete(0, 'end')
        self.scale_entry.insert(0, str(self.scale_uv))
        self._update_plot()

    # ── Sync detection ─────────────────────────────────────────────

    def _compute_stim_events(self):
        """Align CSV stimulus events to EDF time using the sync pulse as reference."""
        self.stim_events = []
        if self.stimulus_path is None or self.sync_time_sec is None:
            return
        try:
            import pandas as pd
            df = pd.read_csv(self.stimulus_path)

            # Find the DAC timestamp of the sync pulse trigger
            sync_rows = df[(df['stim_type'] == 'manual_sync_pulse') &
                           df['start_time'].notna()]
            if sync_rows.empty:
                logger.warning("No manual_sync_pulse row found — cannot align events")
                return
            sync_dac_time = float(sync_rows.iloc[0]['start_time'])
            offset = self.sync_time_sec - sync_dac_time

            skip = {'manual_sync_pulse', 'sync_detection', 'session_note'}
            for _, row in df.iterrows():
                stype = str(row.get('stim_type', ''))
                if stype in skip:
                    continue
                try:
                    if pd.isna(row.get('start_time')):
                        continue
                    t_start = float(row['start_time']) + offset
                    t_end = (float(row['end_time']) + offset
                             if pd.notna(row.get('end_time')) else None)
                    self.stim_events.append({
                        'stim_type': stype,
                        'edf_start': t_start,
                        'edf_end':   t_end,
                        'notes':     str(row.get('notes', '')),
                    })
                except (ValueError, TypeError):
                    continue

            logger.info(f"Aligned {len(self.stim_events)} stimulus events to EDF timeline "
                        f"(offset={offset:.3f}s)")
        except Exception as e:
            logger.warning(f"Could not compute stim events: {e}")

    def _load_existing_sync(self):
        """If the stimulus CSV already has a sync_detection row, load it silently."""
        if self.stimulus_path is None:
            return
        try:
            import pandas as pd
            df = pd.read_csv(self.stimulus_path)
            if 'stim_type' not in df.columns:
                return
            hits = df[df['stim_type'] == 'sync_detection']
            if hits.empty:
                return
            row = hits.iloc[0]
            start = float(row.get('start_time', float('nan')))
            end   = float(row.get('end_time',   float('nan')))
            if not math.isnan(start):
                self.sync_time_sec = start
                self.sync_end_sec  = end if not math.isnan(end) else None
                self.sync_status_label.config(
                    text=f"Sync loaded from CSV @ {start:.2f}s", foreground='blue')
                self.sync_btn.config(state='disabled', text="Sync Detected")
                logger.info(f"Loaded existing sync detection from CSV: {start:.3f}s")
                self._compute_stim_events()
        except Exception as e:
            logger.warning(f"Could not load existing sync from CSV: {e}")


    def _detect_sync(self):
        if self.stimulus_path is None:
            messagebox.showwarning(
                "No CSV Selected",
                "Please select a stimulus CSV file before detecting the sync pulse.\n"
                "The result will be saved into that file.",
                parent=self.win)
            return

        self.sync_btn.config(state='disabled', text="Detecting…")
        self.win.update_idletasks()

        try:
            result = self.parser.detect_sync_pulse(ch_name='DC7')
        except Exception as e:
            messagebox.showerror("Detection Error", str(e), parent=self.win)
            self.sync_btn.config(state='normal', text="Detect Sync")
            return

        if result is None:
            messagebox.showwarning(
                "Not Found",
                "No sync pulse detected on DC7.\n"
                "Check that the correct channel is present and the signal is large enough.",
                parent=self.win)
            return

        self.sync_time_sec = result['start_sec']
        self.sync_end_sec = result['end_sec']

        # Jump viewer to the sync pulse
        max_t = max(0.0, self.total_duration - self.window_sec)
        self.t0 = max(0.0, min(self.sync_time_sec - self.window_sec / 2, max_t))
        self.time_slider.set(self.t0)

        # Save to CSV
        try:
            self._save_sync_to_csv(result['start_sec'], result['end_sec'],
                                   result['channel'])
            status = f"Sync @ {result['start_sec']:.2f}s  (saved to CSV)"
            self.sync_status_label.config(text=status, foreground='green')
            self.sync_btn.config(state='disabled', text="Sync Detected")
            self._compute_stim_events()
            logger.info(f"Sync detection saved: {status}")
        except Exception as e:
            self.sync_status_label.config(
                text=f"Detected @ {result['start_sec']:.2f}s  (CSV save failed)",
                foreground='red')
            logger.error(f"Failed to save sync detection to CSV: {e}", exc_info=True)

        self._update_plot()

    def _save_sync_to_csv(self, start_sec: float, end_sec: float, channel: str):
        import pandas as pd

        df = pd.read_csv(self.stimulus_path)
        patient_id = (df['patient_id'].dropna().iloc[0]
                      if not df.empty and 'patient_id' in df.columns
                      else 'unknown')

        note = f"Sync pulse detected on {channel}"
        row = {
            'patient_id': patient_id,
            'date': datetime.date.today().isoformat(),
            'stim_type': 'sync_detection',
            'notes': note,
            'start_time': start_sec,
            'end_time': end_sec,
        }
        pd.DataFrame([row]).to_csv(
            self.stimulus_path, mode='a', header=False, index=False)

    # ── Plotting ───────────────────────────────────────────────────

    def _on_channel_toggle(self):
        self._update_plot()

    def _update_plot(self):
        active = [name for name in self.ch_names if self.ch_vars[name].get()]
        self.ax.clear()

        if not active:
            self.ax.set_title("No channels selected")
            self.canvas.draw_idle()
            return

        data, times = self.parser.get_data_segment(
            start_sec=self.t0,
            duration_sec=self.window_sec,
            ch_names=active
        )

        if data.size == 0:
            self.ax.set_title("No data in range")
            self.canvas.draw_idle()
            return

        data_uv = data * 1e6  # V -> µV

        # Apply active notch filters
        for freq, var in self.notch_vars.items():
            if var.get() and freq < self.sfreq / 2:
                b, a = iirnotch(freq, Q=30, fs=self.sfreq)
                data_uv = filtfilt(b, a, data_uv, axis=1)

        n_ch = len(active)
        offsets = np.arange(n_ch) * self.scale_uv
        colors = plt.cm.tab20(np.linspace(0, 1, max(n_ch, 1)))

        # X-axis: clock time or relative seconds
        if self.use_clock_time.get() and self.meas_date is not None:
            def _clock_fmt(x, pos):
                t = self.meas_date + datetime.timedelta(seconds=float(x))
                return t.strftime('%H:%M:%S')
            xlabel = "Clock Time"
            self.ax.xaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(_clock_fmt))
        else:
            xlabel = "Time (s)"
            self.ax.xaxis.set_major_formatter(
                matplotlib.ticker.ScalarFormatter())

        for i in range(n_ch):
            trace = data_uv[i] - np.mean(data_uv[i])  # remove DC for display
            self.ax.plot(times, trace + offsets[i],
                         linewidth=0.7, color=colors[i])

        self.ax.set_yticks(offsets)
        self.ax.set_yticklabels(active, fontsize=8)
        self.ax.set_xlim(times[0], times[-1])
        self.ax.set_ylim(-self.scale_uv, offsets[-1] + self.scale_uv)
        self.ax.set_xlabel(xlabel)
        self.ax.grid(True, alpha=0.3)

        # Stimulus event annotations
        legend_seen: set = set()
        for ev in self.stim_events:
            t = ev['edf_start']
            if not (times[0] <= t <= times[-1]):
                continue
            stype = ev['stim_type']
            color = STIM_COLORS.get(stype, '#888888')
            label = (STIMULUS_TYPE_DISPLAY_NAMES.get(stype, stype)
                     if stype not in legend_seen else '_')
            legend_seen.add(stype)
            self.ax.axvline(t, color=color, linewidth=0.7, alpha=0.5, label=label)
            if ev['edf_end'] is not None:
                end = min(ev['edf_end'], times[-1])
                self.ax.axvspan(t, end, color=color, alpha=0.04)

        if legend_seen:
            self.ax.legend(loc='upper right', fontsize=7, framealpha=0.8,
                           handlelength=1, borderpad=0.5)

        # Sync pulse annotation — single marker line + subtle span
        if self.sync_time_sec is not None and times[0] <= self.sync_time_sec <= times[-1]:
            self.ax.axvline(self.sync_time_sec, color='red', linewidth=0.8,
                            linestyle='--', alpha=0.35)
            if self.sync_end_sec is not None:
                end = min(self.sync_end_sec, times[-1])
                self.ax.axvspan(self.sync_time_sec, end, alpha=0.03, color='red')

            y_top = offsets[-1] + self.scale_uv * 0.85
            if self.use_clock_time.get() and self.meas_date is not None:
                t_label = (self.meas_date +
                           datetime.timedelta(seconds=self.sync_time_sec)
                           ).strftime('%H:%M:%S')
            else:
                t_label = f"{self.sync_time_sec:.2f}s"
            self.ax.text(self.sync_time_sec, y_top, f" sync {t_label}",
                         color='red', fontsize=7, alpha=0.5, va='top', clip_on=True)

        self.canvas.draw_idle()

    # ── Cleanup ────────────────────────────────────────────────────

    def _on_close(self):
        plt.close(self.fig)
        self.win.destroy()
