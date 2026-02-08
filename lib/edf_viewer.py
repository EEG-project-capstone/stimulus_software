# lib/edf_viewer.py

"""Interactive EDF signal viewer with channel selection."""

import tkinter as tk
from tkinter import ttk
import logging

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np

from lib.edf_parser import EDFParser

logger = logging.getLogger('eeg_stimulus.edf_viewer')

DEFAULT_WINDOW_SEC = 10
MIN_WINDOW_SEC = 1
MAX_WINDOW_SEC = 60
DEFAULT_SCALE_UV = 100  # microvolts per channel spacing


class EDFViewerWindow:
    """Toplevel window for interactive EDF viewing with channel selection."""

    def __init__(self, parent, parser: EDFParser):
        self.parent = parent
        self.parser = parser
        self.info = parser.get_info_summary()

        self.ch_names = self.info['ch_names']
        self.sfreq = self.info['sfreq']
        self.total_duration = self.info['duration']

        self.window_sec = DEFAULT_WINDOW_SEC
        self.t0 = 0.0
        self.scale_uv = DEFAULT_SCALE_UV

        # Channel visibility: {name: BooleanVar}
        self.ch_vars = {}

        self._build_ui()
        self._update_plot()

    def _build_ui(self):
        self.win = tk.Toplevel(self.parent)
        self.win.title(f"EDF Viewer: {self.parser.edf_path.name}")
        self.win.geometry("1400x800")
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        main = ttk.PanedWindow(self.win, orient=tk.HORIZONTAL)
        main.pack(fill='both', expand=True)

        # Left: channel panel
        self._build_channel_panel(main)

        # Right: plot + controls
        right = ttk.Frame(main)
        main.add(right, weight=4)

        self._build_plot_area(right)
        self._build_controls(right)

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
        canvas = tk.Canvas(frame, width=160, highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient='vertical', command=canvas.yview)
        self._ch_inner = ttk.Frame(canvas)

        self._ch_inner.bind(
            '<Configure>',
            lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=self._ch_inner, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side='left', fill='both', expand=True, padx=(5, 0), pady=5)
        scrollbar.pack(side='right', fill='y', pady=5)

        for name in self.ch_names:
            var = tk.BooleanVar(value=True)
            self.ch_vars[name] = var
            ttk.Checkbutton(
                self._ch_inner, text=name, variable=var,
                command=self._on_channel_toggle
            ).pack(anchor='w')

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

    # ── Controls ───────────────────────────────────────────────────

    def _build_controls(self, parent):
        ctrl = ttk.Frame(parent)
        ctrl.pack(fill='x', padx=10, pady=6)

        # Time slider
        ttk.Label(ctrl, text="Time (s):").pack(side='left')
        max_t = max(0, self.total_duration - self.window_sec)
        self.time_slider = tk.Scale(
            ctrl, from_=0, to=max_t, orient=tk.HORIZONTAL,
            resolution=1, command=self._on_scroll, length=500)
        self.time_slider.pack(side='left', fill='x', expand=True, padx=(4, 10))

        # Window size
        ttk.Label(ctrl, text="Window (s):").pack(side='left')
        self.window_spin = tk.Spinbox(
            ctrl, from_=MIN_WINDOW_SEC, to=MAX_WINDOW_SEC,
            width=4, command=self._on_window_change)
        self.window_spin.delete(0, 'end')
        self.window_spin.insert(0, str(self.window_sec))
        self.window_spin.bind('<Return>', lambda e: self._on_window_change())
        self.window_spin.pack(side='left', padx=(4, 10))

        # Amplitude scale
        ttk.Label(ctrl, text="Scale:").pack(side='left')
        ttk.Button(ctrl, text="-", width=2,
                   command=self._scale_down).pack(side='left', padx=2)
        self.scale_label = ttk.Label(ctrl, text=f"{self.scale_uv} \u00b5V")
        self.scale_label.pack(side='left', padx=2)
        ttk.Button(ctrl, text="+", width=2,
                   command=self._scale_up).pack(side='left', padx=2)

    def _on_scroll(self, value):
        self.t0 = float(value)
        self._update_plot()

    def _on_window_change(self):
        try:
            val = int(float(self.window_spin.get()))
            val = max(MIN_WINDOW_SEC, min(MAX_WINDOW_SEC, val))
        except ValueError:
            val = DEFAULT_WINDOW_SEC
        self.window_sec = val

        max_t = max(0, self.total_duration - self.window_sec)
        self.time_slider.configure(to=max_t)
        if self.t0 > max_t:
            self.t0 = max_t
            self.time_slider.set(self.t0)

        self._update_plot()

    def _scale_up(self):
        self.scale_uv = max(10, self.scale_uv // 2)
        self.scale_label.config(text=f"{self.scale_uv} \u00b5V")
        self._update_plot()

    def _scale_down(self):
        self.scale_uv = min(2000, self.scale_uv * 2)
        self.scale_label.config(text=f"{self.scale_uv} \u00b5V")
        self._update_plot()

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
        n_ch = len(active)
        offsets = np.arange(n_ch) * self.scale_uv
        colors = plt.cm.tab20(np.linspace(0, 1, max(n_ch, 1)))

        for i in range(n_ch):
            self.ax.plot(times, data_uv[i] + offsets[i],
                         linewidth=0.7, color=colors[i])

        self.ax.set_yticks(offsets)
        self.ax.set_yticklabels(active, fontsize=8)
        self.ax.set_xlim(times[0], times[-1])
        self.ax.set_ylim(-self.scale_uv, offsets[-1] + self.scale_uv)
        self.ax.set_xlabel("Time (s)")
        self.ax.grid(True, alpha=0.3)

        self.canvas.draw_idle()

    # ── Cleanup ────────────────────────────────────────────────────

    def _on_close(self):
        plt.close(self.fig)
        self.win.destroy()
