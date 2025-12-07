# Copyright (c) mrmilbe

"""Interactive tuning GUI for rb_waveform_lab.

- Lets you load one audio file.
- Provides sliders/fields for key WaveformAnalysisConfig and render params.
- Renders a 4-band Rekordbox-style waveform preview with zoom.
- Prints simple performance timings for analysis and rendering.

Run with:

    python tuning_gui.py
"""

from __future__ import annotations

import json
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from time import perf_counter
from typing import Optional

from pathlib import Path

import numpy as np
from PIL import ImageTk

from rb_waveform_lab.analysis import analyze_bands, analysis_from_rb_waveform, WaveformAnalysis
from rb_waveform_lab.config import (
    DEFAULT_ANALYSIS_CONFIG,
    DEFAULT_COLOR_CONFIG,
    DEFAULT_RENDER_CONFIG,
    WaveformAnalysisConfig,
    WaveformColorConfig,
    WaveformRenderConfig,
    parse_band_order,
    config_to_dict,
    dict_to_config,
)
from rb_waveform_lab.analysis import resolve_audio_path
from rb_waveform_lab.ANLZ import (
    analyze_anlz_folder,
    extract_beat_grid_downbeats,
    load_audio_for_analysis,
    load_waveform_for_rekordbox,
)
from rb_waveform_lab.antialias import (
    AntialiasCache,
    get_antialiased_waveform,
    reset_antialias_cache,
)
from rb_waveform_lab.playhead import (
    PrerenderCache,
    TimingInfo,
    WindowPlan,
    compute_timing_info,
    compute_window_plan,
    finalize_preview_image,
    get_pan_slider_state,
    render_window_image,
    reset_prerender_cache,
)
from rb_waveform_lab.rkbx_link_listener import DeckEvent, RekordboxLinkListener, format_link_status


# Discrete zoom levels in seconds (index 0 = most zoomed in)
# Fixed absolute durations so zoom behaves consistently across tracks.
ZOOM_LEVELS_SECONDS = [256, 196, 128, 96, 64, 48, 32, 24, 16]
NUM_ZOOM_STEPS = len(ZOOM_LEVELS_SECONDS) - 1


DEFAULT_DECK_PATH = Path(
    r"C:/Users/chris/AppData/Roaming/Pioneer/rekordbox/share/PIONEER/USBANLZ/a82/484d8-6208-4fd8-9406-4393f48abd78"
)

LIBRARY_SEARCH_ROOT: Optional[Path] = Path(r"C:\Rekordbox")
if not LIBRARY_SEARCH_ROOT.is_dir():
    LIBRARY_SEARCH_ROOT = None


class TuningApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("rb_waveform_lab Tuning GUI")

        # State
        self.audio_path: Optional[Path] = None
        self.resolved_path: Optional[Path] = None
        self.signal = None
        self.sr: int = 0
        self.duration: float = 0.0

        self.resolved_path_var = tk.StringVar(value="")

        self.analysis_cfg: WaveformAnalysisConfig = DEFAULT_ANALYSIS_CONFIG
        self.color_cfg: WaveformColorConfig = DEFAULT_COLOR_CONFIG
        self.render_cfg: WaveformRenderConfig = DEFAULT_RENDER_CONFIG

        self.mode_var = tk.StringVar(value="analysis")
        self.current_analysis_raw: Optional[WaveformAnalysis] = None
        self.current_image_tk: Optional[ImageTk.PhotoImage] = None
        self.cached_anlz_waveform: Optional[np.ndarray] = None
        self.cached_anlz_duration: Optional[float] = None
        self.beat_grid: Optional[list] = None
        self.original_bpm: Optional[float] = None  # BPM from ANLZ file

        self.render_smoothing_var = tk.IntVar(value=5)
        self.render_detail_var = tk.IntVar(value=20000)  # Max prerender detail
        self.render_low_gain_var = tk.DoubleVar(value=1.0)
        self.render_lowmid_gain_var = tk.DoubleVar(value=1.0)
        self.render_midhigh_gain_var = tk.DoubleVar(value=1.0)
        self.render_high_gain_var = tk.DoubleVar(value=1.0)
        self.analysis_widgets = []
        self.live_render_var = tk.BooleanVar(value=True)
        self.prerender_cache = PrerenderCache()
        self.antialias_cache = AntialiasCache()
        self.use_link_var = tk.BooleanVar(value=False)
        self.link_deck_var = tk.IntVar(value=0)
        self.link_status_var = tk.StringVar(value="RB Link: disabled")
        self.link_listener: Optional[RekordboxLinkListener] = None
        self._link_poll_job: Optional[str] = None
        self.link_last_anlz_path: Optional[str] = None
        self.link_current_time: Optional[float] = None
        self.link_time_scale: float = 1.0
        self.link_current_bpm: Optional[float] = None  # Live BPM from OSC
        self.link_original_bpm: Optional[float] = None  # Original BPM from OSC
        self.link_poll_interval_ms = 50
        self._is_rendering: bool = False  # Frame skip flag

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._update_analysis_widget_state()

        # If a default ANLZ folder exists, prefill and auto-load it (no popup)
        if DEFAULT_DECK_PATH.is_dir():
            self.audio_entry.delete(0, tk.END)
            self.audio_entry.insert(0, str(DEFAULT_DECK_PATH))

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(1, weight=1)

        # --- Top: file + control buttons ---
        top = ttk.Frame(main)
        top.grid(row=0, column=0, columnspan=2, sticky="ew")
        top.columnconfigure(0, weight=1)

        path_row = ttk.Frame(top)
        path_row.grid(row=0, column=0, columnspan=6, sticky="ew")
        path_row.columnconfigure(1, weight=1)
        ttk.Label(path_row, text="ANLZ folder:").grid(row=0, column=0, sticky="w")
        self.audio_entry = ttk.Entry(path_row)
        self.audio_entry.grid(row=0, column=1, sticky="ew", padx=(4, 4))
        ttk.Button(path_row, text="Browse...", command=self.on_browse).grid(row=0, column=2, padx=(4, 0))
        ttk.Button(path_row, text="Load", command=self.on_load_audio).grid(row=0, column=3)
        ttk.Button(path_row, text="Save Config", command=self.on_save_config).grid(row=0, column=4, padx=(8, 0))
        ttk.Button(path_row, text="Load Config", command=self.on_load_config).grid(row=0, column=5, padx=(4, 0))

        mode_row = ttk.Frame(top)
        mode_row.grid(row=1, column=0, columnspan=4, sticky="w", pady=(4, 0))
        ttk.Label(mode_row, text="Source:").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            mode_row,
            text="Analyze audio",
            value="analysis",
            variable=self.mode_var,
            command=self._on_mode_change,
        ).grid(row=0, column=1, padx=(6, 0))
        ttk.Radiobutton(
            mode_row,
            text="Use Rekordbox PWV",
            value="rekordbox",
            variable=self.mode_var,
            command=self._on_mode_change,
        ).grid(row=0, column=2, padx=(6, 0))

        link_row = ttk.Frame(top)
        link_row.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(4, 0))
        self.link_check = ttk.Checkbutton(
            link_row,
            text="Follow Rekordbox Link",
            variable=self.use_link_var,
            command=self._on_link_toggle,
        )
        self.link_check.grid(row=0, column=0, sticky="w")
        ttk.Label(link_row, text="Deck:").grid(row=0, column=1, padx=(8, 0))
        self.deck_combo_var = tk.StringVar(value=str(self.link_deck_var.get()))
        deck_combo = ttk.Combobox(
            link_row,
            values=["0", "1", "2", "3"],
            width=4,
            state="readonly",
            textvariable=self.deck_combo_var,
        )
        deck_combo.grid(row=0, column=2, sticky="w")
        deck_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_link_deck_change(self.deck_combo_var.get()))
        self.deck_combo = deck_combo
        link_row.columnconfigure(3, weight=1)
        ttk.Label(link_row, textvariable=self.link_status_var).grid(row=0, column=3, sticky="w", padx=(10, 0))

        # --- Left: parameters ---
        params = ttk.Frame(main)
        params.grid(row=1, column=0, sticky="nsw", pady=(8, 0))

        self._build_analysis_params(params)
        self._build_render_params(params)

        # --- Right: preview + timings ---
        right = ttk.Frame(main)
        right.grid(row=1, column=1, sticky="nsew", padx=(8, 0))

        # Zoom and pan controls
        zoom_frame = ttk.Frame(right)
        zoom_frame.grid(row=0, column=0, sticky="ew")

        ttk.Label(zoom_frame, text="Zoom (seconds):").grid(row=0, column=0, sticky="w")
        # Zoom slider: discrete steps, left = most zoomed in, right = full song
        self.zoom_step_var = tk.IntVar(value=NUM_ZOOM_STEPS)  # Start at full track
        self.zoom_slider = ttk.Scale(
            zoom_frame,
            from_=0,
            to=NUM_ZOOM_STEPS,
            orient="horizontal",
            variable=self.zoom_step_var,
            command=self._on_zoom_step_change,
        )
        self.zoom_slider.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        # Show current zoom window length in seconds
        self.zoom_label_var = tk.StringVar(value="full")
        self.zoom_entry = ttk.Entry(zoom_frame, width=6, textvariable=self.zoom_label_var, state="readonly")
        self.zoom_entry.grid(row=0, column=2, sticky="w", padx=(4, 0))

        ttk.Label(zoom_frame, text="Pan:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.pan_var = tk.DoubleVar(value=0.0)
        self.pan_slider = ttk.Scale(
            zoom_frame,
            from_=0.0,
            to=1.0,
            orient="horizontal",
            variable=self.pan_var,
            command=lambda _v: self._rerender_if_ready(),
        )
        self.pan_slider.grid(row=1, column=1, sticky="ew", padx=(4, 0), pady=(4, 0))
        zoom_frame.columnconfigure(1, weight=1)

        self.live_render_check = ttk.Checkbutton(
            zoom_frame,
            text="Live render",
            variable=self.live_render_var,
            command=self._on_live_render_toggle,
        )
        self.live_render_check.grid(row=2, column=0, columnspan=3, sticky="w", pady=(4, 0))

        # Preview label with image
        self.preview_label = ttk.Label(right, relief="sunken")
        self.preview_label.grid(row=1, column=0, sticky="nsew", pady=(4, 4))
        resolved_label = ttk.Label(
            right,
            textvariable=self.resolved_path_var,
            anchor="w",
            justify="left",
            wraplength=600,
        )
        resolved_label.grid(row=2, column=0, sticky="ew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        # Run + timings
        bottom = ttk.Frame(right)
        bottom.grid(row=3, column=0, sticky="ew")
        bottom.columnconfigure(0, weight=1)
        self.timing_label = ttk.Label(bottom, text="Analysis: - | Render: -")
        self.timing_label.grid(row=0, column=0, sticky="w")
        self.bin_count_label = ttk.Label(bottom, text="Bins: -")
        self.bin_count_label.grid(row=0, column=1, sticky="e")

    def _build_analysis_params(self, parent: ttk.Frame) -> None:
        grp = ttk.LabelFrame(parent, text="Analysis parameters")
        self.analysis_group = grp
        grp.grid(row=0, column=0, sticky="nw", pady=(0, 8))

        # Target bins
        self.bins_var = tk.IntVar(value=self.analysis_cfg.target_bins)
        ttk.Label(grp, text="Target bins").grid(row=0, column=0, sticky="w")
        bins_entry = ttk.Entry(grp, textvariable=self.bins_var, width=8)
        bins_entry.grid(row=0, column=1, sticky="w", padx=(4, 0))
        bins_entry.bind("<KeyRelease>", lambda _e: self._analysis_params_changed())
        bins_entry.bind("<FocusOut>", lambda _e: self._analysis_params_changed())
        self.analysis_widgets.append(bins_entry)

        # Helper for creating a paired slider + entry that stay in sync
        def add_slider_with_entry(
            row: int,
            label: str,
            var: tk.DoubleVar,
            from_: float,
            to: float,
            step: float,
            formatter=str,
        ):
            ttk.Label(grp, text=label).grid(row=row, column=0, sticky="w")
            entry = ttk.Entry(grp, width=8)
            entry.grid(row=row, column=2, sticky="w", padx=(4, 0))

            def on_slider_change(value: str) -> None:
                v = float(value)
                entry.delete(0, tk.END)
                entry.insert(0, formatter(round(v, 2)))
                self._analysis_params_changed()

            scale = ttk.Scale(
                grp,
                from_=from_,
                to=to,
                orient="horizontal",
                variable=var,
                command=on_slider_change,
            )
            scale.grid(row=row, column=1, sticky="ew", padx=(4, 0))
            grp.columnconfigure(1, weight=1)
            self.analysis_widgets.append(scale)
            self.analysis_widgets.append(entry)

            def on_entry_change(event: tk.Event) -> None:
                text = entry.get().strip()
                try:
                    v = float(text)
                except ValueError:
                    return
                v = max(from_, min(to, v))
                var.set(v)
                entry.delete(0, tk.END)
                entry.insert(0, formatter(round(v, 2)))
                self._analysis_params_changed()

            entry.bind("<KeyRelease>", on_entry_change)
            entry.bind("<FocusOut>", on_entry_change)

            # Initialize entry from initial var value
            entry.insert(0, formatter(round(var.get(), 2)))
            return scale, entry

        # Cutoffs
        self.low_cut_var = tk.DoubleVar(value=self.analysis_cfg.low_cutoff_hz)
        self.lowmid_cut_var = tk.DoubleVar(value=self.analysis_cfg.lowmid_cutoff_hz)
        self.midhigh_cut_var = tk.DoubleVar(value=self.analysis_cfg.midhigh_cutoff_hz)

        add_slider_with_entry(1, "Low cutoff Hz", self.low_cut_var, 20.0, 400.0, 1.0)
        add_slider_with_entry(2, "Low-mid cutoff Hz", self.lowmid_cut_var, 100.0, 3000.0, 10.0)
        add_slider_with_entry(3, "Mid-high cutoff Hz", self.midhigh_cut_var, 500.0, 10000.0, 10.0)

        self.comp_strength_var = tk.DoubleVar(value=self.analysis_cfg.compression_strength)
        add_slider_with_entry(4, "Compression strength", self.comp_strength_var, 0.1, 10.0, 0.05)

    def _build_render_params(self, parent: ttk.Frame) -> None:
        grp = ttk.LabelFrame(parent, text="Render parameters")
        grp.grid(row=1, column=0, sticky="nw")

        self.height_var = tk.IntVar(value=self.render_cfg.image_height)
        ttk.Label(grp, text="Image height").grid(row=0, column=0, sticky="w")
        def _on_height_change(_v: str) -> None:
            self._mark_prerender_dirty()
            self._rerender_if_ready()

        height_scale = ttk.Scale(
            grp,
            from_=64,
            to=512,
            orient="horizontal",
            variable=self.height_var,
            command=_on_height_change,
        )
        height_scale.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        grp.columnconfigure(1, weight=1)

        # Band draw order (comma-separated short names: l,lm,mh,h)
        ttk.Label(grp, text="Band order").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.band_order_var = tk.StringVar(value=self.color_cfg.band_order_string)
        band_entry = ttk.Entry(grp, textvariable=self.band_order_var, width=24)
        band_entry.grid(row=1, column=1, columnspan=1, sticky="ew", padx=(4, 0), pady=(4, 0))

        def on_band_order_change(event: tk.Event) -> None:
            text = self.band_order_var.get().strip()
            # Store raw string; parsing is done when updating configs.
            self.color_cfg.band_order_string = text
            self._mark_prerender_dirty()
            self._rerender_if_ready()

        band_entry.bind("<KeyRelease>", on_band_order_change)
        band_entry.bind("<FocusOut>", on_band_order_change)

        # Overview vs symmetric overlaid vs vertically stacked
        self.overview_var = tk.BooleanVar(value=self.color_cfg.overview_mode)
        overview_chk = ttk.Checkbutton(
            grp,
            text="Overview mode",
            variable=self.overview_var,
            command=lambda: (self._mark_prerender_dirty(), self._rerender_if_ready()),
        )
        overview_chk.grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))

        self.stack_bands_var = tk.BooleanVar(value=self.color_cfg.stack_bands)
        stack_chk = ttk.Checkbutton(
            grp,
            text="Stack bands (vertical)",
            variable=self.stack_bands_var,
            command=lambda: (self._mark_prerender_dirty(), self._rerender_if_ready()),
        )
        stack_chk.grid(row=3, column=0, columnspan=2, sticky="w")

        self.beat_grid_var = tk.BooleanVar(value=False)
        beatgrid_chk = ttk.Checkbutton(
            grp,
            text="Beat Grid",
            variable=self.beat_grid_var,
            command=lambda: (self._mark_prerender_dirty(), self._rerender_if_ready()),
        )
        beatgrid_chk.grid(row=4, column=0, columnspan=2, sticky="w")

        ttk.Separator(grp).grid(row=5, column=0, columnspan=2, sticky="ew", pady=(6, 6))

        def add_render_slider(
            row: int,
            label: str,
            var: tk.Variable,
            from_: float,
            to: float,
            fmt: str = "{:.2f}",
        ) -> None:
            ttk.Label(grp, text=label).grid(row=row, column=0, sticky="w")
            value_label = ttk.Label(grp, text=fmt.format(var.get()))
            value_label.grid(row=row, column=2, sticky="w", padx=(6, 0))

            def on_change(value: str) -> None:
                var.set(float(value))
                value_label.configure(text=fmt.format(float(value)))
                self._mark_prerender_dirty()
                self._rerender_if_ready()

            scale = ttk.Scale(
                grp,
                from_=from_,
                to=to,
                orient="horizontal",
                variable=var,
                command=on_change,
            )
            scale.grid(row=row, column=1, sticky="ew", padx=(4, 0))

        add_render_slider(5, "Smoothing (bins)", self.render_smoothing_var, 1.0, 63.0, "{:.0f}")
        add_render_slider(6, "Max detail (px)", self.render_detail_var, 2000.0, 30000.0, "{:.0f}")
        add_render_slider(7, "Low gain", self.render_low_gain_var, 0.1, 5.0)
        add_render_slider(8, "Low-mid gain", self.render_lowmid_gain_var, 0.1, 5.0)
        add_render_slider(9, "Mid-high gain", self.render_midhigh_gain_var, 0.1, 5.0)
        add_render_slider(10, "High gain", self.render_high_gain_var, 0.1, 5.0)

    def _on_close(self) -> None:
        self.use_link_var.set(False)
        self._stop_link_listener()
        self.root.destroy()

    def _link_follow_active(self) -> bool:
        return bool(self.use_link_var.get())

    def _on_link_toggle(self) -> None:
        if self.use_link_var.get():
            try:
                self._start_link_listener()
            except OSError as exc:
                messagebox.showerror("Error", f"Failed to start Rekordbox Link listener:\n{exc}")
                self.use_link_var.set(False)
            else:
                self._rerender_if_ready()
        else:
            self._stop_link_listener()
            self.link_current_time = None
            self.link_time_scale = 1.0
            self._mark_prerender_dirty()
            self._rerender_if_ready()

    def _on_link_deck_change(self, value: str) -> None:
        try:
            deck = int(value)
        except ValueError:
            return
        self.link_deck_var.set(deck)
        self.link_last_anlz_path = None
        self.link_current_time = None
        self.link_time_scale = 1.0
        self.link_status_var.set(f"RB Link deck {deck}: waiting for data")
        self.deck_combo_var.set(value)
        if self.use_link_var.get() and self.link_listener is not None:
            state = self.link_listener.get_state(deck)
            snapshot = state.snapshot("snapshot")
            self._handle_link_event(snapshot)

    def _start_link_listener(self) -> None:
        if self.link_listener is None:
            self.link_listener = RekordboxLinkListener()
        self.link_listener.start()
        self.link_status_var.set(f"RB Link: listening on {self.link_listener.port}")
        self.link_last_anlz_path = None
        self._schedule_link_poll()
        self._on_link_deck_change(self.deck_combo_var.get())

    def _stop_link_listener(self) -> None:
        self._cancel_link_poll()
        if self.link_listener is not None:
            self.link_listener.stop()
            self.link_listener = None
        self.link_status_var.set("RB Link: disabled")

    def _schedule_link_poll(self) -> None:
        self._cancel_link_poll()
        self._poll_link_events()

    def _cancel_link_poll(self) -> None:
        if self._link_poll_job is not None:
            self.root.after_cancel(self._link_poll_job)
            self._link_poll_job = None

    def _on_live_render_toggle(self) -> None:
        self._mark_prerender_dirty()
        self._rerender_if_ready()

    def _on_mode_change(self) -> None:
        mode = self.mode_var.get()
        if mode == "rekordbox":
            if self.cached_anlz_waveform is not None:
                self.current_analysis_raw = analysis_from_rb_waveform(
                    self.cached_anlz_waveform, self.cached_anlz_duration
                )
                self._mark_prerender_dirty()
                self._render_from_current_analysis()
        else:
            self.current_analysis_raw = None
            self._mark_prerender_dirty()
            if self.signal is not None:
                self.on_run()
        self._update_analysis_widget_state()

    def _update_analysis_widget_state(self) -> None:
        state = "disabled" if self.mode_var.get() == "rekordbox" else "normal"
        for widget in getattr(self, "analysis_widgets", []):
            try:
                widget.configure(state=state)
            except tk.TclError:
                continue

    def _mark_prerender_dirty(self) -> None:
        reset_prerender_cache(self.prerender_cache)
        reset_antialias_cache(self.antialias_cache)

    def _on_zoom_step_change(self, _value: str) -> None:
        """Handle discrete zoom step change - snap to integer and rerender."""
        # Snap to nearest integer step
        step = round(self.zoom_step_var.get())
        self.zoom_step_var.set(step)
        self._rerender_if_ready()

    def _get_zoom_window_seconds(self) -> float:
        """Return the desired visible window duration in seconds for this zoom step."""
        step = int(round(self.zoom_step_var.get()))
        step = max(0, min(NUM_ZOOM_STEPS, step))
        return float(ZOOM_LEVELS_SECONDS[step])

    def _rerender_if_ready(self) -> None:
        """Re-render preview when sliders move, if audio is loaded.

        Avoids popping error dialogs while still giving live feedback.
        """
        if self.current_analysis_raw is None:
            if self.mode_var.get() == "analysis" and self.signal is not None:
                self.on_run()
            elif self.mode_var.get() == "rekordbox" and self.cached_anlz_waveform is not None:
                self.current_analysis_raw = analysis_from_rb_waveform(
                    self.cached_anlz_waveform, self.cached_anlz_duration
                )
                self._render_from_current_analysis()
            return
        try:
            self._render_from_current_analysis()
        except Exception:
            # Swallow errors on live updates; user can always press the
            # Analyze + Render button for an explicit run with errors shown.
            pass

    def _render_from_current_analysis(self) -> Optional[float]:
        analysis = self.current_analysis_raw
        if analysis is None:
            return None

        self._update_configs_from_widgets()

        timing = compute_timing_info(analysis, fallback_duration=self.duration)
        if timing.n_bins == 0:
            return None

        link_follow = self._link_follow_active()
        link_time = self.link_current_time if link_follow else None
        link_scale = self.link_time_scale if link_follow else 1.0

        preview_w = max(400, self.preview_label.winfo_width() or 0)
        self.render_cfg.image_width = preview_w
        use_live_render = bool(self.live_render_var.get())

        # Compute window plan from full-resolution analysis using fixed-duration zoom
        window_seconds = self._get_zoom_window_seconds()
        plan = compute_window_plan(
            total_duration=timing.total_duration,
            n_bins=timing.n_bins,
            seconds_per_bin=timing.seconds_per_bin,
            window_duration=window_seconds,
            pan_fraction=float(self.pan_var.get()),
            link_follow=link_follow,
            link_time=link_time,
            link_scale=link_scale,
        )

        # Get antialiased waveform for live render mode (operates on visible window)
        # Prerender uses full-resolution analysis directly
        if use_live_render:
            render_analysis = get_antialiased_waveform(
                self.antialias_cache,
                analysis,
                plan.start_bin,
                plan.window_bins,
                preview_w,
            )
            # Recompute timing from antialiased window
            render_timing = compute_timing_info(render_analysis, fallback_duration=plan.window_duration)
            # Create adjusted plan for live render - analysis is already the extracted window starting at bin 0
            render_plan = WindowPlan(
                start_bin=0,
                window_bins=len(render_analysis.low),
                window_duration=plan.window_duration,
                start_time=plan.start_time,
                zoom_label=plan.zoom_label,
                pan_enabled=plan.pan_enabled,
                playhead_fraction=plan.playhead_fraction,
            )
        else:
            render_analysis = analysis
            render_timing = timing
            render_plan = plan

        # Update UI based on plan (show fixed window duration in seconds)
        self.zoom_label_var.set(f"{window_seconds:.0f}s")
        should_reset_pan, slider_state = get_pan_slider_state(plan, timing.n_bins, link_follow)
        if should_reset_pan:
            self.pan_var.set(0.0)
        if self.pan_slider is not None:
            self.pan_slider.configure(state=slider_state)

        t0 = perf_counter()
        img, cache_out = render_window_image(
            render_analysis,
            self.color_cfg,
            self.render_cfg,
            render_plan,
            seconds_per_bin=render_plan.scaled_seconds_per_bin,
            preview_width=preview_w,
            total_duration=render_timing.total_duration,
            use_live_render=use_live_render,
            cache=self.prerender_cache,
            zoom_fraction=1.0,
            link_scale=link_scale,
            beat_grid=self.beat_grid,
            show_beat_grid=bool(self.beat_grid_var.get()),
        )
        self.prerender_cache = cache_out
        t1 = perf_counter()
        render_ms = (t1 - t0) * 1000.0

        # For live render, the rendered window is described by render_plan;
        # for prerender, plan and render_plan are identical.
        effective_plan = render_plan

        img = finalize_preview_image(
            img,
            target_width=preview_w,
            target_height=self.render_cfg.image_height,
            draw_playhead=link_follow,
            playhead_fraction=effective_plan.playhead_fraction,
        )
        self.current_image_tk = ImageTk.PhotoImage(img)
        self.preview_label.configure(image=self.current_image_tk)
        
        # Get BPM info for debug display
        # Priority: OSC > ANLZ file
        original_bpm = 0.0
        current_bpm = 0.0
        
        if link_follow:
            # Use BPM from OSC when Link is active
            original_bpm = self.link_original_bpm or 0.0
            current_bpm = self.link_current_bpm or 0.0
        else:
            # Use BPM from ANLZ file when no Link
            original_bpm = self.original_bpm or 0.0
            current_bpm = original_bpm
        
        # Get scaled duration
        scaled_duration = timing.total_duration * link_scale if link_follow else timing.total_duration
        
        debug_text = (
            f"Bins: {effective_plan.window_bins} / {timing.n_bins} | "
            f"Dur: {timing.total_duration:.2f}s → {scaled_duration:.2f}s | "
            f"BPM: {original_bpm:.1f} → {current_bpm:.1f} | "
            f"s/bin: {timing.seconds_per_bin:.4f}"
        )
        self.bin_count_label.configure(text=debug_text)
        return render_ms

    def _poll_link_events(self) -> None:
        self._link_poll_job = None
        if not self.use_link_var.get() or self.link_listener is None:
            return
        
        # Frame skipping: if still rendering from previous frame, skip this poll
        if self._is_rendering:
            self._link_poll_job = self.root.after(self.link_poll_interval_ms, self._poll_link_events)
            return
        
        listener = self.link_listener
        while True:
            event = listener.get_event(timeout=0.0)
            if event is None:
                break
            self._handle_link_event(event)
        self._link_poll_job = self.root.after(self.link_poll_interval_ms, self._poll_link_events)

    def _handle_link_event(self, event: DeckEvent) -> None:
        if event.deck != self.link_deck_var.get():
            return
        previous_time = self.link_current_time
        previous_scale = self.link_time_scale
        scale_changed = False
        if event.time_seconds is not None:
            self.link_current_time = event.time_seconds
        
        # Store BPM values from OSC
        if event.current_bpm is not None:
            self.link_current_bpm = event.current_bpm
        if event.original_bpm is not None:
            self.link_original_bpm = event.original_bpm
        
        event_scale = getattr(event, "time_scale", None)
        if event_scale is not None:
            if event_scale != self.link_time_scale:
                self.link_time_scale = event_scale
                scale_changed = True
                self._mark_prerender_dirty()  # Invalidate cache on BPM change
        status = format_link_status(event.deck, self.link_current_time, event.anlz_path)
        self.link_status_var.set(status)
        if event.anlz_path and event.anlz_path != self.link_last_anlz_path:
            self.link_last_anlz_path = event.anlz_path
            print(
                f"[RB Link] Deck {event.deck} new ANLZ path: {event.anlz_path} (source={event.source})"
            )
            self.audio_entry.delete(0, tk.END)
            self.audio_entry.insert(0, event.anlz_path)
            self._load_anlz_folder(Path(event.anlz_path), silent=True)
        if self._link_follow_active() and (
            self.link_current_time is not None
            and (previous_time != self.link_current_time or previous_scale != self.link_time_scale)
        ):
            self._is_rendering = True
            try:
                self._rerender_if_ready()
            finally:
                self._is_rendering = False

    def _report_link_error(self, message: str, silent: bool) -> None:
        log = f"[RB Link] {message}"
        print(log)
        if silent:
            self.link_status_var.set(f"RB Link: {message}")
        else:
            messagebox.showerror("Error", message)

    def _load_anlz_folder(self, folder: Path, silent: bool = False) -> bool:
        self.resolved_path_var.set("")
        self.resolved_path = None
        if not folder.is_dir():
            self._report_link_error(f"Folder not found: {folder}", silent)
            return False
        try:
            analysis_info = analyze_anlz_folder(folder)
        except FileNotFoundError:
            self._report_link_error(f"ANLZ folder not found: {folder}", silent)
            return False
        except Exception as exc:
            self._report_link_error(f"Failed to read ANLZ folder:\n{exc}", silent)
            return False

        self.cached_anlz_waveform = analysis_info.waveform
        self.cached_anlz_duration = analysis_info.duration
        
        # Extract beat grid downbeats (every 4th beat - bar boundaries)
        self.beat_grid = extract_beat_grid_downbeats(folder)
        
        # Extract original BPM from beat grid (stored in ANLZ .DAT file)
        self.original_bpm = None
        if self.beat_grid:
            for entry in self.beat_grid:
                if entry.bpm is not None and entry.bpm > 0:
                    self.original_bpm = entry.bpm
                    break
        
        # Resolve audio path from raw Rekordbox path
        self.resolved_path = resolve_audio_path(analysis_info.song_path, LIBRARY_SEARCH_ROOT)
        if self.resolved_path is not None:
            self.resolved_path_var.set(f"Resolved path: {self.resolved_path}")
        else:
            self.resolved_path_var.set("Resolved path: (not found)")

        mode = self.mode_var.get()
        try:
            if mode == "analysis":
                if self.resolved_path is None:
                    raise FileNotFoundError("Audio file not found for analysis mode")
                audio_data = load_audio_for_analysis(
                    self.resolved_path, self.analysis_cfg.target_sample_rate
                )
                self.audio_path = audio_data.audio_path
                self.signal = audio_data.signal
                self.sr = audio_data.sample_rate
                self.duration = audio_data.duration
            else:
                waveform_data = load_waveform_for_rekordbox(
                    analysis_info, self.resolved_path
                )
                self.audio_path = waveform_data.audio_path
                self.signal = None
                self.sr = 0
                self.duration = waveform_data.duration
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            self._report_link_error(str(exc), silent)
            return False

        self.root.after(120, self.on_run)
        self._mark_prerender_dirty()
        return True

    def _analysis_params_changed(self) -> None:
        if self.mode_var.get() != "analysis":
            return
        self.current_analysis_raw = None
        self._mark_prerender_dirty()
        if self.signal is None:
            return
        try:
            self.on_run()
        except Exception:
            pass

    # Event handlers
    def on_browse(self) -> None:
        path = filedialog.askdirectory(title="Select ANLZ folder")
        if path:
            self.audio_entry.delete(0, tk.END)
            self.audio_entry.insert(0, path)

    def on_load_audio(self) -> None:
        path_str = self.audio_entry.get().strip()
        if not path_str:
            messagebox.showerror("Error", "Please select an ANLZ folder first.")
            return
        self._load_anlz_folder(Path(path_str), silent=False)

    def on_save_config(self) -> None:
        """Save current config to JSON file."""
        file_path = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile="waveform_config.json",
        )
        if not file_path:
            return
        
        try:
            # Update configs from current widget state
            self._update_configs_from_widgets()
            
            # Serialize to dict
            config_dict = config_to_dict(self.analysis_cfg, self.color_cfg, self.render_cfg)
            
            # Write to file
            with open(file_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            messagebox.showinfo("Success", f"Configuration saved to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration:\n{e}")

    def on_load_config(self) -> None:
        """Load config from JSON file."""
        file_path = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not file_path:
            return
        
        try:
            # Read from file
            with open(file_path, 'r') as f:
                config_dict = json.load(f)
            
            # Deserialize
            self.analysis_cfg, self.color_cfg, self.render_cfg = dict_to_config(config_dict)
            
            # Update all widgets to reflect loaded config
            self._update_widgets_from_configs()
            
            # Mark prerender as dirty and rerender
            self._mark_prerender_dirty()
            self._rerender_if_ready()
            
            messagebox.showinfo("Success", f"Configuration loaded from:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load configuration:\n{e}")

    def _update_widgets_from_configs(self) -> None:
        """Update all widget values from current config objects."""
        # Analysis config
        self.bins_var.set(self.analysis_cfg.target_bins)
        self.low_cut_var.set(self.analysis_cfg.low_cutoff_hz)
        self.lowmid_cut_var.set(self.analysis_cfg.lowmid_cutoff_hz)
        self.midhigh_cut_var.set(self.analysis_cfg.midhigh_cutoff_hz)
        self.comp_strength_var.set(self.analysis_cfg.compression_strength)
        
        # Render config
        self.height_var.set(self.render_cfg.image_height)
        self.render_smoothing_var.set(self.render_cfg.smoothing_bins)
        self.render_detail_var.set(self.render_cfg.prerender_detail)
        self.render_low_gain_var.set(self.render_cfg.low_gain)
        self.render_lowmid_gain_var.set(self.render_cfg.lowmid_gain)
        self.render_midhigh_gain_var.set(self.render_cfg.midhigh_gain)
        self.render_high_gain_var.set(self.render_cfg.high_gain)
        
        # Color config
        if hasattr(self, "band_order_var"):
            self.band_order_var.set(self.color_cfg.band_order_string)
        if hasattr(self, "overview_var"):
            self.overview_var.set(1 if self.color_cfg.overview_mode else 0)
        if hasattr(self, "stack_bands_var"):
            self.stack_bands_var.set(1 if self.color_cfg.stack_bands else 0)

    def _update_configs_from_widgets(self) -> None:
        # Update analysis config from widget state
        cfg = self.analysis_cfg
        cfg.target_bins = max(100, int(self.bins_var.get()))
        cfg.low_cutoff_hz = float(self.low_cut_var.get())
        cfg.lowmid_cutoff_hz = float(self.lowmid_cut_var.get())
        cfg.midhigh_cutoff_hz = float(self.midhigh_cut_var.get())
        # Keep high cutoff at midhigh cutoff
        cfg.high_cutoff_hz = cfg.midhigh_cutoff_hz
        cfg.compression_strength = float(self.comp_strength_var.get())
        cfg.low_band_gain = 1.0
        cfg.lowmid_band_gain = 1.0
        cfg.midhigh_band_gain = 1.0
        cfg.high_band_gain = 1.0
        cfg.smoothing_window_bins = 1

        # Render config
        self.render_cfg.image_height = int(self.height_var.get())
        self.render_cfg.smoothing_bins = int(round(self.render_smoothing_var.get()))
        self.render_cfg.prerender_detail = int(round(self.render_detail_var.get()))
        self.render_cfg.low_gain = float(self.render_low_gain_var.get())
        self.render_cfg.lowmid_gain = float(self.render_lowmid_gain_var.get())
        self.render_cfg.midhigh_gain = float(self.render_midhigh_gain_var.get())
        self.render_cfg.high_gain = float(self.render_high_gain_var.get())
        # Band order parsing
        raw_order = self.band_order_var.get() if hasattr(self, "band_order_var") else ""
        self.color_cfg.band_order, self.color_cfg.band_order_string = parse_band_order(raw_order)
        self.color_cfg.overview_mode = bool(self.overview_var.get())
        self.color_cfg.stack_bands = bool(self.stack_bands_var.get())
        # Width follows window size for preview; no explicit slider.


    def on_run(self) -> None:
        self._update_configs_from_widgets()

        mode = self.mode_var.get()
        analysis_ms = 0.0

        if mode == "analysis":
            if self.signal is None:
                messagebox.showerror("Error", "Load an ANLZ folder first (analysis mode).")
                return
            t0 = perf_counter()
            raw = analyze_bands(self.signal, self.sr, self.duration, self.analysis_cfg)
            t1 = perf_counter()
            analysis_ms = (t1 - t0) * 1000.0
            self.current_analysis_raw = raw
            self._mark_prerender_dirty()
        else:
            if self.cached_anlz_waveform is None:
                messagebox.showerror(
                    "Error", "No Rekordbox waveform loaded. Load an ANLZ folder first."
                )
                return
            self.current_analysis_raw = analysis_from_rb_waveform(
                self.cached_anlz_waveform, self.cached_anlz_duration
            )
            self._mark_prerender_dirty()

        render_ms = self._render_from_current_analysis()
        if render_ms is None:
            return

        analysis_text = f"{analysis_ms:.1f} ms" if mode == "analysis" else "-"
        self.timing_label.configure(
            text=f"Analysis: {analysis_text} | Render: {render_ms:.1f} ms"
        )

def main() -> None:
    root = tk.Tk()
    app = TuningApp(root)
    root.geometry("1200x600")
    root.mainloop()


if __name__ == "__main__":  # pragma: no cover
    main()
