# Copyright (c) HorstHorstmann

"""GUI for tweaking Rekordbox PWV waveforms coming directly from .2EX/.EXT files.

Unlike ``tuning_gui.py`` this variant skips audio analysis entirely. It loads the
already-analysed 3-band PWV data from Rekordbox cache files (2EX preferred,
EXT as fallback) and lets you experiment with rendering parameters only:

- Per-band gain (Low / Mid / High)
- Simple smoothing window
- Band order (l,m,h permutations)
- Overview mode and stacked display toggles
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageTk
from rb_waveform_lab.ANLZ import analyze_anlz_folder

DEFAULT_DECK_PATH = Path(
    r"C:/Users/chris/AppData/Roaming/Pioneer/rekordbox/share/PIONEER/USBANLZ/a82/484d8-6208-4fd8-9406-4393f48abd78"
)

LIBRARY_SEARCH_ROOT: Optional[Path] = Path(r"C:\Rekordbox")
if not LIBRARY_SEARCH_ROOT.is_dir():
    LIBRARY_SEARCH_ROOT = None


COLORS = {
    "l": (0, 120, 255),
    "m": (255, 120, 0),
    "h": (255, 255, 255),
}
BAND_INDEX = {"l": 0, "m": 1, "h": 2}
ASSUMED_SECONDS_PER_FRAME = 1.0 / 75.0  # Rekordbox preview bins are ~75 Hz





def smooth_rows(data: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return data
    kernel = np.ones(window, dtype=np.float32) / float(window)
    smoothed = np.vstack(
        [
            np.convolve(data[:, idx], kernel, mode="same")
            for idx in range(data.shape[1])
        ]
    ).T
    return smoothed


def normalize_bands(data: np.ndarray) -> np.ndarray:
    max_vals = np.max(data, axis=0, keepdims=True)
    max_vals[max_vals == 0.0] = 1.0
    return data / max_vals


def resample_for_width(data: np.ndarray, width: int) -> np.ndarray:
    if data.shape[0] == width:
        return data
    idx = np.linspace(0, data.shape[0] - 1, width).astype(int)
    return data[idx]


def _format_time_label(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs_int = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs_int:02d}"
    if minutes:
        return f"{minutes}:{secs_int:02d}"
    if seconds >= 10:
        return f"{int(round(seconds))}s"
    # Keep a single decimal for short spans
    return f"{seconds:.1f}s"


def _draw_time_axis(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    axis_start: Optional[float],
    axis_duration: Optional[float],
) -> None:
    if not axis_duration or axis_duration <= 0:
        return

    axis_y = height - 1
    draw.line((0, axis_y, width, axis_y), fill=(80, 80, 80), width=1)
    ticks = 8
    for i in range(ticks + 1):
        x = int(round(i * (width - 1) / ticks))
        draw.line((x, axis_y, x, axis_y - 4), fill=(120, 120, 120), width=1)
        current = (axis_start or 0.0) + axis_duration * (i / ticks)
        label = _format_time_label(current)
        bbox = draw.textbbox((0, 0), label)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        label_x = max(0, min(width - text_w, x - text_w // 2))
        label_y = max(0, axis_y - 4 - text_h)
        draw.text((label_x, label_y), label, fill=(180, 180, 180))


def render_preview(
    data: np.ndarray,
    gains: tuple[float, float, float],
    smoothing: int,
    band_order: list[str],
    overview: bool,
    stack: bool,
    width: int,
    height: int,
    axis_start: Optional[float],
    axis_duration: Optional[float],
) -> Image.Image:
    if data.size == 0:
        return Image.new("RGB", (width, height), (5, 5, 5))

    processed = np.clip(data, 0.0, None)
    processed = smooth_rows(processed, max(1, smoothing))
    processed = normalize_bands(processed)
    processed *= np.array(gains, dtype=np.float32)
    processed = np.clip(processed, 0.0, 1.0)
    processed = resample_for_width(processed, max(width, 10))

    axis_reserved = 26 if axis_duration else 0
    margin = 6
    usable_h = max(1, height - axis_reserved - 2 * margin)

    img = Image.new("RGB", (width, height), (5, 5, 5))
    draw = ImageDraw.Draw(img)

    if stack:
        band_count = max(1, len(band_order))
        lane_height = max(4, usable_h // band_count)
        for idx, band_key in enumerate(band_order):
            color = COLORS[band_key]
            band_vals = processed[:, BAND_INDEX[band_key]]
            lane_top = margin + idx * lane_height
            lane_bottom = lane_top + lane_height - 1
            usable_lane = max(1, lane_height - 2)
            heights = (band_vals * usable_lane).astype(int)
            for x, value in enumerate(heights):
                if value <= 0:
                    continue
                y0 = lane_bottom
                y1 = max(lane_top, lane_bottom - value)
                draw.line((x, y0, x, y1), fill=color)
    else:
        center_y = margin + usable_h // 2
        half_h = max(1, usable_h // 2)
        full_range = usable_h
        for band_key in band_order:
            color = COLORS[band_key]
            band_vals = processed[:, BAND_INDEX[band_key]]
            scale = full_range if overview else half_h
            heights = (band_vals * scale).astype(int)
            for x, value in enumerate(heights):
                if value <= 0:
                    continue
                if overview:
                    base = margin + usable_h
                    draw.line((x, base, x, base - value), fill=color)
                else:
                    draw.line((x, center_y + value, x, center_y - value), fill=color)

    _draw_time_axis(draw, width, height, axis_start, axis_duration)
    return img


class TwoExTuningApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("2EX Waveform Tuning")

        self.folder_path: Optional[Path] = None
        self.waveform: Optional[np.ndarray] = None
        self.duration_seconds: Optional[float] = None
        self.seconds_per_frame: float = ASSUMED_SECONDS_PER_FRAME
        self.resolved_path: Optional[Path] = None
        self.resolved_path_var = tk.StringVar(value="")

        self.low_gain = tk.DoubleVar(value=1.0)
        self.mid_gain = tk.DoubleVar(value=1.0)
        self.high_gain = tk.DoubleVar(value=1.0)
        self.smoothing = tk.IntVar(value=5)
        self.band_order = tk.StringVar(value="l,m,h")
        self.overview_mode = tk.BooleanVar(value=True)
        self.stack_bands = tk.BooleanVar(value=False)
        self.zoom_var = tk.DoubleVar(value=1.0)
        self.pan_var = tk.DoubleVar(value=0.0)
        self.zoom_label = tk.StringVar(value="full")

        self.image_tk: Optional[ImageTk.PhotoImage] = None
        self.pan_slider: Optional[ttk.Scale] = None

        self._build_ui()
        self._prefill_default()

    # --- UI -----------------------------------------------------------------
    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=10)
        frame.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        # Path row
        path_row = ttk.Frame(frame)
        path_row.grid(row=0, column=0, columnspan=2, sticky="ew")
        path_row.columnconfigure(1, weight=1)

        ttk.Label(path_row, text="ANLZ folder:").grid(row=0, column=0, sticky="w")
        self.path_entry = ttk.Entry(path_row)
        self.path_entry.grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(path_row, text="Browse", command=self._on_browse).grid(row=0, column=2)
        ttk.Button(path_row, text="Load", command=self._on_load).grid(row=0, column=3, padx=(4, 0))

        # Controls
        controls = ttk.LabelFrame(frame, text="Render controls")
        controls.grid(row=1, column=0, sticky="nsw", pady=(10, 0))

        self._add_slider(controls, "Low gain", self.low_gain, 0.1, 5.0, 0)
        self._add_slider(controls, "Mid gain", self.mid_gain, 0.1, 5.0, 1)
        self._add_slider(controls, "High gain", self.high_gain, 0.1, 5.0, 2)
        self._add_slider(controls, "Smoothing window", self.smoothing, 1, 101, 3, integer=True)

        ttk.Label(controls, text="Band order (l,m,h)").grid(row=4, column=0, sticky="w", pady=(8, 0))
        band_entry = ttk.Entry(controls, textvariable=self.band_order)
        band_entry.grid(row=4, column=1, sticky="ew", padx=(4, 0), pady=(8, 0))
        band_entry.bind("<KeyRelease>", lambda _e: self._rerender())
        controls.columnconfigure(1, weight=1)

        ttk.Checkbutton(
            controls,
            text="Overview mode",
            variable=self.overview_mode,
            command=self._rerender,
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(6, 0))

        ttk.Checkbutton(
            controls,
            text="Stack bands",
            variable=self.stack_bands,
            command=self._rerender,
        ).grid(row=6, column=0, columnspan=2, sticky="w")

        # Preview
        preview_frame = ttk.Frame(frame)
        preview_frame.grid(row=1, column=1, sticky="nsew", padx=(10, 0))
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(1, weight=1)

        zoom_frame = ttk.Frame(preview_frame)
        zoom_frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        zoom_frame.columnconfigure(1, weight=1)

        ttk.Label(zoom_frame, text="Zoom window").grid(row=0, column=0, sticky="w")
        self.zoom_slider = ttk.Scale(
            zoom_frame,
            from_=0.0,
            to=1.0,
            orient="horizontal",
            variable=self.zoom_var,
            command=lambda _v: self._rerender(),
        )
        self.zoom_slider.grid(row=0, column=1, sticky="ew", padx=(6, 4))
        ttk.Label(zoom_frame, textvariable=self.zoom_label, width=8).grid(row=0, column=2, sticky="e")

        ttk.Label(zoom_frame, text="Pan").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.pan_slider = ttk.Scale(
            zoom_frame,
            from_=0.0,
            to=1.0,
            orient="horizontal",
            variable=self.pan_var,
            command=lambda _v: self._rerender(),
        )
        self.pan_slider.grid(row=1, column=1, sticky="ew", padx=(6, 4), pady=(4, 0))
        self.pan_slider.configure(state="disabled")

        self.preview_label = ttk.Label(preview_frame, relief="sunken")
        self.preview_label.grid(row=1, column=0, sticky="nsew")
        resolved_label = ttk.Label(
            preview_frame,
            textvariable=self.resolved_path_var,
            anchor="w",
            justify="left",
            wraplength=600,
        )
        resolved_label.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(1, weight=1)

    def _add_slider(
        self,
        parent: ttk.Frame,
        label: str,
        var: tk.Variable,
        from_: float,
        to: float,
        row: int,
        integer: bool = False,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        slider = ttk.Scale(
            parent,
            from_=from_,
            to=to,
            orient="horizontal",
            variable=var,
            command=lambda _v: self._rerender(),
        )
        slider.grid(row=row, column=1, sticky="ew", padx=(4, 0))
        if integer:
            var.set(int(var.get()))

    # --- Data handling -------------------------------------------------------
    def _prefill_default(self) -> None:
        if DEFAULT_DECK_PATH.is_dir():
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, str(DEFAULT_DECK_PATH))
            self._on_load()

    def _on_browse(self) -> None:
        chosen = filedialog.askdirectory(title="Select ANLZ folder")
        if chosen:
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, chosen)

    def _on_load(self) -> None:
        folder = Path(self.path_entry.get().strip())
        try:
            result = analyze_anlz_folder(folder, LIBRARY_SEARCH_ROOT)
        except FileNotFoundError:
            messagebox.showerror("Missing folder", f"Folder not found: {folder}")
            return

        if result.waveform is None or result.waveform.size == 0:
            messagebox.showerror("Parse error", f"Failed to read PWV data inside {folder}")
            return

        self.folder_path = folder
        self.waveform = result.waveform
        self.duration_seconds = result.duration
        self.resolved_path = result.resolved_path
        if self.resolved_path is not None:
            self.resolved_path_var.set(f"Resolved path: {self.resolved_path}")
        else:
            self.resolved_path_var.set("Resolved path: (not found)")
        if self.duration_seconds and self.waveform is not None and self.waveform.size:
            self.seconds_per_frame = float(self.duration_seconds) / float(self.waveform.shape[0])
        else:
            self.seconds_per_frame = ASSUMED_SECONDS_PER_FRAME
        self.zoom_var.set(1.0)
        self.pan_var.set(0.0)
        self.zoom_label.set("full")
        if self.pan_slider is not None:
            self.pan_slider.configure(state="disabled")
        # Delay the first render so Tk has time to size the preview label
        self.root.after(120, self._rerender)

    # --- Rendering -----------------------------------------------------------
    def _rerender(self) -> None:
        if self.waveform is None:
            return

        bands = [token.strip().lower() for token in self.band_order.get().split(",") if token.strip()]
        if len(bands) != 3 or any(token not in ("l", "m", "h") for token in bands):
            bands = ["l", "m", "h"]
            self.band_order.set("l,m,h")

        total_frames = self.waveform.shape[0]
        if total_frames == 0:
            return

        seconds_per_frame = max(self.seconds_per_frame, 1e-3)
        total_seconds = total_frames * seconds_per_frame
        zoom_frac = float(self.zoom_var.get())

        if zoom_frac >= 0.999:
            window_frames = total_frames
        else:
            min_window = min(5.0, total_seconds)
            zoom_seconds = min_window + (total_seconds - min_window) * zoom_frac
            window_frames = max(10, int(round(zoom_seconds / seconds_per_frame)))
        window_frames = min(total_frames, window_frames)

        if window_frames >= total_frames:
            start_idx = 0
            if self.pan_slider is not None:
                self.pan_slider.configure(state="disabled")
            self.pan_var.set(0.0)
        else:
            if self.pan_slider is not None:
                self.pan_slider.configure(state="normal")
            pan = max(0.0, min(1.0, float(self.pan_var.get())))
            max_start = total_frames - window_frames
            start_idx = int(round(pan * max_start))

        end_idx = start_idx + window_frames
        window_data = self.waveform[start_idx:end_idx]
        window_duration = window_frames * seconds_per_frame
        axis_start = start_idx * seconds_per_frame

        if window_frames >= total_frames:
            self.zoom_label.set("full")
        else:
            self.zoom_label.set(_format_time_label(window_duration))

        self.preview_label.update_idletasks()
        preview_w = self.preview_label.winfo_width()
        if preview_w < 50:
            preview_w = max(600, self.root.winfo_width() - 320)
        preview_h = 300
        gains = (float(self.low_gain.get()), float(self.mid_gain.get()), float(self.high_gain.get()))
        smoothing = int(self.smoothing.get())

        img = render_preview(
            window_data,
            gains,
            smoothing,
            bands,
            self.overview_mode.get(),
            self.stack_bands.get(),
            preview_w,
            preview_h,
            axis_start,
            window_duration,
        )

        self.image_tk = ImageTk.PhotoImage(img)
        self.preview_label.configure(image=self.image_tk)


def main() -> None:
    root = tk.Tk()
    app = TwoExTuningApp(root)
    root.geometry("1200x600")
    root.mainloop()


if __name__ == "__main__":  # pragma: no cover
    main()
