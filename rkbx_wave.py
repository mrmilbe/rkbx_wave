# Copyright (c) mrmilbe

"""WaveformSync GUI - Dual deck display for Rekordbox Link.

Displays two decks vertically with live waveform sync.
Uses saved configuration files for waveform rendering.

Run with:
    python rkbx_wave.py
    or
    rkbx_wave (after pip install)
"""

from __future__ import annotations

import sys

# Windows-only check
if sys.platform != "win32":
    print("Error: rkbx_wave is Windows-only due to rkbx_link.exe dependency.")
    print("This application requires Windows to communicate with Rekordbox.")
    sys.exit(1)

import ctypes
import json
import shutil
import time
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, colorchooser
from pathlib import Path
from typing import Optional
import os

from rb_waveform_core.rb_waveform_gpu import WaveformGpuWindow, DeckFrame

from rb_waveform_core.config import (
    DEFAULT_COLOR_CONFIG,
    DEFAULT_RENDER_CONFIG,
    WaveformColorConfig,
    WaveformRenderConfig,
    config_to_dict,
    dict_to_config,
    parse_band_order,
)
from rb_waveform_core.deck_controller import DeckController
from rb_waveform_core.rkbx_link_listener import DeckEvent
from rb_waveform_core.rb_memory_reader import RekordboxMemoryReader


LIBRARY_SEARCH_ROOT: Optional[Path] = Path(r"C:\Rekordbox")
if not LIBRARY_SEARCH_ROOT.is_dir():
    LIBRARY_SEARCH_ROOT = None


# ---------------------------------------------------------------------------
# Tooltip helper class
# ---------------------------------------------------------------------------
class ToolTip:
    """Simple tooltip that appears on hover after a short delay."""
    
    def __init__(self, widget: tk.Widget, text: str, delay: int = 500):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tip_window: Optional[tk.Toplevel] = None
        self.scheduled_id: Optional[str] = None
        widget.bind("<Enter>", self._schedule_show)
        widget.bind("<Leave>", self._hide)
        widget.bind("<ButtonPress>", self._hide)
    
    def _schedule_show(self, event=None) -> None:
        self._cancel_scheduled()
        self.scheduled_id = self.widget.after(self.delay, self._show)
    
    def _cancel_scheduled(self) -> None:
        if self.scheduled_id:
            self.widget.after_cancel(self.scheduled_id)
            self.scheduled_id = None
    
    def _show(self, event=None) -> None:
        if self.tip_window:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw, text=self.text, justify="left", background="#ffffe0",
            relief="solid", borderwidth=1, font=("Segoe UI", 9)
        )
        label.pack()
    
    def _hide(self, event=None) -> None:
        self._cancel_scheduled()
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


# User config directory in AppData
USER_CONFIG_DIR = Path(os.environ.get("APPDATA", Path.home())) / "rkbx_wave"
USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
LAST_CONFIG_PATH = USER_CONFIG_DIR / "last_config.txt"
USER_DEFAULT_CONFIG_PATH = USER_CONFIG_DIR / "default_config.json"


def _get_package_data_path() -> Path:
    """Path to the bundled package data (rb_waveform_core/data).

    Works both from a source checkout and an installed wheel: rkbx_wave.py sits
    next to the rb_waveform_core package in either case.
    """
    return Path(__file__).parent / "rb_waveform_core" / "data"


def _get_default_config_path() -> Path:
    """Get path to default_config.json, copying to AppData if needed."""
    # If user already has a copy in AppData, use it
    if USER_DEFAULT_CONFIG_PATH.exists():
        return USER_DEFAULT_CONFIG_PATH
    
    # Try to copy from package data to AppData
    package_config = _get_package_data_path() / "default_config.json"
    if package_config.exists():
        try:
            shutil.copy2(package_config, USER_DEFAULT_CONFIG_PATH)
            print(f"[Config] Copied default config to {USER_DEFAULT_CONFIG_PATH}")
            return USER_DEFAULT_CONFIG_PATH
        except Exception as e:
            print(f"[Config] Failed to copy default config: {e}")
            return package_config
    
    # Last resort: return path even if doesn't exist (will use hardcoded defaults)
    return USER_DEFAULT_CONFIG_PATH



DEFAULT_CONFIG_PATH = _get_default_config_path()

# Match tuning_gui: discrete zoom levels in seconds
ZOOM_LEVELS_SECONDS = [256, 196, 128, 96, 64, 48, 32, 24, 16]
NUM_ZOOM_STEPS = len(ZOOM_LEVELS_SECONDS) - 1

# Friendly labels for Windows virtual-key codes used by the zoom hotkeys.
_VK_NAMES = {
    0x6B: "Num +", 0x6D: "Num -", 0x6A: "Num *", 0x6F: "Num /",
    0xBB: "+", 0xBD: "-", 0x21: "PgUp", 0x22: "PgDn",
    0x26: "Up", 0x28: "Down", 0x25: "Left", 0x27: "Right",
    0x20: "Space", 0x09: "Tab",
}


def _vk_label(vk: int) -> str:
    """Human-readable label for a Windows virtual-key code."""
    if vk in _VK_NAMES:
        return _VK_NAMES[vk]
    if 0x30 <= vk <= 0x39 or 0x41 <= vk <= 0x5A:  # 0-9, A-Z
        return chr(vk)
    if 0x60 <= vk <= 0x69:  # numpad 0-9
        return f"Num {vk - 0x60}"
    if 0x70 <= vk <= 0x7B:  # F1-F12
        return f"F{vk - 0x6F}"
    return f"0x{vk:02X}"

# GPU waveform window: the full-track texture is built once per track/config.
# Height is the texture's vertical resolution (GPU scales it to the viewport).
# pixels-per-second sets horizontal detail; capped per track to fit GPU limits.
GPU_TEXTURE_HEIGHT = 256
GPU_BASE_PIXELS_PER_SECOND = 50.0
GPU_TICK_MS = 16  # ~60 fps render/poll tick
# Build the texture at this multiple of the on-screen resolution; the GPU then
# downsamples with linear filtering = horizontal anti-aliasing (less "pixely").
GPU_OVERSAMPLE = 2.0


class WaveformSyncApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("WaveformSync - Dual Deck Display")
        
        # Shared config (loaded from file)
        self.color_cfg: WaveformColorConfig = DEFAULT_COLOR_CONFIG
        self.render_cfg: WaveformRenderConfig = DEFAULT_RENDER_CONFIG
        
        # Support up to deck_count decks (from config)
        self.deck_count = self.render_cfg.deck_count
        self.decks = {i: DeckController(LIBRARY_SEARCH_ROOT) for i in range(self.deck_count)}

        # GPU waveform window (created after UI is built)
        self.gpu: Optional[WaveformGpuWindow] = None

        # Playhead time extrapolation per deck (smooth scroll between link updates):
        # measure playback rate from consecutive time updates, then project forward.
        self._deck_play_time = {i: None for i in range(self.deck_count)}
        self._deck_play_wall = {i: None for i in range(self.deck_count)}
        self._deck_play_rate = {i: 0.0 for i in range(self.deck_count)}

        # Texture-build state: textures are sized to the on-screen resolution for
        # the current zoom + window width, so they are rebuilt when either changes.
        self._tex_build_zoom: Optional[int] = None
        self._tex_build_width: Optional[int] = None
        self._resize_pending_w: Optional[int] = None
        self._resize_pending_at: float = 0.0

        # Link listener / main tick
        self.link_listener: Optional[RekordboxMemoryReader] = None
        self._tick_job: Optional[str] = None
        
        # UI variables
        self.overview_var = tk.BooleanVar(value=False)
        self.stack_bands_var = tk.BooleanVar(value=False)
        self.rgb_var = tk.BooleanVar(value=True)  # RGB (per-column band blend) waveform, on by default
        self.beat_grid_var = tk.BooleanVar(value=True)  # Enable beat grid by default
        self.cue_var = tk.BooleanVar(value=True)  # Show hot + memory cue markers
        # Beat-grid marker style (live overlay, synced from config)
        self.beat_width_var = tk.DoubleVar(value=self.render_cfg.beat_width)
        self.beat_triangles_var = tk.BooleanVar(value=self.render_cfg.beat_triangles)
        # Global zoom hotkeys (work while Rekordbox is focused); keys come from config
        self.numpad_zoom_var = tk.BooleanVar(value=True)
        self._zoom_key_down = {"in": False, "out": False}
        # GPU waveform window options. Border ON by default so the window has a real
        # titlebar and is natively movable/resizable; turn it off for a clean overlay
        # (borderless can't be dragged - reposition with the border on first).
        self.gpu_on_top_var = tk.BooleanVar(value=True)
        self.gpu_border_var = tk.BooleanVar(value=True)
        self.zoom_var = tk.IntVar(value=0)  # index into fixed zoom levels
        self.link_status_var = tk.StringVar(value="RB Link: disabled")
        
        # Tuning panel variables
        self.tune_panel_visible = tk.BooleanVar(value=False)
        self.band_order_var = tk.StringVar(value="l,m,h")  # Default 3-band order
        # Band gain variables (context-sensitive - will be synced based on overview mode)
        self.low_gain_var = tk.DoubleVar(value=1.0)
        self.mid_gain_var = tk.DoubleVar(value=1.0)
        self.high_gain_var = tk.DoubleVar(value=1.0)
        # Smoothing variable (context-sensitive)
        self.smoothing_var = tk.IntVar(value=1)
        
        self._load_last_or_default_config()
        self._build_ui()
        self._create_gpu_window()
        self._start_link_listener()

        # Waveforms live in the separate GPU window now; the tk window only holds
        # the compact controls + (optional) tune panel + deck info labels, so it
        # sizes to its content (≈ tune-box width).
        self._resize_to_content()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _resize_to_content(self) -> None:
        """Shrink/grow the tk window to fit its current widgets."""
        self.root.update_idletasks()
        w = max(self.root.winfo_reqwidth(), 1)
        h = max(self.root.winfo_reqheight(), 1)
        self.root.geometry(f"{w}x{h}")

    def _create_gpu_window(self) -> None:
        """Create the standalone GPU waveform window and load any current tracks."""
        screen_w = self.root.winfo_screenwidth()
        win_w = int(screen_w * 0.8)
        win_h = self.deck_count * 220
        bg = tuple(self.render_cfg.background_color)
        self.gpu = WaveformGpuWindow(
            self.deck_count,
            size=(win_w, win_h),
            background=bg,
            beat_color=tuple(self.color_cfg.beat_color),
            beat_end_color=tuple(self.color_cfg.beat_end_color),
            beat_width=self.render_cfg.beat_width,
            beat_triangles=self.render_cfg.beat_triangles,
            borderless=not self.gpu_border_var.get(),
            always_on_top=self.gpu_on_top_var.get(),
            on_close=self._on_gpu_window_closed,
        )
        # Borderless windows can't be dragged by a titlebar, so place it at the top
        # centre of the screen by default (a sensible spot for an always-on-top bar).
        try:
            self.gpu.window.position = ((screen_w - win_w) // 2, 100)
        except Exception:
            pass
        print(f"[GPU] Renderer hardware-accelerated: {self.gpu.hardware}")
        self._on_gpu_window_opts()
        self._refresh_all_textures()

    def _on_gpu_window_opts(self) -> None:
        """Apply the GPU window border / always-on-top toggles."""
        if self.gpu is None:
            return
        self.gpu.set_borderless(not self.gpu_border_var.get())
        self.gpu.set_always_on_top(self.gpu_on_top_var.get())
    

    def _load_last_or_default_config(self) -> None:
        """Load configuration from last_config.txt if it exists, else from default."""
        config_path = DEFAULT_CONFIG_PATH
        if LAST_CONFIG_PATH.exists():
            try:
                with open(LAST_CONFIG_PATH, 'r') as f:
                    last_path = f.read().strip()
                if last_path and Path(last_path).exists():
                    config_path = Path(last_path)
            except Exception as e:
                print(f"[Config] Failed to read last_config.txt: {e}")
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                self.color_cfg, self.render_cfg = dict_to_config(config_dict)
                self.deck_count = self.render_cfg.deck_count
                print(f"[Config] Loaded from {config_path}")
            except Exception as e:
                print(f"[Config] Failed to load config: {e}")
        self._sync_tuning_vars_from_config()
        self._sync_visual_vars_from_config()

    def _sync_visual_vars_from_config(self) -> None:
        """Sync the visual-mode toggles + beat-grid style vars from current config."""
        self.overview_var.set(self.color_cfg.overview_mode)
        self.stack_bands_var.set(self.color_cfg.stack_bands)
        self.rgb_var.set(getattr(self.color_cfg, "rgb_mode", True))
        self.beat_width_var.set(self.render_cfg.beat_width)
        self.beat_triangles_var.set(self.render_cfg.beat_triangles)

    def _build_ui(self) -> None:
        self.root.configure(bg="black")
        
        # Create style for black frame
        style = ttk.Style()
        style.configure("Black.TFrame", background="black")
        style.configure("Black.TLabel", background="black", foreground="white")
        style.configure("Tune.TFrame", background="#1a1a1a")
        style.configure("Tune.TLabel", background="#1a1a1a", foreground="white")
        style.configure("Tune.TButton", background="#333333")
        style.configure("Tune.TCheckbutton", background="#1a1a1a", foreground="white")
        style.map("Tune.TCheckbutton", background=[("active", "#1a1a1a")])
        
        main = ttk.Frame(self.root, style="Black.TFrame")
        main.pack(fill="both", expand=True, padx=2, pady=2)
        main.columnconfigure(1, weight=1)  # Waveform area expands
        main.rowconfigure(1, weight=1)  # Content area expands
        
        # Top controls (spans both columns). Compact multi-row layout (~tune width):
        #   row0: Zoom slider + Decks    row1: Overview / Stack / RGB
        #   row2: Beat Grid / On Top / Border   row3: Tune / Load / Save
        controls = ttk.Frame(main)
        controls.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))

        r0 = ttk.Frame(controls)
        r0.pack(fill="x", pady=1)
        ttk.Label(r0, text="Zoom:").pack(side="left", padx=(0, 4))
        zoom_slider = ttk.Scale(
            r0, from_=0, to=NUM_ZOOM_STEPS, orient="horizontal",
            variable=self.zoom_var, command=lambda _v: self._on_zoom_change(),
        )
        zoom_slider.configure(length=110)
        zoom_slider.pack(side="left", padx=(0, 8))
        ToolTip(zoom_slider, "Visible time window: left=zoomed out, right=zoomed in\nNumpad +/- also zoom while Rekordbox is focused")
        ttk.Label(r0, text="Decks:").pack(side="left", padx=(0, 4))
        self.deck_count_var = tk.StringVar(value=str(self.deck_count))
        deck_combo = ttk.Combobox(r0, textvariable=self.deck_count_var, values=["2", "4"], width=3, state="readonly")
        deck_combo.pack(side="left")
        deck_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_deck_count_change())
        ToolTip(deck_combo, "Number of active decks (restarts the memory reader)")

        r1 = ttk.Frame(controls)
        r1.pack(fill="x", pady=1)
        overview_cb = ttk.Checkbutton(r1, text="Overview", variable=self.overview_var, command=self._on_visual_change)
        overview_cb.pack(side="left", padx=(0, 6))
        ToolTip(overview_cb, "Rekordbox-style overview, band order adjustable")
        stack_cb = ttk.Checkbutton(r1, text="Stack", variable=self.stack_bands_var, command=self._on_visual_change)
        stack_cb.pack(side="left", padx=(0, 6))
        ToolTip(stack_cb, "Show each band in its own horizontal lane")
        rgb_cb = ttk.Checkbutton(r1, text="RGB", variable=self.rgb_var, command=self._on_visual_change)
        rgb_cb.pack(side="left", padx=(0, 6))
        ToolTip(rgb_cb, "Single waveform coloured by per-column band blend (rekordbox-style RGB)")

        r2 = ttk.Frame(controls)
        r2.pack(fill="x", pady=1)
        beatgrid_cb = ttk.Checkbutton(r2, text="Beat Grid", variable=self.beat_grid_var, command=self._on_visual_change)
        beatgrid_cb.pack(side="left", padx=(0, 6))
        ToolTip(beatgrid_cb, "Show downbeat markers (bar boundaries) on the waveform")
        cues_cb = ttk.Checkbutton(r2, text="Cues", variable=self.cue_var, command=self._on_visual_change)
        cues_cb.pack(side="left", padx=(0, 6))
        ToolTip(cues_cb, "Show hot cues + memory cues (from the Rekordbox DB, or a USB stick's ANLZ files)")
        ontop_cb = ttk.Checkbutton(r2, text="On Top", variable=self.gpu_on_top_var, command=self._on_gpu_window_opts)
        ontop_cb.pack(side="left", padx=(0, 6))
        ToolTip(ontop_cb, "Keep the waveform window above other windows")
        border_cb = ttk.Checkbutton(r2, text="Border", variable=self.gpu_border_var, command=self._on_gpu_window_opts)
        border_cb.pack(side="left", padx=(0, 6))
        ToolTip(border_cb, "Show the window border (needed to move/resize); off = clean overlay")

        r3 = ttk.Frame(controls)
        r3.pack(fill="x", pady=1)
        tune_btn = ttk.Button(r3, text="Tune", width=7, command=self._toggle_tune_panel)
        tune_btn.pack(side="left", padx=(0, 4))
        ToolTip(tune_btn, "Open/close tuning panel for smoothing, gains, and beat grid")
        load_btn = ttk.Button(r3, text="Load", width=7, command=self._on_load_config)
        load_btn.pack(side="left", padx=(0, 4))
        ToolTip(load_btn, "Load waveform color/render settings from a JSON file")
        save_btn = ttk.Button(r3, text="Save", width=7, command=self._on_save_config)
        save_btn.pack(side="left")
        ToolTip(save_btn, "Save current waveform settings to a JSON file")

        ttk.Label(controls, textvariable=self.link_status_var).pack(anchor="w", pady=(4, 0))

        # Tuning panel (left side, collapsible)
        self.tune_panel = ttk.Frame(main, style="Tune.TFrame", width=200)
        # Don't grid it yet - will be shown/hidden by toggle
        self._build_tune_panel()
        
        # Info area (right side). The waveforms themselves render in the separate
        # GPU window; here we only show the per-deck track name.
        self.waveform_area = ttk.Frame(main, style="Black.TFrame")
        self.waveform_area.grid(row=1, column=1, sticky="nsew")
        self.waveform_area.columnconfigure(0, weight=1)

        self.deck_infos = {}
        for i in range(self.deck_count):
            self.deck_infos[i] = ttk.Label(self.waveform_area, text=f"Deck {i+1}: No track loaded", anchor="w", style="Black.TLabel")
            self.deck_infos[i].grid(row=i, column=0, sticky="ew", pady=(0, 2))

    def _build_tune_panel(self) -> None:
        """Build the tuning subpanel contents."""
        panel = self.tune_panel
        row = 0
        
        # Title
        ttk.Label(panel, text="Tuning", style="Tune.TLabel", font=("TkDefaultFont", 10, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=8, pady=(8, 12)
        )
        row += 1

        # (Render Mode dropdown removed: the GPU sizes textures by zoom now, so the
        # old fixed render-height/interpolation modes no longer affect the display.)

        # Zoom (shares zoom_var with the main slider + numpad hotkey)
        ttk.Label(panel, text="Zoom:", style="Tune.TLabel").grid(row=row, column=0, sticky="w", padx=8, pady=2)
        zoom_scale = ttk.Scale(
            panel, from_=0, to=NUM_ZOOM_STEPS, orient="horizontal",
            variable=self.zoom_var, length=100, command=lambda _v: self._on_zoom_change(),
        )
        zoom_scale.grid(row=row, column=1, sticky="w", padx=8, pady=2)
        ToolTip(zoom_scale, "Visible time window (also numpad +/-): left=zoomed out, right=zoomed in")
        row += 1

        # Band Order (3-band only)
        ttk.Label(panel, text="Band Order (3):", style="Tune.TLabel").grid(row=row, column=0, sticky="w", padx=8, pady=2)
        band_order_entry = ttk.Entry(panel, textvariable=self.band_order_var, width=12)
        band_order_entry.grid(row=row, column=1, sticky="w", padx=8, pady=2)
        band_order_entry.bind("<Return>", lambda e: self._on_band_order_change())
        band_order_entry.bind("<FocusOut>", lambda e: self._on_band_order_change())
        ToolTip(band_order_entry, "Drawing order (back→front): l=low, m=mid, h=high\nExample: 'l,m,h' draws low first, high on top")
        row += 1
        
        # Band order hint
        ttk.Label(panel, text="(l,m,h)", style="Tune.TLabel", foreground="gray").grid(
            row=row, column=1, sticky="w", padx=8, pady=0
        )
        row += 1
        
        # RGB-mode band colours (used by the RGB waveform mode)
        ttk.Label(panel, text="RGB Colors:", style="Tune.TLabel").grid(row=row, column=0, sticky="w", padx=8, pady=2)
        rgb_btn_frame = ttk.Frame(panel, style="Tune.TFrame")
        rgb_btn_frame.grid(row=row, column=1, sticky="w", padx=8, pady=2)
        self._rgb_color_btns = {}
        for band in ("low", "mid", "high"):
            btn = tk.Button(
                rgb_btn_frame,
                text=band[0].upper(),
                width=2,
                relief="raised",
                command=lambda b=band: self._on_pick_rgb_color(b),
            )
            btn.pack(side="left", padx=2)
            ToolTip(btn, f"RGB-mode colour for the {band} band (click to change)")
            self._rgb_color_btns[band] = btn
        self._update_rgb_color_buttons()
        row += 1

        # Separator
        ttk.Separator(panel, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", padx=8, pady=8)
        row += 1

        # Smoothing slider (context-sensitive)
        self.smoothing_context_label = ttk.Label(panel, text="Smoothing (Default):", style="Tune.TLabel")
        self.smoothing_context_label.grid(row=row, column=0, sticky="w", padx=8, pady=2)
        row += 1
        
        smoothing_scale = ttk.Scale(panel, from_=1, to=63, orient="horizontal", variable=self.smoothing_var, length=100)
        smoothing_scale.grid(row=row, column=0, columnspan=2, sticky="ew", padx=8, pady=1)
        smoothing_scale.bind("<ButtonRelease-1>", lambda e: self._on_smoothing_change())
        smoothing_scale.bind("<B1-Motion>", lambda e: self._on_smoothing_change())  # Live update while dragging
        ToolTip(smoothing_scale, "Moving average window (bins): higher = smoother waveform\nOverview mode uses separate smoothing setting")
        row += 1
        
        # Separator
        ttk.Separator(panel, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", padx=8, pady=8)
        row += 1
        
        # Context label for gains
        self.gain_context_label = ttk.Label(panel, text="Gains (Default):", style="Tune.TLabel")
        self.gain_context_label.grid(row=row, column=0, columnspan=2, sticky="w", padx=8, pady=(2, 4))
        row += 1
        
        # Low gain slider
        ttk.Label(panel, text="Low:", style="Tune.TLabel").grid(row=row, column=0, sticky="w", padx=8, pady=1)
        low_scale = ttk.Scale(panel, from_=0.0, to=3.0, orient="horizontal", variable=self.low_gain_var, length=100)
        low_scale.grid(row=row, column=1, sticky="w", padx=8, pady=1)
        low_scale.bind("<ButtonRelease-1>", lambda e: self._on_gain_change())
        low_scale.bind("<B1-Motion>", lambda e: self._on_gain_change())  # Live update while dragging
        ToolTip(low_scale, "Low frequency band amplitude multiplier (0–3×)")
        row += 1
        
        # Mid gain slider  
        ttk.Label(panel, text="Mid:", style="Tune.TLabel").grid(row=row, column=0, sticky="w", padx=8, pady=1)
        mid_scale = ttk.Scale(panel, from_=0.0, to=3.0, orient="horizontal", variable=self.mid_gain_var, length=100)
        mid_scale.grid(row=row, column=1, sticky="w", padx=8, pady=1)
        mid_scale.bind("<ButtonRelease-1>", lambda e: self._on_gain_change())
        mid_scale.bind("<B1-Motion>", lambda e: self._on_gain_change())  # Live update while dragging
        ToolTip(mid_scale, "Mid frequency band amplitude multiplier (0–3×)")
        row += 1
        
        # High gain slider
        ttk.Label(panel, text="High:", style="Tune.TLabel").grid(row=row, column=0, sticky="w", padx=8, pady=1)
        high_scale = ttk.Scale(panel, from_=0.0, to=3.0, orient="horizontal", variable=self.high_gain_var, length=100)
        high_scale.grid(row=row, column=1, sticky="w", padx=8, pady=1)
        high_scale.bind("<ButtonRelease-1>", lambda e: self._on_gain_change())
        high_scale.bind("<B1-Motion>", lambda e: self._on_gain_change())  # Live update while dragging
        ToolTip(high_scale, "High frequency band amplitude multiplier (0–3×)")
        row += 1

        # Separator
        ttk.Separator(panel, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", padx=8, pady=8)
        row += 1

        # --- Beat grid section ---
        ttk.Label(panel, text="Beat Grid", style="Tune.TLabel", font=("TkDefaultFont", 9, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 4)
        )
        row += 1

        ttk.Label(panel, text="Width:", style="Tune.TLabel").grid(row=row, column=0, sticky="w", padx=8, pady=1)
        beat_width_scale = ttk.Scale(panel, from_=1.0, to=8.0, orient="horizontal", variable=self.beat_width_var, length=100)
        beat_width_scale.grid(row=row, column=1, sticky="w", padx=8, pady=1)
        beat_width_scale.bind("<ButtonRelease-1>", lambda e: self._on_beat_style_change())
        beat_width_scale.bind("<B1-Motion>", lambda e: self._on_beat_style_change())
        ToolTip(beat_width_scale, "Thickness of each beat-grid line, in pixels")
        row += 1

        ttk.Label(panel, text="Color:", style="Tune.TLabel").grid(row=row, column=0, sticky="w", padx=8, pady=2)
        self._beat_color_btn = tk.Button(panel, text=" ", width=3, relief="raised", command=self._on_pick_beat_color)
        self._beat_color_btn.grid(row=row, column=1, sticky="w", padx=8, pady=2)
        ToolTip(self._beat_color_btn, "Beat-grid line colour (click to change)")
        self._update_beat_color_button()
        row += 1

        markers_cb = ttk.Checkbutton(
            panel, text="Beat End Markers", variable=self.beat_triangles_var,
            command=self._on_beat_style_change, style="Tune.TCheckbutton",
        )
        markers_cb.grid(row=row, column=0, columnspan=2, sticky="w", padx=8, pady=1)
        ToolTip(markers_cb, "Draw Rekordbox-style triangle markers at the top and bottom of each beat line")
        row += 1

        ttk.Label(panel, text="Marker Color:", style="Tune.TLabel").grid(row=row, column=0, sticky="w", padx=8, pady=2)
        self._beat_end_color_btn = tk.Button(panel, text=" ", width=3, relief="raised", command=self._on_pick_beat_end_color)
        self._beat_end_color_btn.grid(row=row, column=1, sticky="w", padx=8, pady=2)
        ToolTip(self._beat_end_color_btn, "Beat end-marker (triangle) colour, independent of the line (click to change)")
        self._update_beat_end_color_button()
        row += 1

        # Separator
        ttk.Separator(panel, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", padx=8, pady=8)
        row += 1

        # --- Zoom keys (global hotkeys, work while Rekordbox is focused) ---
        ttk.Label(panel, text="Zoom Keys", style="Tune.TLabel", font=("TkDefaultFont", 9, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 4)
        )
        row += 1

        enable_cb = ttk.Checkbutton(
            panel, text="Enabled", variable=self.numpad_zoom_var, style="Tune.TCheckbutton",
        )
        enable_cb.grid(row=row, column=0, columnspan=2, sticky="w", padx=8, pady=1)
        ToolTip(enable_cb, "Zoom with the assigned keys even while Rekordbox is in the foreground")
        row += 1

        ttk.Label(panel, text="Zoom In:", style="Tune.TLabel").grid(row=row, column=0, sticky="w", padx=8, pady=1)
        self._zoom_in_key_btn = ttk.Button(panel, width=10, command=lambda: self._capture_zoom_key("in"))
        self._zoom_in_key_btn.grid(row=row, column=1, sticky="w", padx=8, pady=1)
        ToolTip(self._zoom_in_key_btn, "Click, then press a key to assign (default numpad +)")
        row += 1

        ttk.Label(panel, text="Zoom Out:", style="Tune.TLabel").grid(row=row, column=0, sticky="w", padx=8, pady=1)
        self._zoom_out_key_btn = ttk.Button(panel, width=10, command=lambda: self._capture_zoom_key("out"))
        self._zoom_out_key_btn.grid(row=row, column=1, sticky="w", padx=8, pady=1)
        ToolTip(self._zoom_out_key_btn, "Click, then press a key to assign (default numpad -)")
        row += 1
        self._update_zoom_key_buttons()

        # Separator
        ttk.Separator(panel, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", padx=8, pady=8)
        row += 1

        # (Load/Save buttons moved to main controls row)
    
    def _toggle_tune_panel(self) -> None:
        """Toggle visibility of tuning panel."""
        if self.tune_panel_visible.get():
            self.tune_panel.grid_forget()
            self.tune_panel_visible.set(False)
        else:
            self.tune_panel.grid(row=1, column=0, sticky="ns", padx=(0, 8))
            self.tune_panel_visible.set(True)
            self._sync_tuning_vars_from_config()
        self._resize_to_content()
    
    def _sync_tuning_vars_from_config(self) -> None:
        """Sync tuning panel variables from current config."""
        # Show correct band order for current mode
        if self.overview_var.get():
            self.band_order_var.set(self.color_cfg.band_order_string_overview)
        else:
            self.band_order_var.set(self.color_cfg.band_order_string_default)
        self._sync_gain_vars_from_config()
    
    def _sync_gain_vars_from_config(self) -> None:
        """Sync gain sliders and smoothing based on current overview mode."""
        is_overview = self.overview_var.get()
        if is_overview:
            self.low_gain_var.set(self.render_cfg.overview_low_gain)
            self.mid_gain_var.set(self.render_cfg.overview_mid_gain)
            self.high_gain_var.set(self.render_cfg.overview_high_gain)
            self.smoothing_var.set(self.render_cfg.overview_smoothing_bins)
            if hasattr(self, 'gain_context_label'):
                self.gain_context_label.configure(text="Gains (Overview):")
            if hasattr(self, 'smoothing_context_label'):
                self.smoothing_context_label.configure(text="Smoothing (Overview):")
        else:
            self.low_gain_var.set(self.render_cfg.low_gain)
            self.mid_gain_var.set(self.render_cfg.mid_gain)
            self.high_gain_var.set(self.render_cfg.high_gain)
            self.smoothing_var.set(self.render_cfg.smoothing_bins)
            if hasattr(self, 'gain_context_label'):
                self.gain_context_label.configure(text="Gains (Default):")
            if hasattr(self, 'smoothing_context_label'):
                self.smoothing_context_label.configure(text="Smoothing (Default):")
    
    def _invalidate_caches_and_render(self) -> None:
        """A color/render config changed - rebuild every deck's GPU texture."""
        self._refresh_all_textures()
    
    def _on_band_order_change(self) -> None:
        """Handle band order entry change."""
        raw = self.band_order_var.get()
        band_order, normalized = parse_band_order(raw)
        # Restrict to 3 bands for this GUI
        if len(band_order) > 3:
            band_order = band_order[:3]
            parts = normalized.split(",")[:3]
            normalized = ",".join(parts)
        # Save to correct config field based on current mode
        if self.overview_var.get():
            self.color_cfg.band_order_string_overview = normalized
        else:
            self.color_cfg.band_order_string_default = normalized
        self.color_cfg.band_order = band_order
        self.band_order_var.set(normalized)
        self._invalidate_caches_and_render()
    
    def _on_gain_change(self) -> None:
        """Handle gain slider change - update appropriate config values."""
        is_overview = self.overview_var.get()
        if is_overview:
            self.render_cfg.overview_low_gain = self.low_gain_var.get()
            self.render_cfg.overview_mid_gain = self.mid_gain_var.get()
            self.render_cfg.overview_high_gain = self.high_gain_var.get()
        else:
            self.render_cfg.low_gain = self.low_gain_var.get()
            self.render_cfg.mid_gain = self.mid_gain_var.get()
            self.render_cfg.high_gain = self.high_gain_var.get()
        self._invalidate_caches_and_render()
    
    def _on_smoothing_change(self) -> None:
        """Handle smoothing slider change - update appropriate config values."""
        is_overview = self.overview_var.get()
        smoothing_val = int(self.smoothing_var.get())
        if is_overview:
            self.render_cfg.overview_smoothing_bins = smoothing_val
        else:
            self.render_cfg.smoothing_bins = smoothing_val
        self._invalidate_caches_and_render()
    
    def _on_save_config(self) -> None:
        """Save current configuration to file and update last_config.txt."""
        file_path = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile="waveform_config.json",
        )
        if not file_path:
            return
        try:
            config_dict = config_to_dict(self.color_cfg, self.render_cfg)
            with open(file_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            # Update last_config.txt
            try:
                with open(LAST_CONFIG_PATH, 'w') as lastf:
                    lastf.write(file_path)
            except Exception as e:
                print(f"[Config] Failed to update last_config.txt: {e}")
            messagebox.showinfo("Success", f"Configuration saved to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration:\n{e}")
    
    def _on_load_config(self) -> None:
        """Load configuration from file and update last_config.txt."""
        file_path = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not file_path:
            return
        try:
            with open(file_path, 'r') as f:
                config_dict = json.load(f)
            self.color_cfg, self.render_cfg = dict_to_config(config_dict)
            # Update last_config.txt
            try:
                with open(LAST_CONFIG_PATH, 'w') as lastf:
                    lastf.write(file_path)
            except Exception as e:
                print(f"[Config] Failed to update last_config.txt: {e}")
            # Sync tuning panel variables
            self._sync_tuning_vars_from_config()
            self._sync_visual_vars_from_config()
            self._update_rgb_color_buttons()
            self._update_beat_color_button()
            self._update_beat_end_color_button()
            self._update_zoom_key_buttons()
            self._apply_beat_style()
            self._invalidate_caches_and_render()
            messagebox.showinfo("Success", f"Configuration loaded from:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load configuration:\n{e}")
    
    def _update_rgb_color_buttons(self) -> None:
        """Paint each RGB color-picker button with its current color."""
        btns = getattr(self, "_rgb_color_btns", None)
        if not btns:
            return
        for band, btn in btns.items():
            r, g, b = getattr(self.color_cfg, f"rgb_{band}_color", (255, 255, 255))
            hexc = f"#{int(r):02x}{int(g):02x}{int(b):02x}"
            fg = "#000000" if (r + g + b) > 384 else "#ffffff"
            btn.configure(bg=hexc, activebackground=hexc, fg=fg)

    def _on_pick_rgb_color(self, band: str) -> None:
        """Open a color chooser for an RGB-mode band color and rebuild textures."""
        attr = f"rgb_{band}_color"
        current = getattr(self.color_cfg, attr, (255, 255, 255))
        rgb, _hexstr = colorchooser.askcolor(color=current, title=f"RGB {band} color")
        if rgb is None:
            return
        setattr(self.color_cfg, attr, tuple(int(c) for c in rgb))
        self._update_rgb_color_buttons()
        self._invalidate_caches_and_render()

    # ------------------------------------------------------------------
    # Beat grid style (live overlay — no texture rebuild needed)
    # ------------------------------------------------------------------
    def _update_beat_color_button(self) -> None:
        btn = getattr(self, "_beat_color_btn", None)
        if btn is None:
            return
        r, g, b = self.color_cfg.beat_color
        hexc = f"#{int(r):02x}{int(g):02x}{int(b):02x}"
        fg = "#000000" if (r + g + b) > 384 else "#ffffff"
        btn.configure(bg=hexc, activebackground=hexc, fg=fg)

    def _on_pick_beat_color(self) -> None:
        rgb, _hex = colorchooser.askcolor(color=self.color_cfg.beat_color, title="Beat grid color")
        if rgb is None:
            return
        self.color_cfg.beat_color = tuple(int(c) for c in rgb)
        self._update_beat_color_button()
        self._apply_beat_style()

    def _update_beat_end_color_button(self) -> None:
        btn = getattr(self, "_beat_end_color_btn", None)
        if btn is None:
            return
        r, g, b = self.color_cfg.beat_end_color
        hexc = f"#{int(r):02x}{int(g):02x}{int(b):02x}"
        fg = "#000000" if (r + g + b) > 384 else "#ffffff"
        btn.configure(bg=hexc, activebackground=hexc, fg=fg)

    def _on_pick_beat_end_color(self) -> None:
        rgb, _hex = colorchooser.askcolor(color=self.color_cfg.beat_end_color, title="Beat end marker color")
        if rgb is None:
            return
        self.color_cfg.beat_end_color = tuple(int(c) for c in rgb)
        self._update_beat_end_color_button()
        self._apply_beat_style()

    def _on_beat_style_change(self) -> None:
        self.render_cfg.beat_width = float(self.beat_width_var.get())
        self.render_cfg.beat_triangles = bool(self.beat_triangles_var.get())
        self._apply_beat_style()

    def _apply_beat_style(self) -> None:
        if self.gpu is None:
            return
        self.gpu.set_beat_style(
            color=tuple(self.color_cfg.beat_color),
            end_color=tuple(self.color_cfg.beat_end_color),
            width=self.render_cfg.beat_width,
            triangles=self.render_cfg.beat_triangles,
        )

    def _on_visual_change(self) -> None:
        """Handle overview/stack/RGB mode toggle."""
        self.color_cfg.overview_mode = bool(self.overview_var.get())
        self.color_cfg.stack_bands = bool(self.stack_bands_var.get())
        self.color_cfg.rgb_mode = bool(self.rgb_var.get())

        # Update band order entry for new mode
        self._sync_tuning_vars_from_config()

        self._invalidate_caches_and_render()
    
    def _on_zoom_change(self) -> None:
        """Snap the zoom slider to an integer step and rebuild textures at the new
        zoom's on-screen resolution (so detail matches the display - no aliasing).
        Scrolling within a zoom level stays a pure GPU draw."""
        step = round(self.zoom_var.get())
        self.zoom_var.set(step)
        if self.gpu is not None and step != self._tex_build_zoom:
            self._refresh_all_textures()

    # ------------------------------------------------------------------
    # GPU texture management (rebuilt only on track / config change)
    # ------------------------------------------------------------------
    def _apply_band_order(self) -> None:
        if self.overview_var.get():
            self.color_cfg.band_order = parse_band_order(self.color_cfg.band_order_string_overview)[0]
        else:
            self.color_cfg.band_order = parse_band_order(self.color_cfg.band_order_string_default)[0]

    def _refresh_all_textures(self) -> None:
        if self.gpu is None:
            return
        for i in range(self.deck_count):
            self._refresh_deck_texture(i)
        # Record the zoom + width these textures were built for (rebuild on change).
        self._tex_build_zoom = int(round(self.zoom_var.get()))
        self._tex_build_width = self.gpu.window.size[0]

    def _refresh_deck_texture(self, deck_num: int) -> None:
        """(Re)build a deck's full-track GPU texture from current analysis + config.

        The texture is sized to the on-screen resolution for the current zoom so
        the GPU scales ~1:1 (no minification aliasing). Beats are passed as overlay
        times, not baked. Peak reduction keeps the waveform full when downsampled.
        """
        if self.gpu is None:
            return
        deck = self.decks[deck_num]
        if deck.analysis is None:
            self.gpu.clear_track(deck_num)
            return
        self._apply_band_order()
        total = deck.get_total_duration()
        win_w = max(1, self.gpu.window.size[0])
        zoom_step = int(max(0, min(NUM_ZOOM_STEPS, self.zoom_var.get())))
        window_seconds = float(ZOOM_LEVELS_SECONDS[zoom_step])
        desired_pps = GPU_OVERSAMPLE * win_w / max(window_seconds, 1e-3)
        pps = self.gpu.clamp_pixels_per_second(desired_pps, total)
        result = deck.build_full_track_image(
            self.color_cfg,
            self.render_cfg,
            height=GPU_TEXTURE_HEIGHT,
            pixels_per_second=pps,
            reduce="max",
        )
        if result is None:
            self.gpu.clear_track(deck_num)
            return
        image, pps_out, total_out = result
        downbeats = deck.get_downbeat_times() if self.beat_grid_var.get() else ()
        cues = deck.get_cue_markers() if self.cue_var.get() else ()
        self.gpu.set_track(deck_num, image, pps_out, total_out, downbeats_sec=downbeats, cues=cues)

    def _on_deck_count_change(self) -> None:
        new_count = int(self.deck_count_var.get())
        if new_count == self.deck_count:
            return
        self._set_deck_count(new_count)

    def _set_deck_count(self, count: int) -> None:
        """Switch deck count: restart the memory reader and rebuild the UI."""
        # 1. Restart memory reader with new deck count
        if self.link_listener is not None:
            self.link_listener.stop()
            self.link_listener = None

        # 2. Update in-memory state
        old_count = self.deck_count
        self.deck_count = count
        self.render_cfg.deck_count = count

        # 3. Rebuild deck controllers (keep loaded tracks for existing deck slots)
        new_decks = {}
        for i in range(count):
            new_decks[i] = self.decks[i] if i < old_count else DeckController(LIBRARY_SEARCH_ROOT)
        self.decks = new_decks

        # 4. Rebuild playhead extrapolation state
        self._deck_play_time = {i: self._deck_play_time.get(i) for i in range(count)}
        self._deck_play_wall = {i: self._deck_play_wall.get(i) for i in range(count)}
        self._deck_play_rate = {i: self._deck_play_rate.get(i, 0.0) for i in range(count)}

        # 5. Rebuild deck info labels
        for lbl in self.deck_infos.values():
            lbl.destroy()
        self.deck_infos = {}
        for i in range(count):
            ctrl = self.decks[i]
            text = f"Deck {i+1}: {ctrl.song_name}" if ctrl.analysis is not None else f"Deck {i+1}: No track loaded"
            lbl = ttk.Label(self.waveform_area, text=text, anchor="w", style="Black.TLabel")
            lbl.grid(row=i, column=0, sticky="ew", pady=(0, 2))
            self.deck_infos[i] = lbl

        # 6. Resize tkinter window
        new_height = 50 + (count * 26) + 20
        self.root.geometry(f"{self.root.winfo_width()}x{new_height}")

        # 7. Recreate GPU window, preserving position and width
        old_pos = None
        old_w = None
        if self.gpu is not None:
            try:
                old_pos = self.gpu.window.position
                old_w = self.gpu.window.size[0]
            except Exception:
                pass
            gpu = self.gpu
            self.gpu = None
            gpu.close()
        self._tex_build_zoom = None
        self._tex_build_width = None
        screen_w = self.root.winfo_screenwidth()
        win_w = old_w if old_w is not None else int(screen_w * 0.8)
        win_h = count * 220
        bg = tuple(self.render_cfg.background_color)
        self.gpu = WaveformGpuWindow(
            count,
            size=(win_w, win_h),
            background=bg,
            beat_color=tuple(self.color_cfg.beat_color),
            beat_end_color=tuple(self.color_cfg.beat_end_color),
            beat_width=self.render_cfg.beat_width,
            beat_triangles=self.render_cfg.beat_triangles,
            borderless=not self.gpu_border_var.get(),
            always_on_top=self.gpu_on_top_var.get(),
            on_close=self._on_gpu_window_closed,
        )
        if old_pos is not None:
            try:
                self.gpu.window.position = old_pos
            except Exception:
                pass
        self._on_gpu_window_opts()
        self._refresh_all_textures()
        self.link_listener = RekordboxMemoryReader(deck_count=count)
        self.link_listener.start()
        self.link_status_var.set("RB Memory: connecting...")
        print(f"[Decks] Switched to {count}-deck mode")

    def _start_link_listener(self) -> None:
        """Start the Rekordbox memory reader and the GPU render/poll tick."""
        self.link_listener = RekordboxMemoryReader(deck_count=self.deck_count)
        self.link_listener.start()
        self.link_status_var.set("RB Memory: connecting...")
        self._tick()

    def _cancel_tick(self) -> None:
        if self._tick_job is not None:
            self.root.after_cancel(self._tick_job)
            self._tick_job = None

    def _tick(self) -> None:
        """~60fps: pump GPU window events, drain link events, draw the GPU frame."""
        self._tick_job = None
        if self.gpu is None:
            return
        # 1) GPU window events (close button, etc.)
        self.gpu.pump()
        if self.gpu is None:  # closed during pump
            return
        # 1b) rebuild textures if the window was resized (debounced)
        self._maybe_rebuild_on_resize()
        # 1c) global numpad +/- zoom (works while Rekordbox has focus)
        self._poll_zoom_hotkey()
        # 2) drain all pending link events into deck state (cheap, no drawing)
        if self.link_listener is not None:
            while True:
                event = self.link_listener.get_event(timeout=0.0)
                if event is None:
                    break
                self._handle_link_event(event)
        # 3) draw the GPU frame (scroll/zoom existing textures - no resize/upload)
        self._render_gpu_frame()
        # 4) periodically re-assert topmost and update connection status (~0.5s)
        self._topmost_tick = (getattr(self, "_topmost_tick", 0) + 1) % 30
        if self._topmost_tick == 0:
            self.gpu.reassert_topmost()
            if self.link_listener is not None:
                status = "RB Memory: connected" if self.link_listener.connected else "RB Memory: waiting..."
                self.link_status_var.set(status)
        self._tick_job = self.root.after(GPU_TICK_MS, self._tick)

    def _poll_zoom_hotkey(self) -> None:
        """Global zoom hotkeys, polled so they work even when the GPU window /
        Rekordbox has focus (the tk window need not be focused). Edge-triggered.
        Keys come from config (default numpad + / -)."""
        if not self.numpad_zoom_var.get():
            return
        try:
            gks = ctypes.windll.user32.GetAsyncKeyState
        except Exception:
            return
        in_down = bool(gks(self.render_cfg.zoom_in_key) & 0x8000)
        out_down = bool(gks(self.render_cfg.zoom_out_key) & 0x8000)
        if in_down and not self._zoom_key_down["in"]:
            self._nudge_zoom(+1)   # zoom in
        if out_down and not self._zoom_key_down["out"]:
            self._nudge_zoom(-1)   # zoom out
        self._zoom_key_down["in"] = in_down
        self._zoom_key_down["out"] = out_down

    def _update_zoom_key_buttons(self) -> None:
        if hasattr(self, "_zoom_in_key_btn"):
            self._zoom_in_key_btn.configure(text=_vk_label(self.render_cfg.zoom_in_key))
        if hasattr(self, "_zoom_out_key_btn"):
            self._zoom_out_key_btn.configure(text=_vk_label(self.render_cfg.zoom_out_key))

    def _capture_zoom_key(self, which: str) -> None:
        """Capture the next key press and assign it as a zoom hotkey. On Windows,
        tk's event.keycode is the Windows virtual-key code GetAsyncKeyState uses."""
        btn = self._zoom_in_key_btn if which == "in" else self._zoom_out_key_btn
        btn.configure(text="press key…")

        def on_key(ev):
            self.root.unbind("<KeyPress>")
            vk = int(ev.keycode)
            if which == "in":
                self.render_cfg.zoom_in_key = vk
            else:
                self.render_cfg.zoom_out_key = vk
            self._update_zoom_key_buttons()

        self.root.bind("<KeyPress>", on_key)
        self.root.focus_force()

    def _nudge_zoom(self, delta: int) -> None:
        """Step the zoom level by `delta` (clamped) and rebuild textures."""
        cur = int(round(self.zoom_var.get()))
        step = max(0, min(NUM_ZOOM_STEPS, cur + delta))
        if step != cur:
            self.zoom_var.set(step)
            self._on_zoom_change()

    def _render_gpu_frame(self) -> None:
        zoom_step = int(max(0, min(NUM_ZOOM_STEPS, self.zoom_var.get())))
        window_seconds = float(ZOOM_LEVELS_SECONDS[zoom_step])
        frames = []
        for i in range(self.deck_count):
            deck = self.decks[i]
            visible = window_seconds * max(deck.time_scale, 1e-6)
            frames.append(DeckFrame(
                current_time=self._deck_display_time(i),
                visible_seconds=visible,
                label=f"Deck {i+1}: {deck.song_name}" if deck.analysis is not None else "",
            ))
        self.gpu.render_frame(frames)

    def _maybe_rebuild_on_resize(self) -> None:
        """Rebuild textures when the GPU window width changes (debounced), so the
        on-screen resolution keeps matching the texture resolution."""
        if self.gpu is None or self._tex_build_width is None:
            return
        cur_w = self.gpu.window.size[0]
        if cur_w == self._tex_build_width:
            self._resize_pending_w = None
            return
        now = time.monotonic()
        if self._resize_pending_w != cur_w:
            self._resize_pending_w = cur_w
            self._resize_pending_at = now
            return
        if now - self._resize_pending_at >= 0.2:  # width stable -> rebuild once
            self._resize_pending_w = None
            self._refresh_all_textures()

    def _deck_display_time(self, deck_num: int) -> Optional[float]:
        """Playhead time, extrapolated from the last link update by measured rate."""
        t = self._deck_play_time[deck_num]
        if t is None:
            return None
        rate = self._deck_play_rate[deck_num]
        if rate <= 0.05:
            return t  # paused / unknown -> hold position
        wall = self._deck_play_wall[deck_num]
        if wall is None:
            return t
        dt = max(0.0, min(time.monotonic() - wall, 0.5))  # cap so lost updates can't run away
        return t + dt * rate

    def _record_play_time(self, deck_num: int, t: float) -> None:
        """Track playback rate from consecutive time updates (for extrapolation)."""
        now = time.monotonic()
        prev_t = self._deck_play_time[deck_num]
        prev_w = self._deck_play_wall[deck_num]
        if prev_t is not None and prev_w is not None:
            dw = now - prev_w
            dt = t - prev_t
            if dw > 1e-3:
                rate = dt / dw
                # Reject seeks/rewinds; clamp to a sane playback-speed range.
                if -0.1 <= rate <= 4.0 and abs(dt) < 5.0:
                    self._deck_play_rate[deck_num] = max(0.0, rate)
                else:
                    self._deck_play_rate[deck_num] = 0.0
        self._deck_play_time[deck_num] = t
        self._deck_play_wall[deck_num] = now

    def _handle_link_event(self, event: DeckEvent) -> None:
        """Route a Rekordbox Link event into the matching deck's state (no drawing)."""
        if event.deck not in self.decks:
            return
        ctrl = self.decks[event.deck]
        info_label = self.deck_infos[event.deck]
        deck_num = event.deck

        if event.time_seconds is not None:
            self._record_play_time(deck_num, float(event.time_seconds))
            ctrl.update_time(event.time_seconds)

        # Live BPM -> time scale (changes scroll speed only; texture is unaffected)
        ctrl.update_live_bpm(getattr(event, "current_bpm", None))
        ctrl.update_time_scale_fallback(getattr(event, "time_scale", None))

        # Load new track if path changed
        if event.anlz_path and (ctrl.anlz_path is None or Path(event.anlz_path) != ctrl.anlz_path):
            print(f"[Deck {deck_num}] New track: {event.anlz_path}")
            self._load_deck_anlz(ctrl, Path(event.anlz_path), info_label, deck_num)

    def _load_deck_anlz(self, ctrl: DeckController, anlz_folder: Path, info_label: ttk.Label, deck_num: int) -> None:
        """Load ANLZ data for a deck and (re)build its GPU texture."""
        if not anlz_folder.is_dir():
            print(f"[Deck {deck_num}] ANLZ folder not found: {anlz_folder}")
            return
        try:
            ctrl.load_anlz(anlz_folder)
            info_label.configure(text=f"Deck {deck_num+1}: {ctrl.song_name}")
            if ctrl.cached_waveform is not None and ctrl.cached_duration is not None:
                print(f"[Deck {deck_num}] Loaded: {len(ctrl.cached_waveform)} bins, {ctrl.cached_duration:.1f}s")
            else:
                print(f"[Deck {deck_num}] Loaded ANLZ")
            # Reset playhead extrapolation for the new track.
            self._deck_play_time[deck_num] = ctrl.current_time
            self._deck_play_wall[deck_num] = time.monotonic()
            self._deck_play_rate[deck_num] = 0.0
            self._refresh_deck_texture(deck_num)
        except Exception as e:
            print(f"[Deck {deck_num}] Failed to load ANLZ: {e}")

    def _on_gpu_window_closed(self) -> None:
        """The GPU waveform window was closed -> shut the whole app down (deferred)."""
        self.root.after(0, self._on_close)

    def _on_close(self) -> None:
        """Clean up on window close."""
        self._cancel_tick()
        if self.gpu is not None:
            gpu = self.gpu
            self.gpu = None
            gpu.close()
        if self.link_listener is not None:
            self.link_listener.stop()
        self.root.destroy()


def main():
    root = tk.Tk()
    root.geometry("1000x800")
    app = WaveformSyncApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
