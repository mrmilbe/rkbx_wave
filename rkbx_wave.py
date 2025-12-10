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

import json
import shutil
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from pathlib import Path
from typing import Optional
import subprocess
import os

from PIL import ImageTk

from rb_waveform_core.config import (
    DEFAULT_COLOR_CONFIG,
    DEFAULT_RENDER_CONFIG,
    WaveformColorConfig,
    WaveformRenderConfig,
    RenderMode,
    config_to_dict,
    dict_to_config,
    parse_band_order,
)
from rb_waveform_core.deck_controller import DeckController
from rb_waveform_core.playhead import reset_prerender_cache
from rb_waveform_core.rkbx_link_listener import DeckEvent, RekordboxLinkListener


LIBRARY_SEARCH_ROOT: Optional[Path] = Path(r"C:\Rekordbox")
if not LIBRARY_SEARCH_ROOT.is_dir():
    LIBRARY_SEARCH_ROOT = None

# User config directory in AppData
USER_CONFIG_DIR = Path(os.environ.get("APPDATA", Path.home())) / "rkbx_wave"
USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
LAST_CONFIG_PATH = USER_CONFIG_DIR / "last_config.txt"
USER_DEFAULT_CONFIG_PATH = USER_CONFIG_DIR / "default_config.json"


def _get_package_data_path() -> Path:
    """Get path to package data directory (installed or development)."""
    # Try installed location first (sys.prefix/rkbx_wave_data)
    installed_path = Path(sys.prefix) / "rkbx_wave_data"
    if installed_path.exists():
        return installed_path
    # Fall back to development directory (next to this file)
    return Path(__file__).parent


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


def _get_rkbx_link_path() -> tuple[Path, Path]:
    """Get rkbx_link directory and executable path."""
    # Try installed location first
    installed_path = Path(sys.prefix) / "rkbx_wave_data" / "rkbx_link"
    if installed_path.exists():
        exe_path = installed_path / "rkbx_link.exe"
        if exe_path.exists():
            return installed_path, exe_path
    
    # Fall back to development directory
    dev_dir = Path(__file__).parent / "rkbx_link"
    if dev_dir.exists():
        exe_path = dev_dir / "rkbx_link.exe"
        if exe_path.exists():
            return dev_dir, exe_path
    
    # Return expected path (will fail at runtime if not found)
    return dev_dir, dev_dir / "rkbx_link.exe"


DEFAULT_CONFIG_PATH = _get_default_config_path()

# Match tuning_gui: discrete zoom levels in seconds
ZOOM_LEVELS_SECONDS = [256, 196, 128, 96, 64, 48, 32, 24, 16]
NUM_ZOOM_STEPS = len(ZOOM_LEVELS_SECONDS) - 1


class WaveformSyncApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("WaveformSync - Dual Deck Display")
        
        # Start rkbx_link.exe
        rkbx_link_dir, rkbx_link_exe = _get_rkbx_link_path()
        if not rkbx_link_exe.exists():
            messagebox.showerror(
                "Missing Dependency",
                f"rkbx_link.exe not found at:\n{rkbx_link_exe}\n\n"
                "Please ensure the package was installed correctly."
            )
            sys.exit(1)
        self.rkbx_link_proc = subprocess.Popen(
            [str(rkbx_link_exe)],
            cwd=str(rkbx_link_dir)
        )
        
        # Shared config (loaded from file)
        self.color_cfg: WaveformColorConfig = DEFAULT_COLOR_CONFIG
        self.render_cfg: WaveformRenderConfig = DEFAULT_RENDER_CONFIG
        
        # Support up to deck_count decks (from config)
        self.deck_count = self.render_cfg.deck_count
        self.decks = {i: DeckController(LIBRARY_SEARCH_ROOT) for i in range(self.deck_count)}
        self.deck_images_tk = {i: None for i in range(self.deck_count)}
        
        # Link listener
        self.link_listener: Optional[RekordboxLinkListener] = None
        self._link_poll_job: Optional[str] = None
        self.link_poll_interval_ms = 50
        self._is_rendering: bool = False
        self._waveform_resize_job: Optional[str] = None
        self._last_waveform_area_size: tuple = (0, 0)  # Track (width, height) of waveform_area
        
        # UI variables
        self.overview_var = tk.BooleanVar(value=False)
        self.stack_bands_var = tk.BooleanVar(value=False)
        self.beat_grid_var = tk.BooleanVar(value=True)  # Enable beat grid by default
        self.zoom_var = tk.IntVar(value=0)  # index into fixed zoom levels
        self.link_status_var = tk.StringVar(value="RB Link: disabled")
        
        # Tuning panel variables
        self.tune_panel_visible = tk.BooleanVar(value=False)
        self.render_mode_var = tk.StringVar(value=RenderMode.DEFAULT.value)
        self.band_order_var = tk.StringVar(value="l,m,h")  # Default 3-band order
        # Band gain variables (context-sensitive - will be synced based on overview mode)
        self.low_gain_var = tk.DoubleVar(value=1.0)
        self.mid_gain_var = tk.DoubleVar(value=1.0)
        self.high_gain_var = tk.DoubleVar(value=1.0)
        # Smoothing variable (context-sensitive)
        self.smoothing_var = tk.IntVar(value=1)
        
        self._load_last_or_default_config()
        self._build_ui()
        self._start_link_listener()
        
        # Calculate window size based on screen and content
        screen_width = self.root.winfo_screenwidth()
        window_width = int(screen_width * 0.8)  # 80% of screen width
        
        # Height: controls (~50px) + 2 decks (128px waveform + ~30px info each) + padding
        window_height = 50 + (2 * (128 + 30)) + 20  # = 50 + 316 + 20 = 386
        
        self.root.geometry(f"{window_width}x{window_height}")
        
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
    

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
    
    def _build_ui(self) -> None:
        self.root.configure(bg="black")
        
        # Create style for black frame
        style = ttk.Style()
        style.configure("Black.TFrame", background="black")
        style.configure("Black.TLabel", background="black", foreground="white")
        style.configure("Tune.TFrame", background="#1a1a1a")
        style.configure("Tune.TLabel", background="#1a1a1a", foreground="white")
        style.configure("Tune.TButton", background="#333333")
        
        main = ttk.Frame(self.root, style="Black.TFrame")
        main.pack(fill="both", expand=True, padx=2, pady=2)
        main.columnconfigure(1, weight=1)  # Waveform area expands
        main.rowconfigure(1, weight=1)  # Content area expands
        
        # Top controls (spans both columns)
        controls = ttk.Frame(main)
        controls.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        
        tune_btn = ttk.Button(controls, text="Tune", command=self._toggle_tune_panel)
        tune_btn.pack(side="left", padx=(0, 8))
        load_btn = ttk.Button(controls, text="Load Config", command=self._on_load_config)
        load_btn.pack(side="left", padx=(0, 2))
        save_btn = ttk.Button(controls, text="Save Config", command=self._on_save_config)
        save_btn.pack(side="left", padx=(0, 8))
        
        ttk.Checkbutton(
            controls,
            text="Overview Mode",
            variable=self.overview_var,
            command=self._on_visual_change,
        ).pack(side="left", padx=(0, 8))
        
        ttk.Checkbutton(
            controls,
            text="Stack Bands",
            variable=self.stack_bands_var,
            command=self._on_visual_change,
        ).pack(side="left", padx=(0, 8))
        
        ttk.Checkbutton(
            controls,
            text="Beat Grid",
            variable=self.beat_grid_var,
            command=self._on_visual_change,
        ).pack(side="left", padx=(0, 8))
        
        ttk.Label(controls, text="Zoom:").pack(side="left", padx=(8, 4))
        zoom_slider = ttk.Scale(
            controls,
            from_=0,
            to=NUM_ZOOM_STEPS,
            orient="horizontal",
            variable=self.zoom_var,
            command=lambda _v: self._on_zoom_change(),
        )
        zoom_slider.pack(side="left", padx=(0, 8))
        zoom_slider.configure(length=150)
        
        ttk.Label(controls, textvariable=self.link_status_var).pack(side="right")
        
        # Tuning panel (left side, collapsible)
        self.tune_panel = ttk.Frame(main, style="Tune.TFrame", width=200)
        # Don't grid it yet - will be shown/hidden by toggle
        self._build_tune_panel()
        
        # Waveform display area (right side)
        self.waveform_area = ttk.Frame(main, style="Black.TFrame")
        self.waveform_area.grid(row=1, column=1, sticky="nsew")
        self.waveform_area.columnconfigure(0, weight=1)
        self.waveform_area.bind("<Configure>", self._on_waveform_area_resize)
        
        # Deck displays (up to deck_count) - use Canvas for hardware-accelerated scaling
        self.deck_canvases = {}
        self.deck_canvas_images = {}  # Store canvas image IDs for updates
        self.deck_infos = {}
        for i in range(self.deck_count):
            # Waveform canvas row gets weight to expand vertically
            self.waveform_area.rowconfigure(2 * i, weight=1)
            canvas = tk.Canvas(self.waveform_area, bg="black", highlightthickness=0)
            canvas.grid(row=2 * i, column=0, sticky="nsew", pady=(2, 1))
            self.deck_canvases[i] = canvas
            self.deck_canvas_images[i] = None
            # Info label row stays fixed height
            self.waveform_area.rowconfigure(2 * i + 1, weight=0)
            self.deck_infos[i] = ttk.Label(self.waveform_area, text=f"Deck {i+1}: No track loaded", anchor="w", style="Black.TLabel")
            self.deck_infos[i].grid(row=2 * i + 1, column=0, sticky="ew", pady=(0, 2))
        
        # Keep deck_labels as alias for compatibility
        self.deck_labels = self.deck_canvases
    
    def _build_tune_panel(self) -> None:
        """Build the tuning subpanel contents."""
        panel = self.tune_panel
        row = 0
        
        # Title
        ttk.Label(panel, text="Tuning", style="Tune.TLabel", font=("TkDefaultFont", 10, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=8, pady=(8, 12)
        )
        row += 1
        
        # Render Mode dropdown
        ttk.Label(panel, text="Render Mode:", style="Tune.TLabel").grid(row=row, column=0, sticky="w", padx=8, pady=2)
        render_mode_combo = ttk.Combobox(
            panel,
            textvariable=self.render_mode_var,
            values=[m.value for m in RenderMode],
            state="readonly",
            width=10,
        )
        render_mode_combo.grid(row=row, column=1, sticky="w", padx=8, pady=2)
        render_mode_combo.bind("<<ComboboxSelected>>", lambda e: self._on_render_mode_change())
        row += 1
        
        # Separator
        ttk.Separator(panel, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", padx=8, pady=8)
        row += 1
        
        # Band Order (3-band only)
        ttk.Label(panel, text="Band Order (3):", style="Tune.TLabel").grid(row=row, column=0, sticky="w", padx=8, pady=2)
        band_order_entry = ttk.Entry(panel, textvariable=self.band_order_var, width=12)
        band_order_entry.grid(row=row, column=1, sticky="w", padx=8, pady=2)
        band_order_entry.bind("<Return>", lambda e: self._on_band_order_change())
        band_order_entry.bind("<FocusOut>", lambda e: self._on_band_order_change())
        row += 1
        
        # Band order hint
        ttk.Label(panel, text="(l,m,h)", style="Tune.TLabel", foreground="gray").grid(
            row=row, column=1, sticky="w", padx=8, pady=0
        )
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
        row += 1
        
        # Mid gain slider  
        ttk.Label(panel, text="Mid:", style="Tune.TLabel").grid(row=row, column=0, sticky="w", padx=8, pady=1)
        mid_scale = ttk.Scale(panel, from_=0.0, to=3.0, orient="horizontal", variable=self.mid_gain_var, length=100)
        mid_scale.grid(row=row, column=1, sticky="w", padx=8, pady=1)
        mid_scale.bind("<ButtonRelease-1>", lambda e: self._on_gain_change())
        mid_scale.bind("<B1-Motion>", lambda e: self._on_gain_change())  # Live update while dragging
        row += 1
        
        # High gain slider
        ttk.Label(panel, text="High:", style="Tune.TLabel").grid(row=row, column=0, sticky="w", padx=8, pady=1)
        high_scale = ttk.Scale(panel, from_=0.0, to=3.0, orient="horizontal", variable=self.high_gain_var, length=100)
        high_scale.grid(row=row, column=1, sticky="w", padx=8, pady=1)
        high_scale.bind("<ButtonRelease-1>", lambda e: self._on_gain_change())
        high_scale.bind("<B1-Motion>", lambda e: self._on_gain_change())  # Live update while dragging
        row += 1
        
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
    
    def _sync_tuning_vars_from_config(self) -> None:
        """Sync tuning panel variables from current config."""
        self.render_mode_var.set(self.render_cfg.render_mode.value)
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
    
    def _on_render_mode_change(self) -> None:
        """Handle render mode change."""
        try:
            mode = RenderMode(self.render_mode_var.get())
            self.render_cfg.render_mode = mode
            self.render_cfg.prerender_detail = mode.get_prerender_detail()
            # Invalidate prerender caches
            for i in range(self.deck_count):
                reset_prerender_cache(self.decks[i].prerender_cache)
            self._render_all_decks()
        except ValueError:
            pass
    
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
        # Invalidate and rerender
        for i in range(self.deck_count):
            reset_prerender_cache(self.decks[i].prerender_cache)
        self._render_all_decks()
    
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
        # Invalidate and rerender
        for i in range(self.deck_count):
            reset_prerender_cache(self.decks[i].prerender_cache)
        self._render_all_decks()
    
    def _on_smoothing_change(self) -> None:
        """Handle smoothing slider change - update appropriate config values."""
        is_overview = self.overview_var.get()
        smoothing_val = int(self.smoothing_var.get())
        if is_overview:
            self.render_cfg.overview_smoothing_bins = smoothing_val
        else:
            self.render_cfg.smoothing_bins = smoothing_val
        # Invalidate and rerender for live preview
        for i in range(self.deck_count):
            reset_prerender_cache(self.decks[i].prerender_cache)
        self._render_all_decks()
    
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
            self.overview_var.set(self.color_cfg.overview_mode)
            self.stack_bands_var.set(self.color_cfg.stack_bands)
            # Clear caches for all decks and rerender
            for i in range(self.deck_count):
                reset_prerender_cache(self.decks[i].prerender_cache)
            self._render_all_decks()
            messagebox.showinfo("Success", f"Configuration loaded from:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load configuration:\n{e}")
    
    def _on_visual_change(self) -> None:
        """Handle overview/stack mode toggle."""
        self.color_cfg.overview_mode = bool(self.overview_var.get())
        self.color_cfg.stack_bands = bool(self.stack_bands_var.get())

        # Update band order entry for new mode
        self._sync_tuning_vars_from_config()

        # Invalidate prerender cache for all decks
        for i in range(self.deck_count):
            reset_prerender_cache(self.decks[i].prerender_cache)

        self._render_all_decks()
    
    def _on_zoom_change(self) -> None:
        """Handle zoom slider change - snap to integer and rerender."""
        step = round(self.zoom_var.get())
        self.zoom_var.set(step)
        self._render_all_decks()
    
    def _render_all_decks(self) -> None:
        """Render all decks."""
        if self._waveform_resize_job is not None:
            self.root.after_cancel(self._waveform_resize_job)
            self._waveform_resize_job = None
        for i in range(self.deck_count):
            deck = self.decks[i]
            if deck.analysis is not None:
                self._render_deck(deck, self.deck_labels[i], deck_num=i)

    def _on_waveform_area_resize(self, event=None) -> None:
        """Throttle waveform renders during window resize."""
        if self._is_rendering:
            return  # Prevent recursive renders
        # Only trigger if size actually changed
        new_size = (self.waveform_area.winfo_width(), self.waveform_area.winfo_height())
        if new_size == self._last_waveform_area_size:
            return
        self._last_waveform_area_size = new_size
        if self._waveform_resize_job is not None:
            self.root.after_cancel(self._waveform_resize_job)
        self._waveform_resize_job = self.root.after(200, self._render_all_decks)
    
    def _start_link_listener(self) -> None:
        """Start Rekordbox Link listener for deck 0."""
        self.link_listener = RekordboxLinkListener()
        self.link_listener.start()
        self.link_status_var.set(f"RB Link: listening on {self.link_listener.port}")
        self._schedule_link_poll()
    
    def _schedule_link_poll(self) -> None:
        self._cancel_link_poll()
        self._poll_link_events()
    
    def _cancel_link_poll(self) -> None:
        if self._link_poll_job is not None:
            self.root.after_cancel(self._link_poll_job)
            self._link_poll_job = None
    
    def _poll_link_events(self) -> None:
        self._link_poll_job = None
        if self.link_listener is None:
            return
        
        # Frame skipping
        if self._is_rendering:
            self._link_poll_job = self.root.after(self.link_poll_interval_ms, self._poll_link_events)
            return
        
        while True:
            event = self.link_listener.get_event(timeout=0.0)
            if event is None:
                break
            self._handle_link_event(event)
        
        self._link_poll_job = self.root.after(self.link_poll_interval_ms, self._poll_link_events)
    
    def _handle_link_event(self, event: DeckEvent) -> None:
        """Handle incoming Rekordbox Link event - route to correct deck controller."""
        if event.deck not in self.decks:
            return
        ctrl = self.decks[event.deck]
        label = self.deck_labels[event.deck]
        info_label = self.deck_infos[event.deck]
        deck_num = event.deck

        # Update time
        if event.time_seconds is not None:
            ctrl.update_time(event.time_seconds)

        # Live BPM -> compute scale; fall back to Link-provided scale if needed
        ctrl.update_live_bpm(getattr(event, "current_bpm", None))
        ctrl.update_time_scale_fallback(getattr(event, "time_scale", None))

        # Load new track if path changed
        if event.anlz_path and (ctrl.anlz_path is None or Path(event.anlz_path) != ctrl.anlz_path):
            print(f"[Deck {deck_num}] New track: {event.anlz_path}")
            self._load_deck_anlz(ctrl, Path(event.anlz_path), info_label, deck_num)

        # Render only this deck with updated time
        if ctrl.analysis is not None and not self._is_rendering:
            self._is_rendering = True
            try:
                self._render_deck(ctrl, label, deck_num=deck_num)
            finally:
                self._is_rendering = False
    
    def _load_deck_anlz(self, ctrl: DeckController, anlz_folder: Path, info_label: ttk.Label, deck_num: int) -> None:
        """Load ANLZ data for a deck controller."""
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
            # Immediately render after loading
            label = self.deck_labels[deck_num]
            self._render_deck(ctrl, label, deck_num=deck_num)
        except Exception as e:
            print(f"[Deck {deck_num}] Failed to load ANLZ: {e}")
    
    def _render_deck(self, ctrl: DeckController, label, deck_num: int) -> None:
        """Render waveform for a specific deck via DeckController."""
        if ctrl.analysis is None:
            return

        canvas = self.deck_canvases[deck_num]
        canvas_w = max(400, canvas.winfo_width() or 800)
        canvas_h = max(64, canvas.winfo_height() or 128)
        
        # Use canvas width for render, but fixed height from RenderMode
        preview_w = canvas_w
        # preview_h is determined by RenderMode in deck_controller, we just pass display height
        preview_h = canvas_h

        zoom_step = int(max(0, min(NUM_ZOOM_STEPS, self.zoom_var.get())))
        window_seconds = float(ZOOM_LEVELS_SECONDS[zoom_step])

        # Set band order for rendering based on current mode
        if self.overview_var.get():
            self.color_cfg.band_order = parse_band_order(self.color_cfg.band_order_string_overview)[0]
        else:
            self.color_cfg.band_order = parse_band_order(self.color_cfg.band_order_string_default)[0]

        result = ctrl.render(
            preview_width=preview_w,
            preview_height=preview_h,
            zoom_seconds=window_seconds,
            color_cfg=self.color_cfg,
            render_cfg=self.render_cfg,
            beat_grid_enabled=bool(self.beat_grid_var.get()),
        )

        # Create PhotoImage from rendered image (at fixed render height)
        img_tk = ImageTk.PhotoImage(result.image)
        self.deck_images_tk[deck_num] = img_tk
        
        # Update canvas - clear old image and create new one centered/stretched
        canvas.delete("all")
        # Place image at center of canvas - canvas will clip if image is larger
        # For vertical scaling: image is at fixed height, canvas shows it at its natural size
        # The image width matches canvas width, so horizontal is 1:1
        # Vertical: image is smaller, centered in canvas
        img_x = canvas_w // 2
        img_y = canvas_h // 2
        self.deck_canvas_images[deck_num] = canvas.create_image(img_x, img_y, image=img_tk, anchor="center")
    
    def _on_close(self) -> None:
        """Clean up on window close."""
        self._cancel_link_poll()
        if self.link_listener is not None:
            self.link_listener.stop()
        # Stop rkbx_link.exe
        if hasattr(self, 'rkbx_link_proc') and self.rkbx_link_proc is not None:
            self.rkbx_link_proc.terminate()
            self.rkbx_link_proc.wait()
        self.root.destroy()


def main():
    root = tk.Tk()
    root.geometry("1000x800")
    app = WaveformSyncApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
