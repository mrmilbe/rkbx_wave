# Copyright (c) HorstHorstmann

"""WaveformSync GUI - Dual deck display for Rekordbox Link.

Displays two decks vertically with live waveform sync.
Uses saved configuration files for waveform rendering.

Run with:
    python waveformsync_gui.py
"""

from __future__ import annotations

import json
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from pathlib import Path
from typing import Optional

from PIL import ImageTk

from rb_waveform_lab.config import (
    DEFAULT_ANALYSIS_CONFIG,
    DEFAULT_COLOR_CONFIG,
    DEFAULT_RENDER_CONFIG,
    WaveformAnalysisConfig,
    WaveformColorConfig,
    WaveformRenderConfig,
    dict_to_config,
)
from rb_waveform_lab.deck_controller import DeckController
from rb_waveform_lab.playhead import reset_prerender_cache
from rb_waveform_lab.rkbx_link_listener import DeckEvent, RekordboxLinkListener


LIBRARY_SEARCH_ROOT: Optional[Path] = Path(r"C:\Rekordbox")
if not LIBRARY_SEARCH_ROOT.is_dir():
    LIBRARY_SEARCH_ROOT = None

DEFAULT_CONFIG_PATH = Path("waveform_config.json")

# Match tuning_gui: discrete zoom levels in seconds
ZOOM_LEVELS_SECONDS = [256, 196, 128, 96, 64, 48, 32, 24, 16]
NUM_ZOOM_STEPS = len(ZOOM_LEVELS_SECONDS) - 1


class WaveformSyncApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("WaveformSync - Dual Deck Display")
        
        # Shared config (loaded from file)
        self.analysis_cfg: WaveformAnalysisConfig = DEFAULT_ANALYSIS_CONFIG
        self.color_cfg: WaveformColorConfig = DEFAULT_COLOR_CONFIG
        self.render_cfg: WaveformRenderConfig = DEFAULT_RENDER_CONFIG
        
        # Dual deck controllers (render + ANLZ/BPM logic)
        self.deck_a = DeckController(LIBRARY_SEARCH_ROOT)
        self.deck_b = DeckController(LIBRARY_SEARCH_ROOT)
        self.deck_a_image_tk: Optional[ImageTk.PhotoImage] = None
        self.deck_b_image_tk: Optional[ImageTk.PhotoImage] = None
        
        # Link listener
        self.link_listener: Optional[RekordboxLinkListener] = None
        self._link_poll_job: Optional[str] = None
        self.link_poll_interval_ms = 50
        self._is_rendering: bool = False
        
        # UI variables
        self.overview_var = tk.BooleanVar(value=False)
        self.stack_bands_var = tk.BooleanVar(value=False)
        self.beat_grid_var = tk.BooleanVar(value=False)
        self.zoom_var = tk.IntVar(value=0)  # index into fixed zoom levels
        self.link_status_var = tk.StringVar(value="RB Link: disabled")
        
        self._load_default_config()
        self._build_ui()
        self._start_link_listener()
        
        # Calculate window size based on screen and content
        screen_width = self.root.winfo_screenwidth()
        window_width = int(screen_width * 0.8)  # 80% of screen width
        
        # Height: controls (~50px) + 2 decks (128px waveform + ~30px info each) + padding
        window_height = 50 + (2 * (128 + 30)) + 20  # = 50 + 316 + 20 = 386
        
        self.root.geometry(f"{window_width}x{window_height}")
        
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _load_default_config(self) -> None:
        """Load configuration from default file if it exists."""
        if DEFAULT_CONFIG_PATH.exists():
            try:
                with open(DEFAULT_CONFIG_PATH, 'r') as f:
                    config_dict = json.load(f)
                self.analysis_cfg, self.color_cfg, self.render_cfg = dict_to_config(config_dict)
                print(f"[Config] Loaded from {DEFAULT_CONFIG_PATH}")
            except Exception as e:
                print(f"[Config] Failed to load default config: {e}")
    
    def _build_ui(self) -> None:
        self.root.configure(bg="black")
        
        # Create style for black frame
        style = ttk.Style()
        style.configure("Black.TFrame", background="black")
        style.configure("Black.TLabel", background="black", foreground="white")
        
        main = ttk.Frame(self.root, style="Black.TFrame")
        main.pack(fill="both", expand=False, padx=2, pady=2)
        main.columnconfigure(0, weight=1)
        
        # Top controls
        controls = ttk.Frame(main)
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        
        ttk.Button(controls, text="Load Config", command=self._on_load_config).pack(side="left", padx=(0, 8))
        
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
        
        # Deck 1 display (no frame border)
        self.deck_a_label = ttk.Label(main, relief="flat", background="black")
        self.deck_a_label.grid(row=1, column=0, sticky="ew", pady=(2, 1))
        
        self.deck_a_info = ttk.Label(main, text="Deck 1: No track loaded", anchor="w", style="Black.TLabel")
        self.deck_a_info.grid(row=2, column=0, sticky="ew", pady=(0, 2))
        
        # Deck 2 display (no frame border)
        self.deck_b_label = ttk.Label(main, relief="flat", background="black")
        self.deck_b_label.grid(row=3, column=0, sticky="ew", pady=(1, 1))
        
        self.deck_b_info = ttk.Label(main, text="Deck 2: No track loaded", anchor="w", style="Black.TLabel")
        self.deck_b_info.grid(row=4, column=0, sticky="ew", pady=(0, 2))
    
    def _on_load_config(self) -> None:
        """Load configuration from file."""
        file_path = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not file_path:
            return
        
        try:
            with open(file_path, 'r') as f:
                config_dict = json.load(f)
            self.analysis_cfg, self.color_cfg, self.render_cfg = dict_to_config(config_dict)
            
            # Clear caches for both decks and rerender
            reset_prerender_cache(self.deck_a.prerender_cache)
            reset_prerender_cache(self.deck_b.prerender_cache)
            
            self._render_all_decks()
            
            messagebox.showinfo("Success", f"Configuration loaded from:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load configuration:\n{e}")
    
    def _on_visual_change(self) -> None:
        """Handle overview/stack mode toggle."""
        self.color_cfg.overview_mode = bool(self.overview_var.get())
        self.color_cfg.stack_bands = bool(self.stack_bands_var.get())
        
        # Invalidate prerender cache for both decks
        reset_prerender_cache(self.deck_a.prerender_cache)
        reset_prerender_cache(self.deck_b.prerender_cache)
        
        self._render_all_decks()
    
    def _on_zoom_change(self) -> None:
        """Handle zoom slider change - snap to integer and rerender."""
        step = round(self.zoom_var.get())
        self.zoom_var.set(step)
        self._render_all_decks()
    
    def _render_all_decks(self) -> None:
        """Render both decks."""
        if self.deck_a.analysis is not None:
            self._render_deck(self.deck_a, self.deck_a_label, is_deck_a=True)
        if self.deck_b.analysis is not None:
            self._render_deck(self.deck_b, self.deck_b_label, is_deck_a=False)
    
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
        if event.deck == 0:
            ctrl = self.deck_a
            label = self.deck_a_label
            info_label = self.deck_a_info
            is_deck_a = True
        elif event.deck == 1:
            ctrl = self.deck_b
            label = self.deck_b_label
            info_label = self.deck_b_info
            is_deck_a = False
        else:
            return

        # Update time
        if event.time_seconds is not None:
            ctrl.update_time(event.time_seconds)

        # Live BPM -> compute scale; fall back to Link-provided scale if needed
        ctrl.update_live_bpm(getattr(event, "current_bpm", None))
        ctrl.update_time_scale_fallback(getattr(event, "time_scale", None))

        # Load new track if path changed
        if event.anlz_path and (ctrl.anlz_path is None or Path(event.anlz_path) != ctrl.anlz_path):
            deck_name = "Deck A" if is_deck_a else "Deck B"
            print(f"[{deck_name}] New track: {event.anlz_path}")
            self._load_deck_anlz(ctrl, Path(event.anlz_path), info_label)

        # Render only this deck with updated time
        if ctrl.analysis is not None and not self._is_rendering:
            self._is_rendering = True
            try:
                self._render_deck(ctrl, label, is_deck_a=is_deck_a)
            finally:
                self._is_rendering = False
    
    def _load_deck_anlz(self, ctrl: DeckController, anlz_folder: Path, info_label: ttk.Label) -> None:
        """Load ANLZ data for a deck controller."""
        deck_name = "Deck A" if ctrl is self.deck_a else "Deck B"
        
        if not anlz_folder.is_dir():
            print(f"[{deck_name}] ANLZ folder not found: {anlz_folder}")
            return
        
        try:
            ctrl.load_anlz(anlz_folder)
            deck_number = 1 if ctrl is self.deck_a else 2
            info_label.configure(text=f"Deck {deck_number}: {ctrl.song_name}")
            if ctrl.cached_waveform is not None and ctrl.cached_duration is not None:
                print(f"[{deck_name}] Loaded: {len(ctrl.cached_waveform)} bins, {ctrl.cached_duration:.1f}s")
            else:
                print(f"[{deck_name}] Loaded ANLZ")
        except Exception as e:
            print(f"[{deck_name}] Failed to load ANLZ: {e}")
    
    def _render_deck(self, ctrl: DeckController, label: ttk.Label, *, is_deck_a: bool) -> None:
        """Render waveform for a specific deck via DeckController."""
        if ctrl.analysis is None:
            return

        preview_w = max(400, label.winfo_width() or 800)
        preview_h = 128
        zoom_step = int(max(0, min(NUM_ZOOM_STEPS, self.zoom_var.get())))
        window_seconds = float(ZOOM_LEVELS_SECONDS[zoom_step])

        result = ctrl.render(
            preview_width=preview_w,
            preview_height=preview_h,
            zoom_seconds=window_seconds,
            color_cfg=self.color_cfg,
            render_cfg=self.render_cfg,
            beat_grid_enabled=bool(self.beat_grid_var.get()),
        )

        img_tk = ImageTk.PhotoImage(result.image)
        if is_deck_a:
            self.deck_a_image_tk = img_tk
        else:
            self.deck_b_image_tk = img_tk
        label.configure(image=img_tk)
    
    def _on_close(self) -> None:
        """Clean up on window close."""
        self._cancel_link_poll()
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
