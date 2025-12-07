# Copyright (c) mrmilbe

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from .ANLZ import analyze_anlz_folder, extract_beat_grid
from .analysis import WaveformAnalysis, analysis_from_rb_waveform, resolve_audio_path
from .playhead import (
    PrerenderCache,
    compute_timing_info,
    compute_window_plan,
    finalize_preview_image,
    render_window_image,
    reset_prerender_cache,
)


@dataclass
class RenderResult:
    image: Image.Image
    plan_start_time: float
    plan_window_duration: float


class DeckController:
    """Reusable deck orchestration: load ANLZ, manage BPM, render windows."""

    def __init__(self, library_root: Optional[Path] = None) -> None:
        self.library_root = library_root
        self.anlz_path: Optional[Path] = None
        self.cached_waveform: Optional[np.ndarray] = None
        self.cached_duration: Optional[float] = None
        self.song_name: str = "Unknown"
        self.resolved_audio_path: Optional[Path] = None
        self.analysis: Optional[WaveformAnalysis] = None
        self.beat_grid_full: Optional[list] = None
        self.beat_grid_downbeats: Optional[list] = None
        self.original_bpm: Optional[float] = None
        self.current_bpm: Optional[float] = None
        self.time_scale: float = 1.0
        self.current_time: Optional[float] = None
        self.prerender_cache = PrerenderCache()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_anlz(self, folder: Path) -> None:
        folder = Path(folder)
        if not folder.is_dir():
            raise FileNotFoundError(f"ANLZ folder not found: {folder}")

        info = analyze_anlz_folder(folder)
        self.anlz_path = folder
        self.cached_waveform = info.waveform
        self.cached_duration = info.duration
        self.song_name = info.song_path.stem if info.song_path else "Unknown"
        self.resolved_audio_path = resolve_audio_path(info.song_path, self.library_root)

        self.analysis = analysis_from_rb_waveform(self.cached_waveform, self.cached_duration)

        self.beat_grid_full = extract_beat_grid(folder)
        self.beat_grid_downbeats = _downbeats_only(self.beat_grid_full)
        self.original_bpm = _infer_original_bpm(self.beat_grid_full)
        self.current_bpm = self.original_bpm
        self.time_scale = 1.0
        self.current_time = 0.0
        reset_prerender_cache(self.prerender_cache)

    # ------------------------------------------------------------------
    # Live updates
    # ------------------------------------------------------------------
    def update_time(self, time_seconds: Optional[float]) -> None:
        self.current_time = time_seconds

    def update_live_bpm(self, bpm: Optional[float]) -> bool:
        changed = False
        if bpm is not None:
            try:
                self.current_bpm = float(bpm)
            except (TypeError, ValueError):
                self.current_bpm = None
            new_scale = _compute_time_scale(self.original_bpm, self.current_bpm)
            if new_scale is not None and not _nearly_equal(new_scale, self.time_scale, 1e-3):
                self.time_scale = new_scale
                reset_prerender_cache(self.prerender_cache)
                changed = True
        return changed

    def update_time_scale_fallback(self, scale: Optional[float]) -> bool:
        # Only use fallback if we cannot compute from BPM values.
        if self.original_bpm and self.current_bpm:
            return False
        if scale is None:
            return False
        try:
            new_scale = float(scale)
        except (TypeError, ValueError):
            return False
        if _nearly_equal(new_scale, self.time_scale, 1e-3):
            return False
        self.time_scale = new_scale
        reset_prerender_cache(self.prerender_cache)
        return True

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(
        self,
        preview_width: int,
        preview_height: int,
        zoom_seconds: float,
        color_cfg,
        render_cfg,
        beat_grid_enabled: bool,
    ) -> RenderResult:
        if self.analysis is None:
            raise RuntimeError("No analysis loaded")

        timing = compute_timing_info(self.analysis, fallback_duration=self.cached_duration or 0.0)
        plan = compute_window_plan(
            total_duration=timing.total_duration,
            n_bins=timing.n_bins,
            seconds_per_bin=timing.seconds_per_bin,
            window_duration=zoom_seconds,
            pan_fraction=0.0,
            link_follow=True,
            link_time=self.current_time,
            link_scale=self.time_scale,
        )

        render_cfg.image_height = preview_height
        img, cache_out = render_window_image(
            self.analysis,
            color_cfg,
            render_cfg,
            plan,
            seconds_per_bin=plan.scaled_seconds_per_bin,
            preview_width=preview_width,
            total_duration=timing.total_duration,
            use_live_render=False,
            cache=self.prerender_cache,
            zoom_fraction=1.0,
            link_scale=self.time_scale,
            beat_grid=self.beat_grid_downbeats if beat_grid_enabled else None,
            show_beat_grid=beat_grid_enabled,
        )
        self.prerender_cache = cache_out

        img = finalize_preview_image(
            img,
            target_width=preview_width,
            target_height=preview_height,
            draw_playhead=True,
            playhead_fraction=plan.playhead_fraction,
        )
        return RenderResult(image=img, plan_start_time=plan.start_time, plan_window_duration=plan.window_duration)


# ----------------------------------------------------------------------
# Helpers (kept local to the controller module)
# ----------------------------------------------------------------------

def _downbeats_only(beat_grid: Optional[list]) -> Optional[list]:
    if not beat_grid:
        return None
    downbeats = [entry for entry in beat_grid if getattr(entry, "beat_number", 0) % 4 == 1]
    return downbeats or None


def _infer_original_bpm(beat_grid: Optional[list]) -> Optional[float]:
    if not beat_grid:
        return None
    for entry in beat_grid:
        try:
            bpm = getattr(entry, "bpm", None)
            if bpm and bpm > 0:
                return float(bpm)
        except (TypeError, ValueError):
            continue
    if len(beat_grid) < 2:
        return None
    intervals = []
    prev = beat_grid[0]
    for current in beat_grid[1:]:
        delta = getattr(current, "time_ms", 0) - getattr(prev, "time_ms", 0)
        if delta > 0:
            intervals.append(delta)
        prev = current
    if not intervals:
        return None
    avg_ms = sum(intervals) / len(intervals)
    if avg_ms <= 0:
        return None
    return 60000.0 / avg_ms


def _compute_time_scale(original_bpm: Optional[float], current_bpm: Optional[float]) -> Optional[float]:
    if original_bpm and original_bpm > 0 and current_bpm and current_bpm > 0:
        # Faster live BPM => scale > 1 (show more time / denser beat markers)
        return float(current_bpm) / float(original_bpm)
    return None



def _nearly_equal(a: float, b: float, eps: float = 1e-4) -> bool:
    return abs(float(a) - float(b)) <= eps
