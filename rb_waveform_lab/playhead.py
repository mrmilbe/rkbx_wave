# Copyright (c) HorstHorstmann

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from .analysis import WaveformAnalysis
from .config import WaveformColorConfig, WaveformRenderConfig
from .render import crop_prerendered_image, prerender_full_waveform, render_waveform_window


@dataclass
class WindowPlan:
    start_bin: int
    window_bins: int
    window_duration: float
    start_time: float
    zoom_label: str
    pan_enabled: bool
    playhead_fraction: Optional[float]
    scaled_seconds_per_bin: float  # Adjusted for link_scale (for waveform rendering)


@dataclass
class TimingInfo:
    """Timing parameters derived from analysis data."""
    total_duration: float
    n_bins: int
    seconds_per_bin: float


def compute_timing_info(
    analysis: WaveformAnalysis,
    fallback_duration: float = 0.0,
) -> TimingInfo:
    """Compute timing info from a WaveformAnalysis.

    Args:
        analysis: The waveform analysis data.
        fallback_duration: Duration to use if analysis has no duration.

    Returns:
        TimingInfo with computed values.
    """
    n_bins = len(analysis.low)
    if n_bins == 0:
        return TimingInfo(total_duration=0.0, n_bins=0, seconds_per_bin=0.0)

    total_duration = analysis.duration_seconds if analysis.duration_seconds > 0 else max(fallback_duration, 1e-3)
    seconds_per_bin = total_duration / max(n_bins, 1)
    # Ensure a minimum to avoid division issues
    seconds_per_bin = max(seconds_per_bin, 1e-6)

    return TimingInfo(
        total_duration=total_duration,
        n_bins=n_bins,
        seconds_per_bin=seconds_per_bin,
    )


@dataclass
class PrerenderCache:
    image: Optional[Image.Image] = None
    width: int = 0
    total_duration: float = 0.0
    seconds_per_bin: float = 0.0
    zoom_fraction: float = 1.0  # Track zoom level for re-render detection
    link_scale: float = 1.0  # Track BPM scale for re-render detection
    dirty: bool = True


def reset_prerender_cache(cache: PrerenderCache) -> None:
    cache.image = None
    cache.width = 0
    cache.total_duration = 0.0
    cache.seconds_per_bin = 0.0
    cache.zoom_fraction = 1.0
    cache.link_scale = 1.0
    cache.dirty = True


def compute_window_plan(
    *,
    total_duration: float,
    n_bins: int,
    seconds_per_bin: float,
    window_duration: float,
    pan_fraction: float,
    link_follow: bool,
    link_time: Optional[float],
    link_scale: float,
) -> WindowPlan:
    sec_per_bin = seconds_per_bin if seconds_per_bin > 1e-9 else 1e-9
    total_duration = max(total_duration, sec_per_bin)
    n_bins = max(1, n_bins)

    # Base window from zoom (seconds at 1.0x BPM)
    base_window = max(1.0, float(window_duration))
    # BPM factor scales visible duration in time
    bpm_factor = max(link_scale, 1e-6)
    visible_duration = base_window * bpm_factor
    window_bins = max(1, int(round(visible_duration / sec_per_bin)))

    desired_start_time = 0.0
    pan_enabled = True
    playhead_fraction: Optional[float] = None

    if link_follow and link_time is not None:
        pan_enabled = False
        # Use link_time directly (seconds on the deck timeline)
        target_time = max(0.0, min(total_duration, link_time))
        desired_start_time = target_time - visible_duration / 2.0
        zoom_label = f"{base_window:.1f}s"
        playhead_fraction = 0.5
    else:
        if window_bins >= n_bins:
            pan_enabled = False
            # Clamp so full track is centered in the window
            visible_duration = min(visible_duration, total_duration)
            desired_start_time = max(0.0, (total_duration - visible_duration) / 2.0)
        else:
            pan = max(0.0, min(1.0, pan_fraction))
            max_start = max(0.0, total_duration - visible_duration)
            desired_start_time = pan * max_start
        zoom_label = f"{base_window:.1f}s"

    # Clamp start so window stays roughly in [0, total_duration]
    desired_start_time = max(-visible_duration, min(total_duration, desired_start_time))
    start_bin = int(np.floor(desired_start_time / sec_per_bin)) if sec_per_bin > 0 else 0

    # Keep seconds_per_bin fixed; BPM only affects visible_duration/window_bins
    scaled_sec_per_bin = sec_per_bin

    return WindowPlan(
        start_bin=start_bin,
        window_bins=window_bins,
        window_duration=visible_duration,
        start_time=desired_start_time,
        zoom_label=zoom_label,
        pan_enabled=pan_enabled,
        playhead_fraction=playhead_fraction,
        scaled_seconds_per_bin=scaled_sec_per_bin,
    )


def get_pan_slider_state(
    plan: WindowPlan,
    n_bins: int,
    link_follow: bool,
) -> Tuple[bool, str]:
    """Determine whether to reset pan and what slider state to use.

    Args:
        plan: The computed window plan.
        n_bins: Total number of bins in the analysis.
        link_follow: Whether link follow mode is active.

    Returns:
        (should_reset_pan, slider_state) where slider_state is "normal" or "disabled".
    """
    should_reset_pan = not link_follow and plan.window_bins >= n_bins
    slider_state = "normal" if (plan.pan_enabled and not link_follow) else "disabled"
    return should_reset_pan, slider_state


def ensure_prerender_cache(
    cache: PrerenderCache,
    *,
    analysis: WaveformAnalysis,
    color_cfg: WaveformColorConfig,
    render_cfg: WaveformRenderConfig,
    total_duration: float,
    seconds_per_bin: float,
    preview_width: int,
    zoom_fraction: float = 1.0,
    beat_grid: Optional[list] = None,
    show_beat_grid: bool = False,
    link_scale: float = 1.0,
) -> None:
    # Fixed-resolution prerender: BPM must not change full-track width.
    # Zoom and BPM only decide which time window we crop from this image.
    BASE_PIXELS_PER_SECOND = 25.0
    calculated_width = int(total_duration * BASE_PIXELS_PER_SECOND)
	
    # Clamp to reasonable bounds
    max_detail = getattr(render_cfg, "prerender_detail", 20000)
    target_prerender_width = max(preview_width, min(calculated_width, max_detail))
	
    needs_prerender = (
        cache.dirty
        or cache.image is None
        or cache.width != target_prerender_width
        or abs(cache.seconds_per_bin - seconds_per_bin) > 1e-9
    )
    if not needs_prerender:
        return
    
    from time import perf_counter
    t0 = perf_counter()
    
    cache.image = prerender_full_waveform(
        analysis,
        color_cfg,
        render_cfg,
        seconds_per_bin=seconds_per_bin,
        target_width=target_prerender_width,
        beat_grid=beat_grid,
        show_beat_grid=show_beat_grid,
        scaled_seconds_per_bin=seconds_per_bin,  # Use original for full prerender
    )
    t1 = perf_counter()
    
    cache.width = target_prerender_width
    cache.total_duration = max(total_duration, 1e-6)
    cache.seconds_per_bin = seconds_per_bin
    cache.link_scale = link_scale
    cache.dirty = False


def render_window_image(
    analysis: WaveformAnalysis,
    color_cfg: WaveformColorConfig,
    render_cfg: WaveformRenderConfig,
    plan: WindowPlan,
    *,
    seconds_per_bin: float,
    preview_width: int,
    total_duration: float,
    use_live_render: bool,
    cache: PrerenderCache,
    zoom_fraction: float = 1.0,
    link_scale: float = 1.0,
    beat_grid: Optional[list] = None,
    show_beat_grid: bool = False,
) -> Tuple[Image.Image, PrerenderCache]:
    if use_live_render:
        # Live render: analysis is the antialiased visible window (starts at bin 0)
        # Render the entire window since it's already extracted and downsampled
        # Note: scaled_seconds_per_bin not used in live render (antialiasing handles scaling)
        image = render_waveform_window(
            analysis,
            color_cfg,
            render_cfg,
            start_bin=0,
            window_bins=len(analysis.low),
            seconds_per_bin=seconds_per_bin,
            beat_grid=beat_grid,
            show_beat_grid=show_beat_grid,
            scaled_seconds_per_bin=seconds_per_bin,
        )
        return image, cache

    # Prerender path - uses cached full-resolution rendering
    ensure_prerender_cache(
        cache,
        analysis=analysis,
        color_cfg=color_cfg,
        render_cfg=render_cfg,
        total_duration=total_duration,
        seconds_per_bin=seconds_per_bin,
        preview_width=preview_width,
        zoom_fraction=zoom_fraction,
        beat_grid=beat_grid,
        show_beat_grid=show_beat_grid,
        link_scale=link_scale,
    )
    if cache.image is None:
        raise RuntimeError("Prerender cache is empty after ensure_prerender_cache")
    
    # FAST PATH: When showing full track (zoom >= 0.95), skip crop entirely
    # The prerender width == preview_width at zoom=1.0, so just return it directly
    # ONLY if window fits track exactly. If window > track (zoomed out), we need crop/pad.
    if zoom_fraction >= 0.95 and cache.image.width == preview_width and plan.window_duration <= total_duration:
        return cache.image, cache  # Direct reference - finalize_preview_image will copy if needed
    
    cropped = crop_prerendered_image(
        cache.image,
        cache.total_duration or total_duration,
        preview_width,
        plan.start_time,
        plan.window_duration,
        render_cfg.background_color,
    )
    return cropped, cache


def draw_playhead_line(
    img: Image.Image,
    playhead_fraction: Optional[float],
    color: Tuple[int, int, int] = (255, 0, 0),
    width: int = 2,
    in_place: bool = False,
) -> Image.Image:
    """Draw a vertical playhead line on the image at the given fraction (0.0-1.0).
    
    Args:
        img: Image to draw on.
        playhead_fraction: Position (0.0-1.0), defaults to 0.5.
        color: Line color RGB tuple.
        width: Line width in pixels.
        in_place: If True, draw on img directly (faster). If False, copy first.
    
    Returns:
        Image with the line drawn.
    """
    from PIL import ImageDraw
    
    if playhead_fraction is None:
        playhead_fraction = 0.5
    
    img_width, img_height = img.size
    center_x = int(round(playhead_fraction * (img_width - 1)))
    
    if in_place:
        draw = ImageDraw.Draw(img)
        draw.line([(center_x, 0), (center_x, img_height)], fill=color, width=width)
        return img
    
    # Use PIL copy + ImageDraw - faster than numpy array conversion for large images
    result = img.copy()
    draw = ImageDraw.Draw(result)
    draw.line([(center_x, 0), (center_x, img_height)], fill=color, width=width)
    return result


def finalize_preview_image(
    img: Image.Image,
    target_width: int,
    target_height: int,
    draw_playhead: bool = False,
    playhead_fraction: Optional[float] = None,
) -> Image.Image:
    """Resize image to target dimensions and optionally draw playhead.

    Args:
        img: Source image from render_window_image.
        target_width: Desired output width.
        target_height: Desired output height.
        draw_playhead: Whether to draw a playhead line.
        playhead_fraction: Position of playhead (0.0-1.0), defaults to 0.5.

    Returns:
        Finalized image ready for display.
    """
    from PIL import Image as PILImage
    
    needs_resize = img.width != target_width or img.height != target_height
    
    if needs_resize:
        # Resize creates a new image, so we can draw playhead in-place
        img = img.resize((target_width, target_height), PILImage.BILINEAR)
        if draw_playhead:
            img = draw_playhead_line(img, playhead_fraction, in_place=True)
    elif draw_playhead:
        # No resize, but we need playhead - must copy to avoid modifying cache
        img = draw_playhead_line(img, playhead_fraction, in_place=False)
    # else: no resize, no playhead - return as-is (reference to cache is OK)
    
    return img
