# Copyright (c) HorstHorstmann

"""Antialiasing cache for waveform display.

Downsamples high-resolution waveforms to screen resolution to prevent aliasing
artifacts during rendering. Caches results for performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .analysis import WaveformAnalysis


@dataclass
class AntialiasCache:
    """Cache for antialiased waveforms.
    
    Stores one downsampled waveform window and invalidates when parameters change.
    """
    cache_key: Optional[Tuple[int, int, int]]  # (start_bin, window_bins, target_pixels)
    antialiased: Optional[WaveformAnalysis]
    
    def __init__(self):
        self.cache_key = None
        self.antialiased = None


def reset_antialias_cache(cache: AntialiasCache) -> None:
    """Clear cached downsampled waveform."""
    cache.cache_key = None
    cache.antialiased = None


def get_antialiased_waveform(
    cache: AntialiasCache,
    analysis: WaveformAnalysis,
    start_bin: int,
    window_bins: int,
    target_pixels: int,
) -> WaveformAnalysis:
    """Downsample visible window to target pixel resolution.
    
    Args:
        cache: AntialiasCache instance for caching.
        analysis: Source waveform analysis.
        start_bin: Starting bin index of visible window.
        window_bins: Number of bins in visible window.
        target_pixels: Target screen width in pixels.
    
    Returns:
        Downsampled WaveformAnalysis for the visible window.
        Returns window as-is if window_bins <= target_pixels (no downsampling needed).
    """
    # Extract visible window first
    total_bins = len(analysis.low)
    end_bin = start_bin + window_bins
    
    # Calculate padding
    pad_left = max(0, -start_bin)
    pad_right = max(0, end_bin - total_bins)
    
    # Calculate valid slice indices
    valid_start = max(0, start_bin)
    valid_end = min(total_bins, end_bin)
    
    # Extract valid data
    if valid_start < valid_end:
        low = analysis.low[valid_start:valid_end]
        lowmid = analysis.lowmid[valid_start:valid_end]
        midhigh = analysis.midhigh[valid_start:valid_end]
        high = analysis.high[valid_start:valid_end]
    else:
        # Window is completely outside valid range
        low = np.zeros(0, dtype=np.float32)
        lowmid = np.zeros(0, dtype=np.float32)
        midhigh = np.zeros(0, dtype=np.float32)
        high = np.zeros(0, dtype=np.float32)

    # Apply padding if needed
    if pad_left > 0 or pad_right > 0:
        low = np.pad(low, (pad_left, pad_right), mode='constant')
        lowmid = np.pad(lowmid, (pad_left, pad_right), mode='constant')
        midhigh = np.pad(midhigh, (pad_left, pad_right), mode='constant')
        high = np.pad(high, (pad_left, pad_right), mode='constant')
    
    actual_window_bins = len(low)
    
    # Pure math: do we need downsampling?
    if actual_window_bins <= target_pixels:
        # No downsampling needed (will upscale during render)
        return WaveformAnalysis(
            low=low,
            lowmid=lowmid,
            midhigh=midhigh,
            high=high,
            duration_seconds=analysis.duration_seconds * actual_window_bins / max(1, total_bins),
            sample_rate=analysis.sample_rate,
            config_version=analysis.config_version,
        )
    
    # Check cache
    cache_key = (start_bin, window_bins, target_pixels)
    if cache.cache_key == cache_key and cache.antialiased is not None:
        return cache.antialiased
    
    # Create analysis object for the window
    window_analysis = WaveformAnalysis(
        low=low,
        lowmid=lowmid,
        midhigh=midhigh,
        high=high,
        duration_seconds=analysis.duration_seconds * actual_window_bins / max(1, total_bins),
        sample_rate=analysis.sample_rate,
        config_version=analysis.config_version,
    )
    
    # Downsample to target_pixels using proper antialiasing
    from .analysis import resample_analysis
    antialiased = resample_analysis(window_analysis, target_pixels)
    
    # Update cache
    cache.cache_key = cache_key
    cache.antialiased = antialiased
    
    return antialiased
