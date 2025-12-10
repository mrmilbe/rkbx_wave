# Copyright (c) mrmilbe

"""Core waveform analysis data structures (PWV conversion only).

For audio analysis pipeline, see rb_waveform_lab.analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class WaveformAnalysis:
	"""Waveform analysis data structure.
	
	Core uses 3 bands matching ANLZ file: low, mid, high
	"""
	low: np.ndarray
	mid: np.ndarray      # ANLZ col 1 - mid frequencies
	high: np.ndarray     # ANLZ col 2 - high frequencies
	duration_seconds: float
	sample_rate: int
	config_version: int


def _resample_bins(x: np.ndarray, target_bins: int) -> np.ndarray:
	"""Linear interpolation resampling (internal helper)."""
	if x.size == 0 or target_bins <= 0:
		return np.zeros(target_bins, dtype=np.float32)
	if len(x) == target_bins:
		return x.astype("float32", copy=False)

	idx = np.linspace(0.0, len(x) - 1, target_bins, dtype=np.float64)
	idx_floor = np.floor(idx).astype(int)
	idx_ceil = np.minimum(idx_floor + 1, len(x) - 1)
	w = idx - idx_floor
	out = (1.0 - w) * x[idx_floor] + w * x[idx_ceil]
	return out.astype("float32")


def analysis_from_rb_waveform(
	waveform: np.ndarray,
	duration: float | None,
) -> WaveformAnalysis:
	"""Convert a Rekordbox PWV waveform array (N x 3) to WaveformAnalysis.

	The PWV format stores low/mid/high as columns 0/1/2. We map:
	  col 0 -> low
	  col 1 -> mid
	  col 2 -> high
	"""
	waveform = np.asarray(waveform)
	if waveform.size == 0:
		return WaveformAnalysis(
			low=np.zeros(0, dtype=np.float32),
			mid=np.zeros(0, dtype=np.float32),
			high=np.zeros(0, dtype=np.float32),
			duration_seconds=float(duration or 0.0),
			sample_rate=0,
			config_version=0,
		)

	data = np.clip(waveform.astype(np.float32), 0.0, None)
	max_vals = np.max(data, axis=0)
	max_vals[max_vals == 0.0] = 1.0
	normalized = data / max_vals
	estimated_duration = duration if (duration and duration > 0.0) else len(data) / 75.0

	return WaveformAnalysis(
		low=normalized[:, 0].astype(np.float32, copy=False),
		mid=normalized[:, 1].astype(np.float32, copy=False),
		high=normalized[:, 2].astype(np.float32, copy=False),
		duration_seconds=float(estimated_duration),
		sample_rate=0,
		config_version=0,
	)


def resolve_audio_path(
	raw_rb_path: Optional[Path],
	library_search_root: Optional[Path] = None
) -> Optional[Path]:
	"""Resolve Rekordbox's raw PPTH path to actual audio file.
	
	Args:
		raw_rb_path: Raw path from Rekordbox ANLZ file (typically Mac-style).
		library_search_root: Root directory to search for audio files.
	
	Returns:
		Resolved Path to audio file, or None if not found.
	"""
	if raw_rb_path is None:
		return None
	
	# Try the raw path as-is (Windows path might work directly)
	if raw_rb_path.is_file():
		return raw_rb_path
	
	# If library search root provided, try finding by filename
	if library_search_root and library_search_root.is_dir():
		target_name = raw_rb_path.name
		for candidate in library_search_root.rglob("*"):
			if candidate.is_file() and candidate.name == target_name:
				return candidate
	
	return None

