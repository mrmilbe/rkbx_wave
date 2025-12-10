# Copyright (c) mrmilbe

"""Audio analysis pipeline for waveform generation (Lab-only).

Full four-band spectral analysis with filtering, compression, and normalization.
For experimental/tuning use only—production code uses Rekordbox PWV data.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

from rb_waveform_core.analysis import WaveformAnalysis
from rb_waveform_lab.config import WaveformAnalysisConfig
from rb_waveform_lab.compression import compress_amplitude, normalize_band


def _design_filter_bank(sr: int, cfg: WaveformAnalysisConfig):
	nyq = 0.5 * sr
	low_hi = cfg.low_cutoff_hz / nyq
	lowmid_lo = cfg.low_cutoff_hz / nyq
	lowmid_hi = cfg.lowmid_cutoff_hz / nyq
	midhigh_lo = cfg.lowmid_cutoff_hz / nyq
	midhigh_hi = cfg.midhigh_cutoff_hz / nyq
	high_lo = cfg.high_cutoff_hz / nyq

	low_sos = butter(4, low_hi, btype="low", output="sos")
	lowmid_sos = butter(4, [lowmid_lo, lowmid_hi], btype="band", output="sos")
	midhigh_sos = butter(4, [midhigh_lo, midhigh_hi], btype="band", output="sos")
	high_sos = butter(4, high_lo, btype="high", output="sos")
	return low_sos, lowmid_sos, midhigh_sos, high_sos


def _frame_rms(x: np.ndarray, frame_size: int) -> np.ndarray:
	if x.size == 0:
		return np.zeros(0, dtype=np.float32)
	n_frames = int(np.ceil(len(x) / frame_size))
	pad = n_frames * frame_size - len(x)
	if pad:
		x = np.pad(x, (0, pad), mode="constant")
	frames = x.reshape(n_frames, frame_size)
	rms = np.sqrt(np.mean(frames ** 2, axis=1, dtype=np.float64)).astype("float32")
	return rms


def _resample_bins(x: np.ndarray, target_bins: int) -> np.ndarray:
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


def _smooth_box(x: np.ndarray, window: int) -> np.ndarray:
	if window <= 1 or x.size == 0:
		return x
	kernel = np.ones(window, dtype=np.float32) / float(window)
	return np.convolve(x, kernel, mode="same").astype("float32")


def analyze_bands(
	mono_signal: np.ndarray,
	sr: int,
	duration_seconds: float,
	cfg: WaveformAnalysisConfig,
) -> WaveformAnalysis:
	"""Full pipeline: filter into bands, frame RMS, bin, smooth, compress, normalize."""

	x = mono_signal.astype("float32", copy=False)
	if cfg.pre_emphasis > 0.0 and x.size > 1:
		alpha = float(cfg.pre_emphasis)
		x = np.concatenate([[x[0]], x[1:] - alpha * x[:-1]]).astype("float32")

	low_sos, lowmid_sos, midhigh_sos, high_sos = _design_filter_bank(sr, cfg)
	low_sig = sosfilt(low_sos, x).astype("float32")
	lowmid_sig = sosfilt(lowmid_sos, x).astype("float32")
	midhigh_sig = sosfilt(midhigh_sos, x).astype("float32")
	high_sig = sosfilt(high_sos, x).astype("float32")

	low_env = _frame_rms(low_sig, cfg.frame_size)
	lowmid_env = _frame_rms(lowmid_sig, cfg.frame_size)
	midhigh_env = _frame_rms(midhigh_sig, cfg.frame_size)
	high_env = _frame_rms(high_sig, cfg.frame_size)

	low_bins = _resample_bins(low_env, cfg.target_bins)
	lowmid_bins = _resample_bins(lowmid_env, cfg.target_bins)
	midhigh_bins = _resample_bins(midhigh_env, cfg.target_bins)
	high_bins = _resample_bins(high_env, cfg.target_bins)

	if cfg.smoothing_window_bins > 1:
		low_bins = _smooth_box(low_bins, cfg.smoothing_window_bins)
		lowmid_bins = _smooth_box(lowmid_bins, cfg.smoothing_window_bins)
		midhigh_bins = _smooth_box(midhigh_bins, cfg.smoothing_window_bins)
		high_bins = _smooth_box(high_bins, cfg.smoothing_window_bins)

	low_bins = compress_amplitude(low_bins, cfg.compression_mode, cfg.compression_strength)
	lowmid_bins = compress_amplitude(lowmid_bins, cfg.compression_mode, cfg.compression_strength)
	midhigh_bins = compress_amplitude(midhigh_bins, cfg.compression_mode, cfg.compression_strength)
	high_bins = compress_amplitude(high_bins, cfg.compression_mode, cfg.compression_strength)

	low_bins = normalize_band(low_bins, cfg.normalization_mode, cfg.normalization_percentile)
	lowmid_bins = normalize_band(lowmid_bins, cfg.normalization_mode, cfg.normalization_percentile)
	midhigh_bins = normalize_band(midhigh_bins, cfg.normalization_mode, cfg.normalization_percentile)
	high_bins = normalize_band(high_bins, cfg.normalization_mode, cfg.normalization_percentile)

	low_bins *= cfg.low_band_gain
	lowmid_bins *= cfg.lowmid_band_gain
	midhigh_bins *= cfg.midhigh_band_gain
	high_bins *= cfg.high_band_gain

	return WaveformAnalysis(
		low=low_bins.astype("float32"),
		lowmid=lowmid_bins.astype("float32"),
		midhigh=midhigh_bins.astype("float32"),
		high=high_bins.astype("float32"),
		duration_seconds=float(duration_seconds),
		sample_rate=int(sr),
		config_version=int(cfg.config_version),
	)


# --- Resampling utilities (moved from core, used by antialias.py) ---

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


def _downsample_bins_averaged(x: np.ndarray, target_bins: int) -> np.ndarray:
	"""Downsample by averaging bins - proper anti-aliasing for display.
	
	When target_bins < len(x), each output bin is the average of multiple input bins.
	This prevents aliasing artifacts when displaying zoomed-out waveforms.
	"""
	if x.size == 0 or target_bins <= 0:
		return np.zeros(target_bins, dtype=np.float32)
	if len(x) == target_bins:
		return x.astype("float32", copy=False)
	if len(x) < target_bins:
		# Upsampling - use linear interpolation
		return _resample_bins(x, target_bins)
	
	# Downsampling - use proper averaging
	n_src = len(x)
	result = np.zeros(target_bins, dtype=np.float32)
	
	# Use cumsum for fast range averaging
	cumsum = np.concatenate(([0.0], np.cumsum(x.astype(np.float64))))
	
	for i in range(target_bins):
		# Map output bin i to source range
		src_start = (i * n_src) / target_bins
		src_end = ((i + 1) * n_src) / target_bins
		
		# Integer bounds
		s_int = int(np.floor(src_start))
		e_int = int(np.ceil(src_end))
		e_int = min(e_int, n_src)
		
		if e_int <= s_int:
			continue
		
		# Simple average of all bins in range
		total = cumsum[e_int] - cumsum[s_int]
		count = e_int - s_int
		result[i] = total / count
	
	return result


def resample_analysis(analysis: WaveformAnalysis, target_bins: int) -> WaveformAnalysis:
	"""Resample a WaveformAnalysis to a different number of bins.
	
	Uses proper averaging when downsampling to prevent aliasing artifacts.
	This is useful for live rendering at screen resolution without wobbling.
	
	Args:
		analysis: Original analysis data.
		target_bins: Desired number of bins in output.
	
	Returns:
		New WaveformAnalysis with resampled data.
	"""
	if target_bins <= 0:
		target_bins = 1
	
	n_bins = len(analysis.low)
	if n_bins == target_bins:
		return analysis  # No change needed
	
	return WaveformAnalysis(
		low=_downsample_bins_averaged(analysis.low, target_bins),
		mid=_downsample_bins_averaged(analysis.mid, target_bins),
		high=_downsample_bins_averaged(analysis.high, target_bins),
		duration_seconds=analysis.duration_seconds,
		sample_rate=analysis.sample_rate,
		config_version=analysis.config_version,
	)
