# Copyright (c) mrmilbe

from __future__ import annotations

from dataclasses import replace
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .analysis import WaveformAnalysis
from .config import WaveformColorConfig, WaveformRenderConfig


def _band_array(analysis: WaveformAnalysis, name: str) -> np.ndarray:
	if name == "low":
		return analysis.low
	if name == "mid":
		return analysis.mid
	if name == "high":
		return analysis.high
	# Legacy aliases for lab compatibility
	if name == "lowmid":
		return analysis.mid  # lowmid -> mid
	if name == "midhigh":
		return analysis.mid  # midhigh -> mid
	raise ValueError(f"Unknown band name: {name}")


def _smooth_array(data: np.ndarray, window: int) -> np.ndarray:
	if window <= 1 or data.size == 0:
		return data.astype(np.float32, copy=False)
	kernel = np.ones(window, dtype=np.float32) / float(window)
	return np.convolve(data.astype(np.float32, copy=False), kernel, mode="same").astype(np.float32, copy=False)


def _apply_gain(data: np.ndarray, gain: float) -> np.ndarray:
	if gain == 1.0:
		return data
	return data * float(gain)


def _extract_band_window(band: np.ndarray, start_bin: int, window_bins: int) -> np.ndarray:
	window_bins = max(0, int(window_bins))
	window = np.zeros(window_bins, dtype=np.float32)
	if window_bins == 0:
		return window
	start = int(start_bin)
	src_start = max(0, start)
	src_end = min(len(band), start + window_bins)
	copy_offset = src_start - start
	copy_len = max(0, src_end - src_start)
	if copy_len == 0 or copy_offset >= window_bins:
		return window
	copy_len = min(copy_len, window_bins - copy_offset)
	window[copy_offset : copy_offset + copy_len] = band[src_start : src_start + copy_len]
	return window


def _build_window_analysis(
	analysis: WaveformAnalysis,
	render_cfg: WaveformRenderConfig,
	start_bin: int,
	window_bins: int,
	seconds_per_bin: float,
	use_overview_gains: bool = False,
) -> WaveformAnalysis:
	window_bins = max(1, int(window_bins))
	sec_per_bin = max(1e-9, float(seconds_per_bin))
	low = _extract_band_window(analysis.low, start_bin, window_bins)
	mid = _extract_band_window(analysis.mid, start_bin, window_bins)
	high = _extract_band_window(analysis.high, start_bin, window_bins)
	
	# Select appropriate smoothing and gains based on overview mode
	if use_overview_gains:
		smoothing = max(1, int(getattr(render_cfg, "overview_smoothing_bins", 30)))
	else:
		smoothing = max(1, int(getattr(render_cfg, "smoothing_bins", 1)))
	
	if use_overview_gains:
		low_gain = getattr(render_cfg, "overview_low_gain", 1.0)
		mid_gain = getattr(render_cfg, "overview_mid_gain", getattr(render_cfg, "overview_midhigh_gain", 1.0))
		high_gain = getattr(render_cfg, "overview_high_gain", 1.0)
	else:
		low_gain = getattr(render_cfg, "low_gain", 1.0)
		mid_gain = getattr(render_cfg, "mid_gain", getattr(render_cfg, "midhigh_gain", 1.0))
		high_gain = getattr(render_cfg, "high_gain", 1.0)
	
	low_processed = _apply_gain(_smooth_array(low, smoothing), low_gain)
	mid_processed = _apply_gain(_smooth_array(mid, smoothing), mid_gain)
	high_processed = _apply_gain(_smooth_array(high, smoothing), high_gain)
	return WaveformAnalysis(
		low=low_processed.astype(np.float32, copy=False),
		mid=mid_processed.astype(np.float32, copy=False),
		high=high_processed.astype(np.float32, copy=False),
		duration_seconds=float(window_bins * sec_per_bin),
		sample_rate=analysis.sample_rate,
		config_version=analysis.config_version,
	)


def _average_columns(values: np.ndarray, start_idx: np.ndarray, end_idx: np.ndarray) -> np.ndarray:
	if values.size == 0 or start_idx.size == 0:
		return np.zeros_like(start_idx, dtype=np.float32)
	float_vals = np.asarray(values, dtype=np.float32)
	prefix = np.concatenate(([0.0], np.cumsum(float_vals, dtype=np.float64)))
	sums = prefix[end_idx] - prefix[start_idx]
	lengths = np.maximum(1, end_idx - start_idx).astype(np.float32)
	return (sums / lengths).astype(np.float32)


def render_waveform_image(
	analysis: WaveformAnalysis,
	color_cfg: WaveformColorConfig,
	render_cfg: WaveformRenderConfig,
	beat_grid: Optional[list] = None,
	show_beat_grid: bool = False,
	seconds_per_bin: float = 0.0,
) -> Image.Image:

	w = render_cfg.image_width
	h = render_cfg.image_height
	margin = render_cfg.margin_px

	img = Image.new("RGB", (w, h), color_cfg.low_color if False else render_cfg.background_color)
	draw = ImageDraw.Draw(img)

	usable_h = h - 2 * margin
	center_y = h // 2
	half_h = usable_h // 2

	n_bins = len(analysis.low)
	if n_bins == 0:
		return img

	column_edges = np.linspace(0.0, n_bins, w + 1)
	column_start = np.floor(column_edges[:-1]).astype(int)
	column_end = np.ceil(column_edges[1:]).astype(int)
	column_start = np.clip(column_start, 0, max(0, n_bins - 1))
	column_end = np.clip(column_end, column_start + 1, n_bins)

	band_color_map = {
		"low": color_cfg.low_color,
		"mid": color_cfg.mid_color,
		"high": color_cfg.high_color,
		# Legacy aliases (for lab compatibility)
		"lowmid": color_cfg.lowmid_color,
		"midhigh": color_cfg.midhigh_color,
	}

	# Use band_order directly - should be 3 bands (low, mid, high)
	render_band_order = [b for b in color_cfg.band_order if b in band_color_map]
	# Limit to 3 bands max
	if len(render_band_order) > 3:
		render_band_order = render_band_order[:3]
	elif len(render_band_order) == 0:
		return img

	if color_cfg.stack_bands:
		# ...existing code for stack_bands...
		band_count = len(render_band_order)
		if band_count == 0:
			return img
		lane_height = usable_h // band_count
		for idx, band_name in enumerate(render_band_order):
			vals = np.clip(_band_array(analysis, band_name), 0.0, 1.0)
			column_values = _average_columns(vals, column_start, column_end)
			heights = (column_values * lane_height).astype(int)
			color = band_color_map[band_name]
			lane_top = margin + idx * lane_height
			lane_bottom = lane_top + lane_height
			for x in range(w):
				v = heights[x]
				if v <= 0:
					continue
				y0 = lane_bottom
				y1 = max(lane_top, lane_bottom - v)
				draw.line((x, int(y0), x, int(y1)), fill=color, width=1)
	elif color_cfg.overview_mode:
		# Rekordbox-style stacked overview rendering
		# Only works for 3 bands: low, mid, high
		# Get arrays for each band
		low_vals = np.clip(_band_array(analysis, "low"), 0.0, 1.0)
		mid_vals = np.clip(_band_array(analysis, "mid"), 0.0, 1.0)
		high_vals = np.clip(_band_array(analysis, "high"), 0.0, 1.0)
		low_col = _average_columns(low_vals, column_start, column_end)
		mid_col = _average_columns(mid_vals, column_start, column_end)
		high_col = _average_columns(high_vals, column_start, column_end)
		# Compute stacked heights
		low_height = (low_col * usable_h).astype(int)
		mid_height = ((low_col + mid_col) * usable_h).astype(int)
		high_height = ((low_col + mid_col + high_col) * usable_h).astype(int)
		# Draw low band (bottom)
		for x in range(w):
			v = low_height[x]
			if v <= 0:
				continue
			y0 = h - margin
			y1 = y0 - v
			draw.line((x, int(y0), x, int(y1)), fill=band_color_map["low"], width=1)
		# Draw mid band (middle, stacked on low)
		for x in range(w):
			v0 = low_height[x]
			v1 = mid_height[x]
			if v1 <= v0:
				continue
			y0 = h - margin - v0
			y1 = h - margin - v1
			draw.line((x, int(y0), x, int(y1)), fill=band_color_map["mid"], width=1)
		# Draw high band (top, stacked on mid)
		for x in range(w):
			v0 = mid_height[x]
			v1 = high_height[x]
			if v1 <= v0:
				continue
			y0 = h - margin - v0
			y1 = h - margin - v1
			draw.line((x, int(y0), x, int(y1)), fill=band_color_map["high"], width=1)
	else:
		# Overlaid/symmetric (non-overview) mode
		for band_name in render_band_order:
			vals = np.clip(_band_array(analysis, band_name), 0.0, 1.0)
			column_values = _average_columns(vals, column_start, column_end)
			color = band_color_map[band_name]
			heights = (column_values * half_h).astype(int)
			for x in range(w):
				v = heights[x]
				if v <= 0:
					continue
				y0 = center_y + v
				y1 = center_y - v
				draw.line((x, int(y0), x, int(y1)), fill=color, width=1)

	if render_cfg.center_line:
		draw.line((0, center_y, w, center_y), fill=(40, 40, 40), width=1)

	# Draw beat grid lines (in front of waveform)
	if show_beat_grid and beat_grid and seconds_per_bin > 0:
		for entry in beat_grid:
			# Convert beat time (ms) to bin position
			beat_time_sec = entry.time_ms / 1000.0
			beat_bin = beat_time_sec / seconds_per_bin
			# Convert bin to pixel x coordinate
			beat_x = int((beat_bin / n_bins) * w)
			if 0 <= beat_x < w:
				draw.line((beat_x, 0, beat_x, h), fill=(255, 255, 255), width=2)
				circle_radius = 2  # 4px diameter
				# Top circle
				draw.ellipse(
					(beat_x - circle_radius, 0 - circle_radius, beat_x + circle_radius, 0 + circle_radius),
					fill=(255, 0, 0)
				)
				# Bottom circle
				draw.ellipse(
					(beat_x - circle_radius, h - circle_radius, beat_x + circle_radius, h + circle_radius),
					fill=(255, 0, 0)
				)

	return img


def render_waveform_window(
	analysis: WaveformAnalysis,
	color_cfg: WaveformColorConfig,
	render_cfg: WaveformRenderConfig,
	*,
	start_bin: int,
	window_bins: int,
	seconds_per_bin: float,
	beat_grid: Optional[list] = None,
	show_beat_grid: bool = False,
	scaled_seconds_per_bin: Optional[float] = None,
) -> Image.Image:
	# Use scaled timing for waveform rendering, original timing for beat grid
	render_spb = scaled_seconds_per_bin if scaled_seconds_per_bin is not None else seconds_per_bin
	use_overview_gains = getattr(color_cfg, "overview_mode", False)
	window_analysis = _build_window_analysis(analysis, render_cfg, start_bin, window_bins, render_spb, use_overview_gains)
	return render_waveform_image(window_analysis, color_cfg, render_cfg, beat_grid=beat_grid, show_beat_grid=show_beat_grid, seconds_per_bin=seconds_per_bin)


def prerender_full_waveform(
	analysis: WaveformAnalysis,
	color_cfg: WaveformColorConfig,
	render_cfg: WaveformRenderConfig,
	*,
	seconds_per_bin: float,
	target_width: int,
	beat_grid: Optional[list] = None,
	show_beat_grid: bool = False,
	scaled_seconds_per_bin: Optional[float] = None,
) -> Image.Image:
	width = max(1, int(target_width))
	temp_cfg = replace(render_cfg, image_width=width)
	n_bins = len(analysis.low)
	if n_bins == 0:
		return Image.new("RGB", (width, render_cfg.image_height), render_cfg.background_color)
	return render_waveform_window(
		analysis,
		color_cfg,
		temp_cfg,
		start_bin=0,
		window_bins=n_bins,
		seconds_per_bin=seconds_per_bin,
		beat_grid=beat_grid,
		show_beat_grid=show_beat_grid,
		scaled_seconds_per_bin=scaled_seconds_per_bin,
	)


def crop_prerendered_image(
	prerender_image: Image.Image,
	total_duration: float,
	preview_width: int,
	desired_start_time: float,
	window_duration: float,
	background_color: Tuple[int, int, int],
) -> Image.Image:
	"""Crop a prerendered image to show the desired time window.
	
	When the prerender is zoom-aware (width = preview_width / zoom), this becomes
	a pure pixel crop with no resize - very fast for live playback.
	"""
	preview_width = max(1, int(preview_width))
	
	# FAST PATH: If showing entire image and dimensions match, just return it
	# This is the zoomed-out case - no crop, no copy needed
	if (prerender_image.width == preview_width 
			and desired_start_time <= 0 
			and window_duration >= total_duration - 0.001):
		return prerender_image  # Direct reference - caller must copy if modifying
	
	if total_duration <= 0 or window_duration <= 0 or prerender_image.width <= 1:
		if prerender_image.width == preview_width:
			return prerender_image
		return prerender_image.resize((preview_width, prerender_image.height), Image.NEAREST)
	
	desired_end_time = desired_start_time + window_duration
	data_start = max(0.0, desired_start_time)
	data_end = min(total_duration, desired_end_time)
	if data_end <= data_start:
		return Image.new("RGB", (preview_width, prerender_image.height), background_color)
	
	# Calculate source crop region
	span = max(total_duration, 1e-9)
	src_left = int(np.floor((data_start / span) * (prerender_image.width - 1)))
	src_right = int(np.ceil((data_end / span) * (prerender_image.width - 1)))
	src_left = max(0, min(prerender_image.width - 1, src_left))
	src_right = max(src_left + 1, min(prerender_image.width, src_right))
	
	# Calculate destination region
	window_safe = max(window_duration, 1e-9)
	dest_left = int(round(((data_start - desired_start_time) / window_safe) * preview_width))
	dest_right = int(round(((data_end - desired_start_time) / window_safe) * preview_width))
	dest_left = max(0, min(preview_width, dest_left))
	dest_right = max(dest_left + 1, min(preview_width, max(dest_left + 1, dest_right)))
	
	dest_width = dest_right - dest_left
	src_width = src_right - src_left
	
	# FAST PATH: Full-width crop with matching dimensions - just crop, no paste
	if dest_left == 0 and dest_width == preview_width and src_width == dest_width:
		return prerender_image.crop((src_left, 0, src_right, prerender_image.height))
	
	# Standard path: crop segment and paste onto output
	output = Image.new("RGB", (preview_width, prerender_image.height), background_color)
	segment = prerender_image.crop((src_left, 0, src_right, prerender_image.height))
	
	# Fast path: if src and dest widths match (zoom-aware prerender), just paste
	if src_width == dest_width:
		output.paste(segment, (dest_left, 0))
	elif abs(src_width - dest_width) <= 2:
		# Within 2 pixels - use NEAREST (rounding errors)
		segment = segment.resize((dest_width, prerender_image.height), Image.NEAREST)
		output.paste(segment, (dest_left, 0))
	else:
		# Fallback: actual resize needed
		segment = segment.resize((dest_width, prerender_image.height), Image.BILINEAR)
		output.paste(segment, (dest_left, 0))
	return output


def save_waveform_png(
	analysis: WaveformAnalysis,
	out_path: str,
	color_cfg: WaveformColorConfig,
	render_cfg: WaveformRenderConfig,
) -> None:
	img = render_waveform_image(analysis, color_cfg, render_cfg)
	img.save(out_path, format="PNG")

