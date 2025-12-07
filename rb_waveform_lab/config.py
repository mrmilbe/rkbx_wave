# Copyright (c) HorstHorstmann

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Tuple


CompressionMode = Literal["log", "power", "soft", "linear"]
NormalizationMode = Literal["peak", "rms", "percentile"]


@dataclass
class WaveformAnalysisConfig:
	target_sample_rate: int = 11025
	frame_size: int = 512
	target_bins: int = 30000

	low_cutoff_hz: float = 150.0       # low:      0–150
	lowmid_cutoff_hz: float = 800.0    # lowmid:   150–800
	midhigh_cutoff_hz: float = 4000.0  # midhigh:  800–4000
	high_cutoff_hz: float = 4000.0     # high:     >4000

	smoothing_window_bins: int = 9

	compression_mode: CompressionMode = "soft"
	compression_strength: float = 0.7

	normalization_mode: NormalizationMode = "percentile"
	normalization_percentile: float = 0.99

	low_band_gain: float = 1.0
	lowmid_band_gain: float = 1.0
	midhigh_band_gain: float = 1.0
	high_band_gain: float = 1.0

	pre_emphasis: float = 0.0
	config_version: int = 1


@dataclass
class WaveformColorConfig:
	low_color: Tuple[int, int, int] = (0, 120, 255)
	lowmid_color: Tuple[int, int, int] = (255, 120, 0)
	midhigh_color: Tuple[int, int, int] = (150, 90, 40)
	high_color: Tuple[int, int, int] = (255, 255, 255)

	band_order: Tuple[str, ...] = ("low", "lowmid", "midhigh", "high")
	overview_mode: bool = False  # asymmetric multi-band overview vs symmetric
	stack_bands: bool = False    # true vertical stacking (no overlap)
	blend_bands: bool = True
	band_order_string: str = "l,lm,mh,h"  # editable order hint (short names)


@dataclass
class WaveformRenderConfig:
	image_width: int = 9600
	image_height: int = 256
	background_color: Tuple[int, int, int] = (5, 5, 5)
	line_width: int = 1
	margin_px: int = 4
	center_line: bool = False
	smoothing_bins: int = 1
	low_gain: float = 1.0
	lowmid_gain: float = 1.0
	midhigh_gain: float = 1.0
	high_gain: float = 1.0
	prerender_detail: int = 20000  # Max prerender width in pixels (dynamic: 5k zoomed in, up to this zoomed out)


DEFAULT_ANALYSIS_CONFIG = WaveformAnalysisConfig()
DEFAULT_COLOR_CONFIG = WaveformColorConfig()
DEFAULT_RENDER_CONFIG = WaveformRenderConfig()


# Short name -> internal band name mapping
BAND_SHORT_NAMES = {"l": "low", "lm": "lowmid", "mh": "midhigh", "h": "high"}
DEFAULT_BAND_ORDER = ("low", "lowmid", "midhigh", "high")


def parse_band_order(raw_string: str) -> Tuple[Tuple[str, ...], str]:
	"""Parse a comma-separated band order string (e.g. 'l,lm,mh,h').

	Returns:
		(band_order_tuple, normalized_string)
		Falls back to default order if nothing valid is parsed.
	"""
	if not raw_string or not raw_string.strip():
		return DEFAULT_BAND_ORDER, "l,lm,mh,h"

	tokens = [t.strip().lower() for t in raw_string.split(",") if t.strip()]
	mapped = [BAND_SHORT_NAMES[t] for t in tokens if t in BAND_SHORT_NAMES]

	if mapped:
		return tuple(mapped), ",".join(tokens)
	return DEFAULT_BAND_ORDER, "l,lm,mh,h"


def config_to_dict(analysis_cfg: WaveformAnalysisConfig, color_cfg: WaveformColorConfig, render_cfg: WaveformRenderConfig) -> dict:
	"""Serialize configs to a dictionary for JSON export."""
	return {
		"analysis": {
			"target_sample_rate": analysis_cfg.target_sample_rate,
			"frame_size": analysis_cfg.frame_size,
			"target_bins": analysis_cfg.target_bins,
			"low_cutoff_hz": analysis_cfg.low_cutoff_hz,
			"lowmid_cutoff_hz": analysis_cfg.lowmid_cutoff_hz,
			"midhigh_cutoff_hz": analysis_cfg.midhigh_cutoff_hz,
			"high_cutoff_hz": analysis_cfg.high_cutoff_hz,
			"smoothing_window_bins": analysis_cfg.smoothing_window_bins,
			"compression_mode": analysis_cfg.compression_mode,
			"compression_strength": analysis_cfg.compression_strength,
			"normalization_mode": analysis_cfg.normalization_mode,
			"normalization_percentile": analysis_cfg.normalization_percentile,
			"low_band_gain": analysis_cfg.low_band_gain,
			"lowmid_band_gain": analysis_cfg.lowmid_band_gain,
			"midhigh_band_gain": analysis_cfg.midhigh_band_gain,
			"high_band_gain": analysis_cfg.high_band_gain,
			"pre_emphasis": analysis_cfg.pre_emphasis,
			"config_version": analysis_cfg.config_version,
		},
		"color": {
			"low_color": list(color_cfg.low_color),
			"lowmid_color": list(color_cfg.lowmid_color),
			"midhigh_color": list(color_cfg.midhigh_color),
			"high_color": list(color_cfg.high_color),
			"band_order_string": color_cfg.band_order_string,
			"overview_mode": color_cfg.overview_mode,
			"stack_bands": color_cfg.stack_bands,
			"blend_bands": color_cfg.blend_bands,
		},
		"render": {
			"image_width": render_cfg.image_width,
			"image_height": render_cfg.image_height,
			"background_color": list(render_cfg.background_color),
			"line_width": render_cfg.line_width,
			"margin_px": render_cfg.margin_px,
			"center_line": render_cfg.center_line,
			"smoothing_bins": render_cfg.smoothing_bins,
			"low_gain": render_cfg.low_gain,
			"lowmid_gain": render_cfg.lowmid_gain,
			"midhigh_gain": render_cfg.midhigh_gain,
			"high_gain": render_cfg.high_gain,
			"prerender_detail": render_cfg.prerender_detail,
		}
	}


def dict_to_config(data: dict) -> tuple[WaveformAnalysisConfig, WaveformColorConfig, WaveformRenderConfig]:
	"""Deserialize configs from a dictionary loaded from JSON."""
	analysis_data = data.get("analysis", {})
	color_data = data.get("color", {})
	render_data = data.get("render", {})
	
	analysis_cfg = WaveformAnalysisConfig(
		target_sample_rate=analysis_data.get("target_sample_rate", DEFAULT_ANALYSIS_CONFIG.target_sample_rate),
		frame_size=analysis_data.get("frame_size", DEFAULT_ANALYSIS_CONFIG.frame_size),
		target_bins=analysis_data.get("target_bins", DEFAULT_ANALYSIS_CONFIG.target_bins),
		low_cutoff_hz=analysis_data.get("low_cutoff_hz", DEFAULT_ANALYSIS_CONFIG.low_cutoff_hz),
		lowmid_cutoff_hz=analysis_data.get("lowmid_cutoff_hz", DEFAULT_ANALYSIS_CONFIG.lowmid_cutoff_hz),
		midhigh_cutoff_hz=analysis_data.get("midhigh_cutoff_hz", DEFAULT_ANALYSIS_CONFIG.midhigh_cutoff_hz),
		high_cutoff_hz=analysis_data.get("high_cutoff_hz", DEFAULT_ANALYSIS_CONFIG.high_cutoff_hz),
		smoothing_window_bins=analysis_data.get("smoothing_window_bins", DEFAULT_ANALYSIS_CONFIG.smoothing_window_bins),
		compression_mode=analysis_data.get("compression_mode", DEFAULT_ANALYSIS_CONFIG.compression_mode),
		compression_strength=analysis_data.get("compression_strength", DEFAULT_ANALYSIS_CONFIG.compression_strength),
		normalization_mode=analysis_data.get("normalization_mode", DEFAULT_ANALYSIS_CONFIG.normalization_mode),
		normalization_percentile=analysis_data.get("normalization_percentile", DEFAULT_ANALYSIS_CONFIG.normalization_percentile),
		low_band_gain=analysis_data.get("low_band_gain", DEFAULT_ANALYSIS_CONFIG.low_band_gain),
		lowmid_band_gain=analysis_data.get("lowmid_band_gain", DEFAULT_ANALYSIS_CONFIG.lowmid_band_gain),
		midhigh_band_gain=analysis_data.get("midhigh_band_gain", DEFAULT_ANALYSIS_CONFIG.midhigh_band_gain),
		high_band_gain=analysis_data.get("high_band_gain", DEFAULT_ANALYSIS_CONFIG.high_band_gain),
		pre_emphasis=analysis_data.get("pre_emphasis", DEFAULT_ANALYSIS_CONFIG.pre_emphasis),
		config_version=analysis_data.get("config_version", DEFAULT_ANALYSIS_CONFIG.config_version),
	)
	
	color_cfg = WaveformColorConfig(
		low_color=tuple(color_data.get("low_color", DEFAULT_COLOR_CONFIG.low_color)),
		lowmid_color=tuple(color_data.get("lowmid_color", DEFAULT_COLOR_CONFIG.lowmid_color)),
		midhigh_color=tuple(color_data.get("midhigh_color", DEFAULT_COLOR_CONFIG.midhigh_color)),
		high_color=tuple(color_data.get("high_color", DEFAULT_COLOR_CONFIG.high_color)),
		band_order_string=color_data.get("band_order_string", DEFAULT_COLOR_CONFIG.band_order_string),
		overview_mode=color_data.get("overview_mode", DEFAULT_COLOR_CONFIG.overview_mode),
		stack_bands=color_data.get("stack_bands", DEFAULT_COLOR_CONFIG.stack_bands),
		blend_bands=color_data.get("blend_bands", DEFAULT_COLOR_CONFIG.blend_bands),
	)
	# Parse band order from string
	color_cfg.band_order, color_cfg.band_order_string = parse_band_order(color_cfg.band_order_string)
	
	render_cfg = WaveformRenderConfig(
		image_width=render_data.get("image_width", DEFAULT_RENDER_CONFIG.image_width),
		image_height=render_data.get("image_height", DEFAULT_RENDER_CONFIG.image_height),
		background_color=tuple(render_data.get("background_color", DEFAULT_RENDER_CONFIG.background_color)),
		line_width=render_data.get("line_width", DEFAULT_RENDER_CONFIG.line_width),
		margin_px=render_data.get("margin_px", DEFAULT_RENDER_CONFIG.margin_px),
		center_line=render_data.get("center_line", DEFAULT_RENDER_CONFIG.center_line),
		smoothing_bins=render_data.get("smoothing_bins", DEFAULT_RENDER_CONFIG.smoothing_bins),
		low_gain=render_data.get("low_gain", DEFAULT_RENDER_CONFIG.low_gain),
		lowmid_gain=render_data.get("lowmid_gain", DEFAULT_RENDER_CONFIG.lowmid_gain),
		midhigh_gain=render_data.get("midhigh_gain", DEFAULT_RENDER_CONFIG.midhigh_gain),
		high_gain=render_data.get("high_gain", DEFAULT_RENDER_CONFIG.high_gain),
		prerender_detail=render_data.get("prerender_detail", DEFAULT_RENDER_CONFIG.prerender_detail),
	)
	
	return analysis_cfg, color_cfg, render_cfg
