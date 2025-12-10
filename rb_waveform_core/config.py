# Copyright (c) mrmilbe

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Tuple


CompressionMode = Literal["log", "power", "soft", "linear"]
NormalizationMode = Literal["peak", "rms", "percentile"]


class RenderMode(Enum):
	"""Render quality modes with associated prerender_detail defaults."""
	QUALITY = "quality"    # High quality: 30000px prerender, 255px height, BILINEAR
	DEFAULT = "default"    # Balanced: 20000px prerender, 127px height, NEAREST
	SPEED = "speed"        # Fast: 5000px prerender, 127px height, NEAREST

	def get_prerender_detail(self) -> int:
		"""Get the default prerender_detail for this mode."""
		return {
			RenderMode.QUALITY: 30000,
			RenderMode.DEFAULT: 20000,
			RenderMode.SPEED: 5000,
		}[self]

	def get_render_height(self) -> int:
		"""Get the fixed render height for this mode (7-bit or 8-bit for waveform data)."""
		return {
			RenderMode.QUALITY: 127,   # 8-bit height for quality
			RenderMode.DEFAULT: 63,   # 7-bit height for speed
			RenderMode.SPEED: 32,     # 7-bit height for speed
		}[self]

	def get_interpolation(self) -> str:
		"""Get the interpolation mode for display scaling."""
		return {
			RenderMode.QUALITY: "BILINEAR",
			RenderMode.DEFAULT: "NEAREST",
			RenderMode.SPEED: "NEAREST",
		}[self]


@dataclass
class WaveformColorConfig:
    low_color: Tuple[int, int, int] = (0, 120, 255)
    mid_color: Tuple[int, int, int] = (150, 90, 40)  # Mid frequencies
    high_color: Tuple[int, int, int] = (255, 255, 255)
    # Legacy color aliases for lab compatibility
    lowmid_color: Tuple[int, int, int] = (255, 120, 0)  # Legacy (unused in core)
    midhigh_color: Tuple[int, int, int] = (150, 90, 40)  # Legacy alias for mid_color

    band_order: Tuple[str, ...] = ("low", "mid", "high")  # 3-band default
    overview_mode: bool = False  # asymmetric multi-band overview vs symmetric
    stack_bands: bool = False    # true vertical stacking (no overlap)
    blend_bands: bool = True
    band_order_string_default: str = "h,m,l"  # default for normal mode
    band_order_string_overview: str = "l,m,h"  # default for overview mode


@dataclass
class WaveformRenderConfig:
	image_width: int = 9600
	image_height: int = 256
	background_color: Tuple[int, int, int] = (5, 5, 5)
	line_width: int = 1
	margin_px: int = 4
	center_line: bool = False
	smoothing_bins: int = 1
	overview_smoothing_bins: int = 30
	# 3-band gains (matching ANLZ file structure)
	low_gain: float = 1.0
	mid_gain: float = 1.0
	high_gain: float = 1.0
	# Legacy gain aliases for config compatibility
	lowmid_gain: float = 1.0    # Legacy (maps to mid)
	midhigh_gain: float = 1.0   # Legacy (maps to mid)
	prerender_detail: int = 20000  # Max prerender width in pixels
	deck_count: int = 2  # Number of decks to display in GUI
	render_mode: RenderMode = RenderMode.DEFAULT
	# Overview mode gains (3-band)
	overview_low_gain: float = 1.0
	overview_mid_gain: float = 1.0
	overview_high_gain: float = 1.0
	# Legacy overview gain aliases
	overview_lowmid_gain: float = 1.0   # Legacy
	overview_midhigh_gain: float = 1.0  # Legacy


DEFAULT_COLOR_CONFIG = WaveformColorConfig()
DEFAULT_RENDER_CONFIG = WaveformRenderConfig()


# Short name -> internal band name mapping
# Core uses 3 bands: l=low, m=mid, h=high (matching ANLZ file)
# Legacy aliases: lm->mid, mh->mid (for lab compatibility)
BAND_SHORT_NAMES = {"l": "low", "m": "mid", "h": "high", "lm": "mid", "mh": "mid"}
DEFAULT_BAND_ORDER = ("low", "mid", "high")


def parse_band_order(raw_string: str) -> Tuple[Tuple[str, ...], str]:
	"""Parse a comma-separated band order string.
	
	Core uses 3 bands: l=low, m=mid, h=high (matching ANLZ file)
	Legacy lm/mh both map to mid.

	Returns:
		(band_order_tuple, normalized_string)
		Falls back to default order if nothing valid is parsed.
	"""
	if not raw_string or not raw_string.strip():
		return DEFAULT_BAND_ORDER, "l,m,h"

	tokens = [t.strip().lower() for t in raw_string.split(",") if t.strip()]
	mapped = [BAND_SHORT_NAMES[t] for t in tokens if t in BAND_SHORT_NAMES]
	
	if mapped:
		# Remove duplicates while preserving order
		seen = set()
		unique_mapped = []
		unique_tokens = []
		for token, band in zip(tokens, mapped):
			if band not in seen:
				seen.add(band)
				unique_mapped.append(band)
				# Normalize to simplified naming: lm->m, mh->m
				if token in ('lm', 'mh'):
					unique_tokens.append('m')
				else:
					unique_tokens.append(token)
		
		if unique_mapped:
			return tuple(unique_mapped), ",".join(unique_tokens)
	
	return DEFAULT_BAND_ORDER, "l,m,h"


def config_to_dict(color_cfg: WaveformColorConfig, render_cfg: WaveformRenderConfig) -> dict:
	"""Serialize configs to a dictionary for JSON export."""
	return {
		"color": {
			"low_color": list(color_cfg.low_color),
			"mid_color": list(color_cfg.mid_color),
			"high_color": list(color_cfg.high_color),
			"band_order_string_default": color_cfg.band_order_string_default,
			"band_order_string_overview": color_cfg.band_order_string_overview,
			"overview_mode": color_cfg.overview_mode,
			"stack_bands": color_cfg.stack_bands,
			"blend_bands": color_cfg.blend_bands,
			# Legacy color fields for compatibility
			"lowmid_color": list(color_cfg.lowmid_color),
			"midhigh_color": list(color_cfg.midhigh_color),
		},
		"render": {
			"image_width": render_cfg.image_width,
			"image_height": render_cfg.image_height,
			"background_color": list(render_cfg.background_color),
			"line_width": render_cfg.line_width,
			"margin_px": render_cfg.margin_px,
			"center_line": render_cfg.center_line,
			"smoothing_bins": render_cfg.smoothing_bins,
			"overview_smoothing_bins": render_cfg.overview_smoothing_bins,
			"low_gain": render_cfg.low_gain,
			"mid_gain": render_cfg.mid_gain,
			"high_gain": render_cfg.high_gain,
			"prerender_detail": render_cfg.prerender_detail,
			"deck_count": render_cfg.deck_count,
			"render_mode": render_cfg.render_mode.value,
			"overview_low_gain": render_cfg.overview_low_gain,
			"overview_mid_gain": render_cfg.overview_mid_gain,
			"overview_high_gain": render_cfg.overview_high_gain,
			# Legacy gain fields for compatibility
			"lowmid_gain": render_cfg.lowmid_gain,
			"midhigh_gain": render_cfg.midhigh_gain,
			"overview_lowmid_gain": render_cfg.overview_lowmid_gain,
			"overview_midhigh_gain": render_cfg.overview_midhigh_gain,
		}
	}


def dict_to_config(data: dict) -> tuple[WaveformColorConfig, WaveformRenderConfig]:
	"""Deserialize configs from a dictionary loaded from JSON."""
	color_data = data.get("color", {})
	render_data = data.get("render", {})
	
	# Backward compatibility: if old "band_order_string" exists, use it for both modes
	legacy_band_order = color_data.get("band_order_string")
	if legacy_band_order:
		default_string = color_data.get("band_order_string_default", legacy_band_order)
		overview_string = color_data.get("band_order_string_overview", legacy_band_order)
	else:
		default_string = color_data.get("band_order_string_default", DEFAULT_COLOR_CONFIG.band_order_string_default)
		overview_string = color_data.get("band_order_string_overview", DEFAULT_COLOR_CONFIG.band_order_string_overview)
	
	color_cfg = WaveformColorConfig(
		low_color=tuple(color_data.get("low_color", DEFAULT_COLOR_CONFIG.low_color)),
		mid_color=tuple(color_data.get("mid_color", color_data.get("midhigh_color", DEFAULT_COLOR_CONFIG.mid_color))),
		high_color=tuple(color_data.get("high_color", DEFAULT_COLOR_CONFIG.high_color)),
		lowmid_color=tuple(color_data.get("lowmid_color", DEFAULT_COLOR_CONFIG.lowmid_color)),
		midhigh_color=tuple(color_data.get("midhigh_color", DEFAULT_COLOR_CONFIG.midhigh_color)),
		band_order_string_default=default_string,
		band_order_string_overview=overview_string,
		overview_mode=color_data.get("overview_mode", DEFAULT_COLOR_CONFIG.overview_mode),
		stack_bands=color_data.get("stack_bands", DEFAULT_COLOR_CONFIG.stack_bands),
		blend_bands=color_data.get("blend_bands", DEFAULT_COLOR_CONFIG.blend_bands),
	)
	# Parse band order from the appropriate mode string
	if color_cfg.overview_mode:
		color_cfg.band_order, _ = parse_band_order(color_cfg.band_order_string_overview)
	else:
		color_cfg.band_order, _ = parse_band_order(color_cfg.band_order_string_default)
	
	# Parse render_mode from string
	render_mode_str = render_data.get("render_mode", DEFAULT_RENDER_CONFIG.render_mode.value)
	try:
		render_mode = RenderMode(render_mode_str)
	except ValueError:
		render_mode = DEFAULT_RENDER_CONFIG.render_mode
	
	render_cfg = WaveformRenderConfig(
		image_width=render_data.get("image_width", DEFAULT_RENDER_CONFIG.image_width),
		image_height=render_data.get("image_height", DEFAULT_RENDER_CONFIG.image_height),
		background_color=tuple(render_data.get("background_color", DEFAULT_RENDER_CONFIG.background_color)),
		line_width=render_data.get("line_width", DEFAULT_RENDER_CONFIG.line_width),
		margin_px=render_data.get("margin_px", DEFAULT_RENDER_CONFIG.margin_px),
		center_line=render_data.get("center_line", DEFAULT_RENDER_CONFIG.center_line),
		smoothing_bins=render_data.get("smoothing_bins", DEFAULT_RENDER_CONFIG.smoothing_bins),
		overview_smoothing_bins=render_data.get("overview_smoothing_bins", DEFAULT_RENDER_CONFIG.overview_smoothing_bins),
		low_gain=render_data.get("low_gain", DEFAULT_RENDER_CONFIG.low_gain),
		mid_gain=render_data.get("mid_gain", render_data.get("midhigh_gain", DEFAULT_RENDER_CONFIG.mid_gain)),
		high_gain=render_data.get("high_gain", DEFAULT_RENDER_CONFIG.high_gain),
		lowmid_gain=render_data.get("lowmid_gain", DEFAULT_RENDER_CONFIG.lowmid_gain),
		midhigh_gain=render_data.get("midhigh_gain", DEFAULT_RENDER_CONFIG.midhigh_gain),
		prerender_detail=render_data.get("prerender_detail", DEFAULT_RENDER_CONFIG.prerender_detail),
		deck_count=render_data.get("deck_count", DEFAULT_RENDER_CONFIG.deck_count),
		render_mode=render_mode,
		overview_low_gain=render_data.get("overview_low_gain", DEFAULT_RENDER_CONFIG.overview_low_gain),
		overview_mid_gain=render_data.get("overview_mid_gain", render_data.get("overview_midhigh_gain", DEFAULT_RENDER_CONFIG.overview_mid_gain)),
		overview_high_gain=render_data.get("overview_high_gain", DEFAULT_RENDER_CONFIG.overview_high_gain),
		overview_lowmid_gain=render_data.get("overview_lowmid_gain", DEFAULT_RENDER_CONFIG.overview_lowmid_gain),
		overview_midhigh_gain=render_data.get("overview_midhigh_gain", DEFAULT_RENDER_CONFIG.overview_midhigh_gain),
	)
	
	return color_cfg, render_cfg
