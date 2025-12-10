# Copyright (c) mrmilbe

"""Lab configuration - extends core config with experimental analysis parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Import core config types and defaults
from rb_waveform_core.config import (
    WaveformColorConfig,
    WaveformRenderConfig,
    DEFAULT_COLOR_CONFIG,
    DEFAULT_RENDER_CONFIG,
    CompressionMode,
    NormalizationMode,
    config_to_dict as core_config_to_dict,
    dict_to_config as core_dict_to_config,
    parse_band_order,
)


@dataclass
class WaveformAnalysisConfig:
    """Audio analysis configuration (lab-only, not used in production core)."""
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


DEFAULT_ANALYSIS_CONFIG = WaveformAnalysisConfig()


def lab_config_to_dict(
    analysis_cfg: WaveformAnalysisConfig,
    color_cfg: WaveformColorConfig,
    render_cfg: WaveformRenderConfig
) -> dict:
    """Serialize lab configs (includes analysis + core configs)."""
    # Start with core config
    result = core_config_to_dict(color_cfg, render_cfg)
    
    # Add analysis section
    result["analysis"] = {
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
    }
    
    return result


def lab_dict_to_config(data: dict) -> tuple[WaveformAnalysisConfig, WaveformColorConfig, WaveformRenderConfig]:
    """Deserialize lab configs (includes analysis + core configs)."""
    # Load core configs
    color_cfg, render_cfg = core_dict_to_config(data)
    
    # Load analysis config
    analysis_data = data.get("analysis", {})
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
    
    return analysis_cfg, color_cfg, render_cfg
