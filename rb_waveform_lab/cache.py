# Copyright (c) HorstHorstmann

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .analysis import WaveformAnalysis


def _analysis_to_dict(wa: WaveformAnalysis) -> Dict[str, Any]:
	return {
		"low": wa.low.astype("float32").tolist(),
		"lowmid": wa.lowmid.astype("float32").tolist(),
		"midhigh": wa.midhigh.astype("float32").tolist(),
		"high": wa.high.astype("float32").tolist(),
		"duration_seconds": float(wa.duration_seconds),
		"sample_rate": int(wa.sample_rate),
		"config_version": int(wa.config_version),
	}


def _analysis_from_dict(d: Dict[str, Any]) -> WaveformAnalysis:
	return WaveformAnalysis(
		low=np.asarray(d["low"], dtype=np.float32),
		lowmid=np.asarray(d["lowmid"], dtype=np.float32),
		midhigh=np.asarray(d["midhigh"], dtype=np.float32),
		high=np.asarray(d["high"], dtype=np.float32),
		duration_seconds=float(d["duration_seconds"]),
		sample_rate=int(d["sample_rate"]),
		config_version=int(d.get("config_version", 1)),
	)


def cache_path_for_audio(audio_path: str) -> Path:
	p = Path(audio_path)
	return p.with_suffix(p.suffix + ".rbwf.json")


def save_analysis(wa: WaveformAnalysis, audio_path: str) -> Path:
	out_path = cache_path_for_audio(audio_path)
	data = _analysis_to_dict(wa)
	out_path.write_text(json.dumps(data, separators=(",", ":")))
	return out_path


def load_analysis(audio_path: str) -> WaveformAnalysis | None:
	path = cache_path_for_audio(audio_path)
	if not path.is_file():
		return None
	try:
		data = json.loads(path.read_text())
	except Exception:
		return None
	return _analysis_from_dict(data)

