# Copyright (c) HorstHorstmann

from __future__ import annotations

from typing import Literal

import numpy as np

from .config import CompressionMode, NormalizationMode


def _safe_max(x: np.ndarray) -> float:
	return float(np.max(x)) if x.size else 0.0


def _safe_rms(x: np.ndarray) -> float:
	return float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0


def _safe_percentile(x: np.ndarray, p: float) -> float:
	return float(np.percentile(x, p * 100.0)) if x.size else 0.0


def compress_amplitude(
	x: np.ndarray,
	mode: CompressionMode,
	strength: float,
) -> np.ndarray:
	"""Apply simple amplitude compression curve.

	Input and output are non-negative amplitudes.
	"""

	x = np.asarray(x, dtype=np.float32)
	if mode == "linear":
		return x

	eps = 1e-8
	if mode == "log":
		alpha = max(strength, 1e-3)
		return np.log1p(alpha * x) / np.log1p(alpha)
	if mode == "power":
		gamma = max(strength, 1e-3)
		return np.power(x, gamma)
	if mode == "soft":
		alpha = max(strength, 1e-3)
		return x / (1.0 + alpha * x)

	return x


def normalize_band(
	x: np.ndarray,
	mode: NormalizationMode,
	percentile: float = 0.99,
) -> np.ndarray:
	"""Normalize band to roughly [0, 1]."""

	x = np.asarray(x, dtype=np.float32)
	if not x.size:
		return x

	if mode == "peak":
		m = _safe_max(x)
	elif mode == "rms":
		m = _safe_rms(x)
	else:  # "percentile"
		m = _safe_percentile(x, percentile)

	if m <= 0.0 or not np.isfinite(m):
		return x
	return x / m

