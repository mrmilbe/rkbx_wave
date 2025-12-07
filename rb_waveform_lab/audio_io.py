# Copyright (c) HorstHorstmann

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.signal import resample_poly
import subprocess


def _decode_with_ffmpeg(path: str, target_sr: int) -> Tuple[np.ndarray, int]:
	"""Decode audio using ffmpeg to mono, target_sr, float32 PCM.

	This relies on an `ffmpeg` binary available on PATH. It uses a
	pipe (no temp files) and asks ffmpeg to output raw 32-bit float
	mono PCM at the desired sample rate.
	"""

	cmd = [
		"ffmpeg",
		"-v",
		"error",
		"-i",
		path,
		"-f",
		"f32le",
		"-ac",
		"1",
		"-ar",
		str(target_sr),
		"-",
	]

	try:
		proc = subprocess.run(
			cmd,
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			check=True,
		)
	except FileNotFoundError as exc:
		raise RuntimeError(
			"ffmpeg binary not found. Ensure ffmpeg is installed and on PATH."
		) from exc
	except subprocess.CalledProcessError as exc:
		raise RuntimeError(
			f"ffmpeg failed to decode '{path}': {exc.stderr.decode(errors='ignore')}"
		) from exc

	raw = proc.stdout
	if not raw:
		raise RuntimeError(f"ffmpeg produced no audio data for '{path}'")

	audio = np.frombuffer(raw, dtype=np.float32)
	return audio, target_sr


def load_mono_resampled(path: str, target_sr: int) -> Tuple[np.ndarray, int, float]:
	"""Load an audio file via ffmpeg, convert to mono+target_sr, return (signal, sr, duration)."""

	mono, sr_out = _decode_with_ffmpeg(path, target_sr)

	# Ensure contiguous float32 array
	mono = np.asarray(mono, dtype=np.float32)

	duration_seconds = float(len(mono)) / float(sr_out) if sr_out > 0 else 0.0
	return mono, sr_out, duration_seconds

