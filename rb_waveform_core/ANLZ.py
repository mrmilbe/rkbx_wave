# Copyright (c) mrmilbe

"""Rekordbox ANLZ file parsing for waveform and beat grid extraction.

Position in data flow:
    ANLZ folder (.DAT/.EXT/.2EX files) → ANLZ.py → analysis.py → WaveformAnalysis

Responsibilities:
    - Parse ANLZ files via pyrekordbox library
    - Extract PWV7 tag (high-res waveform: N×3 uint8 array at 150 fps)
    - Extract PPTH tag (audio file path)
    - Extract PQTZ tag (beat grid with time/beat_number/bpm)
    - Extract PCOB tag (hot cues - debug only)

File priority:
    - PWV7 waveform: .2EX preferred (has extended color data)
    - Beat grid: .DAT preferred (authoritative beat data)

Usage:
    result = analyze_anlz_folder(folder)
    analysis = analysis_from_rb_waveform(result.waveform, result.duration)
    beat_grid = extract_beat_grid(folder)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, List, Dict, Any

import numpy as np
from pyrekordbox.anlz import AnlzFile


# Debug flag for verbose cue extraction output
DEBUG_CUES = False


@dataclass(frozen=True)
class AnlzAnalysisResult:
    """Structured response for ANLZ lookups."""

    waveform: Optional[np.ndarray]
    song_path: Optional[Path]  # Raw path from PPTH tag (not resolved)
    duration: Optional[float]  # Calculated from PWV at 150 fps


@dataclass(frozen=True)
class LoadedAudioData:
    """Audio data loaded for analysis mode."""
    signal: np.ndarray
    sample_rate: int
    duration: float
    audio_path: Path


@dataclass(frozen=True)
class LoadedWaveformData:
    """Waveform data loaded for rekordbox mode."""
    waveform: np.ndarray
    duration: float
    audio_path: Optional[Path]


@dataclass(frozen=True)
class BeatGridEntry:
    """Single beat grid entry from PQTZ tag."""
    time_ms: float  # Time in milliseconds
    beat_number: int  # Beat ordinal/number
    bpm: Optional[float] = None  # Tempo at this entry if available


def extract_cues(folder: Path | str) -> Optional[List[Dict[str, Any]]]:
    """Extract hot cues from ANLZ folder via PCOB tag.

    Returns a list of dicts with keys: time_ms, color, cue_num, label.
    If no cues are found or parsing fails, returns None.
    """
    folder_path = Path(folder)
    if not folder_path.is_dir():
        return None

    # Prefer .DAT for beat grid, but cues may live in .2EX only on some setups.
    # Try DAT/EXT/2EX via beatgrid picker first, then fall back to generic picker.
    anlz_file = _pick_anlz_file_for_beatgrid(folder_path)
    if anlz_file is None:
        anlz_file = _pick_anlz_file(folder_path)
    if anlz_file is None:
        return None

    try:
        anlz = AnlzFile.parse_file(str(anlz_file))
        tags = getattr(anlz, "tags", [])
    except Exception:
        return None

    # Debug: list all tag types so we can see if PCOB is present
    if DEBUG_CUES:
        print(f"[extract_cues] ANLZ file: {anlz_file}")
        unique_types = sorted({type(tag).__name__ for tag in tags}) if tags else []
        print(f"[extract_cues] Tag types: {', '.join(unique_types) if unique_types else '(none)'}")

    cues: List[Dict[str, Any]] = []
    for tag in tags:
        if not type(tag).__name__.startswith("PCOB"):
            continue
        if DEBUG_CUES:
            print(f"[extract_cues] Found PCOB tag: {type(tag).__name__}")
        try:
            content = tag.struct.content
        except AttributeError:
            if DEBUG_CUES:
                print("[extract_cues]   PCOB tag missing struct.content")
            continue

        entries = getattr(content, "entries", None)
        if not entries:
            if DEBUG_CUES:
                print("[extract_cues]   PCOB has no entries")
            continue

        for e in entries:
            # Debug: show raw entry attributes to understand available fields
            if DEBUG_CUES:
                attrs = {k: getattr(e, k) for k in dir(e) if not k.startswith("_")}
                print(f"[extract_cues]   Entry attrs: {attrs}")

            cue_time = getattr(e, "time", None)
            cue_color = getattr(e, "color", None)
            cue_num = getattr(e, "cue_num", None)
            label_id = getattr(e, "label", None)

            if cue_time is None:
                continue

            cues.append(
                {
                    "time_ms": float(cue_time),
                    "color": cue_color,
                    "cue_num": cue_num,
                    "label": label_id,
                }
            )

    if not cues:
        return None
    return cues


def load_audio_for_analysis(
    audio_path: Path,
    target_sample_rate: int,
) -> LoadedAudioData:
    """Load audio file for analysis mode.

    Args:
        audio_path: Resolved path to audio file (use resolve_audio_path from analysis module).
        target_sample_rate: Sample rate to resample audio to.

    Returns:
        LoadedAudioData with signal, sample rate, duration, and path.

    Raises:
        FileNotFoundError: If audio file doesn't exist.
        RuntimeError: If audio loading fails.
    """
    from .audio_io import load_mono_resampled

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        signal, sr, duration = load_mono_resampled(str(audio_path), target_sample_rate)
    except Exception as exc:
        raise RuntimeError(f"Failed to load audio via ffmpeg: {exc}") from exc

    return LoadedAudioData(
        signal=signal,
        sample_rate=sr,
        duration=duration,
        audio_path=audio_path,
    )


def load_waveform_for_rekordbox(
    analysis_result: AnlzAnalysisResult,
    resolved_audio_path: Optional[Path] = None,
) -> LoadedWaveformData:
    """Extract waveform data from ANLZ analysis result for rekordbox mode.

    Args:
        analysis_result: Result from analyze_anlz_folder.
        resolved_audio_path: Optional resolved audio file path.

    Returns:
        LoadedWaveformData with waveform, duration (from PWV), and optional path.

    Raises:
        ValueError: If folder does not contain PWV waveform data.
    """
    if analysis_result.waveform is None:
        raise ValueError("This folder does not contain PWV waveform data.")

    return LoadedWaveformData(
        waveform=analysis_result.waveform,
        duration=float(analysis_result.duration or 0.0),
        audio_path=resolved_audio_path,
    )


def analyze_anlz_folder(folder: Path | str) -> AnlzAnalysisResult:
    """Return waveform, raw Rekordbox path, and duration from ANLZ folder.
    
    Duration is calculated from PWV waveform at 150 frames/second.
    Path is the raw PPTH value (not resolved to actual file).
    """
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise FileNotFoundError(f"ANLZ folder not found: {folder_path}")

    anlz_file = _pick_anlz_file(folder_path)
    if anlz_file is None:
        return AnlzAnalysisResult(None, None, None)

    return _parse_anlz_file(anlz_file)


def extract_beat_grid(folder: Path | str) -> Optional[list[BeatGridEntry]]:
    """Extract beat grid from ANLZ folder.
    
    Returns list of beat grid entries from PQTZ tag, or None if not available.
    Common for streaming tracks to not have a beat grid.
    
    Note: Prefers .DAT files over .2EX for beat grid extraction as DAT files
    contain the authoritative beat grid data.
    
    Args:
        folder: Path to ANLZ folder containing .DAT, .EXT or .2EX files.
        
    Returns:
        List of BeatGridEntry objects, or None if no beat grid found.
    """
    folder_path = Path(folder)
    if not folder_path.is_dir():
        return None

    anlz_file = _pick_anlz_file_for_beatgrid(folder_path)
    if anlz_file is None:
        return None

    try:
        tags = getattr(AnlzFile.parse_file(str(anlz_file)), "tags", None)
    except Exception:
        return None

    if not tags:
        return None

    return _extract_beat_grid(tags)


def extract_beat_grid_downbeats(folder: Path | str) -> Optional[list[BeatGridEntry]]:
    """Extract beat grid downbeats (every 4th beat) from ANLZ folder.
    
    Returns list of downbeat entries (beat_number % 4 == 1) from PQTZ tag.
    These represent bar boundaries / measure starts.
    
    Args:
        folder: Path to ANLZ folder containing .DAT, .EXT or .2EX files.
        
    Returns:
        List of BeatGridEntry objects for downbeats only, or None if no beat grid found.
    """
    beat_grid = extract_beat_grid(folder)
    if beat_grid is None:
        return None
    
    # Filter for downbeats (Rekordbox numbers beats starting from 1)
    return [entry for entry in beat_grid if entry.beat_number % 4 == 1]


def _pick_anlz_file(folder: Path) -> Optional[Path]:
    """Prefer .2EX files, fall back to .EXT when needed."""

    two_ex = sorted(folder.glob("*.2EX"))
    if two_ex:
        return two_ex[0]
    ext_files = sorted(folder.glob("*.EXT"))
    return ext_files[0] if ext_files else None


def _pick_anlz_file_for_beatgrid(folder: Path) -> Optional[Path]:
    """Prefer .DAT files for beat grid, fall back to .EXT, then .2EX."""
    
    dat_files = sorted(folder.glob("*.DAT"))
    if dat_files:
        return dat_files[0]
    
    ext_files = sorted(folder.glob("*.EXT"))
    if ext_files:
        return ext_files[0]
    
    two_ex = sorted(folder.glob("*.2EX"))
    return two_ex[0] if two_ex else None


def _parse_anlz_file(anlz_path: Path) -> AnlzAnalysisResult:
    try:
        tags = getattr(AnlzFile.parse_file(str(anlz_path)), "tags", None)
    except Exception:
        return AnlzAnalysisResult(None, None, None)

    if not tags:
        return AnlzAnalysisResult(None, None, None)

    song_path = _extract_rb_path(tags)
    waveform = _extract_waveform(tags)
    
    # Calculate duration from PWV waveform (150 frames per second)
    duration = None
    if waveform is not None and len(waveform) > 0:
        duration = len(waveform) / 150.0
    
    return AnlzAnalysisResult(waveform, song_path, duration)


def _extract_rb_path(tags: Iterable[object]) -> Optional[Path]:
    for tag in tags:
        cls_name = type(tag).__name__
        if cls_name.startswith("PPTH"):
            try:
                path_str = tag.struct.content.path
                if path_str:
                    return Path(path_str)
            except Exception:
                continue
    return None


def _extract_waveform(tags: Iterable[object]) -> Optional[np.ndarray]:
    for wanted in ("PWV7", "PWV6", "PWV5"):
        for tag in tags:
            if not type(tag).__name__.startswith(wanted):
                continue
            try:
                raw_entries = tag.struct.content.entries
            except AttributeError:
                continue
            if isinstance(raw_entries, (bytes, bytearray, memoryview)):
                byte_data = bytes(raw_entries)
            else:
                try:
                    byte_data = bytes(int(val) & 0xFF for val in raw_entries)
                except TypeError:
                    continue
            data = np.frombuffer(byte_data, dtype=np.uint8)
            if data.size % 3 != 0:
                continue
            return data.reshape(-1, 3).astype(np.float32)
    return None


def _extract_beat_grid(tags: Iterable[object]) -> Optional[list[BeatGridEntry]]:
    """Extract beat grid entries from PQTZ tag."""
    for tag in tags:
        if not type(tag).__name__.startswith("PQTZ"):
            continue
        
        try:
            entries = tag.struct.content.entries
        except AttributeError:
            continue
        
        if not entries:
            continue
        
        beat_grid = []
        for entry in entries:
            try:
                # Try to extract time and beat number
                # Format varies by Rekordbox version, but time and beat are always present
                time_ms = float(getattr(entry, 'time', 0))
                beat_number = int(getattr(entry, 'beat', 0))
                bpm_raw = getattr(entry, 'tempo', None)
                bpm: Optional[float] = None
                if bpm_raw is not None:
                    try:
                        bpm = float(bpm_raw)
                        if bpm > 400.0:  # Rekordbox stores tempo as BPM*100
                            bpm /= 100.0
                        if bpm <= 0:
                            bpm = None
                    except (TypeError, ValueError):
                        bpm = None
                
                beat_grid.append(BeatGridEntry(
                    time_ms=time_ms,
                    beat_number=beat_number,
                    bpm=bpm
                ))
            except (AttributeError, TypeError, ValueError):
                continue
        
        if beat_grid:
            return beat_grid
    
    return None


# Library resolution moved to analysis.py
# Use resolve_audio_path() from analysis module for path resolution
