# Copyright (c) mrmilbe

"""Value scanning stages for rb_offset_detect.

Each stage narrows process memory down to a small set of candidate *final
addresses* for the deck the user loaded (Rekordbox "deck 1" = index 0):

  - BPM (f32):         scan for the displayed BPM, then re-filter after a pitch
                       change so only addresses that tracked the change remain.
  - playhead (i64):    scan for the sample position after N seconds of playback,
                       then keep addresses that advanced by ~the expected delta.
  - ANLZ path (str):   scan for the known folder GUID we pre-picked; string hits
                       are unique, so no second round is needed.

These functions are pure with respect to the UI: detect.py collects the user
input and snapshots and calls in here.
"""

from __future__ import annotations

import struct
from typing import Dict, List

from rb_offset_detect.common import MemorySnapshot, SAMPLE_RATE


# ---------------------------------------------------------------------------
# Live re-reads (cheap point reads against the running process)
# ---------------------------------------------------------------------------

def _read_f32(pm, addr: int):
    try:
        return struct.unpack_from("<f", pm.read_bytes(addr, 4))[0]
    except Exception:
        return None


def _read_i64(pm, addr: int):
    try:
        return struct.unpack_from("<q", pm.read_bytes(addr, 8))[0]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# BPM (f32)
# ---------------------------------------------------------------------------

def bpm_round1(snap: MemorySnapshot, bpm: float, tol: float = 0.5) -> List[int]:
    """Initial BPM candidates from a snapshot."""
    return snap.scan_f32(bpm, tol)


def bpm_round2(pm, addrs: List[int], new_bpm: float, tol: float = 0.5) -> List[int]:
    """Keep only candidates whose live value now matches the changed BPM."""
    out = []
    for a in addrs:
        v = _read_f32(pm, a)
        if v is not None and abs(v - new_bpm) <= tol:
            out.append(a)
    return out


# ---------------------------------------------------------------------------
# Playhead / sample position (i64)
# ---------------------------------------------------------------------------

def sample_round1(pm, snap: MemorySnapshot, position_seconds: float, tol_seconds: float = 0.7) -> Dict[int, int]:
    """Candidates near `position_seconds` worth of samples.

    Returns {addr: actual_value} — each candidate's real value is read live (the
    deck is stopped, so values are stable) so round 2 can measure a precise delta
    regardless of the absolute scale.
    """
    target = int(position_seconds * SAMPLE_RATE)
    tol = int(tol_seconds * SAMPLE_RATE)
    actual: Dict[int, int] = {}
    for a in snap.scan_i64(target, tol):
        v = _read_i64(pm, a)
        if v is not None:
            actual[a] = v
    return actual


def sample_round2(pm, prev: Dict[int, int], delta_seconds: float, tol_seconds: float = 0.7) -> List[int]:
    """Keep candidates that advanced by ~delta_seconds worth of samples.

    `delta_seconds` is the difference between the two absolute playhead positions
    the user reported (pos2 - pos1), so the user can just read the deck display.
    """
    expected = delta_seconds * SAMPLE_RATE
    tol = tol_seconds * SAMPLE_RATE
    out = []
    for a, prev_val in prev.items():
        v = _read_i64(pm, a)
        if v is None:
            continue
        if abs((v - prev_val) - expected) <= tol:
            out.append(a)
    return out


# ---------------------------------------------------------------------------
# ANLZ path (string)
# ---------------------------------------------------------------------------

def anlz_path_addresses(snap: MemorySnapshot, folder_guid: str) -> List[int]:
    """String-start addresses for memory holding the pre-picked ANLZ folder GUID."""
    needle = folder_guid.encode("utf-8", errors="ignore")
    return snap.scan_string_starts(needle)
