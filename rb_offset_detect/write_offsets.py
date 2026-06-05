# Copyright (c) mrmilbe

"""Verification + offsets-file writing for rb_offset_detect.

Resolves candidate chains against the live process to confirm they read sane
values, assembles a version block in the exact format used by
rb_waveform_core/data/offsets, and appends it to the offsets file.

Only bpm / sample_pos / anlz_path are detected (what rkbx_wave consumes). The
masterdeck and per-deck track_info lines required by the file layout are carried
over verbatim from the reference block and flagged as unverified — rkbx_wave
never dereferences them.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Dict, List, Optional

from rb_offset_detect.common import (
    SAMPLE_RATE,
    Pointer,
    format_pointer,
)
from rb_waveform_core.rb_memory_reader import _read_ntstring, _resolve_chain


def resolve_value(pm, base: int, chain: Pointer, kind: str):
    """Resolve a chain and read its value. kind in {bpm, sample_pos, anlz_path}."""
    derefs, final = chain
    try:
        addr = _resolve_chain(pm, base, derefs, final)
    except Exception:
        return None
    try:
        if kind == "bpm":
            return struct.unpack_from("<f", pm.read_bytes(addr, 4))[0]
        if kind == "sample_pos":
            return struct.unpack_from("<q", pm.read_bytes(addr, 8))[0]
        if kind == "anlz_path":
            return _read_ntstring(pm, addr, maxlen=500)
    except Exception:
        return None
    return None


def describe_value(kind: str, value) -> str:
    """Human-readable + plausibility hint for a resolved value."""
    if value is None:
        return "unreadable"
    if kind == "bpm":
        ok = 40.0 <= value <= 260.0
        return f"{value:.2f} BPM {'OK' if ok else '(?)'}"
    if kind == "sample_pos":
        secs = value / SAMPLE_RATE
        ok = 0 <= secs <= 36000
        return f"{secs:.2f}s {'OK' if ok else '(?)'}"
    if kind == "anlz_path":
        ok = ("USBANLZ" in value) or value.upper().endswith(".DAT")
        return f"{value!r} {'OK' if ok else '(?)'}"
    return str(value)


def verify_field(pm, base: int, kind: str, chains: List[Pointer]) -> List[str]:
    """Resolve every deck's chain for a field; return per-deck description lines."""
    out = []
    for d, chain in enumerate(chains):
        val = resolve_value(pm, base, chain, kind)
        out.append(f"  deck {d}: {format_pointer(*chain)}  -> {describe_value(kind, val)}")
    return out


def assemble_block(
    version: str,
    field_chains: Dict[str, List[Pointer]],
    reference_block: dict,
) -> List[str]:
    """Build the offsets-file lines for a new version.

    field_chains needs keys: bpm, sample_pos, anlz_path (each a list per deck).
    masterdeck + track_info are carried from the reference block.
    """
    lines = [version, reference_block["masterdeck"]]
    for d, deck in enumerate(reference_block["decks"]):
        lines.append(format_pointer(*field_chains["bpm"][d]))
        lines.append(format_pointer(*field_chains["sample_pos"][d]))
        lines.append(deck["track_info"])  # carried, unused by rkbx_wave
        lines.append(format_pointer(*field_chains["anlz_path"][d]))
    return lines


def append_block(path: Path, version: str, lines: List[str], reference_version: str) -> None:
    """Append a new version block to the offsets file, separated by a blank line.

    A leading comment documents which fields were detected vs. carried over.
    Blocks are separated by 2+ blank lines (matches the runtime parser).
    """
    note = (
        f"# {version}: bpm/sample_pos/anlz_path detected by rb_offset_detect; "
        f"masterdeck/track_info carried from {reference_version} (unused by rkbx_wave)"
    )
    existing = path.read_text(encoding="utf-8").rstrip("\n")
    block_text = "\n".join([note, *lines])
    path.write_text(f"{existing}\n\n\n{block_text}\n", encoding="utf-8")


def preview_block(version: str, lines: List[str], reference_version: str) -> str:
    note = (
        f"# {version}: bpm/sample_pos/anlz_path detected; "
        f"masterdeck/track_info carried from {reference_version}"
    )
    return "\n".join([note, *lines])
