# Copyright (c) mrmilbe

"""Small helper to inspect Rekordbox ANLZ metadata for debugging.

Usage:
    python 2ex_debug.py [deck-folder]

If no folder is provided, the DEFAULT_DECK_PATH from tuning_gui_2ex.py is
used. The script looks for the first .2EX file (preferred) and the first
.DAT file inside that folder, parses them via pyrekordbox, and prints the
available tag types. For the 2EX file it also reports waveform lengths in
bytes and frames to help reason about time-axis scaling.
"""

from __future__ import annotations

import sys
from pathlib import Path
import struct
from typing import Iterable, Optional, Sequence

from pyrekordbox.anlz import AnlzFile

from rb_waveform_core.ANLZ import extract_beat_grid, analyze_anlz_folder

DEFAULT_DECK_PATH = Path(
    r"C:/Users/chris/AppData/Roaming/Pioneer/rekordbox/share/PIONEER/USBANLZ/a82/484d8-6208-4fd8-9406-4393f48abd78"
)


def test_beat_grid_extraction(deck_path: Path) -> None:
    """Test beat grid extraction from ANLZ folder."""
    print(f"\n{'='*60}")
    print(f"Testing Beat Grid Extraction")
    print(f"{'='*60}")
    print(f"Deck Path: {deck_path}")
    print()
    
    if not deck_path.exists():
        print(f"❌ Path does not exist!")
        return
    print(f"{deck_path} exists.")
    # Get song info
    analysis_result = analyze_anlz_folder(deck_path)
    if analysis_result.song_path:
        print(f"Song Name: {analysis_result.song_path.stem}")
        print(f"Song Path: {analysis_result.song_path}")
    else:
        print("Song Name: Unknown")
        print("Song Path: Not available")
    print()
    
    beat_grid = extract_beat_grid(deck_path)
    
    if beat_grid is None:
        print("❌ No beat grid found (common for streaming tracks)")
        return
    
    print(f"✅ Found {len(beat_grid)} beat grid entries")
    print()
    
    # Print first 10 and last 5 entries
    print("First 10 beats:")
    for entry in beat_grid[:10]:
        time_sec = entry.time_ms / 1000.0
        bpm_info = f" @ {entry.bpm:.2f} BPM" if entry.bpm else ""
        print(f"  Beat {entry.beat_number:4d} at {entry.time_ms:8.1f}ms ({time_sec:6.2f}s){bpm_info}")
    
    if len(beat_grid) > 15:
        print("\n  ...")
        print("\nLast 5 beats:")
        for entry in beat_grid[-5:]:
            time_sec = entry.time_ms / 1000.0
            bpm_info = f" @ {entry.bpm:.2f} BPM" if entry.bpm else ""
            print(f"  Beat {entry.beat_number:4d} at {entry.time_ms:8.1f}ms ({time_sec:6.2f}s){bpm_info}")
    
    # Calculate average BPM
    if len(beat_grid) >= 2:
        print()
        first = beat_grid[0]
        last = beat_grid[-1]
        duration_ms = last.time_ms - first.time_ms
        num_beats = len(beat_grid)
        print(f"{duration_ms=}, {num_beats=}")
        if duration_ms > 0 and num_beats > 0:
            avg_bpm = (num_beats / (duration_ms / 1000.0)) * 60.0
            print(f"Average BPM (from spacing): {avg_bpm:.2f}")
    # Show first non-null BPM from tag if available
    tempo_candidates = [entry.bpm for entry in beat_grid if entry.bpm]
    if tempo_candidates:
        print(f"Tagged BPM: {tempo_candidates[0]:.2f}")
        print(f"Total duration: {duration_ms / 1000.0:.2f}s")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    deck_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DECK_PATH
    test_beat_grid_extraction(deck_path)
