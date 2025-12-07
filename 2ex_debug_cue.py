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

from rb_waveform_lab.ANLZ import analyze_anlz_folder, extract_cues

DEFAULT_DECK_PATH = Path(
    r"C:/Users/chris/AppData/Roaming/Pioneer/rekordbox/share/PIONEER/USBANLZ/a82/484d8-6208-4fd8-9406-4393f48abd78"
)


def test_cues(deck_path: Path) -> None:
    """Test cue extraction from ANLZ folder."""
    print(f"\n{'='*60}")
    print("Testing Cue Extraction")
    print(f"{'='*60}")
    print(f"Deck Path: {deck_path}")
    print()

    if not deck_path.exists():
        print("❌ Path does not exist!")
        return

    print(f"{deck_path} exists.")

    # Song info
    analysis_result = analyze_anlz_folder(deck_path)
    if analysis_result.song_path:
        print(f"Song Name: {analysis_result.song_path.stem}")
        print(f"Song Path: {analysis_result.song_path}")
    else:
        print("Song Name: Unknown")
        print("Song Path: Not available")
    print()

    print()
    print("Cues:")
    cues = extract_cues(deck_path)
    if not cues:
        print("  (no cues found)")
    else:
        print(f"✅ Found {len(cues)} cues")
        for cue in cues:
            time_sec = cue["time_ms"] / 1000.0
            print(
                f"  Cue {cue.get('cue_num')} at {cue['time_ms']:8.1f}ms "
                f"({time_sec:6.2f}s), color={cue.get('color')}, label={cue.get('label')}"
            )

    print(f"{'='*60}\n")


if __name__ == "__main__":
    deck_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DECK_PATH
    test_cues(deck_path)

