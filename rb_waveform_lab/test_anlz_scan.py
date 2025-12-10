# Copyright (c) mrmilbe

"""Scan folders for .2EX files and extract waveform info."""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path
from rb_waveform_core.ANLZ import analyze_anlz_folder

def scan(root: str, max_files: int = 10):
    root_path = Path(root)
    count = 0
    for folder in root_path.rglob("*.2EX"):
        anlz_folder = folder.parent
        try:
            result = analyze_anlz_folder(anlz_folder,lib_folder=Path(r"C:\Rekordbox"))
            wf_shape = result.waveform.shape if result.waveform is not None else None
            dur_calc=wf_shape[0]/150
            print(f"{count+1}. {anlz_folder.name}")
            print(f"   Song: {result.song_path}")
            print(f"   Resolved: {result.resolved_path}")
            print(f"   Duration: {result.duration:.1f}s" if result.duration else "   Duration: ?")
            print(f"   Calculated Duration: {dur_calc:.1f}s" if result.waveform is not None else "   Calculated Duration: ?")
            print(f"   Waveform: {wf_shape}")
            print()
            count += 1
            if count >= max_files:
                break
        except Exception as e:
            print(f"   Error: {e}")
    print(f"Scanned {count} files.")

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\chris\AppData\Roaming\Pioneer\rekordbox\share\PIONEER\USBANLZ"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    scan(folder, 20)
