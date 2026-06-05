# Copyright (c) mrmilbe

"""rb_offset_detect — semi-automated Rekordbox memory-offset discovery.

Run with Rekordbox open:

    python -m rb_offset_detect.detect
    (or)  python rb_offset_detect/detect.py

It walks you through loading a known song and a few play/stop/pitch actions,
scans process memory for the resulting values, traces the pointer chains back to
module-relative roots (reusing the 7.2.2 chain structure as a hint), then writes
a new version block to rb_waveform_core/data/offsets.

Detects bpm / playhead / ANLZ-path offsets (what rkbx_wave uses). Windows only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Allow running as a loose script (python rb_offset_detect/detect.py).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rb_offset_detect import scan
from rb_offset_detect import pointer_scan as ptr
from rb_offset_detect import write_offsets as wr
from rb_offset_detect.common import (
    MemorySnapshot,
    REFERENCE_VERSION,
    Pointer,
    connect_rekordbox,
    detect_version,
    enum_readable_regions,
    load_reference_block,
    offsets_file_path,
    pick_anlz_dat,
)

LOADED_DECK = 0  # Rekordbox "deck 1" maps to offset index 0


# ---------------------------------------------------------------------------
# Small UI helpers
# ---------------------------------------------------------------------------

def _prompt(msg: str) -> str:
    return input(msg).strip()


def _prompt_float(msg: str) -> Optional[float]:
    raw = _prompt(msg)
    try:
        return float(raw.replace(",", "."))
    except ValueError:
        print("  ! not a number")
        return None


def _hr(title: str) -> None:
    print(f"\n== {title} ==")


def take_snapshot(pm) -> MemorySnapshot:
    regions = enum_readable_regions(pm.process_handle)
    snap = MemorySnapshot(pm, regions)
    print(f"  (snapshot: {len(regions)} regions, {snap.total_bytes / 1e6:.0f} MB)")
    return snap


def _pick_root(roots: List, label: str) -> Optional[int]:
    """roots is [(root, votes)] or [((root, final), votes)]; show + pick top."""
    if not roots:
        print(f"  ! no {label} root candidates found")
        return None
    print(f"  {label} root candidates (root: votes):")
    for cand, votes in roots[:6]:
        if isinstance(cand, tuple):
            print(f"    0x{cand[0]:X} (final 0x{cand[1]:X}): {votes}")
        else:
            print(f"    0x{cand:X}: {votes}")
    top = roots[0][0]
    return top[0] if isinstance(top, tuple) else top


# ---------------------------------------------------------------------------
# Detection stages
# ---------------------------------------------------------------------------

def detect_bpm(pm, base: int, ref: dict) -> Optional[List[Pointer]]:
    _hr("BPM")
    bpm_a = _prompt_float("Enter the BPM showing on deck 1 (e.g. 128.00): ")
    if bpm_a is None:
        return None
    print("  scanning memory...")
    snap = take_snapshot(pm)
    cand = scan.bpm_round1(snap, bpm_a)
    print(f"  round 1: {len(cand)} candidates")
    if not cand:
        return None
    bpm_b = _prompt_float("Now nudge the pitch/tempo and enter the NEW BPM: ")
    if bpm_b is None:
        return None
    cand = scan.bpm_round2(pm, cand, bpm_b)
    print(f"  round 2: {len(cand)} candidates")
    roots = ptr.trace_field_roots(snap, base, "bpm", cand, ref, LOADED_DECK)
    if not roots:
        print("  hint trace empty — trying final-offset sweep...")
        roots = ptr.blind_trace_roots(snap, base, cand, ref, "bpm", LOADED_DECK)
    root = _pick_root(roots, "bpm")
    return ptr.build_deck_chains("bpm", root, ref) if root is not None else None


def detect_sample(pm, base: int, ref: dict) -> Optional[List[Pointer]]:
    _hr("Playhead / sample position")
    print("  On deck 1: PLAY a few seconds, then STOP. Read the playhead time off the deck.")
    pos1 = _prompt_float("  Playhead position now, in seconds (deck shows 0:18 -> enter 18): ")
    if pos1 is None:
        return None
    print("  scanning memory...")
    snap = take_snapshot(pm)
    prev = scan.sample_round1(pm, snap, pos1)
    print(f"  round 1: {len(prev)} candidates")
    if not prev:
        return None
    pos2 = _prompt_float("  Now PLAY further (do NOT cue back), STOP, and enter the new playhead seconds: ")
    if pos2 is None:
        return None
    if pos2 <= pos1:
        print("  ! position must be later than the first; aborting sample stage")
        return None
    cand = scan.sample_round2(pm, prev, pos2 - pos1)
    print(f"  round 2: {len(cand)} candidates")
    roots = ptr.trace_field_roots(snap, base, "sample_pos", cand, ref, LOADED_DECK)
    if not roots:
        print("  hint trace empty — trying final-offset sweep...")
        roots = ptr.blind_trace_roots(snap, base, cand, ref, "sample_pos", LOADED_DECK)
    root = _pick_root(roots, "sample_pos")
    return ptr.build_deck_chains("sample_pos", root, ref) if root is not None else None


def detect_anlz(pm, base: int, ref: dict, folder: Path) -> Optional[List[Pointer]]:
    _hr("ANLZ path")
    print("  scanning memory for the loaded track's ANLZ folder...")
    snap = take_snapshot(pm)
    cand = scan.anlz_path_addresses(snap, folder.name)
    print(f"  found {len(cand)} string candidates")
    if not cand:
        print("  ! the loaded track's ANLZ path was not found in memory — is it loaded on deck 1?")
        return None
    roots = ptr.trace_field_roots(snap, base, "anlz_path", cand, ref, LOADED_DECK)
    if not roots:
        print("  hint trace empty — trying final-offset sweep...")
        roots = ptr.blind_trace_roots(snap, base, cand, ref, "anlz_path", LOADED_DECK)
    root = _pick_root(roots, "anlz_path")
    return ptr.build_deck_chains("anlz_path", root, ref) if root is not None else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Detect Rekordbox memory offsets")
    parser.add_argument("--reference", default=REFERENCE_VERSION,
                        help="reference version whose chain structure is used as a hint")
    parser.add_argument("--label", default=None,
                        help="version label to write (defaults to the detected version)")
    args = parser.parse_args()

    print("rb_offset_detect — connect Rekordbox first.\n")
    try:
        pm, base = connect_rekordbox()
    except Exception as e:
        print(f"Could not connect to Rekordbox: {e}")
        return 1
    print(f"Connected (module base 0x{base:X})")

    detected = detect_version(pm)
    print(f"Detected Rekordbox version: {detected or 'unknown'}")
    try:
        ref = load_reference_block(args.reference)
    except Exception as e:
        print(f"Could not load reference block {args.reference}: {e}")
        return 1
    print(f"Using reference structure from {ref['version']} ({len(ref['decks'])} decks)")

    label = args.label or detected
    if not label:
        label = _prompt("Enter a version label to write (e.g. 7.3.1): ")
        if not label:
            print("No version label — aborting.")
            return 1

    picked = pick_anlz_dat()
    if picked is None:
        print("Could not find any ANLZ .DAT under the Rekordbox USBANLZ folder.")
        return 1
    folder, song = picked
    print(f"\nPicked ANLZ folder: {folder}")
    print(f'  -> song: "{song}"')
    _prompt(f'In Rekordbox, search for and load "{song}" onto deck 1, press PLAY, then press Enter...')

    field_chains: Dict[str, List[Pointer]] = {}
    for name, fn in (("bpm", lambda: detect_bpm(pm, base, ref)),
                     ("sample_pos", lambda: detect_sample(pm, base, ref)),
                     ("anlz_path", lambda: detect_anlz(pm, base, ref, folder))):
        chains = fn()
        if chains is None:
            print(f"\n! {name} detection failed. Aborting (nothing written).")
            return 1
        field_chains[name] = chains

    # Verify everything live.
    _hr("Verify")
    for kind in ("bpm", "sample_pos", "anlz_path"):
        print(f"{kind}:")
        for line in wr.verify_field(pm, base, kind, field_chains[kind]):
            print(line)

    block_lines = wr.assemble_block(label, field_chains, ref)
    _hr("Result")
    print(wr.preview_block(label, block_lines, ref["version"]))

    target = offsets_file_path()
    ans = _prompt(f'\nAppend version "{label}" to {target}? [y/N] ')
    if ans.lower() == "y":
        wr.append_block(target, label, block_lines, ref["version"])
        print("Written. Restart rkbx_wave to use the new version.")
    else:
        print("Not written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
