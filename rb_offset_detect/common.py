# Copyright (c) mrmilbe

"""Shared primitives for rb_offset_detect.

Offline tooling to discover Rekordbox memory offsets for a new app version by:
  1. value-scanning process memory for a known BPM / playhead / path, then
  2. tracing the pointer chain backwards to a module-relative root, reusing the
     known 7.2.2 inner-chain *structure* as a hint.

Only stdlib + already-present deps (pymem, psutil, numpy, pyrekordbox via
rb_waveform_core). Windows-only (ReadProcessMemory / VirtualQueryEx).

The 7.2.2 reference block is the single source of truth for the inner-chain
hints and the per-deck offset strides; it is read from the same offsets file the
runtime uses (rb_waveform_core/data/offsets).
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Make rb_waveform_core importable when run as a script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rb_waveform_core.rb_memory_reader import (  # noqa: E402
    SAMPLE_RATE,
    _OFFSETS_FILE_CANDIDATES,
    _detect_rekordbox_version,
    _read_offset_blocks,
)
from rb_waveform_core.ANLZ import analyze_anlz_folder  # noqa: E402

Pointer = Tuple[List[int], int]   # (deref_offsets, final_offset)

REFERENCE_VERSION = "7.2.2"

# Plausible range for a module-relative root offset inside rekordbox.exe.
ROOT_MIN = 0x0000_0000
ROOT_MAX = 0x1000_0000


# ---------------------------------------------------------------------------
# Win32 memory enumeration
# ---------------------------------------------------------------------------

MEM_COMMIT = 0x1000
MEM_IMAGE = 0x1000000
MEM_PRIVATE = 0x20000
# Readable protections (low byte): READONLY, READWRITE, WRITECOPY, EXECUTE_READ,
# EXECUTE_READWRITE, EXECUTE_WRITECOPY.
_READABLE = {0x02, 0x04, 0x08, 0x20, 0x40, 0x80}
_PAGE_GUARD = 0x100


class MEMORY_BASIC_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BaseAddress", ctypes.c_ulonglong),
        ("AllocationBase", ctypes.c_ulonglong),
        ("AllocationProtect", wintypes.DWORD),
        ("__alignment1", wintypes.DWORD),
        ("RegionSize", ctypes.c_ulonglong),
        ("State", wintypes.DWORD),
        ("Protect", wintypes.DWORD),
        ("Type", wintypes.DWORD),
        ("__alignment2", wintypes.DWORD),
    ]


def _virtual_query():
    fn = ctypes.windll.kernel32.VirtualQueryEx
    fn.restype = ctypes.c_size_t
    fn.argtypes = [
        wintypes.HANDLE,
        ctypes.c_void_p,
        ctypes.POINTER(MEMORY_BASIC_INFORMATION),
        ctypes.c_size_t,
    ]
    return fn


def enum_readable_regions(handle: int) -> List[Tuple[int, int]]:
    """Return [(base, size)] of committed, readable, non-guard PRIVATE/IMAGE regions.

    MAPPED (file-backed) regions are skipped — the structs we want live in the
    heap (PRIVATE) and the module's data sections (IMAGE).
    """
    vq = _virtual_query()
    mbi = MEMORY_BASIC_INFORMATION()
    addr = 0x10000
    max_addr = 0x7FFF_FFFF_FFFF
    regions: List[Tuple[int, int]] = []
    while addr < max_addr:
        if not vq(handle, ctypes.c_void_p(addr), ctypes.byref(mbi), ctypes.sizeof(mbi)):
            break
        base, size = int(mbi.BaseAddress), int(mbi.RegionSize)
        if size <= 0:
            break
        protect = mbi.Protect & 0xFF
        if (
            mbi.State == MEM_COMMIT
            and protect in _READABLE
            and not (mbi.Protect & _PAGE_GUARD)
            and mbi.Type in (MEM_PRIVATE, MEM_IMAGE)
        ):
            regions.append((base, size))
        addr = base + size
    return regions


# ---------------------------------------------------------------------------
# Process connection + bulk reads
# ---------------------------------------------------------------------------

def connect_rekordbox() -> Tuple[object, int]:
    """Open rekordbox.exe and return (pymem instance, module base)."""
    import pymem
    import pymem.process

    pm = pymem.Pymem("rekordbox.exe")
    module = pymem.process.module_from_name(pm.process_handle, "rekordbox.exe")
    if not module:
        raise RuntimeError("Could not locate rekordbox.exe module base")
    return pm, module.lpBaseOfDll


def _read_region(pm, base: int, size: int) -> bytes:
    """Read a region, chunking around unreadable pages (zero-filled to keep addressing)."""
    try:
        return pm.read_bytes(base, size)
    except Exception:
        pass
    out = bytearray(size)
    chunk = 0x100000  # 1 MiB
    for off in range(0, size, chunk):
        n = min(chunk, size - off)
        try:
            out[off:off + n] = pm.read_bytes(base + off, n)
        except Exception:
            pass  # leave zero-filled
    return bytes(out)


class MemorySnapshot:
    """A captured set of readable regions, with cached uint64 views for ptr scans."""

    def __init__(self, pm, regions: List[Tuple[int, int]]):
        self.bases: List[int] = []
        self.buffers: List[bytes] = []
        self._u64: List[Optional[np.ndarray]] = []
        total = 0
        for base, size in regions:
            data = _read_region(pm, base, size)
            self.bases.append(base)
            self.buffers.append(data)
            self._u64.append(None)
            total += len(data)
        self.total_bytes = total

    def iter_buffers(self):
        return zip(self.bases, self.buffers)

    def _u64_view(self, idx: int) -> np.ndarray:
        if self._u64[idx] is None:
            data = self.buffers[idx]
            n = len(data) - (len(data) % 8)
            self._u64[idx] = np.frombuffer(data, dtype="<u8", count=n // 8)
        return self._u64[idx]

    # -- value scans -------------------------------------------------------

    def scan_f32(self, target: float, tol: float) -> List[int]:
        hits: List[int] = []
        with np.errstate(invalid="ignore", over="ignore"):
            for idx, base in enumerate(self.bases):
                data = self.buffers[idx]
                n = len(data) - (len(data) % 4)
                if n == 0:
                    continue
                arr = np.frombuffer(data, dtype="<f4", count=n // 4)
                mask = np.isfinite(arr) & (np.abs(arr - target) <= tol)
                for i in np.nonzero(mask)[0]:
                    hits.append(base + int(i) * 4)
        return hits

    def scan_i64(self, target: int, tol: int) -> List[int]:
        hits: List[int] = []
        lo, hi = target - tol, target + tol
        for idx, base in enumerate(self.bases):
            arr = self._u64_view(idx).view("<i8")
            mask = (arr >= lo) & (arr <= hi)
            for i in np.nonzero(mask)[0]:
                hits.append(base + int(i) * 8)
        return hits

    def scan_string_starts(self, needle: bytes) -> List[int]:
        """Addresses of the start of null-terminated strings containing `needle`."""
        hits: List[int] = []
        for base, data in self.iter_buffers():
            start = 0
            while True:
                idx = data.find(needle, start)
                if idx < 0:
                    break
                nul = data.rfind(b"\x00", 0, idx)
                hits.append(base + (nul + 1 if nul >= 0 else 0))
                start = idx + 1
        return hits

    # -- pointer scan ------------------------------------------------------

    def find_ptr_holders(self, value: int) -> List[int]:
        """Addresses holding the 8-byte value `value` (8-byte-aligned scan)."""
        hits: List[int] = []
        v = np.uint64(value & 0xFFFFFFFFFFFFFFFF)
        for idx, base in enumerate(self.bases):
            arr = self._u64_view(idx)
            for i in np.nonzero(arr == v)[0]:
                hits.append(base + int(i) * 8)
        return hits


def trace_roots(
    snap: MemorySnapshot,
    base: int,
    final_addr: int,
    inner_offsets: List[int],
    final_offset: int,
    max_candidates: int = 4096,
) -> List[int]:
    """Reverse a pointer chain from a known final address to module-relative roots.

    Mirrors the forward resolution in rb_memory_reader._resolve_chain:
        addr = base
        for o in [root, *inner_offsets]: addr = *(addr + o)
        final_addr = addr + final_offset
    Returns candidate `root` offsets (relative to `base`).
    """
    # Work backwards through inner_offsets (reverse order), then the root level.
    level = [final_addr - final_offset]            # P_last (struct addr)
    for off in reversed(inner_offsets):
        nxt: List[int] = []
        for pv in level:
            for holder in snap.find_ptr_holders(pv):
                nxt.append(holder - off)           # P_{k-1}
        # de-dup, bound
        level = list(dict.fromkeys(nxt))[:max_candidates]
        if not level:
            return []
    roots: List[int] = []
    for p1 in level:
        for holder in snap.find_ptr_holders(p1):
            root = holder - base
            if ROOT_MIN <= root <= ROOT_MAX:
                roots.append(root)
    return list(dict.fromkeys(roots))


# ---------------------------------------------------------------------------
# Pointer-line formatting + reference block
# ---------------------------------------------------------------------------

def parse_pointer_line(line: str) -> Pointer:
    parts = [int(tok, 16) for tok in line.split()]
    return parts[:-1], parts[-1]


def format_pointer(derefs: List[int], final: int) -> str:
    return " ".join(f"{v:X}" for v in (*derefs, final))


def offsets_file_path() -> Path:
    """First existing offsets file (prefers the packaged rb_waveform_core copy)."""
    for cand in _OFFSETS_FILE_CANDIDATES:
        if cand.exists():
            return cand
    return _OFFSETS_FILE_CANDIDATES[0]


def load_reference_block(version: str = REFERENCE_VERSION) -> dict:
    """Read the reference version's raw pointer lines from the offsets file.

    Returns {
        "version": str,
        "masterdeck": str,              # raw line
        "decks": [ {bpm, sample_pos, track_info, anlz_path}, ... ]  # raw lines
    }
    Per-deck order matches rb_waveform_core/rb_memory_reader (4 lines / deck).
    """
    text = offsets_file_path().read_text(encoding="utf-8")
    for block in _read_offset_blocks(text):
        if block and block[0] == version:
            rest = block[2:]
            decks = []
            for i in range(0, len(rest) - 3, 4):
                decks.append({
                    "bpm": rest[i],
                    "sample_pos": rest[i + 1],
                    "track_info": rest[i + 2],
                    "anlz_path": rest[i + 3],
                })
            return {"version": block[0], "masterdeck": block[1], "decks": decks}
    raise ValueError(f"Reference version {version} not found in offsets file")


# ---------------------------------------------------------------------------
# Song picker (reuses the runtime ANLZ parser)
# ---------------------------------------------------------------------------

def usbanlz_root() -> Optional[Path]:
    appdata = Path.home() / "AppData" / "Roaming"
    import os
    appdata = Path(os.environ.get("APPDATA", str(appdata)))
    root = appdata / "Pioneer" / "rekordbox" / "share" / "PIONEER" / "USBANLZ"
    return root if root.is_dir() else None


def pick_anlz_dat(root: Optional[Path] = None) -> Optional[Tuple[Path, str]]:
    """Pick an ANLZ folder that has a parseable song name.

    Returns (anlz_folder, song_name) — song_name is the PPTH filename. The folder
    path itself is what will appear in Rekordbox's memory for the loaded deck.
    """
    root = root or usbanlz_root()
    if root is None:
        return None
    for dat in root.rglob("ANLZ0000.DAT"):
        folder = dat.parent
        try:
            result = analyze_anlz_folder(folder)
        except Exception:
            continue
        if result.song_path is not None:
            return folder, Path(result.song_path).name
    return None


def detect_version(pm) -> Optional[str]:
    return _detect_rekordbox_version(pm.process_id)
