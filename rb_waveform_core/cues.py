# Copyright (c) mrmilbe

"""Hot-cue / memory-cue extraction for a loaded track.

Where cues live depends on the source:
  - Local library tracks: the ANLZ files under the rekordbox AppData ``share``
    folder do NOT contain cues (Rekordbox keeps them in its encrypted master.db
    and only writes them into ANLZ on export). So for local tracks we read them
    from master.db via pyrekordbox, mapping the ANLZ folder -> ContentID.
  - Exported Pioneer USB sticks (audio under ``X:\\Contents``): the ANLZ files on
    the stick are populated on export, so we read the cues straight from them.

Both hot cues and memory cues are returned.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from pyrekordbox.anlz import AnlzFile

from .ANLZ import _pick_anlz_file_for_beatgrid, _pick_anlz_file


# Fixed hot-cue colours by slot (A=1 .. H=8).
_HOTCUE_SLOT_COLORS = {
    1: (230, 40, 40),     # A red
    2: (50, 200, 215),    # B light blue / turquoise
    3: (45, 125, 45),     # C dark grass green
    4: (160, 60, 205),    # D purple
    5: (70, 230, 70),     # E bright green
    6: (255, 145, 20),    # F orange
    7: (35, 70, 205),     # G dark blue
    8: (240, 220, 45),    # H yellow
}
_DEFAULT_HOTCUE_RGB = (0, 200, 130)    # fallback when slot is unknown
_DEFAULT_MEMORY_RGB = (255, 190, 0)    # amber, for memory cues


@dataclass(frozen=True)
class CueMarker:
    """A single cue point to draw on the waveform."""
    time_sec: float
    color: tuple          # (r, g, b)
    is_memory: bool        # True = memory cue, False = hot cue
    number: int = 0        # hot-cue slot (1-8) when applicable


# Local rekordbox analysis root (cues are NOT stored in these ANLZ files).
_LOCAL_SHARE = Path(os.environ.get("APPDATA", str(Path.home()))) / "Pioneer" / "rekordbox" / "share"

_DB = None
_DB_FAILED = False


def _marker_color(is_memory: bool, number: int) -> tuple:
    """Memory cues are amber; hot cues are coloured by their slot (A-H)."""
    if is_memory:
        return _DEFAULT_MEMORY_RGB
    return _HOTCUE_SLOT_COLORS.get(number, _DEFAULT_HOTCUE_RGB)


def _is_local(folder: Path) -> bool:
    try:
        return folder.resolve().is_relative_to(_LOCAL_SHARE.resolve())
    except (ValueError, OSError):
        return False


def get_track_cues(folder: Path) -> List[CueMarker]:
    """Return hot + memory cues for a track's ANLZ folder (local DB or stick)."""
    folder = Path(folder)
    try:
        if _is_local(folder):
            return _cues_from_db(folder)
        return _cues_from_anlz(folder)
    except Exception as e:
        print(f"[Cues] Failed to load cues for {folder}: {e}")
        return []


# ---------------------------------------------------------------------------
# Local library: Rekordbox master.db via pyrekordbox
# ---------------------------------------------------------------------------

def _get_db():
    global _DB, _DB_FAILED
    if _DB is None and not _DB_FAILED:
        try:
            from pyrekordbox import Rekordbox6Database
            _DB = Rekordbox6Database()
        except Exception as e:
            print(f"[Cues] Could not open Rekordbox database: {e}")
            _DB_FAILED = True
    return _DB


def _cues_from_db(folder: Path) -> List[CueMarker]:
    db = _get_db()
    if db is None:
        return []
    from pyrekordbox.db6.tables import DjmdContent

    # ANLZ folder names are unique content GUIDs; match the AnalysisDataPath.
    content = (
        db.session.query(DjmdContent)
        .filter(DjmdContent.AnalysisDataPath.like(f"%/{folder.name}/%"))
        .first()
    )
    if content is None:
        return []

    out: List[CueMarker] = []
    for cue in db.get_cue(ContentID=content.ID):
        in_ms = getattr(cue, "InMsec", None)
        if in_ms is None:
            continue
        is_mem = bool(getattr(cue, "is_memory_cue", False))
        number = int(getattr(cue, "Kind", 0) or 0)
        out.append(CueMarker(
            time_sec=float(in_ms) / 1000.0,
            color=_marker_color(is_mem, number),
            is_memory=is_mem,
            number=number,
        ))
    return out


# ---------------------------------------------------------------------------
# Exported USB stick: cues live in the (populated) ANLZ files
# ---------------------------------------------------------------------------

def _cues_from_anlz(folder: Path) -> List[CueMarker]:
    anlz_file = _pick_anlz_file_for_beatgrid(folder) or _pick_anlz_file(folder)
    if anlz_file is None:
        return []
    try:
        tags = getattr(AnlzFile.parse_file(str(anlz_file)), "tags", []) or []
    except Exception:
        return []

    out: List[CueMarker] = []
    seen = set()
    for tag in tags:
        name = type(tag).__name__
        if not (name.startswith("PCOB") or name.startswith("PCO2")):
            continue
        content = getattr(getattr(tag, "struct", None), "content", None)
        if content is None:
            continue
        # Memory vs hot from the tag's type/cue_type enum.
        type_val = getattr(content, "type", None) or getattr(content, "cue_type", None)
        is_mem = str(type_val).endswith("memory") or str(type_val) == "0"
        for e in getattr(content, "entries", None) or []:
            t = getattr(e, "time", None)
            if t is None:
                continue
            time_sec = float(t) / 1000.0
            key = (round(time_sec, 3), is_mem)
            if key in seen:
                continue
            seen.add(key)
            number = int(getattr(e, "hot_cue", 0) or 0)
            out.append(CueMarker(time_sec, _marker_color(is_mem, number), is_mem, number))
    return out
