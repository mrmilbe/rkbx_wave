# Copyright (c) mrmilbe

#!/usr/bin/env python3
"""Simple, low-noise CLI harness for :mod:`rkbx_link_listener`."""

from typing import Iterable, List
import time

from rb_waveform_lab.rkbx_link_listener import DeckEvent, RekordboxLinkListener


# Edit this tuple locally when you want to tweak the output columns.
FIELDS_TO_SHOW = (
    "deck",
    "source",
    "time",
    "anlz",
    "bpm",
    "time-scale",
)

FIELD_TO_SOURCES = {
    #"time": ("time",),
    "anlz": ("anlz_path",),
    #"bpm": ("bpm_current", "bpm_original"),
    #"time-scale": ("bpm_current", "bpm_original"),
}


def format_event(event: DeckEvent, fields: Iterable[str]) -> str:
    parts: List[str] = []
    for field in fields:
        if field == "deck":
            parts.append(f"Deck {event.deck}")
        elif field == "source":
            parts.append(f"src={event.source}")
        elif field == "time":
            time_val = "-" if event.time_seconds is None else f"{event.time_seconds:7.2f}s"
            parts.append(f"t={time_val}")
        elif field == "anlz":
            path_val = event.anlz_path or "(none)"
            parts.append(f"ANLZ={path_val}")
        elif field == "bpm":
            cur = "-" if event.current_bpm is None else f"{event.current_bpm:6.2f}"
            orig = "-" if event.original_bpm is None else f"{event.original_bpm:6.2f}"
            parts.append(f"BPM={cur}/{orig}")
        elif field == "time-scale":
            ts = "-" if event.time_scale is None else f"{event.time_scale:5.3f}"
            parts.append(f"scale={ts}")
    return " | ".join(parts)


def main() -> None:
    listener = RekordboxLinkListener()
    listener.start()
    print(f"Listening for Rekordbox Link events on {listener.ip}:{listener.port}")
    print("Press Ctrl+C to stop.")
    sources_of_interest = set()
    for field in FIELDS_TO_SHOW:
        sources_of_interest.update(FIELD_TO_SOURCES.get(field, ()))
    try:
        while True:
            event = listener.get_event(timeout=1.0)
            if event is None:
                continue
            if sources_of_interest and event.source not in sources_of_interest:
                continue
            print(format_event(event, FIELDS_TO_SHOW))
    except KeyboardInterrupt:
        print("\nStopping listener...")
    finally:
        listener.stop()
        time.sleep(0.1)

if __name__ == "__main__":
    main()
