# Copyright (c) mrmilbe

"""Reverse pointer-chain tracing for rb_offset_detect.

Given candidate final addresses (from scan.py) for the loaded deck, trace the
pointer chain backwards to a module-relative root, using the reference version's
inner-chain *structure* as a hint. In 7.2.2 all decks of a field share the same
root (only a second-level offset strides per deck), so finding the root from the
loaded deck lets us construct every deck's chain by reusing the reference strides.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

from rb_offset_detect.common import (
    MemorySnapshot,
    Pointer,
    parse_pointer_line,
    trace_roots,
)


def inner_hint(reference_block: dict, field: str, deck_index: int = 0) -> Tuple[List[int], int]:
    """(inner_offsets, final_offset) for a field/deck from the reference block.

    inner_offsets excludes the module-relative root (the first deref offset).
    """
    line = reference_block["decks"][deck_index][field]
    derefs, final = parse_pointer_line(line)
    return derefs[1:], final


def trace_field_roots(
    snap: MemorySnapshot,
    base: int,
    field: str,
    final_addrs: List[int],
    reference_block: dict,
    deck_index: int = 0,
) -> List[Tuple[int, int]]:
    """Trace candidate final addresses to roots, ranked by how often each recurs.

    Returns [(root, votes)] sorted by votes desc. A root that the true final
    address resolves to will typically out-vote spurious string/value hits.
    """
    inner, final = inner_hint(reference_block, field, deck_index)
    votes: Counter = Counter()
    for addr in final_addrs:
        for root in trace_roots(snap, base, addr, inner, final):
            votes[root] += 1
    return votes.most_common()


def blind_trace_roots(
    snap: MemorySnapshot,
    base: int,
    final_addrs: List[int],
    reference_block: dict,
    field: str,
    deck_index: int = 0,
    final_window: int = 0x80,
) -> List[Tuple[int, int]]:
    """Fallback: reuse the reference inner chain but sweep the final offset.

    Covers the common case where a struct grew and only the terminal field
    offset shifted. Sweeps final_offset over reference_final ± final_window
    (8-byte steps for ints/ptrs; 4 for the f32 BPM field).
    """
    inner, ref_final = inner_hint(reference_block, field, deck_index)
    step = 4 if field == "bpm" else 8
    votes: Counter = Counter()
    finals = range(max(0, ref_final - final_window), ref_final + final_window + 1, step)
    for addr in final_addrs:
        for fo in finals:
            for root in trace_roots(snap, base, addr, inner, fo):
                votes[(root, fo)] += 1
    # Flatten: report best (root, final) pairs as (root, votes) keyed separately.
    return votes.most_common()


def build_deck_chains(field: str, root: int, reference_block: dict) -> List[Pointer]:
    """Construct every deck's chain for `field` by swapping the new root into the
    reference per-deck deref lists (preserving per-deck strides + final offset)."""
    chains: List[Pointer] = []
    for deck in reference_block["decks"]:
        derefs, final = parse_pointer_line(deck[field])
        chains.append(([root, *derefs[1:]], final))
    return chains
