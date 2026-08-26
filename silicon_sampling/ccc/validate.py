"""Checks that must pass before the Climate Change Challenge is sampled."""

from __future__ import annotations

import random
import re

import pandas as pd

from ..survey.render import walk
from ..survey.slots import IntSlot
from . import instrument as inst
from . import outcomes as oc
from .paths import RECODED_CSV

_RANGE = re.compile(r"Whole number from -?\d+ to -?\d+")


def check_every_slider_states_its_range() -> dict:
    """No integer slot may reach a model without its numeric range.

    The defect this guards against cost an entire sampling round on two other
    studies: models asked for a number with no range stated answered on an
    implicit 0-10 scale, putting 80-94% of 0-100 answers at 10 or below against
    8-31% for real participants.
    """
    silent = []
    total = 0
    for condition in inst.conditions():
        for event, payload in walk(inst.elements_for(condition, random.Random(0))):
            if event != "slot" or not isinstance(payload, IntSlot):
                continue
            total += 1
            if not _RANGE.search(payload.describe() or ""):
                silent.append((condition, payload.id))
    return {"int_slots": total, "without_range": silent}


def check_composites_reproduce() -> dict:
    """Our item lists must rebuild the published composites exactly.

    Run against the released data with the reverse-coding *not* applied, because
    the released item columns already carry it. A mismatch means the item list is
    wrong, which would corrupt every effect built on that outcome.
    """
    frame = pd.read_csv(RECODED_CSV, encoding="utf-8-sig", low_memory=False)
    columns = inst.data_columns()
    worst = {}
    for name, (items, _) in oc.COMPOSITES.items():
        mapped = [columns.get(item, item) for item in items]
        if name not in frame.columns or any(c not in frame.columns for c in mapped):
            worst[name] = None
            continue
        values = frame[mapped].apply(pd.to_numeric, errors="coerce")
        rebuilt = values.sum(axis=1) if name in oc.SUMMED else values.mean(axis=1)
        reference = pd.to_numeric(frame[name], errors="coerce")
        keep = rebuilt.notna() & reference.notna()
        worst[name] = float((rebuilt[keep] - reference[keep]).abs().max())
    return worst


def main() -> int:
    ranges = check_every_slider_states_its_range()
    print(f"integer slots: {ranges['int_slots']}")
    print(f"  without a stated range: {len(ranges['without_range'])}")
    for condition, slot in ranges["without_range"][:20]:
        print(f"    {condition}: {slot}")
    print("\ncomposite reconstruction, max |difference| against the released column:")
    bad = 0
    for name, diff in check_composites_reproduce().items():
        if diff is None:
            print(f"  {name:18s} NOT CHECKABLE")
            bad += 1
        else:
            flag = "" if diff < 1e-6 else "   <-- MISMATCH"
            bad += diff >= 1e-6
            print(f"  {name:18s} {diff:.2e}{flag}")
    failed = len(ranges["without_range"]) + bad
    print("\nVERDICT:", "clean" if failed == 0 else f"{failed} problem(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
