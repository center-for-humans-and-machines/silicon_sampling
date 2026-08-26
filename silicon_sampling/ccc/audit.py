"""Per-arm media loss for the Climate Change Challenge.

Written from the fidelity verification rather than inferred from tag counts: an
image's *count* says nothing about whether the argument depends on it, and half of
one earlier study's audit claims were written without opening the file. Each rating
here traces to something checked — the prose read, or the image fetched and
measured. See ``docs/reports/ccc_validation/``.
"""

from __future__ import annotations

import pandas as pd

from . import instrument as inst
from .paths import MODALITY_AUDIT

REASONS = {
    "Consensus Framing 2": "pure text, single page",
    "Gains Framing": "no images; the longest stimulus at 4,693 characters",
    "Free Market Framing": "no images; loses only the pacing of 11 one-sentence pages",
    "Control Neckties": "one image, URL live, fetched and verified a decorative photo with no prose reference",
    "Control Baseball": "one image, URL live, fetched and verified a decorative photo",
    "Control Dances": "one image, URL live, fetched and verified a decorative photo",
    "Consensus Framing 1": "one dead image between two paragraphs that both state the 97% figure verbatim; high redundancy",
    "Binding Framing": "one dead 800x182 masthead above the slogan, which is separately present as text; not asserted decorative",
    "High Social Distance Framing": "two dead images; the 800x65 strip under a 16:9 photo is probably a caption rendered as an image, so one line of read text is likely lost",
    "Purity Framing": "image load-bearing but recoverable: the prose points at it deictically and the live image measures a 3.2x saturation collapse across halves, matching the clear/hazy pair the text claims",
    "Warmth Framing": "no images, but the arm's active ingredient is a writing task (median dwell 130.8 s against 17.0 s on page 1); kept as a free-text slot, and its human answer was never released",
    "Dire But Solvable Framing": "unresolved: two image-only questions at the tops of pages 3 and 4, no captions, no prose reference, absent from the SI, URLs dead, yet humans dwelled 20.8 s and 15.0 s on them",
}


def table() -> pd.DataFrame:
    rows = []
    for condition in inst.conditions():
        rows.append(
            {
                "condition": condition,
                "usable": "yes",
                "media_loss": inst.MEDIA_LOSS.get(condition),
                "image_note_supplied": condition in inst.ARM_IMAGE_NOTES,
                "reason": REASONS.get(condition, ""),
            }
        )
    for condition, why in inst.DROPPED_ARMS.items():
        rows.append(
            {
                "condition": condition,
                "usable": "no",
                "media_loss": 3,
                "image_note_supplied": False,
                "reason": why,
            }
        )
    return pd.DataFrame(rows).sort_values(["media_loss", "condition"])


def main() -> int:
    frame = table()
    MODALITY_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(MODALITY_AUDIT, index=False)
    pd.set_option("display.width", 200)
    print(frame[["condition", "usable", "media_loss"]].to_string(index=False))
    print(f"\nwrote {MODALITY_AUDIT}")
    return 0
