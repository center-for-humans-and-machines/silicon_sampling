"""The study's nine scored outcomes, built from raw items exactly as it builds them.

Every formula here is transcribed from the study's own ``SDC - Script - Step 2.R``,
because the whole exercise is comparing our sample against *their* estimates: a
composite built even slightly differently would make the comparison meaningless.

The construction is verified rather than trusted — :func:`verify_against_published`
recomputes all nine from the study's raw responses and checks them against the
published columns.

Note the polarity. Six of the nine are reverse-scored so that **high is bad**:
more animosity, more support for undemocratic practices, more distrust. Getting a
sign wrong here would silently invert directional agreement, which is why the
verification is not optional.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: The nine scored outcomes and their scale ranges (all already 0-100).
OUTCOMES = {
    "PA": 100.0,
    "ADA": 100.0,
    "SPV": 100.0,
    "SUC": 100.0,
    "OppBip": 100.0,
    "SocDistrust": 100.0,
    "SocDis": 100.0,
    "BEPF": 100.0,
    "Composite": 100.0,
}

#: Human-readable names, from the study's own outcome table.
LABELS = {
    "PA": "Partisan Animosity",
    "ADA": "Support for Undemocratic Practices",
    "SPV": "Support for Partisan Violence",
    "SUC": "Support for Undemocratic Candidates",
    "OppBip": "Opposition to Bipartisan Cooperation",
    "SocDistrust": "Social Distrust",
    "SocDis": "Social Distance",
    "BEPF": "Biased Evaluation of Politicized Facts",
    "Composite": "Composite of Outcomes",
}

#: Raw items each outcome needs, by the slot ids our transcripts use.
REQUIRED_ITEMS = (
    "PA_Fth_Rep",
    "PA_Fth_Dem",
    "PA_DG",
    "ADA_1",
    "ADA_2",
    "ADA_3",
    "ADA_4",
    "SPV_1",
    "SPV_2",
    "SPV_3",
    "SPV_4",
    "SUC_1",
    "SUC_2",
    "SUC_3",
    "SUC_4 ",
    "SupBip_1",
    "SupBip_2",
    "SocDis_1",
    "SocDis_2",
    "SocTru",
    "BEPF_R1",
    "BEPF_R2",
    "BEPF_R3",
    "BEPF_R4",
    "BEPF_D1",
    "BEPF_D2",
    "BEPF_D3",
    "BEPF_D4",
)


def _num(frame: pd.DataFrame, name: str) -> pd.Series:
    return (
        pd.to_numeric(frame[name], errors="coerce")
        if name in frame.columns
        else pd.Series(np.nan, index=frame.index)
    )


def compute(frame: pd.DataFrame, inparty: str = "inparty") -> pd.DataFrame:
    """Add the nine outcomes to a respondent-level frame of raw items.

    ``frame`` uses the *slot* names (``ADA_1``); the study's own export suffixes
    them (``ADA_1_1``), and both spellings are accepted so this works on our
    sample and on theirs.
    """
    data = frame.copy()

    def item(*names: str) -> pd.Series:
        for name in names:
            if name in data.columns:
                return _num(data, name)
        return pd.Series(np.nan, index=data.index)

    republican = data[inparty].astype(str).str.startswith("Republican")

    fth_rep = item("PA_Fth_Rep", "PA_Fth_Rep_1")
    fth_dem = item("PA_Fth_Dem", "PA_Fth_Dem_1")
    # Feeling thermometers are warmth toward each party; animosity is the
    # reverse of warmth toward the *out*party, so which item is used flips with
    # the respondent's own party.
    data["PA_Out"] = np.where(republican, 100 - fth_dem, 100 - fth_rep)
    # The dictator game gives 0-50 cents to an outpartisan; animosity is what is
    # withheld, doubled onto the 0-100 scale.
    data["PA_DG_scaled"] = (50 - item("PA_DG", "PA_DG_1")) * 2
    data["PA"] = (data["PA_Out"] + data["PA_DG_scaled"]) / 2

    data["ADA"] = sum(item(f"ADA_{i}", f"ADA_{i}_1") for i in range(1, 5)) / 4
    data["SPV"] = (
        item("SPV_1", "SPV_1_2")
        + item("SPV_2", "SPV_2_2")
        + item("SPV_3", "SPV_3_1")
        + item("SPV_4", "SPV_4_2")
    ) / 4
    data["SUC"] = (
        item("SUC_1", "SUC_1_1")
        + item("SUC_2", "SUC_2_1")
        + item("SUC_3", "SUC_3_1")
        + item("SUC_4 ", "SUC_4_", "SUC_4__1")
    ) / 4

    data["OppBip"] = (
        100 - (item("SupBip_1", "SupBip_1_1") + item("SupBip_2", "SupBip_2_1")) / 2
    )
    data["SocDis"] = (
        100 - (item("SocDis_1", "SocDis_1_1") + item("SocDis_2", "SocDis_2_1")) / 2
    )
    data["SocDistrust"] = 100 - item("SocTru", "SocTru_1")

    bepf_r = sum(item(f"BEPF_R{i}", f"BEPF_R{i}_1") for i in range(1, 5)) / 4
    bepf_d = sum(item(f"BEPF_D{i}", f"BEPF_D{i}_1") for i in range(1, 5)) / 4
    # Each partisan is shown the battery about their own side; bias is the
    # reverse of how critically they evaluate it.
    data["BEPF"] = np.where(republican, 100 - bepf_r, 100 - bepf_d)

    data["Composite"] = data[
        ["PA", "ADA", "SPV", "SUC", "OppBip", "SocDis", "SocDistrust", "BEPF"]
    ].mean(axis=1)
    return data


def verify_against_published(
    published: pd.DataFrame, raw: pd.DataFrame, tolerance: float = 1e-6
) -> pd.DataFrame:
    """Recompute the outcomes from raw responses and check the published columns.

    A sign or an item flipped here would invert the comparison the whole report
    rests on, so this runs before anything is scored.
    """
    recomputed = compute(raw.assign(inparty=published["Inparty_Person"]))
    rows = []
    for outcome in OUTCOMES:
        if outcome not in published.columns:
            continue
        theirs = pd.to_numeric(published[outcome], errors="coerce")
        ours = pd.to_numeric(recomputed[outcome], errors="coerce")
        both = theirs.notna() & ours.notna()
        difference = (theirs[both] - ours[both]).abs()
        rows.append(
            {
                "outcome": outcome,
                "n_compared": int(both.sum()),
                "n_published": int(theirs.notna().sum()),
                "max_abs_diff": (
                    float(difference.max()) if len(difference) else float("nan")
                ),
                "matches": bool(len(difference) and difference.max() < tolerance),
            }
        )
    return pd.DataFrame(rows)
