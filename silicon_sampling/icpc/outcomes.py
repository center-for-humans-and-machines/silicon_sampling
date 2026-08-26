"""The study's four outcomes, built the way its own cleaning script builds them.

Unlike the Voelkel tournament, this study publishes no composite and no reverse
scoring: all four outcomes point the same way, higher is more pro-climate, and
three of them are read straight off the raw items.  The one that is not — the
effortful task — is a *count*, and that matters more than it sounds, because it
is the only outcome here whose scale is neither 0-100 nor binary and the only one
where a treatment effect in the human data runs the *opposite* way to the other
three.

The constructions were reverse-engineered from the two published files and then
verified against each other rather than taken on trust.  Doell et al.'s raw
export and Vlasceanu et al.'s cleaned extract come from one data collection, so
recomputing the cleaned columns from the raw ones is a real check with a real
failure mode; :func:`verify_against_published` runs it over all 58,928 shared
respondents and reports the maximum absolute discrepancy per outcome.  It is
zero for all four.

Two details that were not obvious and are worth stating:

* ``Share`` was recoded before publication.  The Qualtrics codes were 1 = willing,
  2 = no social media, 3 = not willing; the cleaning script maps 3 to 0, so the
  published column reads 0 = not willing, 1 = willing, 2 = does not use social
  media.  The analysis extract then drops the third category to ``NA``, which is
  what makes the sharing outcome binary rather than three-valued.
* ``WEPTcc`` is the plain sum of the eight ``WEPT<n>confirm`` flags, not a count of
  pages that met the 90%-correct rule the instrument threatened.  Accepting a page
  is what counts.  The sum treats a *missing* flag as a refusal, which is the
  study's own convention and not a shortcut: the eight questions are chained, so
  page 5 is missing precisely because page 4 was declined, and 18 respondents who
  never reached the block at all are published as ``WEPTcc = 0``.  Only a frame
  carrying no WEPT column whatsoever yields ``NaN``.

A note on what the verification can and cannot see.  Comparing a battery *mean*
against the published composite is invariant to any permutation of the items
inside it, so it says nothing about whether item 1 of the transcript is item 1 of
the export — and that binding is exactly what a hand transcription gets wrong.
:func:`verify_items_against_published` therefore compares the thirteen belief and
policy items one at a time, and the item-to-column binding itself is held to the
``.qsf`` and the codebook in ``tests/test_icpc.py``.  The two checks answer
different questions and neither substitutes for the other.
"""

from __future__ import annotations

import numpy as np

from ..lazy import lazy_module

#: Imported on first use, so the sampler stays importable in the
#: Muse-Glimmer vLLM container, which ships no pandas.  Only the analysis
#: functions in this module touch it.
pd = lazy_module("pandas")

#: Outcome -> scale range, in the units :func:`treatment_effects` divides by so
#: that every effect is expressed in percentage points of its own scale.
OUTCOMES: dict[str, float] = {
    "belief": 100.0,
    "policy": 100.0,
    "sharing": 1.0,
    "wept": 8.0,
}

LABELS = {
    "belief": "Belief in climate change",
    "policy": "Support for climate policy",
    "sharing": "Willingness to share climate information",
    "wept": "Effortful task (WEPT pages completed)",
}

#: Raw items each outcome is built from, by the slot ids the transcripts use.
BELIEF_ITEMS = tuple(f"Belief.in.CC_{i}" for i in (1, 2, 4, 5))
POLICY_ITEMS = tuple(f"CC_policy_{i}" for i in (1, 2, 3, 5, 6, 7, 8, 9, 10))
WEPT_ITEMS = tuple(f"WEPT{i}confirm" for i in range(1, 9))
SHARE_ITEM = "Share"

REQUIRED_ITEMS = BELIEF_ITEMS + POLICY_ITEMS + WEPT_ITEMS + (SHARE_ITEM,)

#: Published names for the same four columns in Vlasceanu et al.'s extract.
PUBLISHED = {
    "belief": tuple(f"Belief{i}" for i in range(1, 5)),
    "policy": tuple(f"Policy{i}" for i in range(1, 10)),
    "sharing": ("SHAREcc",),
    "wept": ("WEPTcc",),
}

#: Raw item -> the cleaned extract's column for that same item.  Vlasceanu et al.
#: renumbered the batteries consecutively (``CC_policy_5`` became ``Policy4``),
#: which is why the pairing has to be written down rather than derived.
PUBLISHED_ITEMS: dict[str, dict[str, str]] = {
    "belief": dict(zip(BELIEF_ITEMS, (f"Belief{i}" for i in range(1, 5)))),
    "policy": dict(zip(POLICY_ITEMS, (f"Policy{i}" for i in range(1, 10)))),
}

#: How the sharing item is coded once the cleaning script has run.
SHARE_CODES = {
    0: "I'm not willing to share that.",
    1: "Yes, I am willing to share this information.",
    2: "I do not use social media.",
}
SHARE_TEXT_TO_CODE = {text: code for code, text in SHARE_CODES.items()}
#: The category the analysis extract drops rather than scores.
SHARE_NOT_APPLICABLE = 2


def _numeric(frame: pd.DataFrame, names) -> pd.DataFrame:
    """The named columns as numbers, with absent ones as all-missing."""
    out = {}
    for name in names:
        if name in frame.columns:
            out[name] = pd.to_numeric(frame[name], errors="coerce")
        else:
            out[name] = pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.DataFrame(out, index=frame.index)


def share_code(values: pd.Series) -> pd.Series:
    """The sharing item as 0/1/2, whether it arrived as a code or as text.

    A silicon sample answers the option *text*; the published data holds the
    recoded number.  Both are accepted so one construction serves both frames.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    text = values.astype(str).str.strip().map(SHARE_TEXT_TO_CODE)
    return numeric.where(numeric.notna(), text)


def compute(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the four outcomes to a respondent-level frame of raw items."""
    data = frame.copy()
    data["belief"] = _numeric(data, BELIEF_ITEMS).mean(axis=1)
    data["policy"] = _numeric(data, POLICY_ITEMS).mean(axis=1)

    codes = share_code(data[SHARE_ITEM]) if SHARE_ITEM in data.columns else None
    if codes is None:
        data["sharing"] = np.nan
    else:
        data["sharing"] = codes.where(codes != SHARE_NOT_APPLICABLE)

    confirms = _numeric(data, WEPT_ITEMS)
    # "yes"/"no" from a sampled transcript, 1/0 from the published data.
    for name in WEPT_ITEMS:
        if name in data.columns and confirms[name].isna().all():
            spelled = data[name].astype(str).str.strip().str.lower()
            confirms[name] = spelled.map({"yes": 1.0, "no": 0.0})
    present = [name for name in WEPT_ITEMS if name in data.columns]
    data["wept"] = (
        confirms.sum(axis=1)
        if present
        else pd.Series(np.nan, index=data.index, dtype="float64")
    )
    return data


def composite(frame: pd.DataFrame) -> pd.Series:
    """The mean of the four outcomes, each first put on 0-100.

    Ours, not the study's — neither publication reports a composite — so it is
    kept out of :data:`OUTCOMES` and off the scoring path.  It exists because a
    single number per arm is the readable summary in a report table.
    """
    parts = [
        pd.to_numeric(frame[name], errors="coerce") / scale * 100
        for name, scale in OUTCOMES.items()
        if name in frame.columns
    ]
    return pd.concat(parts, axis=1).mean(axis=1) if parts else pd.Series(dtype=float)


def _merge_on_response_id(raw: pd.DataFrame, published: pd.DataFrame) -> pd.DataFrame:
    """The two published files joined, with missing ids dropped first.

    512 of the cleaned rows and 512 of the raw rows carry a missing
    ``ResponseId``, and a merge on missing keys matches every one of them to
    every other: 58,928 rows become 321,072 and any comparison after that is
    meaningless.  Dropping them costs nothing — a row with no id cannot be
    checked against anything anyway.
    """
    return raw.dropna(subset=["ResponseId"]).merge(
        published.dropna(subset=["ResponseId"]),
        on="ResponseId",
        suffixes=("", "_pub"),
    )


def _agreement(
    theirs: pd.Series, mine: pd.Series, tolerance: float
) -> dict[str, object]:
    """One comparison row, counting the rows each side has and the other lacks.

    The count columns are the point.  An earlier version reported only the
    intersection, so a row the study scored and we left missing simply vanished
    from the comparison and ``matches`` stayed ``True`` — which is how 18
    respondents published as ``WEPTcc = 0`` went unnoticed behind a maximum
    absolute difference of zero.  Coverage is part of agreeing.
    """
    both = theirs.notna() & mine.notna()
    difference = (theirs[both] - mine[both]).abs()
    worst = float(difference.max()) if len(difference) else float("nan")
    only_published = int((theirs.notna() & mine.isna()).sum())
    only_ours = int((mine.notna() & theirs.isna()).sum())
    return {
        "n_compared": int(both.sum()),
        "n_published": int(theirs.notna().sum()),
        "n_ours": int(mine.notna().sum()),
        "only_published": only_published,
        "only_ours": only_ours,
        "max_abs_diff": worst,
        "matches": bool(
            len(difference)
            and worst <= tolerance
            and not only_published
            and not only_ours
        ),
    }


def verify_against_published(
    raw: pd.DataFrame, published: pd.DataFrame, tolerance: float = 1e-9
) -> pd.DataFrame:
    """Recompute the four outcomes from Doell's raw items and check Vlasceanu's.

    A battery mean is invariant to permuting the items inside it, so this check
    is blind to the item-to-column binding by construction; pair it with
    :func:`verify_items_against_published`.
    """
    merged = _merge_on_response_id(raw, published)
    ours = compute(merged)
    rows = []
    for outcome, columns in PUBLISHED.items():
        if len(columns) == 1:
            theirs = pd.to_numeric(merged[columns[0]], errors="coerce")
        else:
            theirs = _numeric(merged, columns).mean(axis=1)
        mine = pd.to_numeric(ours[outcome], errors="coerce")
        rows.append({"outcome": outcome, **_agreement(theirs, mine, tolerance)})
    return pd.DataFrame(rows)


def verify_items_against_published(
    raw: pd.DataFrame, published: pd.DataFrame, tolerance: float = 1e-9
) -> pd.DataFrame:
    """The same check one item at a time, which is what a permutation shows up in.

    Every belief and policy item, not a spot check: if the raw export's
    ``CC_policy_5`` and the cleaned extract's ``Policy4`` disagree respondent by
    respondent, the two files do not share an item order and nothing built on
    either one can be trusted.  The composite check cannot see this, because the
    mean of nine sliders is the same whichever slider was which.
    """
    merged = _merge_on_response_id(raw, published)
    rows = []
    for outcome, pairs in PUBLISHED_ITEMS.items():
        for item, column in pairs.items():
            mine = pd.to_numeric(merged.get(item), errors="coerce")
            theirs = pd.to_numeric(merged.get(column), errors="coerce")
            rows.append(
                {
                    "outcome": outcome,
                    "item": item,
                    "published_column": column,
                    "mean_ours": float(mine.mean()),
                    "mean_published": float(theirs.mean()),
                    **_agreement(theirs, mine, tolerance),
                }
            )
    return pd.DataFrame(rows)
