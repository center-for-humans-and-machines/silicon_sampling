"""The study's outcomes, built from raw items exactly as the study builds them.

Every formula here is transcribed from the authors' own
``Advocacy_Cleaning_main.ipynb`` (fetched with the rest of the materials, and
sitting in ``data/Goldwert/Materials/code``), because the point of the exercise
is comparing a silicon sample against *their* estimates: a composite built even
slightly differently makes the comparison meaningless.  The construction is
verified rather than trusted — :func:`verify_against_published` recomputes all
four preregistered composites from the raw items in the published file and checks
them against the published columns.

Everything on this scale is oriented so that **high is more advocacy**, which
makes the sign of an intervention effect readable without a lookup table.  Three
things about the scales are worth stating before anyone treats them as clean
latent measures.

**Eleven items are sliders that start at zero.** The survey set
``SliderStartPositions`` to 0 with ``CustomStart``, so a respondent who never
touched the control is recorded as an exact 0 rather than as missing.  ``march``,
``conversation``, ``flyless``, ``lessbeef``, ``bank_raw``, ``pol_candidate``,
``pol_campaign``, ``belief_1``, ``policy_1``, ``Pefficacy`` and ``Cefficacy`` all
carry a non-response spike at the bottom of their range that cannot be told apart
from a real "definitely not".

**The two attitude items are reverse-labelled, and their polarity cannot be
settled.** ``belief_1`` and ``policy_1`` put "Very much so" at the left of the
bar and "Not at all" at the right, the opposite of every other slider in the
instrument.  Three pieces of evidence about which direction respondents actually
answered in do not agree.  The labels say a high number means *less* belief.  The
party gap says the opposite, though only just: Democrats score 63.3 on
``belief_1`` against Republicans' 57.5 once the zero spike is dropped, and on
``policy_1`` there is *no* party gap at all (56.6 against 57.5) in a country where
support for a fossil-fuel transition is among the most polarised questions there
is.  The display-position slope agrees with the labels: every other outcome falls
as it is shown later — ``march`` from 48 to 39 — while these two *rise*, 48.7 to
57.2 and 45.0 to 55.1, which is what a reverse-scored item looks like under the
same fatigue.  The paper reports neither item.  They are kept here, flagged, and
excluded from :data:`SCORED`, because calibrating a Pfänder attitude outcome
against a scale whose sign is unknown is worse than not calibrating it.

**Missingness is heavy and correlated with condition and with display position.**
Non-null counts run from 23,846 on ``march`` to 14,069 on ``flyless``, and three
columns — ``letter``, ``newsletter`` and ``donation_bin`` — were zero-filled to
all 31,324 rows by their construction even though their source items stop at
about 23,400.  A mean taken over the full column is not a mean over the people
who saw the question.

Two shapes of missingness live under that sentence and only one of them is
reproducible.  The heavy non-null counts on ``flyless``, ``lessbeef`` and
``pol_candidate`` are *opt-outs*: each of those sliders carried a "Not
Applicable" checkbox and a respondent who already did not fly ticked it, which
the cleaning script read as missing.  A synthetic respondent can do the same —
see :class:`~silicon_sampling.goldwert.convert.EscapableIntSlot`, which prints
the escape in the survey's own words and records ``NaN`` where the published file
has ``NaN`` — so ``lifestyle_changes``, whose only two members are ``flylessN``
and ``lessbeefN``, comes out missing for the same *kind* of respondent even
though the rate will not match.

The zero-fills are not reproducible at all, and ``newsletter`` is the one that
costs something, because unlike ``letter`` it is scored twice: standalone in
:data:`SCORED` at weight 1.0 and again as one of the four in
``public_awareness``.  4,140 of the 19,141 kept-arm rows are zeros contributed by
respondents who never reached the signup form, reach is *lowest* in the control
arm (69.2% against 65.1%-83.3% across the treatments), and the resulting bias has
a known direction: the zero-fill inflates the mean absolute arm effect by 1.95x
and flips its sign on four of the ten arms.  A silicon sample has reach 1.0 and
so estimates the reach-conditional effect — attenuated by about half against the
published all-rows one, wrong-signed on those four arms, and about +0.07 high in
level.  :func:`~silicon_sampling.goldwert.score.newsletter_contribution` is the
table; the point of stating it here is that ``SCORED`` and ``COMPOSITES`` below
both carry the column and neither can be written any other way, because the
comparison is against the authors' published construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Item:
    """One measured item, and everything needed to read its column honestly."""

    column: str
    label: str
    kind: str
    lo: float
    hi: float
    note: str = ""


#: Every outcome item of the advocacy battery, by its published column name.
ITEMS: tuple[Item, ...] = (
    Item(
        "petition",
        "Signed the EDF methane petition",
        "binary",
        0,
        1,
        "self-report after a live EDF action page; survey codes 4/5 recoded to 1/0",
    ),
    Item(
        "letter",
        "Wrote a deliverable letter to a representative",
        "binary",
        0,
        1,
        "0/1 code of the free-text letter by GPT-3.5 then human check; the text "
        "itself was de-identified out of the published file, and the column is "
        "zero-filled to all 31,324 rows",
    ),
    Item(
        "pol_candidate",
        "Commits to supporting climate candidates",
        "slider",
        0,
        100,
        'slider starting at 0, with a "Not Applicable / Not Eligible to Vote" '
        "escape that becomes missing (8,993 cases)",
    ),
    Item(
        "pol_campaign",
        "Willing to join a campaign lobbying officials",
        "slider",
        0,
        100,
        "slider starting at 0",
    ),
    Item(
        "newsletter1",
        "Signed up to the 350.org newsletter",
        "binary",
        0,
        1,
        "self-report after the organisation's own embedded signup form",
    ),
    Item(
        "newsletter2",
        "Signed up to Citizens' Climate Lobby",
        "binary",
        0,
        1,
        "self-report after the organisation's own embedded signup form",
    ),
    Item(
        "newsletter",
        "Signed up to either newsletter",
        "binary",
        0,
        1,
        "OR of the two, and zero-filled to all 31,324 rows by that construction",
    ),
    Item(
        "donation",
        "Dollars given to an environmental organisation",
        "dollars",
        0,
        10,
        "allocation of a $10 bonus between self and the organisation, with 100 "
        "participants' choices actually paid out",
    ),
    Item("donation_keep", "Dollars kept", "dollars", 0, 10, "10 minus donation"),
    Item(
        "donation_bin",
        "Gave anything at all",
        "binary",
        0,
        1,
        "donation > 0, and zero-filled to all 31,324 rows",
    ),
    Item(
        "march",
        "Commits to attending a climate demonstration",
        "slider",
        0,
        100,
        "slider starting at 0",
    ),
    Item(
        "conversation",
        "Commits to talking about climate with close others",
        "slider",
        0,
        100,
        "slider starting at 0",
    ),
    Item(
        "flyless",
        "Commits to flying less this year",
        "slider",
        0,
        100,
        'slider starting at 0, with a "Not Applicable" escape that becomes '
        "missing; only 14,069 usable values, the sparsest item in the battery",
    ),
    Item(
        "lessbeef",
        "Commits to eating less red meat",
        "slider",
        0,
        100,
        'slider starting at 0, with a "Not Applicable" escape that becomes missing',
    ),
    Item(
        "video",
        "Shared a climate video on social media",
        "binary",
        0,
        1,
        '"I do not have social media" becomes missing (about 4,900 cases)',
    ),
    Item(
        "bank_raw",
        "Commits to moving money out of a fossil-funding bank",
        "slider",
        0,
        100,
        "slider starting at 0",
    ),
    Item(
        "bank",
        "Same, among respondents whose bank scored badly",
        "slider",
        0,
        100,
        'bank_raw blanked for the 4,109 whose bank scored "great" or "good", for '
        "whom the commitment is not meaningful",
    ),
    Item(
        "bankscore",
        "What score bank.green gave their bank",
        "choice",
        1,
        6,
        "requires the respondent to look up their own real bank on a live site, so "
        "this item is not meaningfully simulable",
    ),
    Item(
        "belief_1",
        "Climate change is a global emergency",
        "slider",
        0,
        100,
        "REVERSE-LABELLED and unreported by the paper; see the module docstring",
    ),
    Item(
        "policy_1",
        "Supports climate mitigative policies",
        "slider",
        0,
        100,
        "REVERSE-LABELLED and unreported by the paper; see the module docstring",
    ),
    Item(
        "Pefficacy",
        "Personal efficacy",
        "slider",
        0,
        100,
        "mediator, slider starting at 0",
    ),
    Item(
        "Cefficacy",
        "Collective efficacy",
        "slider",
        0,
        100,
        "mediator, slider starting at 0",
    ),
)

BY_COLUMN = {item.column: item for item in ITEMS}

#: The ten emotion ratings, all 0-100.
EMOTIONS = (
    "Anger",
    "Sadness",
    "Fear",
    "Guilt",
    "Hope",
    "Pride",
    "Disappointment",
    "Anxiety",
    "Joy",
    "Disgust",
)
POSITIVE_EMOTIONS = ("Hope", "Pride", "Joy")
NEGATIVE_EMOTIONS = (
    "Anger",
    "Sadness",
    "Fear",
    "Guilt",
    "Disappointment",
    "Anxiety",
    "Disgust",
)

#: The four preregistered composites, as the cleaning notebook defines them:
#: the mean of items first rescaled onto 0-1.
COMPOSITES: dict[str, tuple[str, ...]] = {
    "public_awareness": ("newsletter", "video", "marchN", "conversationN"),
    "political_advocacy": ("petition", "letter", "pol_campaignN", "pol_candidateN"),
    "financial_advocacy": ("donationN", "bankN"),
    "lifestyle_changes": ("flylessN", "lessbeefN"),
}

#: ``political_advocacy`` minus ``letter``: the same composite over the three of its
#: four items that a silicon sample can actually produce.
#:
#: Reported alongside the preregistered one, never instead of it.  ``letter`` is a
#: GPT-3.5 judgement of free text that was de-identified out of the published file,
#: so the classifier can be neither re-run nor inspected, and 42% of its zeros are
#: respondents who never reached the page rather than respondents who wrote nothing
#: — a kind of zero a sampled respondent structurally cannot have.  Any rule that
#: fills the column with a near-constant is affinely equivalent to dropping it, so
#: reporting the drop explicitly is the honest version of what the rule does; see
#: :func:`~silicon_sampling.goldwert.score.letter_contribution` for how much of the
#: between-arm signal goes with it.
LETTER_FREE: tuple[str, ...] = ("petition", "pol_campaignN", "pol_candidateN")

#: Column -> divisor that puts it on 0-1, exactly as the notebook does it.
NORMALIZED = {
    "marchN": ("march", 100.0),
    "conversationN": ("conversation", 100.0),
    "pol_campaignN": ("pol_campaign", 100.0),
    "pol_candidateN": ("pol_candidate", 100.0),
    "bankN": ("bank", 100.0),
    "flylessN": ("flyless", 100.0),
    "lessbeefN": ("lessbeef", 100.0),
    "donationN": ("donation", 10.0),
}

#: What a silicon sample is scored on, and each outcome's scale range.  The four
#: preregistered composites, the two behaviours that are the whole reason this
#: study is in the calibration set, and the two efficacy mediators.  Notably
#: absent: ``belief_1``, ``policy_1`` and ``bankscore``, for the reasons above.
SCORED: dict[str, float] = {
    "public_awareness": 1.0,
    "political_advocacy": 1.0,
    "financial_advocacy": 1.0,
    "lifestyle_changes": 1.0,
    "donation": 10.0,
    "newsletter": 1.0,
    "petition": 1.0,
    "march": 100.0,
    "conversation": 100.0,
    "Pefficacy": 100.0,
    "Cefficacy": 100.0,
}

LABELS = {
    "public_awareness": "Public awareness advocacy",
    "political_advocacy": "Political advocacy",
    "financial_advocacy": "Financial advocacy",
    "lifestyle_changes": "Lifestyle change commitments",
    "donation": "Donation ($ of 10 to an environmental organisation)",
    "newsletter": "Newsletter signup (either organisation)",
    "petition": "Petition signature",
    "march": "Commitment to attend a demonstration",
    "conversation": "Commitment to talk about climate",
    "Pefficacy": "Personal efficacy",
    "Cefficacy": "Collective efficacy",
}

#: Items every scored outcome needs, by the slot ids our transcripts use.
REQUIRED_ITEMS = (
    "petition",
    "letter",
    "pol_candidate",
    "pol_campaign",
    "newsletter1",
    "newsletter2",
    "donation",
    "donation_keep",
    "march",
    "conversation",
    "flyless",
    "lessbeef",
    "video",
    "bankscore",
    "bank_raw",
    "Pefficacy",
    "Cefficacy",
)


def _num(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def derive_items(frame: pd.DataFrame) -> pd.DataFrame:
    """Recode the raw items the way the cleaning notebook does.

    Written to be idempotent, so it can run on a silicon sample holding only raw
    slot answers *and* on the published file, which already carries the derived
    columns.  Anything already present is left alone: recoding a recoded column
    would, for instance, turn ``petition`` from 1 into missing.
    """
    data = frame.copy()

    # The survey stored these yes/no items as 4 and 5.
    for column in ("petition", "newsletter1", "newsletter2"):
        raw = _num(data, column)
        if raw.isin([4, 5]).any():
            data[column] = raw.replace({4: 1, 5: 0})

    raw_video = _num(data, "video")
    if raw_video.isin([2, 4]).any():
        data["video"] = raw_video.replace({2: 0}).replace({4: np.nan})

    if "newsletter" not in data.columns:
        data["newsletter"] = (
            (_num(data, "newsletter1") == 1) | (_num(data, "newsletter2") == 1)
        ).astype(int)
    if "donation_bin" not in data.columns:
        data["donation_bin"] = np.where(_num(data, "donation") > 0, 1, 0)
    if "donation_keep" not in data.columns:
        data["donation_keep"] = 10 - _num(data, "donation")
    if "bank" not in data.columns:
        # The commitment to move your money is not meaningful for someone whose
        # bank already scored well, so the study blanks it for them.
        data["bank"] = _num(data, "bank_raw").where(
            ~_num(data, "bankscore").isin([1, 2]), np.nan
        )
    return data


def compute(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the normalised items, the four composites and the emotion means."""
    data = derive_items(frame)
    for name, (source, divisor) in NORMALIZED.items():
        data[name] = _num(data, source) / divisor
    for name, parts in COMPOSITES.items():
        data[name] = sum(_num(data, part) for part in parts) / len(parts)
    data["political_advocacy_no_letter"] = sum(
        _num(data, part) for part in LETTER_FREE
    ) / len(LETTER_FREE)
    data["pos_emo"] = sum(_num(data, e) / 100 for e in POSITIVE_EMOTIONS) / len(
        POSITIVE_EMOTIONS
    )
    data["neg_emo"] = sum(_num(data, e) / 100 for e in NEGATIVE_EMOTIONS) / len(
        NEGATIVE_EMOTIONS
    )
    return data


def verify_against_published(
    published: pd.DataFrame, tolerance: float = 1e-9
) -> pd.DataFrame:
    """Recompute every derived column from the raw items and check the published one.

    The published file carries both the raw items and the authors' derived
    columns, which makes this a closed test: if a divisor or a member item were
    wrong here, the recomputed column would not match theirs, and every effect
    estimate downstream would be quietly off.
    """
    checked = (
        *NORMALIZED,
        *COMPOSITES,
        "pos_emo",
        "neg_emo",
        "newsletter",
        "donation_bin",
        "donation_keep",
        "bank",
    )
    # Recompute from the raw items only. Leaving the authors' own derived columns
    # in place would make :func:`derive_items` pass them straight through and the
    # check would compare each column against itself.
    raw = published.drop(columns=[c for c in checked if c in published.columns])
    recomputed = compute(raw)
    rows = []
    for name in checked:
        if name not in published.columns:
            continue
        theirs = pd.to_numeric(published[name], errors="coerce")
        ours = pd.to_numeric(recomputed[name], errors="coerce")
        both = theirs.notna() & ours.notna()
        difference = (theirs[both] - ours[both]).abs()
        rows.append(
            {
                "column": name,
                "n_compared": int(both.sum()),
                "n_published": int(theirs.notna().sum()),
                "n_missing_mismatch": int((theirs.notna() != ours.notna()).sum()),
                "max_abs_diff": float(difference.max()) if len(difference) else np.nan,
                "matches": bool(len(difference) and difference.max() < tolerance),
            }
        )
    return pd.DataFrame(rows)


def coverage(frame: pd.DataFrame) -> pd.DataFrame:
    """Non-null counts for every item and outcome: the missingness picture in one table."""
    rows = []
    for name in [item.column for item in ITEMS] + list(SCORED) + list(EMOTIONS):
        if name not in frame.columns:
            continue
        values = pd.to_numeric(frame[name], errors="coerce")
        item = BY_COLUMN.get(name)
        rows.append(
            {
                "column": name,
                "n_observed": int(values.notna().sum()),
                "share_observed": round(float(values.notna().mean()), 4),
                "mean": round(float(values.mean()), 4),
                "at_scale_floor": (
                    int((values == 0).sum()) if item and item.kind == "slider" else None
                ),
                "note": item.note if item else "",
            }
        )
    return pd.DataFrame(rows).drop_duplicates(subset="column").reset_index(drop=True)


@dataclass(frozen=True)
class Anchor:
    """One Goldwert item offered as a calibration anchor for a Pfänder outcome."""

    goldwert_column: str
    goldwert_scale: str
    pfander_outcome: str
    pfander_scale: str
    #: ``near_identical``, ``adjacent`` or ``conceptual``.
    closeness: str
    what_it_anchors: str
    wording_gap: str


#: What this study can and cannot calibrate for the Pfänder target.
#:
#: The two ``near_identical`` rows are the reason the package exists.  Everything
#: below them is a conceptual neighbour at best, and six of Pfänder's thirteen
#: outcomes — the whole multidimensional trust battery, single-item trust and
#: distrust, institutional trust, funding perceptions and scientists' role in
#: policy making — have no counterpart in this study at all.  No Goldwert arm is
#: about the trustworthiness of scientists.
PFANDER_ANCHORS: tuple[Anchor, ...] = (
    Anchor(
        "donation",
        "0-10 whole dollars",
        "donation_ams",
        "0-10 whole dollars",
        "near_identical",
        "control-arm level, the shape of the distribution (including the mass at $0 "
        "and at $10), and the size an intervention effect can reach",
        'Goldwert allocates between the respondent and an unnamed "environmental '
        'organization" and adds a group-contingent match — the pool doubles if at '
        "least half of participants give $5 or more. Pfänder names the recipient, the "
        "American Meteorological Society, describes it as a scientific society rather "
        "than an advocacy group, and has no match. Both randomly realise 100 "
        "participants' choices as real money.",
    ),
    Anchor(
        "newsletter",
        "0/1 (either of two organisations)",
        "newsletter_signup",
        "0/1",
        "near_identical",
        "baseline signup rate and the size an intervention effect can reach",
        "Goldwert embeds the organisations' own signup forms in the survey and asks "
        "twice — 350.org and Citizens' Climate Lobby, both advocacy groups — then "
        "takes the OR, which mechanically sits above either single rate. Pfänder links "
        "out to one science-communication newsletter, Katharine Hayhoe's \"Talking "
        'Climate". Use `newsletter1` alone for a single-organisation comparison.',
    ),
    Anchor(
        "petition",
        "0/1",
        "behavior_mean",
        "0-100 mean of 6 intention items",
        "adjacent",
        "how far a real behavioural measure moves compared with stated intentions",
        "Goldwert's is a real signature on a live Environmental Defense Fund action "
        "page; Pfänder's behaviour outcome is six self-reported intentions.",
    ),
    Anchor(
        "conversation",
        "0-100 slider",
        "behavior_mean",
        "0-100 mean of 6 intention items",
        "conceptual",
        "the level and effect size of a talk-about-climate commitment",
        "Pfänder's `individual_talk_1` is one of the six items in `behavior_mean`; "
        "this is the same construct on a slider that starts at 0.",
    ),
    Anchor(
        "march",
        "0-100 slider",
        "behavior_mean",
        "0-100 mean of 6 intention items",
        "conceptual",
        "the level of a costly collective-action commitment",
        "no Pfänder item asks about demonstrations; this bounds the collective-action "
        "end of the behaviour scale.",
    ),
    Anchor(
        "Cefficacy",
        "0-100 slider",
        "concern_mean",
        "0-100 mean of 3 items",
        "conceptual",
        "a mediator level only; not a substitute for the concern battery",
        "efficacy and concern are different constructs. Offered because Pfänder's "
        "concern items have no closer neighbour here.",
    ),
    Anchor(
        "belief_1",
        "0-100 slider, REVERSE-LABELLED",
        "belief_post",
        "0-100",
        "unusable",
        "nothing — listed so that it is not reached for by accident",
        'same scale and adjacent content ("climate change is a global emergency" '
        'against "human activities are causing climate change"), but the item\'s '
        "polarity cannot be established and the paper does not report it. See the "
        "module docstring.",
    ),
    Anchor(
        "policy_1",
        "0-100 slider, REVERSE-LABELLED",
        "policy_general",
        "0-100",
        "unusable",
        "nothing — listed so that it is not reached for by accident",
        "there is no Democrat-Republican gap on this item, which for a fossil-fuel "
        "transition question means the measure, not the country.",
    ),
)

#: Pfänder outcomes with no counterpart anywhere in this study.
PFANDER_UNCOVERED = (
    "trust_multidimensional",
    "trust_post",
    "distrust_post",
    "funding_perceptions",
    "policy_role_mean",
    "inst_trust_mean",
)


def pfander_anchor_table() -> pd.DataFrame:
    """The mapping as a table, one row per anchor."""
    from dataclasses import asdict

    return pd.DataFrame([asdict(anchor) for anchor in PFANDER_ANCHORS])


def anchor_levels(frame: pd.DataFrame, condition_col: str = "condName") -> pd.DataFrame:
    """Per-arm level of each usable anchor, on the observed rows only.

    ``newsletter`` and ``donation_bin`` are zero-filled to all 31,324 rows by their
    construction, so a mean over the whole column understates the signup rate by
    about ten points.  This restricts each anchor to the respondents who reached
    its page, which is the number a silicon sample should be compared against.
    """
    data = compute(frame)
    reached_newsletter = data["newsletter1"].notna() | data["newsletter2"].notna()
    rows = []
    for arm, group in data.groupby(condition_col):
        reached = reached_newsletter.loc[group.index]
        donation = pd.to_numeric(group["donation"], errors="coerce").dropna()
        rows.append(
            {
                "condName": arm,
                "n_assigned": len(group),
                "donation_n": len(donation),
                "donation_mean": round(float(donation.mean()), 4),
                "donation_share_zero": round(float((donation == 0).mean()), 4),
                "donation_share_all_ten": round(float((donation == 10).mean()), 4),
                "newsletter_n_reached": int(reached.sum()),
                "newsletter_rate_reached": round(
                    float(group.loc[reached, "newsletter"].mean()), 4
                ),
                "newsletter_rate_zero_filled": round(
                    float(group["newsletter"].mean()), 4
                ),
                "newsletter1_rate_350org": round(
                    float(pd.to_numeric(group["newsletter1"], errors="coerce").mean()),
                    4,
                ),
                "newsletter2_rate_ccl": round(
                    float(pd.to_numeric(group["newsletter2"], errors="coerce").mean()),
                    4,
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("donation_mean", ascending=False)
