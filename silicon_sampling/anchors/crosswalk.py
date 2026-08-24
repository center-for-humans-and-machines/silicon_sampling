"""Which external item stands in for which Pfänder item, and how honestly.

Pfänder publishes no human data, so an external anchor is the only way its
control-arm *levels* can be checked against anything real.  The danger is that
borrowing a level from a differently worded question is a silent error: it
produces a plausible number that moves the response-distribution and
demographic-baseline metrics in the wrong direction, and unlike a coding bug it
leaves no trace.  So every mapping is written down here as data — the two item
texts side by side, the source's response scale, and a grade — rather than being
implied by a dictionary of numbers somewhere.  A reader can disagree with a grade
without reading any code, which is the point.

## The grades

``verbatim``
    Same stem, same referent, and endpoint labels that mean the same thing; only
    the granularity of the response scale differs.  Nothing in either source
    reaches this bar, which is worth knowing.

``near``
    Small, named differences of stem or referent that plausibly leave the level
    within a few points — "scientists" for "climate scientists", "politicians"
    for "policy makers", an added qualifying clause.

``construct-only``
    Measures the construct, but with a difference that can plausibly move the
    level by more than the break-even anchor error: a unipolar support scale
    against a bipolar one, one item standing in for a multi-item composite,
    partial coverage of a composite's items, a different time frame.

``unusable``
    No defensible conversion exists — most often because the source item is
    categorical in a way that has no ordered latent scale to map onto.

Only ``verbatim`` and ``near`` are offered as anchors by default.  The
``construct-only`` and ``unusable`` rows are kept because a search that found
nothing is a result: without them the report would read as if we had never looked
at ``belief_post``.

## Three sources, and what a grade is *not* about

TISP and CCAM are post-stratified: both publish a weight that reweights the
respondents to population margins, and a level taken from them is an estimate of a
population level.  Goldwert is not.  It is a quota-matched CloudResearch Connect
panel with no weight in the file, and its own authors write that "despite our
sample approximating the United States population on age, race, gender, and
ethnicity, it is not a truly representative sample, and might embed biases
associated with online panel samples", vouching explicitly for
*between-condition* comparisons rather than for levels.  That is precisely the
distinction this package turns on, so it is a grading input in its own right,
separate from wording.

It is worth being clear that a grade here answers one question — *can this item's
level stand in for the Pfänder item's level* — and that this is a different
question from the one Goldwert's own ``PFANDER_ANCHORS`` table answers with
``near_identical``.  That table is right that the donation is the same scale, the
same units and the same real-money mechanism; two rows here nonetheless come out
``construct-only``, because identity of *instrument* is not transfer of *level*.
The disagreement is about which question is being graded, not about the facts.

## The referent shift, and why it is not free

Every ``TRUST_SCI_*`` row is graded ``near`` and not ``verbatim`` for one reason:
TISP asks about *scientists*, Pfänder about *climate scientists*.  That is not a
cosmetic difference in the US, and TISP itself can size it — the same respondents
answered ``TRUST_PEW`` (confidence in scientists) and ``CLIM_TRUST`` (trust in
climate scientists), and the weighted gap is 3.9 points on the converted scale.
:mod:`silicon_sampling.anchors.levels` subtracts it from the trust battery and
from nothing else, because it was measured on a trust item pair and generalises no
further than that.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..pfander.outcomes import OUTCOMES

#: Grades from best to worst.  Order is the API: ``at_least`` slices this list.
GRADES = ("verbatim", "near", "construct-only", "unusable")

#: How a source item reaches Pfänder's scale.  ``likert`` converts from a fixed
#: number of response options, ``native`` means the item is already on that scale
#: and the identity is the honest transform, and ``none`` means no defensible
#: transform exists — an unordered category set, or a slider whose polarity cannot
#: be established — so no number is produced at all.
CONVERSIONS = ("likert", "native", "none")

#: The grade at or above which an anchor is offered without the caller asking.
DEFAULT_MIN_GRADE = "near"


@dataclass(frozen=True)
class Entry:
    """One (Pfänder item, source item) pair, graded.

    ``group`` names the candidate anchor this row belongs to.  It is declared
    rather than inferred because a source can offer both the *components* of a
    composite and a rival single item for the same outcome — TISP does exactly that
    for ``trust_post`` — and grouping by source alone would let the rival item drag
    the battery's grade down with it.
    """

    pfander_outcome: str
    pfander_item: str
    pfander_text: str
    pfander_scale: str
    source: str
    source_item: str
    source_text: str
    source_scale: str
    source_options: int | None
    grade: str
    note: str
    group: str
    conversion: str = "likert"
    missing_codes: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.grade not in GRADES:
            raise ValueError(f"unknown grade {self.grade!r}")
        if self.pfander_outcome not in OUTCOMES:
            raise ValueError(f"not a scored outcome: {self.pfander_outcome!r}")
        if self.conversion not in CONVERSIONS:
            raise ValueError(f"unknown conversion {self.conversion!r}")
        # The two fields have to agree or a measurement silently takes the wrong
        # path: a likert row without an option count cannot be converted, and a
        # native or unconvertible row has no option count to offer.
        if self.conversion == "likert" and not (self.source_options or 0) >= 2:
            raise ValueError(f"{self.source_item}: likert needs >= 2 options")
        if self.conversion != "likert" and self.source_options is not None:
            raise ValueError(f"{self.source_item}: {self.conversion} takes no options")


_SLIDER = "0-100 slider"
_LIKERT5_AGREE = "5-point, endpoints Strongly disagree / Strongly agree"
_LIKERT4_SUPPORT = (
    "4-point fully labelled, Strongly oppose / Somewhat oppose / "
    "Somewhat support / Strongly support"
)
_REFERENT = "referent is scientists in general, not climate scientists"

#: Stem, source stem and grade for the twelve-item Besley trust battery.
_TRUST_BATTERY = (
    (
        "trust_competence_1",
        "How incompetent or competent are most climate scientists?",
        "Very incompetent / Very competent",
        "TRUST_SCI_expert",
        "How expert or inexpert are most scientists?",
        "Very inexpert / Very expert",
        "near",
        f"adjective differs (expert for competent); {_REFERENT}",
    ),
    (
        "trust_competence_2",
        "How unintelligent or intelligent are most climate scientists?",
        "Very unintelligent / Very intelligent",
        "TRUST_SCI_intellig",
        "How intelligent or unintelligent are most scientists?",
        "Very unintelligent / Very intelligent",
        "near",
        f"stem identical apart from adjective order; {_REFERENT}",
    ),
    (
        "trust_competence_3",
        "How unqualified or qualified are most climate scientists?",
        "Very unqualified / Very qualified",
        "TRUST_SCI_qualified",
        (
            "How qualified or unqualified are most scientists when it comes to "
            "conducting high-quality research?"
        ),
        "Very unqualified / Very qualified",
        "near",
        f"source adds 'when it comes to conducting high-quality research'; {_REFERENT}",
    ),
    (
        "trust_integrity_1",
        "How dishonest or honest are most climate scientists?",
        "Very dishonest / Very honest",
        "TRUST_SCI_honest",
        "How honest or dishonest are most scientists?",
        "Very dishonest / Very honest",
        "near",
        f"stem identical apart from adjective order; {_REFERENT}",
    ),
    (
        "trust_integrity_2",
        "How unethical or ethical are most climate scientists?",
        "Very unethical / Very ethical",
        "TRUST_SCI_ethical",
        "How ethical or unethical are most scientists?",
        "Very unethical / Very ethical",
        "near",
        f"stem identical apart from adjective order; {_REFERENT}",
    ),
    (
        "trust_integrity_3",
        "How insincere or sincere are most climate scientists?",
        "Very insincere / Very sincere",
        "TRUST_SCI_sincere",
        "How sincere or insincere are most scientists?",
        "Very insincere / Very sincere",
        "near",
        f"stem identical apart from adjective order; {_REFERENT}",
    ),
    (
        "trust_benevolence_1",
        (
            "How unconcerned or concerned are most climate scientists about "
            "people's wellbeing?"
        ),
        "Very unconcerned / Very concerned",
        "TRUST_SCI_concerned",
        "How concerned or not concerned are most scientists about people's wellbeing?",
        "Not concerned / Very concerned",
        "near",
        f"source's low pole is 'Not concerned' rather than 'Very unconcerned'; {_REFERENT}",
    ),
    (
        "trust_benevolence_2",
        "How uneager or eager are most climate scientists to improve others' lives?",
        "Very uneager / Very eager",
        "TRUST_SCI_improve",
        "How eager or uneager are most scientists to improve others' lives?",
        "Very uneager / Very eager",
        "near",
        f"stem identical apart from adjective order; {_REFERENT}",
    ),
    (
        "trust_benevolence_3",
        (
            "How inconsiderate or considerate are most climate scientists of "
            "others' interests?"
        ),
        "Very inconsiderate / Very considerate",
        "TRUST_SCI_otherint",
        "How considerate or inconsiderate are most scientists of others' interests?",
        "Very inconsiderate / Very considerate",
        "near",
        f"stem identical apart from adjective order; {_REFERENT}",
    ),
    (
        "trust_openness_1",
        "How open, if at all, are most climate scientists to feedback?",
        "Not open at all / Very open",
        "TRUST_SCI_open",
        "How open are most scientists to feedback?",
        "Not open / Very open",
        "near",
        f"stem identical apart from 'if at all'; {_REFERENT}",
    ),
    (
        "trust_openness_2",
        "How unwilling or willing are most climate scientists to be transparent?",
        "Very unwilling / Very willing",
        "TRUST_SCI_trans",
        "How willing or unwilling are most scientists to be transparent?",
        "Very unwilling / Very willing",
        "near",
        f"stem identical apart from adjective order; {_REFERENT}",
    ),
    (
        "trust_openness_3",
        (
            "How much or how little attention do climate scientists pay to other "
            "people's views?"
        ),
        "Very little attention / A great deal of attention",
        "TRUST_SCI_otherviews",
        "How much or little attention do scientists pay to others' views?",
        "Very little attention / Very much attention",
        "near",
        f"stem near-identical; {_REFERENT}",
    ),
)

#: Stem, source stem and grade for the four science-in-policy norm items.
_POLICY_ROLE = (
    (
        "policy_1_1",
        (
            "Climate scientists should work closely with policy makers to integrate "
            "scientific results into policy-making."
        ),
        "NORMPERC_integrate",
        (
            "Scientists should work closely with politicians to integrate scientific "
            "results into policy-making."
        ),
        "'politicians' for 'policy makers'; " + _REFERENT,
    ),
    (
        "policy_2_1",
        "Climate scientists should actively advocate for specific policies.",
        "NORMPERC_advocate",
        "Scientists should actively advocate for specific policies.",
        _REFERENT + " (only difference)",
    ),
    (
        "policy_3_1",
        "Climate scientists should communicate their findings to policy makers.",
        "NORMPERC_communicate",
        "Scientists should communicate their findings to politicians.",
        "'politicians' for 'policy makers'; " + _REFERENT,
    ),
    (
        "policy_4_1",
        "Climate scientists should be more involved in the policy-making process.",
        "NORMPERC_involved",
        "Scientists should be more involved in the policy-making process.",
        _REFERENT + " (only difference)",
    ),
)

#: The five TISP climate-policy items that have a Pfänder counterpart, in order.
_POLICY_SUPPORT = (
    (
        "policy_specific_1_1",
        "Raising taxes on fossil fuels (e.g., gas, oil, coal)",
        "CLIM_POLSUPPORT_fueltax",
        "Raising carbon taxes on gas and fossil fuels or coal",
    ),
    (
        "policy_specific_2_1",
        "Expanding infrastructure for public transportation",
        "CLIM_POLSUPPORT_publictransport",
        "Expanding infrastructure for public transportation",
    ),
    (
        "policy_specific_3_1",
        "Increasing the use of sustainable energy such as wind and solar energy",
        "CLIM_POLSUPPORT_sustenergy",
        "Increasing the use of sustainable energy such as wind and solar energy",
    ),
    (
        "policy_specific_4_1",
        "Protecting forested and land areas",
        "CLIM_POLSUPPORT_protection",
        "Protecting forested and land areas",
    ),
    (
        "policy_specific_5_1",
        "Increasing taxes on carbon-intensive foods (e.g., beef and dairy products)",
        "CLIM_POLSUPPORT_foodtax",
        "Increasing taxes on carbon intense foods (e.g., beef and dairy products)",
    ),
)

_UNIPOLAR_NOTE = (
    "stem matches, but the source scale is unipolar support intensity "
    "(Not at all / Moderately / Very much, plus Not applicable) against Pfänder's "
    "bipolar Strongly oppose - Strongly support slider, and covers 5 of the 7 items"
)


def _trust_entries() -> tuple[Entry, ...]:
    return tuple(
        Entry(
            pfander_outcome="trust_multidimensional",
            pfander_item=item,
            pfander_text=stem,
            pfander_scale=f"{_SLIDER}, endpoints {poles}",
            source="TISP",
            source_item=source_item,
            source_text=source_stem,
            source_scale=f"5-point fully labelled, endpoints {source_poles}",
            source_options=5,
            grade=grade,
            note=note,
            group="TISP TRUST_SCI battery",
        )
        for (
            item,
            stem,
            poles,
            source_item,
            source_stem,
            source_poles,
            grade,
            note,
        ) in _TRUST_BATTERY
    )


def _policy_role_entries() -> tuple[Entry, ...]:
    return tuple(
        Entry(
            pfander_outcome="policy_role_mean",
            pfander_item=item,
            pfander_text=stem,
            pfander_scale=f"{_SLIDER}, endpoints Strongly disagree / Strongly agree",
            source="TISP",
            source_item=source_item,
            source_text=source_stem,
            source_scale=_LIKERT5_AGREE,
            source_options=5,
            grade="near",
            note=note,
            group="TISP NORMPERC battery",
        )
        for item, stem, source_item, source_stem, note in _POLICY_ROLE
    )


def _policy_support_entries() -> tuple[Entry, ...]:
    return tuple(
        Entry(
            pfander_outcome="policy_specific_mean",
            pfander_item=item,
            pfander_text=stem,
            pfander_scale=f"{_SLIDER}, endpoints Strongly oppose / Strongly support",
            source="TISP",
            source_item=source_item,
            source_text=source_stem,
            source_scale=(
                "3-point fully labelled (Not at all / Moderately / Very much) "
                "plus Not applicable, coded 4 and dropped"
            ),
            source_options=3,
            grade="construct-only",
            note=_UNIPOLAR_NOTE,
            group="TISP CLIM_POLSUPPORT items",
            missing_codes=(4.0,),
        )
        for item, stem, source_item, source_stem in _POLICY_SUPPORT
    )


_OTHERS = (
    Entry(
        pfander_outcome="trust_post",
        pfander_item="trust_post_1",
        pfander_text="How much do you trust climate scientists?",
        pfander_scale=f"{_SLIDER}, endpoints not at all / very strongly",
        source="TISP",
        source_item="CLIM_TRUST",
        group="TISP CLIM_TRUST",
        source_text=(
            "To what extent do you trust scientists in your country who work on "
            "climate change?"
        ),
        source_scale="5-point, endpoints Not at all / Very strongly",
        source_options=5,
        grade="near",
        note=(
            "endpoint labels identical and the referent is climate scientists; "
            "stem paraphrased and restricted to scientists in the respondent's country"
        ),
    ),
    Entry(
        pfander_outcome="trust_post",
        pfander_item="trust_post_1",
        pfander_text="How much do you trust climate scientists?",
        pfander_scale=f"{_SLIDER}, endpoints not at all / very strongly",
        source="TISP",
        source_item="TRUST_PEW",
        group="TISP TRUST_PEW",
        source_text=(
            "How much confidence do you have in scientists to act in the best "
            "interests of the public?"
        ),
        source_scale=(
            "5-point, endpoints No confidence at all / A great deal of confidence"
        ),
        source_options=5,
        grade="construct-only",
        note=(
            "confidence to act in the public interest, scientists in general; kept "
            "because its gap from CLIM_TRUST is what sizes the referent shift"
        ),
    ),
    Entry(
        pfander_outcome="concern_mean",
        pfander_item="concern_1_1",
        pfander_text="How concerned are you about climate change?",
        pfander_scale=f"{_SLIDER}, endpoints Not at all / Extremely",
        source="CCAM",
        source_item="worry",
        group="CCAM worry",
        source_text="How worried are you about global warming?",
        source_scale=(
            "4-point fully labelled, Not at all worried / Not very worried / "
            "Somewhat worried / Very worried"
        ),
        source_options=4,
        grade="construct-only",
        note=(
            "worry for concern and global warming for climate change would be near on "
            "their own, but this is one item standing in for a three-item composite "
            "whose other two (seriousness, importance relative to other issues) are "
            "absent; CCAM's own priority item sits 5.6 points below worry, so a "
            "worry-only anchor overstates the composite"
        ),
        missing_codes=(-1.0, 0.0),
    ),
    Entry(
        pfander_outcome="policy_general",
        pfander_item="policy_general_1",
        pfander_text=(
            'How much do you oppose or support the following statement? "The U.S. '
            'government should do more to reduce global warming"'
        ),
        pfander_scale=f"{_SLIDER}, endpoints Strongly oppose / Strongly support",
        source="CCAM",
        source_item="priority",
        group="CCAM priority",
        source_text=(
            "[Global warming] Do you think each of these issues should be a low, "
            "medium, high, or very high priority for the president and Congress?"
        ),
        source_scale="4-point fully labelled, Low / Medium / High / Very high",
        source_options=4,
        grade="construct-only",
        note=(
            "priority for federal action, not support for a statement about it; "
            "CCAM's transition_economy (a specific 2050 clean-energy target, 58.3) "
            "is the nearest support-scaled alternative and is no closer in construct"
        ),
        missing_codes=(-1.0, 0.0),
    ),
    Entry(
        pfander_outcome="behavior_mean",
        pfander_item="individual_talk_1",
        pfander_text=(
            "Talk to friends and family about the importance of climate change "
            "(how likely in the next twelve months)"
        ),
        pfander_scale=f"{_SLIDER}, endpoints Not likely at all / Extremely likely",
        source="CCAM",
        source_item="discuss_GW",
        group="CCAM discuss_GW",
        source_text="How often do you discuss global warming with your family and friends?",
        source_scale="4-point fully labelled, Never / Rarely / Occasionally / Often",
        source_options=4,
        grade="construct-only",
        note=(
            "reported frequency of past discussion against stated likelihood over the "
            "next twelve months, and one item standing in for a six-item composite "
            "whose other five (diet, transport, solar, flying, donating) have no "
            "counterpart in either source"
        ),
        missing_codes=(-1.0, 0.0),
    ),
    Entry(
        pfander_outcome="policy_specific_mean",
        pfander_item="policy_specific_1_1",
        pfander_text="Raising taxes on fossil fuels (e.g., gas, oil, coal)",
        pfander_scale=f"{_SLIDER}, endpoints Strongly oppose / Strongly support",
        source="CCAM",
        source_item="reduce_tax",
        group="CCAM reduce_tax",
        source_text=(
            "Require fossil fuel companies to pay a carbon tax and use the money to "
            "reduce other taxes (such as income tax) by an equal amount."
        ),
        source_scale=_LIKERT4_SUPPORT,
        source_options=4,
        grade="construct-only",
        note=(
            "revenue-neutral carbon tax on companies, not a tax on fuels; kept "
            "because it is the cross-source check on the TISP fuel-tax anchor"
        ),
        missing_codes=(-1.0, 0.0),
    ),
    Entry(
        pfander_outcome="policy_specific_mean",
        pfander_item="policy_specific_3_1",
        pfander_text=(
            "Increasing the use of sustainable energy such as wind and solar energy"
        ),
        pfander_scale=f"{_SLIDER}, endpoints Strongly oppose / Strongly support",
        source="CCAM",
        source_item="generate_renewable",
        group="CCAM generate_renewable",
        source_text=(
            "Generate renewable energy (solar and wind) on public land in the U.S."
        ),
        source_scale=_LIKERT4_SUPPORT,
        source_options=4,
        grade="construct-only",
        note=(
            "restricted to public land; kept because it is the cross-source check on "
            "the TISP sustainable-energy anchor"
        ),
        missing_codes=(-1.0, 0.0),
    ),
    Entry(
        pfander_outcome="belief_post",
        pfander_item="belief_post_1",
        pfander_text=(
            'How accurate do you think this statement is? "Human activities are '
            'causing climate change."'
        ),
        pfander_scale=f"{_SLIDER}, endpoints not at all accurate / extremely accurate",
        source="CCAM",
        source_item="cause_recoded",
        group="CCAM cause_recoded",
        source_text="Assuming global warming is happening, do you think it is...",
        source_scale=(
            "unordered categories: mostly human activities / mostly natural changes / "
            "other / global warming is not happening / don't know"
        ),
        source_options=None,
        conversion="none",
        grade="unusable",
        note=(
            "a cause attribution and a conditional one at that, not a rated accuracy; "
            "scoring the categories 0/50/100 would invent the anchor rather than "
            "measure it, and CCAM's happening item is a yes/no proportion, which is "
            "not a slider mean either"
        ),
    ),
    Entry(
        pfander_outcome="funding_perceptions",
        pfander_item="funding_5 reversed (100 - funding_5)",
        pfander_text=(
            "Do you think the federal government is spending too much, too little or "
            "about the right amount of money on climate change research?"
        ),
        pfander_scale=(
            f"{_SLIDER}, 0 = far too little, 50 = about the right amount, "
            "100 = far too much; the outcome is 100 - funding_5, so high means "
            "wants more funding, which is the orientation the source item already has"
        ),
        source="CCAM",
        source_item="fund_research",
        group="CCAM fund_research",
        source_text=(
            "Fund more research into renewable energy sources, such as solar and wind "
            "power."
        ),
        source_scale=_LIKERT4_SUPPORT,
        source_options=4,
        grade="unusable",
        note=(
            "support for funding renewables, not a judgement of whether current "
            "federal climate-research spending is too much or too little; the Pfänder "
            "item is centred on an adequacy midpoint that the source scale has no "
            "point for.  Orientation is already that of the outcome (high means "
            "wants more funding), so the figure needs no reversal - it is the "
            "construct that fails, not the coding"
        ),
        missing_codes=(-1.0, 0.0),
    ),
)


_MATCH_NOTE = (
    "same scale, same units and the same real-money mechanism (both studies pay "
    "out 100 randomly chosen participants), which is as close as any source item "
    "in this crosswalk gets. Downgraded anyway on three counts. Goldwert adds a "
    "group-contingent match - the pool doubles if at least half of participants "
    "give $5 or more - and the control arm's distribution shows exactly what that "
    "buys: 29.6% of respondents give precisely $5, against 1.5-3.6% at each of "
    "$1, $2, $3, $4, $6, $7, $8 and $9. Pfander has no match, so that mode cannot "
    "exist there, and the mass it holds is worth roughly half a dollar of the mean "
    "on an order-of-magnitude reckoning - about the whole error budget. The "
    "recipient differs in kind: an unnamed environmental advocacy organisation "
    "against the named American Meteorological Society, framed as a scientific "
    "society, which in the US moves who is willing to give and in the opposite "
    "direction. And the sample carries no weight."
)

_GOLDWERT = (
    Entry(
        pfander_outcome="donation_ams",
        pfander_item="donation",
        pfander_text=(
            "Allocation of a $10 bonus to the American Meteorological Society, a "
            "non-profit, non-partisan society of 12,000 scientists that supports "
            "climate change research"
        ),
        pfander_scale="0-10 whole dollars; 100 participants' choices paid out",
        source="Goldwert",
        source_item="donation",
        source_text=(
            "Allocation of a $10 bonus between the respondent and an environmental "
            "organization, with the pool doubled if at least half of participants "
            "give $5 or more"
        ),
        source_scale="0-10 whole dollars; 100 participants' choices paid out",
        source_options=None,
        conversion="native",
        grade="construct-only",
        note=_MATCH_NOTE,
        group="Goldwert donation",
    ),
    Entry(
        pfander_outcome="newsletter_signup",
        pfander_item="newsletter",
        pfander_text=(
            'Did you subscribe to the "Talking Climate" newsletter on the previous '
            "page? (an outbound link to climate scientist Katharine Hayhoe's "
            "newsletter, opened in a new tab)"
        ),
        pfander_scale="0/1",
        source="Goldwert",
        source_item="newsletter1",
        source_text=(
            "Signed up to the 350.org newsletter through the organisation's own "
            "signup form, embedded in the survey page"
        ),
        source_scale="0/1",
        source_options=None,
        conversion="native",
        grade="construct-only",
        note=(
            "the same binary act on the same scale, but the friction and the "
            "organisation both differ in ways that routinely halve or double a "
            "signup rate: an advocacy group's form embedded in the page against an "
            "outbound link to a scientist's newsletter that the respondent must "
            "open in a new tab and come back from. Goldwert's own two embedded "
            "advocacy forms already differ by 2.4 points (350.org 0.243, Citizens' "
            "Climate Lobby 0.218), which is a floor on the item-level variability, "
            "and a link-out is a far larger change than one organisation for "
            "another. `newsletter1` is used rather than the `newsletter` column, "
            "which is the OR of the two forms and so mechanically exceeds either. "
            "Note also that `tier1.calibrate` applies `levels` to continuous "
            "outcomes only, so a level for this outcome cannot reach a submission "
            "through the validated path even if it were graded higher."
        ),
        group="Goldwert newsletter1",
    ),
    Entry(
        pfander_outcome="behavior_mean",
        pfander_item="individual_talk_1",
        pfander_text=(
            "Talk to friends and family about the importance of climate change "
            "(how likely in the next twelve months)"
        ),
        pfander_scale=f"{_SLIDER}, endpoints Not likely at all / Extremely likely",
        source="Goldwert",
        source_item="conversation",
        source_text=("Commitment to talking about climate change with close others"),
        source_scale="0-100 slider whose handle starts at 0",
        source_options=None,
        conversion="native",
        grade="construct-only",
        note=(
            "closest wording match anywhere for one of the six behaviour items, and "
            "already on a 0-100 slider, but the handle starts at 0 so a respondent "
            "who never touched it is recorded as a hard 0 rather than as missing - "
            "the level is a mixture of intentions and non-responses in unknown "
            "proportion. Still one item of six, and the sample carries no weight. "
            "Kept because it is a third independent estimate of a behaviour-"
            "intention level, against CCAM's discuss_GW"
        ),
        group="Goldwert conversation",
    ),
    Entry(
        pfander_outcome="behavior_mean",
        pfander_item="behavior_mean (no counterpart item)",
        pfander_text=(
            "Mean of six stated likelihoods over the next twelve months: eat less "
            "meat, use transport alternatives, install solar, fly less, talk to "
            "others, donate to an environmental NGO"
        ),
        pfander_scale=f"{_SLIDER} mean of six items",
        source="Goldwert",
        source_item="petition",
        source_text=(
            "Signed the Environmental Defense Fund methane petition on the "
            "organisation's own live action page"
        ),
        source_scale="0/1",
        source_options=None,
        conversion="none",
        grade="unusable",
        note=(
            "a completed real action against a mean of stated intentions. The "
            "obstacle is the same one that rules out CCAM's happening item for "
            "belief_post: a proportion is not the mean of a 0-100 intention scale, "
            "and there is no defensible transform between them. Recorded because "
            "Goldwert's own anchor table offers it, graded adjacent, for the "
            "different purpose of bounding how far a real behaviour moves"
        ),
        group="Goldwert petition",
    ),
    Entry(
        pfander_outcome="belief_post",
        pfander_item="belief_post_1",
        pfander_text=(
            'How accurate do you think this statement is? "Human activities are '
            'causing climate change."'
        ),
        pfander_scale=f"{_SLIDER}, endpoints not at all accurate / extremely accurate",
        source="Goldwert",
        source_item="belief_1",
        source_text="Belief that climate change is happening and human-caused",
        source_scale=(
            "0-100 slider, reverse-labelled (Very much so at the left, Not at all "
            "at the right), handle starting at 0"
        ),
        source_options=None,
        conversion="none",
        grade="unusable",
        note=(
            "the closest belief item in any of the three sources and it still "
            "cannot be used: its polarity is not established. The labels say high "
            "means less belief, the party gap says the opposite (Democrats 63.3, "
            "Republicans 57.5) and the display-position slope agrees with the "
            "labels, rising 48.7 to 57.2 where every other outcome falls. The "
            "authors report neither this item nor policy_1. An anchor whose sign is "
            "unknown is worse than none, so no number is produced"
        ),
        group="Goldwert belief_1",
    ),
    Entry(
        pfander_outcome="policy_general",
        pfander_item="policy_general_1",
        pfander_text=(
            'How much do you oppose or support the following statement? "The U.S. '
            'government should do more to reduce global warming"'
        ),
        pfander_scale=f"{_SLIDER}, endpoints Strongly oppose / Strongly support",
        source="Goldwert",
        source_item="policy_1",
        source_text="Support for a transition away from fossil fuels",
        source_scale=(
            "0-100 slider, reverse-labelled like belief_1, handle starting at 0"
        ),
        source_options=None,
        conversion="none",
        grade="unusable",
        note=(
            "same unresolved polarity as belief_1, and the evidence is worse here: "
            "there is no party gap at all (Democrats 56.6, Republicans 57.5) on one "
            "of the most polarised questions in US politics, which says the item is "
            "not measuring what its name says in a single consistent direction"
        ),
        group="Goldwert policy_1",
    ),
)

#: Every (Pfänder item, source item) pair considered, graded.
CROSSWALK: tuple[Entry, ...] = (
    _trust_entries()
    + _OTHERS[:2]
    + _policy_role_entries()
    + _policy_support_entries()
    + _OTHERS[2:]
    + _GOLDWERT
)

#: Outcomes for which neither source carries a candidate item at all, and why.
UNMATCHED: dict[str, str] = {
    "distrust_post": (
        "no distrust item in either source; distrust is not 100 minus trust, which is "
        "the reason Pfänder measures both"
    ),
    "inst_trust_mean": (
        "no source item asks about the EPA, NASA, NOAA, universities or the federal "
        "government; TISP's CLIM_GOV battery is about government conduct on climate, "
        "not trust in named institutions"
    ),
}


def grade_rank(grade: str) -> int:
    return GRADES.index(grade)


def at_least(grade: str, entries: tuple[Entry, ...] = CROSSWALK) -> tuple[Entry, ...]:
    """Entries graded ``grade`` or better."""
    if grade not in GRADES:
        raise ValueError(f"unknown grade {grade!r}")
    limit = grade_rank(grade)
    return tuple(entry for entry in entries if grade_rank(entry.grade) <= limit)


def groups(
    entries: tuple[Entry, ...] = CROSSWALK,
) -> dict[tuple[str, str], tuple[Entry, ...]]:
    """Entries bundled by (Pfänder outcome, group), the unit an anchor is built from.

    A composite outcome is anchored by a *set* of source items or not at all, and
    two sources may both offer a set for the same outcome, so the group — not the
    row — is what :mod:`silicon_sampling.anchors.levels` chooses between.
    """
    out: dict[tuple[str, str], list[Entry]] = {}
    for entry in entries:
        out.setdefault((entry.pfander_outcome, entry.group), []).append(entry)
    return {key: tuple(value) for key, value in out.items()}


def group_grade(entries: tuple[Entry, ...]) -> str:
    """A group is only as good as its worst item."""
    return max((entry.grade for entry in entries), key=grade_rank)


def to_frame(entries: tuple[Entry, ...] = CROSSWALK):
    """The crosswalk as a DataFrame, for the report table and for eyeballing."""
    import pandas as pd

    return pd.DataFrame([entry.__dict__ for entry in entries])
