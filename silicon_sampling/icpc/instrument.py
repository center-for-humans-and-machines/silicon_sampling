"""The ICPC instrument, assembled per respondent.

The ``.qsf`` flow is unusually legible for a twelve-arm tournament, and every
branch in it matters here:

**The control arm reads its filler text before the shared definition.** The
Dickens excerpt sits in a ``cond == 1`` branch placed *above* the block that
defines "climate change" for everybody; the eleven treated arms get the
definition first and their stimulus after it.  A missingness analysis of the raw
data reached the same conclusion independently, which is why the two agree.

**Three outcome blocks are shuffled as blocks.** Belief, policy support and the
sharing item are drawn by a subset-of-three randomiser, so all six orders occur
and the ``FL_17_DO`` column records which one a respondent got.  The effortful
task always follows them.

**The control arm answers extra measures at the end**, its two blocks also in
random order (``FL_32_DO``), and inside the second of them Qualtrics draws one of
nine wordings of the same willingness item.  Those extras are covariates for the
original paper; here they matter because they are the reason a control transcript
is longer than a treated one.

Two arms are named differently by the study's two publications: Doell calls them
``Letter2Future`` and ``Identity-Social-Norms-Intervention`` where Vlasceanu calls
them ``LetterFutureGen`` and ``WorkTogetherNorm``.  Doell's spellings are the keys
throughout this package, because Doell's export is the file everything is scored
against; the aliases are carried on the arm so a join against the cleaned extract
does not need a lookup table written somewhere else.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..survey.elements import Block as TranscriptBlock
from ..survey.elements import Text
from ..vlasceanu import content_shared as shared
from ..vlasceanu.country import UNITED_STATES, CountryProfile
from .convert import Converted, convert_qsf_block, convert_screens, loaded


@dataclass(frozen=True)
class Arm:
    """One of the twelve experimental arms."""

    #: The ``cond`` code in the published data, 1-12.
    code: int
    #: Doell's ``condName``: the key this package uses everywhere.
    key: str
    #: Vlasceanu's ``condName``, where the two papers disagree.
    alias: str
    #: Descriptive title, from the paper.  Never shown to a respondent.
    title: str
    #: File-name stem for the rendered template.
    slug: str
    #: ``.qsf`` block holding the stimulus, or ``None`` for the control filler.
    block: str | None = None
    #: Extra ``.qsf`` blocks this arm alone sees, in flow order.
    extra_blocks: tuple[str, ...] = field(default_factory=tuple)


ARMS: tuple[Arm, ...] = (
    Arm(
        1,
        "Control",
        "Control",
        "Control (Dickens excerpt)",
        "control",
        "1. Control Distracter",
    ),
    Arm(
        2,
        "Identity-Social-Norms-Intervention",
        "WorkTogetherNorm",
        "Working together / social norms flyer",
        "work_together_norm",
        "2. Identity-Social-Norms-Intervention",
    ),
    Arm(
        3,
        "NegativeEmotions",
        "NegativeEmotions",
        "Writing about negative emotions",
        "negative_emotions",
        "3. Negative-Emotion-Intervention",
    ),
    Arm(
        4,
        "SciConsens",
        "SciConsens",
        "Scientific consensus",
        "sci_consensus",
        "4. Scientific Consensus Intervention",
    ),
    Arm(
        5,
        "CollectAction",
        "CollectAction",
        "Collective action / dynamic norms of efficacy",
        "collective_action",
        "5. Collective Action Intervention_New",
    ),
    Arm(
        6,
        "SystemJust",
        "SystemJust",
        "System justification",
        "system_justification",
        "6. System Justification Intervention",
    ),
    Arm(
        7,
        "PsychDistance",
        "PsychDistance",
        "Decreasing psychological distance",
        "psych_distance",
        "7. Decreasing Psychological Distance Intervention",
    ),
    Arm(
        8,
        "PluralIgnorance",
        "PluralIgnorance",
        "Correcting pluralistic ignorance",
        "plural_ignorance",
        "8. Correcting Pluralistic Ignorance Intervention",
    ),
    Arm(
        9,
        "Letter2Future",
        "LetterFutureGen",
        "Letter to future generations",
        "letter_future_generations",
        "9. A Letter to Future GenerationsV2",
    ),
    Arm(
        10,
        "DynamicNorm",
        "DynamicNorm",
        "Dynamic social norms",
        "dynamic_norms",
        "10. Dynamic Social Norms",
    ),
    Arm(
        11,
        "FutureSelfCont",
        "FutureSelfCont",
        "Future self-continuity",
        "future_self_continuity",
        "11. Future Self-Continuity Intervention",
    ),
    Arm(
        12,
        "BindingMoral",
        "BindingMoral",
        "Binding moral foundations",
        "binding_moral",
        "12. A Binding Moral Foundations Intervention_v1Globe",
    ),
)

BY_KEY = {arm.key: arm for arm in ARMS}
BY_CODE = {arm.code: arm for arm in ARMS}
BY_ALIAS = {arm.alias: arm for arm in ARMS}

#: Arm keys, in ``cond`` order.  The control arm is first because it is the
#: reference every treatment effect is taken against.
CONDITIONS: tuple[str, ...] = tuple(arm.key for arm in ARMS)
CONTROL = "Control"

#: The three outcome blocks Qualtrics shuffles as a set.
DV_BLOCKS: tuple[str, ...] = ("belief", "policy", "sharing")

#: The two control-only extra blocks, also shuffled.
CONTROL_EXTRA_BLOCKS: tuple[str, ...] = ("control_ivs", "terms_probing")

#: ``.qsf`` blocks the control-only extras come from, for the modality audit.
CONTROL_EXTRA_QSF = (
    "1. Control Condition IVs",
    "1. Control Condition IV - terms probing",
)


def dv_order(rng: random.Random) -> list[str]:
    """One respondent's order over the three outcome blocks."""
    order = list(DV_BLOCKS)
    rng.shuffle(order)
    return order


def extras_order(rng: random.Random) -> list[str]:
    """One control respondent's order over the two extra-measure blocks."""
    order = list(CONTROL_EXTRA_BLOCKS)
    rng.shuffle(order)
    return order


# --------------------------------------------------------------------------- #
# panel record
# --------------------------------------------------------------------------- #

#: The demographics, as an echo id and the label the panel record prints.
#:
#: This instrument asks its demographics **last**, after every outcome — unlike
#: the Pfänder and Voelkel instruments, which ask them before the stimulus.  A
#: synthetic respondent walking the instrument in order therefore reaches the
#: belief battery knowing nothing whatever about itself, and every profile in an
#: arm becomes exchangeable: the sample can reproduce a mean but cannot reproduce
#: a subgroup, and prefilling the demographics buys nothing because they are read
#: only after the answers that were supposed to depend on them.
#:
#: The fix is the standard one and it is stated rather than smuggled: a panel
#: record, printed above the consent form, carrying exactly the fields the panel
#: supplier knew before the session began.  It is **not part of the instrument**,
#: it is marked as such on the page, and ``assemble(..., panel_header=False)``
#: renders the instrument without it for anyone who wants the unmodified thing.
PANEL_FIELDS: tuple[tuple[str, str], ...] = (
    ("panel_gender", "Gender"),
    ("panel_age", "Age"),
    ("panel_education", "Years of formal education completed"),
    ("panel_income", "Total yearly family/household income"),
    ("panel_ses", "Position on the social ladder, 1 (worst off) to 10 (best off)"),
    (
        "panel_politics_social",
        "Political orientation on social issues, 0 (extremely liberal/left-wing)"
        " to 100 (extremely conservative/right-wing)",
    ),
    (
        "panel_politics_economic",
        "Political orientation on economic issues, same 0-100 scale",
    ),
)


def panel_block() -> TranscriptBlock:
    """The panel-supplied record, printed above the consent form."""
    lines = [f"{label}: <<={field}>>" for field, label in PANEL_FIELDS]
    return TranscriptBlock(
        key="panel_record",
        title="Panel record",
        note=(
            "supplied by the survey panel before the session; not a screen the "
            "participant filled in here"
        ),
        elements=[
            Text("PARTICIPANT PANEL RECORD", style="head"),
            Text("\n".join(lines)),
        ],
    )


def _shared_block(block, page_break: bool = True) -> tuple[TranscriptBlock, dict]:
    converted = convert_screens(block, page_break=page_break)
    return (
        TranscriptBlock(
            key=block.key,
            title=block.title,
            elements=converted.elements,
            note=block.note,
        ),
        converted.data_columns,
    )


def _qsf_block(description: str) -> tuple[TranscriptBlock, dict]:
    converted = convert_qsf_block(description)
    return (
        TranscriptBlock(
            key=_key(description), title=description, elements=converted.elements
        ),
        converted.data_columns,
    )


def _key(description: str) -> str:
    return "".join(
        character if character.isalnum() else "_" for character in description.lower()
    ).strip("_")


def assemble(
    arm: Arm | str,
    *,
    battery: list[str] | None = None,
    extras: list[str] | None = None,
    probe_index: int = 0,
    country: CountryProfile = UNITED_STATES,
    panel_header: bool = True,
) -> Converted:
    """The full element sequence one respondent walks through, plus its columns.

    ``battery`` and ``extras`` are the respondent's own randomised block orders;
    ``probe_index`` picks which of the nine terms-probing wordings the control
    respondent was shown.
    """
    if isinstance(arm, str):
        arm = BY_KEY[arm]
    battery = list(battery) if battery is not None else list(DV_BLOCKS)
    extras = list(extras) if extras is not None else list(CONTROL_EXTRA_BLOCKS)

    elements: list = []
    columns: dict[str, str] = {}

    def add(pair: tuple[TranscriptBlock, dict]) -> None:
        block, mapped = pair
        elements.append(block)
        columns.update(mapped)

    if panel_header:
        elements.append(panel_block())
    add(_shared_block(shared.CONSENT, page_break=False))
    # The control arm reads its filler text *before* the shared definition.
    if arm.code == 1 and arm.block:
        add(_qsf_block(arm.block))
    add(_shared_block(shared.INTRO))
    if arm.code != 1 and arm.block:
        add(_qsf_block(arm.block))
    for extra in arm.extra_blocks:
        add(_qsf_block(extra))

    dv_sources = {
        "belief": lambda: _shared_block(shared.BELIEF),
        "policy": lambda: _shared_block(shared.POLICY),
        "sharing": lambda: _shared_block(shared.sharing_block(arm.code)),
    }
    for name in battery:
        add(dv_sources[name]())

    add(_shared_block(shared.WEPT))

    if arm.code == 1:
        extra_sources = {
            "control_ivs": lambda: _shared_block(shared.CONTROL_EXTRA_IVS),
            "terms_probing": lambda: _shared_block(
                shared.terms_probing_block(probe_index)
            ),
        }
        for name in extras:
            add(extra_sources[name]())

    add(_shared_block(shared.demographics_block(country)))
    add(_shared_block(shared.DEBRIEF))
    return Converted(elements=elements, data_columns=columns)


def elements_for(arm: Arm | str, **kwargs) -> list:
    """Just the elements, for callers that do not need the column map."""
    return assemble(arm, **kwargs).elements


def data_columns() -> dict[str, str]:
    """Slot id -> published data column, across every arm."""
    columns: dict[str, str] = {}
    for arm in ARMS:
        columns.update(assemble(arm).data_columns)
    return columns


def header(profile_id: str, arm: Arm | str) -> str:
    """The transcript preamble, dated to the study's own fielding window."""
    if isinstance(arm, str):
        arm = BY_KEY[arm]
    return "\n".join(
        [
            "=" * 78,
            " INTERNATIONAL CLIMATE PSYCHOLOGY COLLABORATION",
            " A 63-country tournament of interventions on climate belief, policy support,",
            " information sharing and effortful pro-environmental behaviour.",
            " Response transcripts, one file per participant.",
            "-" * 78,
            f" File           : responses/{profile_id}.txt",
            f" Participant ID : {profile_id}",
            f" Condition      : {arm.code:02d}",
            " Instrument     : United States quota sample, fielded July-October 2022",
            " Note           : Verbatim record of one session, screens in the order they were",
            '                  displayed. Lines beginning "Response:" hold what the participant',
            "                  entered.",
            "=" * 78,
        ]
    )


# --------------------------------------------------------------------------- #
# modality audit
# --------------------------------------------------------------------------- #

#: Tags whose presence in a stimulus means a text transcript cannot stand in for
#: it.  Static images can be described; the rest cannot.
BLOCKING_MEDIA = ("video", "audio", "iframe", "script")


def modality_rows() -> list[dict]:
    """One row per arm: what its stimulus is made of, and whether text can hold it.

    The decision rule is the one the Voelkel audit used, with the one difference
    that decides this study: a *static image* is kept, because a picture can be
    described in words and the description put on the screen where the picture
    was.  Video, audio, an iframe or a script cannot be, and an arm carrying any
    of them would have to be dropped.
    """
    state = loaded()
    rows = []
    for arm in ARMS:
        blocks = [
            state.by_description[name]
            for name in ([arm.block] if arm.block else []) + list(arm.extra_blocks)
        ]
        if arm.code == 1:
            blocks += [state.by_description[name] for name in CONTROL_EXTRA_QSF]
        media = {"video": 0, "audio": 0, "image": 0, "iframe": 0, "script": 0}
        chars = 0
        screens = 0
        for block in blocks:
            for name, count in state.survey.block_media(block).items():
                media[name] += count
            chars += len(state.survey.block_text(block))
            screens += 1 + sum(
                1
                for entry in state.block_elements[block.bid]
                if entry.get("Type") == "Page Break"
            )
        blocking = [name for name in BLOCKING_MEDIA if media[name]]
        if blocking:
            decision, reason = "drop", "stimulus uses " + ", ".join(blocking)
        elif media["image"]:
            decision = "keep"
            reason = (
                f"text plus {media['image']} static image(s), each described in words"
            )
        else:
            decision, reason = "keep", "pure text, non-interactive"
        rows.append(
            {
                "cond": arm.code,
                "condition": arm.key,
                "alias": arm.alias,
                "decision": decision,
                "reason": reason,
                "n_blocks": len(blocks),
                "n_screens": screens,
                "chars": chars,
                **media,
                "blocks": "; ".join(block.description for block in blocks),
            }
        )
    return rows
