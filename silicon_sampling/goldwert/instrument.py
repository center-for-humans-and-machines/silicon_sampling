"""The climate-advocacy megastudy instrument, reassembled from a scattered export.

Nothing about this study's materials is in one place, and the reassembly is the
substance of this module.

**The eighteen arms are eighteen separate files.** The authors published one
Qualtrics export per arm rather than one survey, so the arm text comes from
whichever of the eighteen files holds it, and the arm's *page structure* comes
from that file's own ``Page Break`` elements.

**The outcome battery survives only as debris.** Two of the eighteen exports are
exports of the whole master survey with everything but their own arm swept into a
single "Trash / Unused Questions" block: 630 questions, flattened, no page
breaks, no flow, no block names. That block is the only surviving copy of the
nine outcome pages, the mediators and the demographics. :data:`BATTERY` names the
questions of each outcome block by hand, because there is nothing left to derive
them from.

**The order of those nine blocks is recoverable, and it matters.** ``DV_order``
in the published file is a nine-way permutation per respondent, and the study's
own analysis shows five-percentage-point swings between first and last position.
The *canonical* order is not lost either: the published file's column order is
the Qualtrics export order, which is builder order, and it puts belief and policy
first and the commitment block last — which is how :data:`DV_BLOCK_ORDER` was
fixed. The video-sharing outcome sits after all nine, then the attention check,
the efficacy mediators, the emotions and the demographics, in that order, because
that is the order those columns appear in.

**Eleven of eighteen arms survive the modality audit.** Five are built around a
video and two around a screenshot of a newspaper article, and neither can be put
into a text transcript without either inventing content or transcribing media
that nothing here has looked at. :data:`ARMS` carries the call and the reason for
every arm; :func:`modality_audit` counts the media behind it.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

from ..survey.elements import Block as TranscriptBlock
from ..survey.elements import Text
from ..survey.slots import Slot
from ..voelkel.qsf import Survey, load_survey
from . import convert as cv
from .paths import MASTER_QSF, arm_qsf

#: The trash block of either master export: the whole survey minus one arm.
TRASH_BLOCK = "BL_9BouSHFCKXQhiGa"


@dataclass(frozen=True)
class Arm:
    """One experimental arm, and whether a text transcript can carry it.

    ``modality`` is one of ``pure_text``, ``text_plus_image``, ``image_of_text``,
    ``video`` or ``other``.  ``usable`` is the audit's call, and ``reason`` says
    why in the terms the audit uses — not a restatement of ``modality``.
    """

    cond: int
    name: str
    qsf: str
    slug: str
    modality: str
    usable: bool
    reason: str


#: Every arm, in the order ``cond`` numbers them.  ``name`` is ``condName`` in the
#: published file, so an arm needs no translation to be joined to the responses.
ARMS: tuple[Arm, ...] = (
    Arm(
        0,
        "Control",
        "Neutral_Control_Condition",
        "control",
        "video",
        True,
        "the only content is a five-minute knot-tying video: semantically null, so "
        "a content-free control screen is a faithful rendering rather than a gap",
    ),
    Arm(
        1,
        "ClimatePolicyLiteracy",
        "Climate_Policy_Literacy",
        "climate_policy_literacy",
        "video",
        False,
        "the stimulus is one hosted video; the three comprehension items ask what "
        "the video said, and nothing else on the page carries its content",
    ),
    Arm(
        2,
        "MispCorrectionRisks",
        "Misperception_Correction_Risks",
        "misperception_correction_risks",
        "text_plus_image",
        True,
        "six correction pages in random order plus a writing prompt; the correction "
        "prose is piped in from the survey flow's embedded data, and so is one of six "
        "charts, selected on the writing page by the respondent's own answer",
    ),
    Arm(
        3,
        "CoBenefits",
        "Co-Benefits",
        "co_benefits",
        "image_of_text",
        False,
        "the whole stimulus is one 2282x7768px infographic styled as a social-media "
        "feed; the 73 words of surrounding text only say to read it",
    ),
    Arm(
        4,
        "GlobalHealthThreat",
        "Global_Health_Threat",
        "global_health_threat",
        "video",
        False,
        "substantial prose either side, but the Lancet video sits in the middle and "
        "the true/false matrix explicitly grades what it said",
    ),
    Arm(
        5,
        "GuiltCollResponsibility",
        "Guilt-Based_Collective_Responsibility",
        "guilt_collective_responsibility",
        "image_of_text",
        False,
        "the evidence is three screenshots of a New York Times article; the "
        "comprehension items quote figures that appear only inside them",
    ),
    Arm(
        6,
        "SystemJustification",
        "System_Justification",
        "system_justification",
        "text_plus_image",
        True,
        "twelve landscape photographs, each one captioned in the prose beside it "
        '("Chicago, Illinois", "Grand Canyon, Arizona"); the argument is the prose',
    ),
    Arm(
        7,
        "EcologicalDisruptions",
        "Connecting_to_Ecological_Disruptions",
        "ecological_disruptions",
        "text_plus_image",
        True,
        "long article excerpts plus four images, one of which is a temperature-anomaly "
        "graph the respondent reasons from — that one item is unanswerable as text, "
        "but its answer is stated on the following page and it is not a scored outcome",
    ),
    Arm(
        8,
        "ShiftFocusIndColl",
        "Shifting_Focus_from_Individual_to_Collective_Action",
        "shift_focus_individual_collective",
        "video",
        False,
        "a recorded talk is the stimulus; all three follow-up items grade it",
    ),
    Arm(
        9,
        "IndStructuralChange",
        "Linking_Individual_and_Structural_Change",
        "linking_individual_structural",
        "text_plus_image",
        True,
        "four images across six live blocks, all either decorative or a chart of a "
        "percentage the adjacent text already states",
    ),
    Arm(
        10,
        "BindingMorals",
        "Binding_Moral_Foundations",
        "binding_moral_foundations",
        "text_plus_image",
        True,
        "four national-park photographs the respondent rates; the prose names and "
        "describes every one, and the ratings are intervention-internal",
    ),
    Arm(
        11,
        "CollEfficacyEmoBenefit",
        "Collective_Efficacy_and_Emotional_Benefit",
        "collective_efficacy_emotional_benefit",
        "text_plus_image",
        True,
        'six illustrative images and one march clip captioned "watch to feel the '
        'zeal"; every argument is stated in the prose, and the yes/no branches '
        'that give a "Wrong!" or "You are right!" screen are reproduced',
    ),
    Arm(
        12,
        "HopeAngerNarratives",
        "Hope_and_Anger_Narratives",
        "hope_anger_narratives",
        "pure_text",
        True,
        "no media at all; two narrative blocks, both shown, in random order",
    ),
    Arm(
        13,
        "ThreatInjustEfficacy",
        "Threat-Injustice-and-Efficacy",
        "threat_injustice_efficacy",
        "text_plus_image",
        True,
        "four uncaptioned photographs on their own screens; the fear-and-efficacy "
        "argument is entirely in the prose around them",
    ),
    Arm(
        14,
        "DynamicAngerNorm",
        "Dynamic_Anger_Norm",
        "dynamic_anger_norm",
        "text_plus_image",
        True,
        "five charts, each one restating a percentage the sentence beside it already "
        'gives ("61% of Americans think..."), so the images are strictly redundant',
    ),
    Arm(
        15,
        "BipartisanEliteCues",
        "Bipartisan_Elite_Cues",
        "bipartisan_elite_cues",
        "video",
        False,
        "the endorsement is two White House speech clips; the text summarises what "
        "each speaker stressed but not what they said",
    ),
    Arm(
        16,
        "ActivistPerspective",
        "Climate_Activist_Perspective_Taking",
        "activist_perspective_taking",
        "video",
        False,
        "a documentary trailer is the entire stimulus; the writing prompt follows it",
    ),
    Arm(
        17,
        "LetterFuture",
        "Letter_to_Future_Generations",
        "letter_to_future_generations",
        "text_plus_image",
        True,
        "the benchmark arm: a three-page framing and a letter-writing task, with one "
        "decorative image",
    ),
)

BY_NAME = {arm.name: arm for arm in ARMS}
CONTROL = "Control"
#: Arms a text transcript can carry, control first.
CONDITIONS: tuple[str, ...] = tuple(arm.name for arm in ARMS if arm.usable)

#: The nine randomised outcome blocks, and what each is called in ``DV_order``.
DV_BLOCK_ORDER: tuple[str, ...] = (
    "BeliefandPolicySupport",
    "Petition",
    "OpenEndedLetter",
    "supportclimaterepelection",
    "Bank",
    "Donation",
    "Attendmarch",
    "Newsletter",
    "Commitment",
)

#: Every reconstructed block of the instrument outside the arms, by the questions
#: it holds.  Named by hand: the trash block these come from kept no block
#: boundaries, so the grouping is evidence from the published column order plus
#: the page timers, not something the file states.
BATTERY: dict[str, tuple[str, ...]] = {
    # QID87 carries the whole consent form as its own question text, so the
    # consent screen is one question, not a display block plus a button.
    "Consent": ("QID87", "QID88"),
    "AttentionCheckColour": ("QID89",),
    "DVIntro": ("QID657",),
    "BeliefandPolicySupport": ("QID504", "QID500", "QID505", "QID503"),
    "Petition": ("QID31", "QID32", "QID77", "QID53"),
    "OpenEndedLetter": ("QID11", "QID76", "QID78"),
    "supportclimaterepelection": ("QID29", "QID69", "QID79"),
    "Bank": ("QID18", "QID19", "QID70", "QID44", "QID80"),
    "Donation": ("QID16", "QID34", "QID81"),
    "Attendmarch": ("QID37", "QID82"),
    "Newsletter": (
        "QID668",
        "QID4",
        "QID3",
        "QID83",
        "QID74",
        "QID71",
        "QID72",
        "QID84",
        "QID73",
    ),
    "Commitment": ("QID17", "QID50", "QID51"),
    "AttnCheck60": ("QID265", "QID266"),
    "VideoShare": ("QID13", "QID15", "QID75", "QID85"),
    "Efficacy": ("QID666", "QID667"),
    "Emotions": ("QID313", "QID314"),
    "Demographics": (
        "QID270",
        "QID271",
        "QID273",
        "QID276",
        "QID664",
        "QID280",
        "QID284",
        "QID285",
    ),
}

#: Shown before the arm.
PRE_BLOCKS: tuple[str, ...] = ("Consent", "AttentionCheckColour")
#: Shown after the nine randomised blocks, in this fixed order.
POST_BLOCKS: tuple[str, ...] = (
    "AttnCheck60",
    "VideoShare",
    "Efficacy",
    "Emotions",
    "Demographics",
)

#: Answers supplied by the profile rather than sampled.  Consent and both
#: attention checks are pre-filled as passed, because the published file contains
#: only respondents who passed them: sampling them would build a selection effect
#: that has no counterpart in the data being predicted.
PREFILLED_ANSWERS = {
    "Q5": "Yes, I am at least 18 years old and I want to participate",
    "AttentionCheck_purp": "Purple",
    "AttnCheck60": "Sixty",
}

#: Arms whose live blocks are presented in a random order by the survey's own
#: randomiser, with the number actually shown.  Both of these show every block.
RANDOMISED_ARM_BLOCKS = {
    "MispCorrectionRisks": 6,
    "HopeAngerNarratives": 2,
}


@dataclass(frozen=True)
class Loaded:
    """The master export, parsed once, plus its raw question payloads."""

    survey: Survey
    payloads: dict
    blocks: dict


@lru_cache(maxsize=1)
def master() -> Loaded:
    """The master export: source of the outcome battery."""
    return _load(MASTER_QSF)


@lru_cache(maxsize=32)
def arm_survey(stem: str) -> Loaded:
    """One arm's own export."""
    return _load(arm_qsf(stem))


def _load(path) -> Loaded:
    survey = load_survey(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return Loaded(
        survey=survey,
        payloads={
            element["PrimaryAttribute"]: element["Payload"]
            for element in payload["SurveyElements"]
            if element["Element"] == "SQ"
        },
        blocks=next(e for e in payload["SurveyElements"] if e["Element"] == "BL")[
            "Payload"
        ],
    )


def live_block_ids(path) -> list[str]:
    """Blocks an arm's own flow actually reaches, in flow order.

    The trash block is unreachable by construction, so this is also how the arm's
    own content is separated from the 630 questions parked beside it.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    flow = next(e for e in payload["SurveyElements"] if e["Element"] == "FL")["Payload"]
    found: list[str] = []

    def walk(node) -> None:
        if isinstance(node, list):
            for child in node:
                walk(child)
            return
        if not isinstance(node, dict):
            return
        if node.get("Type") in {"Standard", "Block"} and node.get("ID"):
            found.append(str(node["ID"]))
        for child in node.get("Flow") or []:
            walk(child)

    walk(flow)
    return found


def flow_embedded(path) -> dict[str, str]:
    """Embedded fields the arm's own survey flow assigns a literal value.

    Only the ones with a value: a field the flow declares as ``Recipient`` is
    filled in at runtime from the respondent, and pretending otherwise would put
    a constant where a personalised string went.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    flow = next(e for e in payload["SurveyElements"] if e["Element"] == "FL")["Payload"]
    values: dict[str, str] = {}

    def walk(node) -> None:
        if isinstance(node, list):
            for child in node:
                walk(child)
            return
        if not isinstance(node, dict):
            return
        for entry in node.get("EmbeddedData") or []:
            field = entry.get("Field")
            value = entry.get("Value")
            if field and value:
                values[str(field)] = str(value)
        for child in node.get("Flow") or []:
            walk(child)

    walk(flow)
    return values


@lru_cache(maxsize=64)
def slot_index(stem: str, prefix: str) -> dict[str, str]:
    """QID -> slot id, so a piped *answer* can name the slot it echoes.

    One arm quotes the respondent's own guess back at them — "You guessed 32% …
    the actual number is 18%" — as ``${q://QID.../ChoiceGroup/...}``.  Without
    this the echo renders as a bare question id, which is a token sequence nobody
    ever saw on screen.

    The id is read off the slot :func:`~silicon_sampling.goldwert.convert.convert_question`
    actually emits rather than recomputed from the export tag, because computing
    it a second way is how an echo comes to name a slot that does not exist.  The
    two rules parted on multi-row text entry: Qualtrics suffixes such a tag with
    the row key and the converter follows it, while a hand-written "sliders,
    matrices and constant sums take the key" rule did not — so
    ``IndStructuralChange``'s three "You guessed …%" screens echoed
    ``..._genderQ`` while the answer sat under ``..._genderQ_6``, and the session
    raised ``KeyError`` the moment it tried to render the page.  A template render
    cannot see this, because a template prints the marker instead of resolving it;
    only driving a session does.  Cached per export because it now converts every
    question in the file to find out.
    """
    state = arm_survey(stem)
    index: dict[str, str] = {}
    for qid, question in state.survey.questions.items():
        payload = state.payloads.get(qid, {})
        produced = cv.convert_question(question, payload, state.survey)
        emitted = next(
            (element.id for element in produced if isinstance(element, Slot)), None
        )
        if emitted is None:
            # A display-only question records nothing, so nothing can echo it;
            # the export tag is kept so the pipe still renders as *something*
            # traceable rather than as a raw QID.
            rows = cv._rows(payload)
            key = rows[0][0] if question.kind in {"Slider", "Matrix", "CS"} else ""
            emitted = cv.published_column(cv.export_column(question, key))
        index[qid] = f"{prefix}{emitted}"
    return index


def arm_blocks(arm: Arm) -> list[tuple[str, str]]:
    """``(block id, description)`` for one arm's live blocks, in flow order."""
    state = arm_survey(arm.qsf)
    return [
        (bid, state.survey.blocks[bid].description)
        for bid in live_block_ids(arm_qsf(arm.qsf))
        if bid in state.survey.blocks
    ]


def arm_elements(arm: Arm, rng: random.Random | None = None) -> list:
    """One arm's stimulus, as transcript blocks in the order it was shown.

    Where the survey randomises the arm's own blocks it randomises the *order*
    and shows all of them, so a draw here is a permutation, never a subset.
    """
    state = arm_survey(arm.qsf)
    blocks = arm_blocks(arm)
    if arm.name in RANDOMISED_ARM_BLOCKS and rng is not None and len(blocks) > 1:
        randomised = blocks[-RANDOMISED_ARM_BLOCKS[arm.name] :]
        head = blocks[: len(blocks) - len(randomised)]
        rng.shuffle(randomised)
        blocks = head + randomised
    embedded = flow_embedded(arm_qsf(arm.qsf))
    out: list = []
    seen: set[str] = set()
    for bid, description in blocks:
        key = _key(description or bid)
        items = cv.block_items(state.blocks, bid)
        converted = cv.convert_run(
            state.survey,
            state.payloads,
            items,
            prefix=f"{arm.slug}__",
            embedded=embedded,
            qid_to_slot=slot_index(arm.qsf, f"{arm.slug}__"),
        )
        if not converted.elements:
            continue
        # Two arms present the same block twice under different framings — hope
        # and anger narratives, the Biden and Portman clips — and Qualtrics gave
        # both copies the same export tags.  When that collides, the block's own
        # name goes into the id rather than one copy quietly overwriting the other.
        if set(converted.data_columns) & seen:
            converted = cv.convert_run(
                state.survey,
                state.payloads,
                items,
                prefix=f"{arm.slug}__{key}__",
                embedded=embedded,
                qid_to_slot=slot_index(arm.qsf, f"{arm.slug}__{key}__"),
            )
        seen |= set(converted.data_columns)
        out.append(
            TranscriptBlock(
                key=key, title=description or bid, elements=list(converted.elements)
            )
        )
    return out


def battery_elements(name: str) -> list:
    """One reconstructed block of the shared instrument."""
    state = master()
    return list(
        cv.convert_run(
            state.survey,
            state.payloads,
            [("Question", qid) for qid in BATTERY[name]],
            page_from_timers=True,
        ).elements
    )


def dv_order(rng: random.Random) -> list[str]:
    """One respondent's draw of the nine-block outcome order."""
    order = list(DV_BLOCK_ORDER)
    rng.shuffle(order)
    return order


def elements_for(
    arm_name: str,
    battery: Sequence[str] | None = None,
    rng: random.Random | None = None,
) -> list:
    """The full element sequence one respondent in this arm walks through."""
    arm = BY_NAME[arm_name]
    elements: list = []
    for name in PRE_BLOCKS:
        elements.append(
            TranscriptBlock(key=_key(name), title=name, elements=battery_elements(name))
        )
    elements.extend(arm_elements(arm, rng))
    elements.append(
        TranscriptBlock(
            key="dv_intro", title="DVIntro", elements=battery_elements("DVIntro")
        )
    )
    for name in battery if battery is not None else list(DV_BLOCK_ORDER):
        elements.append(
            TranscriptBlock(key=_key(name), title=name, elements=battery_elements(name))
        )
    for name in POST_BLOCKS:
        elements.append(
            TranscriptBlock(key=_key(name), title=name, elements=battery_elements(name))
        )
    return elements


def data_columns() -> dict[str, str]:
    """Slot id -> published data column, across the whole instrument.

    Slot ids *are* the published column names (see
    :data:`~silicon_sampling.goldwert.convert.PUBLISHED_COLUMN`), so this is an
    identity map by construction — kept as a function because that is the
    property worth asserting in a test.
    """
    columns: dict[str, str] = {}
    for name in BATTERY:
        columns.update(
            cv.convert_run(
                master().survey,
                master().payloads,
                [("Question", qid) for qid in BATTERY[name]],
                page_from_timers=True,
            ).data_columns
        )
    return columns


def modality_audit() -> list[dict]:
    """One row per arm: what its live blocks contain, and the keep-or-drop call.

    The media counts include Qualtrics *graphic* questions, whose asset lives in
    ``Payload["Graphics"]`` rather than in an ``<img>`` tag.  Missing those was
    the difference between "Co-Benefits is 73 words of text" and "Co-Benefits is
    one enormous infographic", so they are counted here explicitly.
    """
    rows = []
    for arm in ARMS:
        state = arm_survey(arm.qsf)
        embedded = flow_embedded(arm_qsf(arm.qsf))
        counts = {"video": 0, "audio": 0, "image": 0, "iframe": 0, "graphic": 0}
        # Media the flow holds as an embedded value and pipes onto a page at
        # runtime: invisible to any scan of the question text, and in one arm it
        # is the only image there is.
        counts["piped_image"] = sum(
            value.lower().count("<img") for value in embedded.values()
        )
        words = 0
        chars = 0
        descriptions = []
        for bid, description in arm_blocks(arm):
            descriptions.append(description or bid)
            block = state.survey.blocks[bid]
            for qid in block.question_ids:
                question = state.survey.questions.get(qid)
                if question is None:
                    continue
                question = cv.resolve_embedded(question, embedded)
                for key, count in question.media().items():
                    if key in counts:
                        counts[key] += count
                if state.payloads.get(qid, {}).get("Graphics"):
                    counts["graphic"] += 1
                words += len(question.text.split())
                chars += len(question.text)
        rows.append(
            {
                "cond": arm.cond,
                "condName": arm.name,
                "qsf": arm.qsf,
                "modality": arm.modality,
                "usable": "yes" if arm.usable else "no",
                "reason": arm.reason,
                "n_blocks": len(descriptions),
                "words": words,
                "chars": chars,
                **counts,
                "blocks": "; ".join(descriptions),
            }
        )
    return rows


def _key(description: str) -> str:
    return "".join(
        character if character.isalnum() else "_" for character in description.lower()
    ).strip("_")


def header(profile_id: str, arm_name: str) -> str:
    """The transcript preamble, dated to the study's own fielding window."""
    arm = BY_NAME[arm_name]
    return "\n".join(
        [
            "=" * 78,
            " CLIMATE ADVOCACY MEGASTUDY",
            " A megastudy of behavioral interventions to catalyze public, political and",
            " financial climate advocacy.",
            " Response transcripts, one file per participant.",
            "-" * 78,
            f" File           : responses/{profile_id}.txt",
            f" Participant ID : {profile_id}",
            f" Condition      : {arm.cond:02d}",
            " Instrument     : US national sample, fielded June 2024",
            " Note           : Verbatim record of one session, screens in the order they were",
            '                  displayed. Lines beginning "Response:" hold what the participant',
            "                  entered.",
            "=" * 78,
        ]
    )


def transition_text() -> Text:  # pragma: no cover - parity with the Voelkel module
    return Text("")
