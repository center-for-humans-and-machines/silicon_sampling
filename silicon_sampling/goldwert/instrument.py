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
into a text transcript without inventing content: their comprehension items grade
what the media said. The eleven that survive still show 46 pictures and clips between them,
and those are described rather than elided --
:mod:`~silicon_sampling.goldwert.images` holds what each one shows, written from
files fetched off the hosts the exports hot-link. :data:`ARMS` carries the call, the
reason and a 0-3 :data:`MEDIA_LOSS` grade for every arm; :func:`modality_audit`
counts the media behind it and reports how much of it reaches words.
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
from ..voelkel.qsf import Survey, load_survey, strip_html
from . import convert as cv
from . import images
from .paths import MASTER_QSF, arm_qsf

#: The trash block of either master export: the whole survey minus one arm.
TRASH_BLOCK = "BL_9BouSHFCKXQhiGa"


#: What each :attr:`Arm.media_loss` grade means.  A *kept* arm's grade is what a
#: synthetic respondent still does not get after every picture has been described,
#: not what it would have lost from a blank placeholder — so 0 is now common and 3
#: is reserved for the arms that are dropped precisely because they earn it.
#:
#: The grades exist to be regressed on, not to be read as a summary.  The project
#: needs to test separately whether models do worse on media-heavy arms, and that
#: test needs a per-arm number that was assigned by looking at the media rather than
#: at the results.
MEDIA_LOSS = {
    0: "nothing lost: the text carries everything the screen carried",
    1: "peripheral loss: affect, atmosphere or a quantity the prose gives in words",
    2: "substantive loss: something the argument leans on is only in the media",
    3: "the intervention's core is not in the text",
}


@dataclass(frozen=True)
class Arm:
    """One experimental arm, and whether a text transcript can carry it.

    ``modality`` is one of ``pure_text``, ``text_plus_image``, ``image_of_text``,
    ``video`` or ``other``.  ``usable`` is the audit's call, and ``reason`` says
    why in the terms the audit uses — not a restatement of ``modality``.
    ``media_loss`` grades what is still missing on the 0-3 scale of
    :data:`MEDIA_LOSS`.
    """

    cond: int
    name: str
    qsf: str
    slug: str
    modality: str
    usable: bool
    reason: str
    media_loss: int = 0


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
        "one five-minute knot-tying video and the instruction to watch it. Its "
        "*content* is null, which is why the arm is kept; the five minutes of "
        "attention it consumed are not, which is why the screen is described rather "
        "than left blank -- every effect in the study is a contrast against this arm",
        1,
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
        3,
    ),
    Arm(
        2,
        "MispCorrectionRisks",
        "Misperception_Correction_Risks",
        "misperception_correction_risks",
        "text_plus_image",
        True,
        "six correction pages in a random order plus a writing prompt; the correction "
        "prose is piped in from the survey flow's embedded data, and so is one of six "
        "news photographs, selected on the writing page by the respondent's own "
        "answer. The prose carries the argument; the photographs illustrate it",
        0,
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
        3,
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
        3,
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
        3,
    ),
    Arm(
        6,
        "SystemJustification",
        "System_Justification",
        "system_justification",
        "text_plus_image",
        True,
        "twelve photographs, eight of them captioned by place in the prose beside "
        'them ("Chicago, Illinois", "Grand Canyon, Arizona"); the argument is the '
        "prose, but four are uncaptioned and two of those four -- a full-frame US "
        "flag and the Macy's Thanksgiving parade -- are the patriotic prime this "
        "intervention works through, so they are described in the transcript",
        1,
    ),
    Arm(
        7,
        "EcologicalDisruptions",
        "Connecting_to_Ecological_Disruptions",
        "ecological_disruptions",
        "text_plus_image",
        True,
        "long article excerpts plus four images; the reasoning task asks which of "
        "four unlabelled contributor charts matches a temperature-anomaly curve, so "
        "all five panels are transcribed and the item is answerable. The row of dead "
        "songbirds the writing prompt points at is described too",
        2,
    ),
    Arm(
        8,
        "ShiftFocusIndColl",
        "Shifting_Focus_from_Individual_to_Collective_Action",
        "shift_focus_individual_collective",
        "video",
        False,
        "a recorded talk is the stimulus; all three follow-up items grade it",
        3,
    ),
    Arm(
        9,
        "IndStructuralChange",
        "Linking_Individual_and_Structural_Change",
        "linking_individual_structural",
        "text_plus_image",
        True,
        "four figures across seven live blocks; three restate a percentage the "
        "adjacent prose already gives, and the fourth is the individual-to-structural "
        'diagram two pages call "the figure above", so all four are described',
        1,
    ),
    Arm(
        10,
        "BindingMorals",
        "Binding_Moral_Foundations",
        "binding_moral_foundations",
        "text_plus_image",
        True,
        "four national-park photographs the respondent is asked to rate, two of the "
        'six items by pointing at "the picture on the right above"; only the Smoky '
        "Mountains pair is described in the prose, so all four are described in the "
        "transcript. The ratings are intervention-internal",
        2,
    ),
    Arm(
        11,
        "CollEfficacyEmoBenefit",
        "Collective_Efficacy_and_Emotional_Benefit",
        "collective_efficacy_emotional_benefit",
        "text_plus_image",
        True,
        "every argument is stated in the prose and the yes/no branches that give a "
        '"Wrong!" or "You are right!" screen are reproduced, but this is the one kept '
        "arm whose pictures are gone: all five graphics were deleted from the media "
        "library and 404 everywhere, and the climate-march clip shown to feel the "
        "zeal of a march is a video, so the emotional-benefit half is thin",
        2,
    ),
    Arm(
        12,
        "HopeAngerNarratives",
        "Hope_and_Anger_Narratives",
        "hope_anger_narratives",
        "pure_text",
        True,
        "no media at all; two narrative blocks, both shown, in random order",
        0,
    ),
    Arm(
        13,
        "ThreatInjustEfficacy",
        "Threat-Injustice-and-Efficacy",
        "threat_injustice_efficacy",
        "text_plus_image",
        True,
        "the fear-and-efficacy argument is in the prose, but two whole screens have "
        "no content except photographs -- three children's protest placards, then "
        "solar installers -- and the item right after the first asks whether it is "
        "fair that children suffer most, so all four are described",
        2,
    ),
    Arm(
        14,
        "DynamicAngerNorm",
        "Dynamic_Anger_Norm",
        "dynamic_anger_norm",
        "text_plus_image",
        True,
        'two gauges strictly redundant with the sentence beside them ("61% of '
        'Americans think..."), two decorative photographs, and one line chart that is '
        "not redundant: it is the rising 2016-2023 trend that makes this a dynamic "
        "norm, and only the picture gives the starting level",
        1,
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
        3,
    ),
    Arm(
        16,
        "ActivistPerspective",
        "Climate_Activist_Perspective_Taking",
        "activist_perspective_taking",
        "video",
        False,
        "a documentary trailer is the entire stimulus; the writing prompt follows it",
        3,
    ),
    Arm(
        17,
        "LetterFuture",
        "Letter_to_Future_Generations",
        "letter_to_future_generations",
        "text_plus_image",
        True,
        "the benchmark arm: a three-page framing and a letter-writing task. Its one "
        "image, a family finding a time capsule, has been deleted from the media "
        "library, but the prose describes that scene in full",
        0,
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
    # Two display screens and the block's page timers were missing from this list
    # until the .qsf flow was walked against it. QID268 is the section's own
    # introduction ("The following section includes some questions about your
    # background...") and QID278 introduces the household questions; without them
    # the demographics arrive with no preamble at all. And with no timers in the
    # list, `page_from_timers` found no page breaks and put all eight questions on
    # one screen, where the respondent answered them on five.
    "Demographics": (
        "QID267",
        "QID268",
        "QID269",
        "QID270",
        "QID271",
        "QID272",
        "QID273",
        "QID275",
        "QID276",
        "QID664",
        "QID277",
        "QID278",
        "QID279",
        "QID280",
        "QID281",
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
#:
#: Kept only as the assertion :func:`arm_block_groups` is checked against — the
#: order itself now comes from the flow, because *counting* the randomised blocks
#: and shuffling that many off the end of the list put the wrong blocks in the
#: randomiser.  ``MispCorrectionRisks`` has nine live blocks and its randomiser
#: holds blocks two to seven; shuffling the last six shuffled blocks four to nine,
#: which are the four remaining corrections *plus the writing prompt and the closing
#: debrief*.  The consequence is plain in the old rendered template: the page
#: reading "To review, climate change can: decrease employment-related income …"
#: and the writing task that follows it landed between the second and third
#: correction, so the summary of six corrections was shown after two of them and the
#: last four arrived after the arm had already been wrapped up.  Employment and
#: Property Destruction, meanwhile, were never randomised at all.
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


@dataclass(frozen=True)
class BlockGroup:
    """A run of an arm's blocks that move together, or do not move at all.

    ``subset`` is how many of ``ids`` the survey actually shows.  Both randomisers
    in the kept arms show all of theirs, but reading the number rather than assuming
    it is the difference between a permutation and a sample.
    """

    randomised: bool
    ids: tuple[str, ...]
    subset: int


def live_block_groups(path) -> list[BlockGroup]:
    """The arm's flow as runs of blocks that move together, in flow order.

    A ``BlockRandomizer`` node in the flow names exactly which blocks it permutes,
    and that is the only trustworthy answer to "which of this arm's screens moved".
    Deriving it any other way — from a count, from a position — is what broke
    ``MispCorrectionRisks``; see :data:`RANDOMISED_ARM_BLOCKS`.  Blocks outside any
    randomiser accumulate into fixed groups, so a caller shuffles within a group and
    never across one.

    The published file agrees with what this reads: ``FL_34_DO`` records each
    ``MispCorrectionRisks`` respondent's own draw, and every one of its 1,717 values
    is a permutation of the six correction blocks and of nothing else; ``FL_62_DO``
    does the same for the two ``HopeAngerNarratives`` blocks.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    flow = next(e for e in payload["SurveyElements"] if e["Element"] == "FL")["Payload"]
    groups: list[BlockGroup] = []

    def blocks_under(node) -> list[str]:
        found: list[str] = []

        def descend(inner) -> None:
            if isinstance(inner, list):
                for child in inner:
                    descend(child)
                return
            if not isinstance(inner, dict):
                return
            if inner.get("Type") in {"Standard", "Block"} and inner.get("ID"):
                found.append(str(inner["ID"]))
            for child in inner.get("Flow") or []:
                descend(child)

        for child in node.get("Flow") or []:
            descend(child)
        return found

    def walk(node) -> None:
        if isinstance(node, list):
            for child in node:
                walk(child)
            return
        if not isinstance(node, dict):
            return
        kind = node.get("Type")
        if kind in {"BlockRandomizer", "Randomizer"}:
            ids = blocks_under(node)
            if ids:
                try:
                    subset = int(node.get("SubSet") or len(ids))
                except (TypeError, ValueError):
                    subset = len(ids)
                groups.append(
                    BlockGroup(True, tuple(ids), max(1, min(subset, len(ids))))
                )
            return
        if kind in {"Standard", "Block"} and node.get("ID"):
            block_id = str(node["ID"])
            if groups and not groups[-1].randomised:
                fixed = groups[-1]
                groups[-1] = BlockGroup(
                    False, fixed.ids + (block_id,), len(fixed.ids) + 1
                )
            else:
                groups.append(BlockGroup(False, (block_id,), 1))
        for child in node.get("Flow") or []:
            walk(child)

    walk(flow)
    return groups


def live_block_ids(path) -> list[str]:
    """Blocks an arm's own flow actually reaches, in flow order.

    The trash block is unreachable by construction, so this is also how the arm's
    own content is separated from the 630 questions parked beside it.
    """
    return [bid for group in live_block_groups(path) for bid in group.ids]


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


def arm_block_order(arm: Arm, rng: random.Random | None = None) -> list[str]:
    """One respondent's draw of this arm's block order, in flow order.

    Shuffling happens *inside* a randomiser's own group and nowhere else, which is
    the property :data:`RANDOMISED_ARM_BLOCKS` explains the loss of.
    """
    order: list[str] = []
    for group in live_block_groups(arm_qsf(arm.qsf)):
        ids = list(group.ids)
        if group.randomised and rng is not None:
            rng.shuffle(ids)
            ids = ids[: group.subset]
        order.extend(ids)
    return order


#: How the six correction pages of ``MispCorrectionRisks`` head themselves:
#: "Question ${e://Field/n} out of ${e://Field/total_questions}".  The flow sets
#: ``n`` to 1 once and the page script increments it on every correction screen
#: (``let n = parseInt("${e://Field/n}") + 1``), so the counter follows the *drawn*
#: order.  Resolving ``n`` from the flow value alone headed all six pages
#: "Question 1 out of 6", which is not a page any respondent saw.
COUNTER_FIELD = "n"


def arm_elements(arm: Arm, rng: random.Random | None = None) -> list:
    """One arm's stimulus, as transcript blocks in the order it was shown.

    Where the survey randomises the arm's own blocks it randomises the *order* and,
    in both arms that do it, shows all of them — so a draw here is a permutation.
    """
    state = arm_survey(arm.qsf)
    names = {
        bid: state.survey.blocks[bid].description
        for bid in state.survey.blocks  # noqa: C416 - explicit for readability
    }
    order = [bid for bid in arm_block_order(arm, rng) if bid in state.survey.blocks]
    embedded = flow_embedded(arm_qsf(arm.qsf))
    counted = 0
    out: list = []
    seen: set[str] = set()
    for bid in order:
        description = names[bid]
        key = _key(description or bid)
        items = cv.block_items(state.blocks, bid)
        page_embedded = embedded
        if COUNTER_FIELD in embedded and _counts_itself(state, items):
            counted += 1
            page_embedded = {**embedded, COUNTER_FIELD: str(counted)}
        graded = _graded_slot(arm, state, items)
        if graded is not None:
            # Redirect this screen's feedback pipe at its own field rather than at
            # the one the survey shares between all six; see FEEDBACK_SUFFIX.
            page_embedded = {
                **page_embedded,
                "text": "${e://Field/" + feedback_field(graded) + "}",
            }
        converted = cv.convert_run(
            state.survey,
            state.payloads,
            items,
            prefix=f"{arm.slug}__",
            embedded=page_embedded,
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
                embedded=page_embedded,
                qid_to_slot=slot_index(arm.qsf, f"{arm.slug}__{key}__"),
            )
        seen |= set(converted.data_columns)
        out.append(
            TranscriptBlock(
                key=key, title=description or bid, elements=list(converted.elements)
            )
        )
    return out


#: The arm whose pages are written by its own page script rather than by the
#: export, and the questions of it that a stand-in has to reproduce.
FEEDBACK_ARM = "MispCorrectionRisks"
#: Export tags of the six correction items, and the summary question after them.
CORRECTION_TAGS = (
    "employment",
    "property",
    "consumer_goods",
    "energy_prices",
    "dependent_care",
    "healthcare_expenses",
)
SUMMARY_TAG = "Q34"
#: Which embedded field pair each option of the summary question selects.
#: Transcribed from that question's own ``addOnPageSubmit`` handler, which reads the
#: choice's *recode* value 1-6 and pastes ``<topic>_text`` into ``choice_text`` and
#: ``<topic>_img`` into ``img``.  Recode order, not display order — they differ,
#: because the six choices carry keys 1, 4, 5, 6, 7, 8.
SUMMARY_TOPIC_BY_RECODE = (
    "employment",
    "property",
    "consumer",
    "energy",
    "childcare",
    "healthcare",
)


#: Suffix of the per-screen field each correction page's feedback line echoes.
#:
#: There has to be one field per screen, and the reason is not cosmetic.  A session
#: re-renders the whole transcript from ``answers`` at every step, so a *single*
#: ``text`` field shared by all six feedback screens does not merely resolve late —
#: it rewrites history: answering the fourth correction item changes what the first
#: feedback screen says, several thousand tokens back.  Nothing in a rendered
#: template can show that, and a driven session only shows it if something is
#: watching the prefix, which is why it surfaced as a failure of the incremental
#: tokenisation check rather than as a wrong-looking transcript.  Naming the field
#: after the item it depends on makes each screen a function of one answer and makes
#: the prefix immutable again.
FEEDBACK_SUFFIX = "__feedback"


def feedback_field(slot_id: str) -> str:
    """The per-screen field the feedback line for ``slot_id`` is echoed from."""
    return f"{slot_id}{FEEDBACK_SUFFIX}"


@lru_cache(maxsize=1)
def correction_feedback() -> dict[str, dict[str, str]]:
    """Slot id -> {option chosen: the line that page then headed itself with}.

    The six correction screens open with ``${e://Field/text}`` in
    ``${e://Field/col}``, and both are set by the page script from the answer just
    given: recode 1 pastes the flow's ``correct`` value in green, recode 0 its
    ``incorrect`` value in red.  Every part of that is in the export — the recode
    table on the question, the two strings in the flow — so the feedback is
    recoverable, and a respondent who answered "Increasing" to the employment item
    should be told they were wrong rather than shown a placeholder.

    The colour is dropped on purpose: it is a hex code in a ``style`` attribute, and
    a transcript that reported "#c0392b" would be reporting the stylesheet.
    """
    arm = BY_NAME[FEEDBACK_ARM]
    state = arm_survey(arm.qsf)
    flow = flow_embedded(arm_qsf(arm.qsf))
    right = str(flow.get("correct") or "").strip()
    wrong = str(flow.get("incorrect") or "").strip()
    table: dict[str, dict[str, str]] = {}
    for qid, question in state.survey.questions.items():
        if question.export_tag not in CORRECTION_TAGS:
            continue
        payload = state.payloads.get(qid, {})
        recodes = payload.get("RecodeValues") or {}
        choices = payload.get("Choices") or {}
        answers = {}
        for key, choice in choices.items():
            label = cv.one_line(str(choice.get("Display", "")))
            answers[label] = right if str(recodes.get(str(key))) == "1" else wrong
        if answers:
            slot = f"{arm.slug}__{cv.published_column(cv.export_column(question))}"
            table[slot] = answers
    return table


@lru_cache(maxsize=1)
def summary_choice_pages() -> tuple[str, dict[str, tuple[str, str]]]:
    """``(slot id, {option: (correction prose, image note)})`` for the writing page.

    The writing screen of ``MispCorrectionRisks`` is three runtime-piped fields:
    ``img``, ``choice_text`` and ``option_text``.  All three are functions of the
    answer the respondent has just given on the previous screen, and the flow holds
    every value they can take, so all three are recoverable — which matters, because
    the page asks the respondent to write about the issue whose correction paragraph
    and photograph it is displaying, and a stand-in note left them writing about a
    blank.
    """
    arm = BY_NAME[FEEDBACK_ARM]
    state = arm_survey(arm.qsf)
    flow = flow_embedded(arm_qsf(arm.qsf))
    question = next(
        q for q in state.survey.questions.values() if q.export_tag == SUMMARY_TAG
    )
    payload = next(
        state.payloads[qid]
        for qid, q in state.survey.questions.items()
        if q is question
    )
    recodes = payload.get("RecodeValues") or {}
    choices = payload.get("Choices") or {}
    table: dict[str, tuple[str, str]] = {}
    for key, choice in choices.items():
        position = int(str(recodes.get(str(key), key)))
        if not 1 <= position <= len(SUMMARY_TOPIC_BY_RECODE):
            continue
        topic = SUMMARY_TOPIC_BY_RECODE[position - 1]
        label = cv.one_line(str(choice.get("Display", "")))
        prose = strip_html(str(flow.get(f"{topic}_text") or "")).strip()
        note = cv.describe_media_html(str(flow.get(f"{topic}_img") or ""))
        table[label] = (prose, note)
    slot = f"{arm.slug}__{cv.published_column(cv.export_column(question))}"
    return slot, table


def _graded_slot(
    arm: Arm, state: Loaded, items: Sequence[tuple[str, str]]
) -> str | None:
    """The slot whose answer this block's feedback screen reacts to, if it has one."""
    if arm.name != FEEDBACK_ARM:
        return None
    for kind, qid in items:
        question = state.survey.questions.get(qid) if kind == "Question" else None
        if question is not None and question.export_tag in CORRECTION_TAGS:
            return f"{arm.slug}__{cv.published_column(cv.export_column(question))}"
    return None


def _counts_itself(state: Loaded, items: Sequence[tuple[str, str]]) -> bool:
    """Whether any question in this block heads itself with the page counter.

    Read off the text rather than listed by block id, so a block that stops using
    the counter stops advancing it.
    """
    marker = "${e://Field/" + COUNTER_FIELD + "}"
    for kind, qid in items:
        question = state.survey.questions.get(qid) if kind == "Question" else None
        if question is not None and marker in question.raw_text:
            return True
    return False


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


def media_keys(arm: Arm) -> list[str]:
    """Every distinct picture or clip this arm's live blocks reference, in order.

    Keyed the way :mod:`~silicon_sampling.goldwert.images` is keyed, so that the
    audit can report *how many of this arm's pictures are actually described* rather
    than only how many there are.  A count is not an audit: the previous version of
    this file reported twelve images for ``SystemJustification`` and called all
    twelve captioned, and both halves of that were produced without opening a file.
    """
    state = arm_survey(arm.qsf)
    embedded = flow_embedded(arm_qsf(arm.qsf))
    found: list[str] = []

    def note(key: str) -> None:
        if key and key not in found:
            found.append(key)

    for bid, _ in arm_blocks(arm):
        for qid in state.survey.blocks[bid].question_ids:
            question = state.survey.questions.get(qid)
            if question is None:
                continue
            raw = cv.resolve_embedded(question, embedded).raw_text
            for match in cv._MEDIA_TAG.finditer(raw):
                window = raw[match.start() : match.start() + 800]
                source = cv._SRC.search(match.group(0)) or cv._SRC.search(window)
                note(cv._asset_key(match.group(0), source.group(1) if source else ""))
            note(str(state.payloads.get(qid, {}).get("Graphics") or ""))
    for value in embedded.values():
        for match in cv._MEDIA_TAG.finditer(value):
            source = cv._SRC.search(match.group(0))
            note(cv._asset_key(match.group(0), source.group(1) if source else ""))
    return found


def modality_audit() -> list[dict]:
    """One row per arm: what its live blocks contain, and the keep-or-drop call.

    The media counts include Qualtrics *graphic* questions, whose asset lives in
    ``Payload["Graphics"]`` rather than in an ``<img>`` tag.  Missing those was
    the difference between "Co-Benefits is 73 words of text" and "Co-Benefits is
    one enormous infographic", so they are counted here explicitly.

    Three columns were added after the counts turned out not to settle anything.
    ``media_loss`` is the 0-3 grade of :data:`MEDIA_LOSS` — the thing a
    media-heaviness analysis needs and a count of ``<img>`` tags is not.
    ``described`` and ``undescribed`` say how many of the arm's distinct assets
    :mod:`~silicon_sampling.goldwert.images` can put into words, so a claim that an
    arm is safe to keep can be checked against whether its pictures are in the file
    at all.
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
        keys = media_keys(arm)
        described = [
            key
            for key in keys
            if images.describe(key) is not None or key in images.MEDIA_ALT
        ]
        rows.append(
            {
                "cond": arm.cond,
                "condName": arm.name,
                "qsf": arm.qsf,
                "modality": arm.modality,
                "usable": "yes" if arm.usable else "no",
                "media_loss": arm.media_loss,
                "media_loss_meaning": MEDIA_LOSS[arm.media_loss],
                "n_assets": len(keys),
                "described": len(described),
                "undescribed": len(keys) - len(described),
                "unrecoverable": sum(1 for key in keys if key in images.UNRECOVERABLE),
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
