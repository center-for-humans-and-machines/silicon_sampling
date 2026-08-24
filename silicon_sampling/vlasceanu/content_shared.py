"""Screens every participant saw, in display order.

Wording is transcribed from ``data/Vlasceanu/master_survey.pdf`` (the text layer
of that file is corrupt, so it was read visually).  Page boundaries come from the
``*_Page.Submit`` timer variables in ``data_raw.csv``; the timer name is recorded
on each :class:`~silicon_sampling.vlasceanu.elements.Screen`.

**Which item goes with which column is not the PDF's to say.**  Reading a matrix
off a page recovers the set of statements but not reliably the order, and the
order is the whole of the binding: the ``items`` pairs below name published data
columns, so a permuted list silently files an answer under a column measuring
something else.  Battery means are invariant to that permutation, which is why it
survived a comparison against the published composites.  Every battery here is
therefore ordered by the *choice codes* in ``data/ICPC/Materials/usa_3.qsf`` —
none of the four matrix questions defines ``RecodeValues``,
``ChoiceDataExportTags`` or ``VariableNaming``, so the export suffix simply *is*
the choice code — cross-checked against ``codebook.xlsx`` and against the
published per-item means.  Where the visual transcription and the ``.qsf``
disagreed on the *wording* too, the ``.qsf`` wins: it is what participants read.
``tests/test_icpc.py`` re-derives the binding from both authorities on every run,
so a hand edit here that breaks it fails rather than quietly mis-scores.

Note on the consent and debriefing screens: the master survey carries New York
University's forms as a *specimen*.  The adaptation manual instructed every team
to substitute their own institution's form, so the names, addresses and study
identifiers below are placeholders rather than something the 63 national samples
actually read.  They are kept verbatim because they are the best available
approximation of that screen; deleting them is a local edit to ``CONSENT``.
"""

from __future__ import annotations

from .country import CountryProfile
from .elements import (
    Block,
    Bullets,
    Choice,
    Echo,
    FreeText,
    Gate,
    Image,
    Matrix,
    MultiChoice,
    Number,
    NumberGrid,
    Screen,
    Slider,
    Text,
)

ADVANCE = "You will be able to advance the page shortly"
ADVANCE_DOT = "You will be able to advance the page shortly."


# --------------------------------------------------------------------------- #
# 1. consent and the first attention check
# --------------------------------------------------------------------------- #

CONSENT = Block(
    key="consent",
    title="Consent",
    screens=[
        Screen(
            timer="consent_timer",
            elements=[
                Image(
                    alt=(
                        "Letterhead of New York University. At the left, the NYU torch "
                        "logo in a box; to its right, in serif type, 'New York "
                        "University', the underlined italic line 'A private university "
                        "in the public service', and then the address block 'Faculty of "
                        "Arts and Science / Department of Psychology / 6 Washington "
                        "Place, Room 550 / New York, NY 10003-6634 / Telephone: (212) "
                        "998-7820 / FAX: (212) 995-4018'."
                    )
                ),
                Text("Consent Form for IRB-FY2022-6272", style="head"),
                Text(
                    "You have been invited to take part in a research study to learn "
                    "more about your beliefs and behaviors. This study will be "
                    "conducted by Kimberly Doell, Department of Psychology, College of "
                    "Arts & Science (CAS), New York University, as a part of her "
                    "Faculty-directed Research. Her faculty sponsor is Professor Jay "
                    "Van Bavel, Department of psychology, CAS, New York University."
                ),
                Text(
                    "If you agree to be in this study, you will be asked to do the following:"
                ),
                Bullets(
                    [
                        "Complete a set of online questionnaires. You may be asked to "
                        "rate different stimuli, or describe a series of activities you "
                        "regularly engage in, or write a short letter.",
                        "You will also be asked to report some personal information, "
                        "including your age, gender, etc.",
                    ]
                ),
                Text(
                    "Participation in this study will involve no more than 30 minutes. "
                    "There are no known risks associated with your participation in "
                    "this research beyond those of everyday life."
                ),
                Text(
                    "Although you will receive no direct benefits, this research may "
                    "help the investigator understand your beliefs and behaviors."
                ),
                Text(
                    "Confidentiality of your research records will be strictly "
                    "maintained by assigning code numbers to each participant so that "
                    "data is never directly linked to individual identity. Your "
                    "information from this study will not be used for future research."
                ),
                Text(
                    "Participation in this study is voluntary. You may refuse to "
                    "participate or withdraw at any time without penalty. For "
                    "questionnaires, or surveys, you have the right to skip or not "
                    "answer any questions you prefer not to answer."
                ),
                Text(
                    "If there is anything about the study or your participation that is "
                    "unclear or that you do not understand, if you have questions or "
                    "wish to report a research-related problem, you may contact "
                    "Kimberly Doell at (212) 992-9627, kd2517@nyu.edu, 6 Washington "
                    "Place New York, NY 10003, or the faculty sponsor, Jay Van Bavel at "
                    "(212) 992-9627, jay.vanbavel@nyu.edu, 6 Washington Place New York, "
                    "NY 10003."
                ),
                Text(
                    "For questions about your rights as a research participant, you may "
                    "contact the University Committee on Activities Involving Human "
                    "Subjects (UCAIHS), New York University, 665 Broadway, Suite 804, "
                    "New York, New York, 10012, at ask.humansubjects@nyu.edu or (212) "
                    "998-4808. Please reference the study # (IRB-FY2022-6272) when "
                    "contacting the IRB (UCAIHS). If you would like to keep a copy of "
                    "this consent form, please print it now for your records."
                ),
                Text(
                    "Please check the box below indicating that you are at least 18 "
                    "years old, and you agree to participate. If you do not want to "
                    "participate, simply close this window."
                ),
                Choice(
                    slot="Q5",
                    options=[
                        "Yes, I am at least 18 years old and I want to participate"
                    ],
                ),
            ],
        ),
        Screen(
            timer=None,
            elements=[
                Text(
                    "The color test you are about to take part in is very simple. "
                    'Please select the color "purple" from the list below. We would '
                    "like to make sure that you are reading these questions carefully."
                ),
                Choice(
                    slot="AttentionCheck_purp",
                    options=["Red", "Yellow", "Green", "Purple", "Blue"],
                ),
            ],
        ),
    ],
    note=(
        "The colour check screened participants: teams were told they could branch "
        "failures straight out of the survey."
    ),
)


# --------------------------------------------------------------------------- #
# 2. the shared climate-change definition
# --------------------------------------------------------------------------- #

INTRO = Block(
    key="intro",
    title="Climate change definition",
    screens=[
        Screen(
            timer="IntroTime",
            elements=[
                Text(
                    "Throughout this survey, you may be asked to read some information, "
                    "report your beliefs or behaviors, or even write a small paragraph."
                ),
                Text(
                    "Before we begin, we would like to clarify what we mean by "
                    '"climate change".'
                ),
                Text(
                    "Climate change is the phenomenon describing the fact that the "
                    "world's average temperature has been increasing over the past 150 "
                    "years and will likely be increasing more in the future.",
                    style="underline",
                ),
                Text(ADVANCE),
            ],
        )
    ],
)


# --------------------------------------------------------------------------- #
# 3. dependent variables (presented in a randomised order)
# --------------------------------------------------------------------------- #

BELIEF = Block(
    key="belief",
    title="Belief in climate change",
    screens=[
        Screen(
            timer="belief_timer",
            elements=[
                Text("Instructions: How accurate do you think these statements are?"),
                Matrix(
                    left="Not at all accurate",
                    right="Extremely accurate",
                    randomised=True,
                    items=[
                        (
                            "Belief.in.CC_1",
                            "Human activities are causing climate change.",
                        ),
                        (
                            "Belief.in.CC_2",
                            "Climate change poses a serious threat to humanity.",
                        ),
                        (
                            "Belief.in.CC_4",
                            "Taking action to fight climate change is necessary to "
                            "avoid a global catastrophe.",
                        ),
                        ("Belief.in.CC_5", "Climate change is a global emergency."),
                    ],
                ),
            ],
        )
    ],
)

POLICY = Block(
    key="policy",
    title="Climate policy support",
    screens=[
        Screen(
            timer="policy_timer",
            elements=[
                Text(
                    "Instructions: Many countries have introduced policies to help "
                    "reduce carbon emissions and help to mitigate the climate crisis. "
                    "This can include the implementation of laws and requirements which "
                    "broadly aim to reduce various greenhouse gasses."
                ),
                Text(
                    "Please indicate your level of agreement with the following statements."
                ),
                Text("I support..."),
                Matrix(
                    left="Not at all",
                    mid="Moderately",
                    right="Very much so",
                    extra="Not Applicable",
                    randomised=True,
                    items=[
                        (
                            "CC_policy_1",
                            "raising carbon taxes on gas/fossil fuels/coal",
                        ),
                        (
                            "CC_policy_2",
                            "significantly expanding infrastructure for public transportation",
                        ),
                        (
                            "CC_policy_3",
                            "increasing the number of charging stations for electric vehicles",
                        ),
                        (
                            "CC_policy_5",
                            "increasing the use of sustainable energy such as wind and solar energy",
                        ),
                        (
                            "CC_policy_6",
                            "increasing taxes on airline companies to offset carbon emissions",
                        ),
                        ("CC_policy_7", "protecting forested and land areas"),
                        ("CC_policy_8", "investing more in green jobs and businesses"),
                        (
                            "CC_policy_9",
                            "introducing laws to keep waterways and oceans clean",
                        ),
                        (
                            "CC_policy_10",
                            "increasing taxes on carbon intense foods (for example meat, and dairy)",
                        ),
                    ],
                ),
            ],
        )
    ],
)


def sharing_block(cond_code: int) -> Block:
    """Social-media sharing.  The post text carries the condition code."""
    return Block(
        key="sharing",
        title="Willingness to share climate information",
        screens=[
            Screen(
                timer="share_timer",
                elements=[
                    Text(
                        "Did you know that removing meat and dairy for only two out of "
                        "three meals per day could decrease food-related carbon "
                        "emissions by 60%? It is an easy way to fight #ClimateChange "
                        f"#ManyLabsClimate{cond_code} source: https://econ.st/3qjvOnn"
                    ),
                    Text(
                        "Are you willing to share this information (above) on your social media?"
                    ),
                    Text(
                        "If yes, please do it now, by copying and pasting the entire message."
                    ),
                    Choice(
                        slot="Share",
                        options=[
                            "Yes, I am willing to share this information.",
                            "I do not use social media.",
                            "I'm not willing to share that.",
                        ],
                    ),
                ],
            ),
            Screen(
                timer="share_timer2",
                condition=(
                    "shown only to respondents who said they were willing to share"
                ),
                gate=Gate("Share", "Yes, I am willing to share this information."),
                elements=[
                    Text(
                        "Please select the platform you posted it on (select all that apply):"
                    ),
                    MultiChoice(
                        slot="Share2",
                        options=[
                            "Twitter",
                            "Facebook",
                            "Instagram",
                            "Other (please specify)",
                            "I am not willing to share this information on social media",
                        ],
                        other_slot="Share2_4_TEXT",
                        allow_none=False,
                    ),
                ],
            ),
        ],
        note=(
            "The '#ManyLabsClimate<n>' tag was piped from the condition code, so the "
            "post text differs by one character between conditions."
        ),
    )


# --------------------------------------------------------------------------- #
# 4. WEPT tree-planting task
# --------------------------------------------------------------------------- #

_WEPT_DEMO_ROWS = (
    (19, 54, 67, 71, 85, 44, 14, 92, 75),
    (74, 78, 73, 24, 23, 26, 81, 75, 64),
)

#: The six rows of ten numbers that WEPT pages 2-8 are drawn from.
_ROW = {
    "A": (21, 61, 97, 46, 98, 54, 17, 96, 74, 38),
    "B": (43, 25, 54, 75, 61, 63, 56, 31, 19, 63),
    "C": (69, 39, 66, 96, 21, 73, 33, 97, 51, 98),
    "D": (38, 87, 29, 81, 26, 32, 14, 16, 52, 64),
    "E": (19, 26, 47, 71, 91, 46, 56, 53, 39, 13),
    "F": (62, 84, 46, 13, 38, 28, 45, 34, 41, 68),
}

#: WEPT page 1 has its own number set; pages 2-8 permute the six rows above.
_WEPT_PAGE_ROWS: dict[int, tuple[tuple[int, ...], ...]] = {
    1: (
        (64, 15, 82, 83, 58, 98, 96, 23, 69, 12),
        (48, 21, 68, 13, 19, 63, 24, 27, 22, 63),
        (18, 88, 37, 73, 39, 66, 43, 27, 93, 22),
        (91, 59, 26, 52, 53, 37, 48, 44, 86, 13),
        (18, 74, 44, 65, 22, 63, 78, 43, 71, 57),
        (42, 74, 63, 92, 31, 58, 73, 28, 27, 56),
    ),
    2: tuple(_ROW[k] for k in "ABCDEF"),
    3: tuple(_ROW[k] for k in "ACDFEB"),
    4: tuple(_ROW[k] for k in "ADEFBC"),
    5: tuple(_ROW[k] for k in "ADEFBC"),
    6: tuple(_ROW[k] for k in "EDFBCA"),
    7: tuple(_ROW[k] for k in "CFEBDA"),
    8: tuple(_ROW[k] for k in "EAFBDC"),
}

#: The donation sentence is worded slightly differently on some pages.
_WEPT_OFFER = {
    1: (
        "The next page will contain 60 numbers, and if you complete this page we "
        "will donate 1 tree to the Eden Reforestation Project."
    ),
    2: (
        "The next page will contain 60 numbers, and if you complete it we will "
        "donate another 1 tree to the Eden Reforestation Project."
    ),
    5: (
        "The next page will contain 60 numbers, and if you complete it we will "
        "donate another 1 tree to the Eden Reforestation Project."
    ),
}
_WEPT_OFFER_DEFAULT = (
    "The next page will contain 60 numbers, and if you complete it we will donate "
    "another tree to the Eden Reforestation Project."
)

_WEPT_TASK_PROMPT = (
    "Identify all those stimuli with an even first digit and an odd second digit. "
    'For example, you should click on "25", because the first digit (2) is even and '
    "the second digit is odd (5)."
)

_WEPT_CAREFUL = (
    "If you decide to complete this page, please do so thoroughly because we can "
    "only count pages that are at least 90% correct. We will not give you feedback, "
    "so please check whether your answers are correct before proceeding to the next "
    "page."
)

_WEPT_DECIDE = (
    "Do you want to complete this page? If you click no, you will proceed to the end "
    "of the survey, and not be allowed to complete any more number identification "
    "tasks."
)


def _tree_pictogram(filled: int) -> Image:
    empty = 8 - filled
    return Image(
        alt=(
            "A pictogram of eight simple tree outlines, arranged four over four. "
            f"{filled} of them are filled in solid green and the remaining {empty} are "
            "left as plain outlines, marking progress through the task."
        )
    )


def _wept_pages() -> list[Screen]:
    screens: list[Screen] = []
    # The two page timers per WEPT page, in the order they appear in the data.
    decision_timers = ["Q46", "Q50", "Q98", "Q102", "Q106", "Q110", "Q114", "Q118"]
    grid_timers = ["Q48", "Q52", "Q100", "Q104", "Q108", "Q112", "Q116", "Q120"]
    for i in range(1, 9):
        screens.append(
            Screen(
                timer=decision_timers[i - 1],
                condition=(
                    None if i == 1 else f"reached only if page {i - 1} was accepted"
                ),
                gate=None if i == 1 else Gate(f"WEPT{i - 1}confirm", "yes"),
                elements=[
                    Text(_WEPT_OFFER.get(i, _WEPT_OFFER_DEFAULT)),
                    _tree_pictogram(i),
                    Text(_WEPT_CAREFUL),
                    Text(_WEPT_DECIDE),
                    Choice(slot=f"WEPT{i}confirm", options=["yes", "no"]),
                ],
            )
        )
        screens.append(
            Screen(
                timer=grid_timers[i - 1],
                condition=f'shown only if the answer to WEPT{i}confirm was "yes"',
                gate=Gate(f"WEPT{i}confirm", "yes"),
                elements=[
                    Text(_WEPT_TASK_PROMPT),
                    NumberGrid(
                        slots=[f"WEPT{i}nums_{r}" for r in range(1, 7)],
                        rows=_WEPT_PAGE_ROWS[i],
                    ),
                ],
            )
        )
    return screens


WEPT = Block(
    key="wept",
    title="Tree-planting task",
    screens=[
        Screen(
            timer="WEPTTimer_0",
            elements=[
                Text(
                    "Instructions: We would like you to complete a number "
                    "identification task. Below, you see a series of two-digit "
                    "numbers. You will need to click the boxes below the target "
                    "numbers. Target numbers are all numbers that consist of an even "
                    "first digit (i.e., 2, 4, 6, 8) and an odd second digit (i.e., 1, "
                    '3, 5, 7, 9). For example, "25" or "83" would be target numbers, '
                    'but "17", "42", or "56" would not be target numbers.'
                ),
                Text(
                    "Please identify all numbers with an even first digit and an odd "
                    "second digit. We will give you feedback if you missed something."
                ),
                NumberGrid(
                    slots=["WEPTdemo1_1", "WEPTdemo2_1"],
                    rows=_WEPT_DEMO_ROWS,
                    randomised=False,
                ),
            ],
            condition="the page could not be advanced until both rows were answered correctly",
        ),
        Screen(
            timer="WEPTtreeTimer",
            elements=[
                Text("Great job!"),
                Text(
                    "Did you know that planting trees is one of the best ways to fight "
                    "climate change? As trees grow, they remove carbon dioxide (a "
                    "greenhouse gas) from the air. They store the carbon in the trees "
                    "and soil, and then release oxygen into the air."
                ),
                Text(
                    "In the following pages, you will have the option to complete "
                    "additional pages of the number-identification task. For each page "
                    "that you correctly complete, we will make a donation of one tree "
                    "to the Eden Reforestation Project, an organization that has "
                    "planted over 830 million trees since its inauguration."
                ),
                Image(
                    alt=(
                        "The logo of Eden Reforestation Projects: a drawing of a "
                        "broad-canopied green tree on sandy ground, beside the words "
                        "'Eden Reforestation Projects'. Next to it, a small square "
                        "badge reading 'Platinum Transparency 2022 - Candid'."
                    )
                ),
                Text(
                    "Please note, these trees will actually be planted in the real "
                    "world, in the future. The more pages you complete, the more trees "
                    "will be planted!"
                ),
                Text(ADVANCE),
            ],
        ),
        Screen(
            timer="WEPTtimer2",
            elements=[
                Text(
                    "It is up to you to decide how much time and effort you want to "
                    "invest in the task. There are a maximum of 8 pages that you can "
                    "complete, and each page will contain 60 numbers."
                ),
                Text(
                    "For each page that you complete, one tree will be planted on your "
                    "behalf. Please note, this is completely voluntary, and completion "
                    "of this task will not impact your compensation in anyway. If you "
                    'want, you can decline checking the numbers (by clicking "no") and '
                    "go directly to the next part of the study. However, please do not "
                    "simply close the survey before you have reached the end of it."
                ),
                Text(ADVANCE),
            ],
        ),
        *_wept_pages(),
    ],
)


# --------------------------------------------------------------------------- #
# 5. extra measures, control condition only
# --------------------------------------------------------------------------- #

CONTROL_EXTRA_IVS = Block(
    key="control_ivs",
    title="Additional measures (control condition only)",
    screens=[
        Screen(
            timer="control_timer1",
            elements=[
                Slider(
                    slot="Trust_sci1_1",
                    stem="On average, how competent are climate change research scientists?",
                    left="Not at all",
                    right="Very much so",
                    extra="No Opinion",
                )
            ],
        ),
        Screen(
            timer="control_timer2",
            elements=[
                Slider(
                    slot="Trust_sci2_1",
                    stem=(
                        "On average, how much do you trust scientific research about "
                        "climate change?"
                    ),
                    left="Not at all",
                    right="Very much so",
                    extra="No Opinion",
                )
            ],
        ),
        Screen(
            timer="control_timer3",
            elements=[
                Slider(
                    slot="Trust_gov_1",
                    stem="On average, how much do you trust your government?",
                    left="Not at all",
                    right="Very much so",
                    extra="No Opinion",
                )
            ],
        ),
        Screen(
            timer="control_timer4",
            elements=[
                Slider(
                    slot="ID_hum_1",
                    stem=(
                        "To what degree do you see yourself as someone who cares about "
                        "human welfare?"
                    ),
                    left="Not at all",
                    right="Very much so",
                )
            ],
        ),
        Screen(
            timer="control_timer5",
            elements=[
                Slider(
                    slot="ID_GC_1",
                    stem="To what degree do you think of yourself as a global citizen?",
                    left="Not at all",
                    right="Very much so",
                )
            ],
        ),
        Screen(
            timer="control_timer6",
            elements=[
                Text("To what degree ..."),
                Matrix(
                    left="Not at all",
                    right="Very much so",
                    randomised=True,
                    items=[
                        (
                            "Enviro_ID_1",
                            "do you see yourself as someone who cares about the natural "
                            "environment",
                        ),
                        (
                            "Enviro_ID_2",
                            "are you pleased to be someone who cares about the natural "
                            "environment",
                        ),
                        (
                            "Enviro_ID_3",
                            "do you feel strong ties with others who care about the "
                            "natural environment",
                        ),
                        (
                            "Enviro_ID_4",
                            "do you identify with others who care about the natural "
                            "environment",
                        ),
                    ],
                ),
            ],
        ),
        Screen(
            timer="control_timer7",
            elements=[
                Text(
                    "Please rate the degree to which you agree/disagree with the "
                    "following statements about yourself:"
                ),
                Matrix(
                    left="Strongly disagree",
                    right="Strongly agree",
                    randomised=True,
                    items=[
                        (
                            "Enviro_motiv_1",
                            "Because of today's politically correct standards, I try to "
                            "appear pro-environmental.",
                        ),
                        (
                            "Enviro_motiv_11",
                            "I try to hide my negative thoughts about pro-environmental "
                            "behavior in order to avoid negative reactions from others.",
                        ),
                        (
                            "Enviro_motiv_12",
                            "If I acted anti-environmental, I would be concerned that "
                            "others would be angry with me.",
                        ),
                        (
                            "Enviro_motiv_13",
                            "I attempt to appear pro-environmental in order to avoid "
                            "disapproval from others.",
                        ),
                        (
                            "Enviro_motiv_14",
                            "I try to act pro-environmental because of pressure from others.",
                        ),
                        (
                            "Enviro_motiv_15",
                            "I attempt to behave pro-environmentally because it is "
                            "personally important to me.",
                        ),
                        (
                            "Enviro_motiv_16",
                            "According to my personal values, acting non-environmental is OK.",
                        ),
                        (
                            "Enviro_motiv_17",
                            "I am personally motivated by my beliefs to be pro-environmental.",
                        ),
                        (
                            "Enviro_motiv_18",
                            "Because of my personal values, I believe that acting "
                            "anti-environmental is wrong.",
                        ),
                        (
                            "Enviro_motiv_20",
                            "Being pro-environmental is important to my self-concept.",
                        ),
                    ],
                ),
            ],
        ),
        Screen(
            timer="control_timer8",
            elements=[
                Text(
                    "Think for a moment about people from your country and their views "
                    "on climate change."
                ),
                Text(
                    "What percentage of people in your country do you think would agree "
                    'with the statement "Climate change is a global emergency"?'
                ),
                Slider(
                    slot="PlurIgnoranceItem_1",
                    left="% Who Agree",
                    right="",
                    stem="Enter your Guess Here",
                ),
            ],
        ),
    ],
)

#: The nine wordings of the terms-probing item.  Each control participant saw
#: exactly one of them, chosen at random.
TERMS_PROBING_ITEMS = [
    ("probe_CC_1", "To what degree are you willing to act to prevent climate change?"),
    ("probe_GW_1", "To what degree are you willing to act to prevent global warming?"),
    ("probe_GH_1", "To what degree are you willing to act to prevent global heating?"),
    (
        "probe_CCrisis_1",
        "To what degree are you willing to act to prevent the climate crisis?",
    ),
    (
        "probe_GE_1",
        "To what degree are you willing to act to reduce the greenhouse effect?",
    ),
    ("probe_CE_1", "To what degree are you willing to act to reduce carbon emissions?"),
    (
        "probe_CP_1",
        "To what degree are you willing to act to reduce greenhouse gasses?",
    ),
    (
        "probe_CEmerg_1",
        "To what degree are you willing to act to prevent the climate emergency?",
    ),
    (
        "probe_CPoll_1",
        "To what degree are you willing to act to reduce carbon pollution?",
    ),
]


def terms_probing_block(item_index: int = 0) -> Block:
    """One randomly drawn terms-probing item (control condition only)."""
    slot, stem = TERMS_PROBING_ITEMS[item_index]
    return Block(
        key="terms_probing",
        title="Terms probing (control condition only)",
        screens=[
            Screen(
                timer="ControlIV_timer",
                elements=[
                    Slider(
                        slot=slot, stem=stem, left="Not at all", right="Very much so"
                    )
                ],
            )
        ],
        note=(
            "Qualtrics drew one of nine wordings at random per participant; the other "
            "eight are listed in TERMS_PROBING_ITEMS. Verified in the data: every "
            "control participant answered exactly zero or one of the nine."
        ),
    )


# --------------------------------------------------------------------------- #
# 6. second attention check, demographics, debriefing
# --------------------------------------------------------------------------- #


def demographics_block(profile: CountryProfile) -> Block:
    return Block(
        key="demographics",
        title="Demographics",
        screens=[
            Screen(
                timer="demographic_timer1",
                elements=[
                    Text(
                        "In the previous section you viewed some information about "
                        "climate change. To indicate you are reading this paragraph, "
                        "please type the word sixty in the text box below."
                    ),
                    FreeText(slot="Attn_60", hint="Type your answer in the box below"),
                ],
            ),
            Screen(
                timer="demographic_timer2",
                elements=[
                    Text("DEMOGRAPHICS", style="head"),
                    Text(
                        "The following section includes some questions about your "
                        "background and demographics. These questions may not seem "
                        "particularly relevant to the tasks that you completed today. "
                        "However, knowing the demographics of the people who take part "
                        "in our research helps us understand who our participant sample "
                        "represents. This is important in understanding the extent to "
                        "which our findings might be specific to certain groups of "
                        "people (e.g., undergraduate students), or whether they might "
                        "generalize to wider populations."
                    ),
                ],
            ),
            Screen(
                timer="demo_timer4",
                elements=[
                    Text("What is your gender?"),
                    Choice(
                        slot="Gender",
                        options=[
                            "Prefer not to say",
                            "Male",
                            "Non-binary/third gender/other",
                            "Female",
                        ],
                        other_slot="Gender_4_TEXT",
                        randomised=True,
                    ),
                ],
            ),
            Screen(
                timer="demo_timer5",
                elements=[
                    Text("How old are you?  (please enter a number)"),
                    Number(slot="Age", hint="Enter your age in years"),
                ],
            ),
            Screen(
                timer="demo_timer6",
                elements=[
                    Text("How many years of formal education have you completed?"),
                    Choice(slot="Education.2", options=list(profile.education_options)),
                ],
            ),
            Screen(
                timer="demo_timer7",
                elements=[
                    Text(
                        "What is your political orientation for the issues listed below?"
                    ),
                    Text(
                        'Please note, by "liberal" we mean classically left-wing, and by '
                        '"conservative", we mean classically right-wing.'
                    ),
                    Matrix(
                        left="Extremely liberal/left-wing",
                        mid="Moderate",
                        right="Extremely conservative/right-wing",
                        extra="Prefer not to respond",
                        randomised=True,
                        items=[
                            (
                                "Politics2_1",
                                "For social issues (e.g., health care, education, etc.)",
                            ),
                            ("Politics2_9", "For economic issues (e.g., taxes)"),
                        ],
                    ),
                ],
            ),
            Screen(
                timer="demo_timer8",
                elements=[
                    Text(
                        "We are also interested in learning about you/your family. "
                        "Please answer the following questions to the best of your "
                        "abilities:"
                    ),
                    Text("What is your total yearly family/household income?"),
                    Choice(slot="Income", options=list(profile.income_options)),
                ],
            ),
            Screen(
                timer="demo_timer9",
                elements=[
                    Text(
                        "Do you own/have access to these items in your home? (check all "
                        "that apply)"
                    ),
                    MultiChoice(
                        slot="Indirect_SES",
                        options=[
                            "Washing machine",
                            "Separate room for kitchen",
                            "Television",
                            "Freezer/deep freeze",
                            "Vacuum cleaner",
                            "Personal computer",
                            "Bathroom",
                        ],
                        randomised=True,
                    ),
                ],
            ),
            Screen(
                timer="demo_timer10",
                elements=[
                    Text(
                        "Instructions: Think of this ladder as representing where people "
                        f"stand in {profile.country}. At the top of the ladder are the "
                        "people who are the best off - those who have the most money, "
                        "the most education, and the most respected jobs. At the bottom "
                        "are the people who are the worst off - those who have the least "
                        "money, least education, the least respected jobs, or no job. "
                        "The higher up you are on this ladder, the closer you are to the "
                        "people at the very top; the lower you are, the closer you are "
                        "to the people at the very bottom."
                    ),
                    Image(
                        alt=(
                            "A drawing of a grey ten-runged ladder, standing at a slight "
                            "angle and leaning to the left."
                        )
                    ),
                    Text("Where would you place yourself on this ladder?"),
                    Text(
                        "Please choose the rung where you think you stand at this time "
                        f"in your life relative to other people in {profile.country}."
                    ),
                    Choice(
                        slot="MacArthur_SES",
                        options=[
                            "Rung 10 (Top) People here are the best off",
                            "Rung 9",
                            "Rung 8",
                            "Rung 7",
                            "Rung 6",
                            "Rung 5",
                            "Rung 4",
                            "Rung 3",
                            "Rung 2",
                            "Rung 1 (Bottom) People here are the worst off",
                        ],
                    ),
                ],
            ),
            Screen(
                timer="demo_timer11",
                elements=[
                    Text(
                        "To the best of your knowledge, what percentage of climate "
                        "scientists have concluded that human-caused climate change is "
                        "happening?"
                    ),
                    Slider(
                        slot="PerceivedSciConsensu_1",
                        left="0",
                        right="100",
                        stem="Percentage",
                    ),
                ],
            ),
            Screen(
                timer="demo_timer12",
                elements=[
                    Text(
                        "Thank you for your participation in our survey. Please let us "
                        "know if you have any comments, questions, or concerns via the "
                        "text box below:"
                    ),
                    FreeText(slot="Comments_pilot"),
                ],
            ),
        ],
    )


DEBRIEF = Block(
    key="debrief",
    title="Debriefing",
    screens=[
        Screen(
            timer="debrief_timer",
            elements=[
                Text("DEBRIEFING FORM", style="head"),
                Text("TITLE OF RESEARCH: Understanding Climate Action"),
                Text("INVESTIGATOR: Prof. Jay Van Bavel"),
                Text("IRB-FY2022-6272"),
                Text("Dear Participant,"),
                Text(
                    "In this study, we are interested in the mechanisms leading people "
                    "to take action against the current climate crisis. To investigate, "
                    "we randomly assigned participants to one of 12 conditions, each of "
                    "them testing a different intervention aimed at stimulating climate "
                    "action (for example, spending time on the tree planting task, "
                    "supporting climate policy, or even belief in climate change). "
                    "Ultimately, the results from this study will allow us to not only "
                    "compare the efficacy of these interventions, but determine which "
                    "ones are the best at promoting different facets of sustainability. "
                    "We are sorry that it was necessary to hide the full experimental "
                    "design from you, but it was necessary that you were naive to this "
                    "experimental design."
                ),
                Text(
                    "Additionally, it should be noted that all information, and figures "
                    "displayed throughout this survey were valid, and based off factual "
                    "information. Thus, we did not expose you to any lies, or deceit "
                    "when it comes to the climate change related information contained "
                    "in this study."
                ),
                Text("Thank you for your participation."),
            ],
        )
    ],
    note="Teams were told to replace or drop this form according to their own ethics approval.",
)


__all__ = [
    "ADVANCE",
    "ADVANCE_DOT",
    "BELIEF",
    "CONSENT",
    "CONTROL_EXTRA_IVS",
    "DEBRIEF",
    "INTRO",
    "POLICY",
    "TERMS_PROBING_ITEMS",
    "WEPT",
    "demographics_block",
    "sharing_block",
    "terms_probing_block",
    "Echo",
    "Bullets",
]
