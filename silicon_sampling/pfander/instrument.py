"""The Silicon Sample Benchmark instrument, as structured transcript elements.

Wording is verbatim from ``survey/questionnaire.txt`` (chronological order,
Qualtrics variable names and legal codes annotated per item), cross-checked
against the rendered ``questionnaire.html``.  Where the two disagree,
``questionnaire.txt`` wins — it is the file the benchmark ships for exactly this
purpose, and it is explicit about the two places the rendered page misleads:
the on-screen party order is Republican / Independent / Democrat / Other, and
race is a single choice ("which race / ethnicity you *most* identify as").

Slot ids are the Qualtrics variable names.  Translation to submission column
names happens once, at export, via ``codebook.csv``.
"""

from __future__ import annotations

from typing import Mapping

from ..survey.elements import Block, Conditional, PageBreak, Text
from ..survey.slots import ChoiceSlot, FreeTextSlot, IntSlot

# --------------------------------------------------------------------------- #
# shared wording
# --------------------------------------------------------------------------- #

SLIDER_BLURB = (
    "Below is a range from 0 to 100. Click on any space within this range and a bar will appear. "
    "Feel free to move that bar around to the number that best represents your answer."
)

AGREE_7 = "1 = Strongly disagree, 2 = Disagree, 3 = Somewhat disagree, 4 = Neither agree nor disagree, 5 = Somewhat agree, 6 = Agree, 7 = Strongly agree."

SURVEY_YEAR = 2026

STATES = (
    "Alabama, Alaska, Arizona, Arkansas, California, Colorado, Connecticut, Delaware, Florida, Georgia, Hawaii, Idaho, "
    "Illinois, Indiana, Iowa, Kansas, Kentucky, Louisiana, Maine, Maryland, Massachusetts, Michigan, Minnesota, "
    "Mississippi, Missouri, Montana, Nebraska, Nevada, New Hampshire, New Jersey, New Mexico, New York, North Carolina, "
    "North Dakota, Ohio, Oklahoma, Oregon, Pennsylvania, Rhode Island, South Carolina, South Dakota, Tennessee, Texas, "
    "Utah, Vermont, Virginia, Washington, West Virginia, Wisconsin, Wyoming, Washington, D.C."
).split(", ")
# "Washington, D.C." survives the split above as two fragments; rebuild it.
STATES = tuple(STATES[:-2] + ["Washington, D.C."])
STATE_OPTIONS = STATES + ("Prefer not to say",)


def slider(slot_id: str, prompt: str, anchors: str, **kwargs) -> IntSlot:
    """A 0-100 Qualtrics slider.  Integers only, per the codebook."""
    return IntSlot(
        id=slot_id, prompt=prompt, anchors=anchors, lo=0, hi=100, max_tokens=6, **kwargs
    )


def likert7(slot_id: str, prompt: str) -> IntSlot:
    return IntSlot(id=slot_id, prompt=prompt, anchors=AGREE_7, lo=1, hi=7, max_tokens=4)


# --------------------------------------------------------------------------- #
# consent and introduction
# --------------------------------------------------------------------------- #

CONSENT = Block(
    key="consent",
    title="Consent form",
    elements=[
        Text("Consent form", style="head"),
        Text(
            "You are invited to participate in a research study that will ask about your opinions and attitudes. "
            "You must be at least 18 years of age to participate. There are no risks associated with this study and "
            "your identity will be kept confidential. We cannot and do not guarantee or promise that you will receive "
            "any benefits from this study."
        ),
        Text("Participation"),
        Text(
            "If you decide to participate in this project, please understand your participation is voluntary and you "
            "may withdraw your consent or discontinue participation at any time without penalty. The alternative is "
            "not to participate. You have the right to refuse to answer particular questions. Your individual privacy "
            "will be maintained in all published and written data resulting from the study."
        ),
        Text("Contact Information"),
        Text(
            "If you have any questions, concerns or complaints about this research, its procedures, risks and "
            "benefits, contact the Protocol Director, Madalina Vlasceanu at mov@stanford.edu. If you wish to contact "
            "someone independent of the researchers, you may email the Stanford Institutional Review Board at "
            "irbnonmed@stanford.edu."
        ),
        Text(
            "If you agree to participate in this research, please click to the next screen and complete the questionnaire."
        ),
        PageBreak(),
        Text(
            "You will need to qualify for this study. You will find out if you qualify shortly."
        ),
        Text(
            "This study takes about 15 minutes to complete. It has several sections requiring attention. To get paid, "
            "please participate until the very end of this study."
        ),
        Text(
            "For us to be able to use your responses, it is crucial that you help and answer the following questions honestly."
        ),
        ChoiceSlot(
            id="filter",
            prompt="Do you agree to pay attention and participate in all sections of this study?",
            options=("Yes", "No"),
            source="prefilled",
            max_tokens=4,
        ),
        ChoiceSlot(
            id="filter_ai",
            prompt=(
                "Do you agree to complete this survey without the use of artificial intelligence (AI) tools "
                "(e.g., ChatGPT) or other automated systems?"
            ),
            options=("Yes", "No"),
            source="prefilled",
            max_tokens=4,
        ),
    ],
)

# --------------------------------------------------------------------------- #
# demographics
# --------------------------------------------------------------------------- #

DEMOGRAPHICS = Block(
    key="demographics",
    title="Demographics",
    elements=[
        PageBreak(),
        Text("Demographics", style="head"),
        ChoiceSlot(
            id="gender",
            prompt="What is your gender?",
            options=("Male", "Female", "Other"),
            source="prefilled",
            max_tokens=4,
        ),
        IntSlot(
            id="year_birth",
            prompt="What is your year of birth?",
            anchors="Four-digit year.",
            lo=1900,
            hi=SURVEY_YEAR - 18,
            source="prefilled",
            max_tokens=6,
        ),
        ChoiceSlot(
            id="race",
            prompt="Please select which race / ethnicity you most identify as.",
            options=(
                "White / Caucasian",
                "Black / African-American",
                "Latino / Hispanic",
                "Asian / Asian-American",
                "Other",
            ),
            source="prefilled",
            max_tokens=10,
        ),
        ChoiceSlot(
            id="education",
            prompt="What is the highest level of school that you have completed?",
            options=(
                "Less than high school",
                "High school diploma / GED",
                "Some college or Associate's degree",
                "Bachelor's degree",
                "Master's degree / Professional degree",
                "Doctorate degree / Ph.D.",
            ),
            max_tokens=14,
        ),
        PageBreak(),
        Text("Have you, personally, ever learned about climate science at…"),
        ChoiceSlot(
            id="education_climate_1",
            prompt="Primary school",
            options=("Yes", "No", "Not applicable"),
            max_tokens=6,
        ),
        ChoiceSlot(
            id="education_climate_2",
            prompt="High school",
            options=("Yes", "No", "Not applicable"),
            max_tokens=6,
        ),
        ChoiceSlot(
            id="education_climate_3",
            prompt="College / University",
            options=("Yes", "No", "Not applicable"),
            max_tokens=6,
        ),
        PageBreak(),
        ChoiceSlot(
            id="income",
            prompt="What is your total yearly family/household income before taxes?",
            options=(
                "Less than $30,000",
                "$30,000 to $55,999",
                "$56,000 to $99,999",
                "$100,000 to $167,999",
                "$168,000 or more",
            ),
            max_tokens=14,
        ),
        ChoiceSlot(
            id="household",
            prompt="How many people currently live in your household, including yourself?",
            options=("1", "2", "3", "4", "5", "6 or more"),
            max_tokens=6,
        ),
        ChoiceSlot(
            id="social_class",
            prompt="How would you describe your social class?",
            options=("Lower class", "Working class", "Middle class", "Upper class"),
            max_tokens=6,
        ),
        ChoiceSlot(
            id="rural",
            prompt="Which of the following best describes the place where you live?",
            options=(
                "A large city",
                "A suburb near a large city",
                "A small city or town",
                "A rural area",
            ),
            max_tokens=10,
        ),
        FreeTextSlot(
            id="zip_code",
            prompt="What is your zip code? (If you are not willing to share your zip code, insert 99999.)",
            hint="Five digits.",
            pattern=r"\d{5}",
            max_tokens=8,
        ),
        PageBreak(),
        IntSlot(
            id="attention1",
            prompt='To help us keep track of who is paying attention, please select "Somewhat disagree" in the options below.',
            anchors=AGREE_7,
            lo=1,
            hi=7,
            source="prefilled",
            max_tokens=4,
        ),
    ],
)

# --------------------------------------------------------------------------- #
# partisan identity, religion, epistemic autonomy
# --------------------------------------------------------------------------- #


def _is_partisan(answers: Mapping[str, object]) -> bool:
    return answers.get("party") in ("Republican", "Democrat")


def _is_christian(answers: Mapping[str, object]) -> bool:
    return answers.get("religion") in (
        "Protestant",
        "Catholic",
        "Mormon",
        "Orthodox Christian",
    )


def _is_religious(answers: Mapping[str, object]) -> bool:
    religion = answers.get("religion")
    return religion is not None and religion != "I am not religious"


IDENTITY = Block(
    key="identity",
    title="Partisan identity and religion",
    elements=[
        PageBreak(),
        ChoiceSlot(
            id="party",
            prompt="Generally speaking, do you usually think of yourself as a Republican, a Democrat, an Independent, or what?",
            # On-screen order per questionnaire.txt; Qualtrics recodes on export.
            options=("Republican", "Independent", "Democrat", "Other"),
            max_tokens=6,
        ),
        Conditional(
            note='party = "Republican" or "Democrat"',
            predicate=_is_partisan,
            elements=[
                Text(SLIDER_BLURB),
                slider(
                    "partisan_importance",
                    "How important is being a <<=party>> to you?",
                    "0 = Not important at all, 50 = Moderately important, 100 = Extremely important.",
                ),
            ],
        ),
        PageBreak(),
        ChoiceSlot(
            id="religion",
            prompt="What is your religion, if any?",
            options=(
                "I am not religious",
                "Protestant",
                "Catholic",
                "Mormon",
                "Orthodox Christian",
                "Jewish",
                "Muslim",
                "Buddhist",
                "Hindu",
                "Other religion",
            ),
            max_tokens=8,
        ),
        Conditional(
            note='religion is "Protestant", "Catholic", "Mormon" or "Orthodox Christian"',
            predicate=_is_christian,
            elements=[
                ChoiceSlot(
                    id="religion_bornagain",
                    prompt='Would you describe yourself as a "born-again" or evangelical Christian?',
                    options=("Yes", "No"),
                    max_tokens=4,
                )
            ],
        ),
        Conditional(
            note='religion is not "I am not religious"',
            predicate=_is_religious,
            elements=[
                Text(SLIDER_BLURB),
                slider(
                    "religiosity",
                    "How religious are you?",
                    "0 = Not at all religious … 100 = Extremely religious.",
                ),
            ],
        ),
        PageBreak(),
        Text(
            "Please indicate how much you agree or disagree with the following statements."
        ),
        likert7("epist_auton_1", "I like to think things through for myself."),
        likert7("epist_auton_2", "I like to figure things out for myself."),
        likert7("epist_auton_3", "I like to make up my own mind about things."),
        likert7(
            "epist_auton_4",
            "I only believe something if I can see for myself that it is true.",
        ),
        likert7(
            "epist_auton_5",
            "I don't go along with the opinions of others without thinking things through for myself.",
        ),
        likert7(
            "epist_auton_6",
            "I have never really questioned the things I have been taught to believe.",
        ),
        PageBreak(),
        FreeTextSlot(
            id="attention2",
            prompt=(
                "Imagine you are playing video games with a friend and at some point your friend says: "
                "\"I don't want to play this game anymore! To make sure that you read the instructions, please write "
                "the word 'attention' in the box below. I really dislike this game, it's the most overrated game "
                'ever." Do you agree with your friend?'
            ),
            hint="Free text.",
            source="prefilled",
            max_tokens=12,
        ),
    ],
)

# --------------------------------------------------------------------------- #
# pre-treatment measures
# --------------------------------------------------------------------------- #

PRE_TREATMENT = Block(
    key="pre_treatment",
    title="Pre-treatment measures",
    elements=[
        PageBreak(),
        Text("Thank you, you have qualified for the study."),
        Text(
            "In the following sections, we're interested in your opinion about climate change and climate scientists."
        ),
        Text(
            "Climate scientists study changes in the Earth's climate over time and how they might affect the planet in "
            "the future. Please keep this definition in mind when filling out this study."
        ),
        Text(
            "Please make sure you do not close this tab until you have finished the study. You must complete the whole "
            "study to collect your payment."
        ),
        PageBreak(),
        Text(SLIDER_BLURB),
        slider(
            "belief_pre",
            'How accurate do you think this statement is? "Human activities are causing climate change."',
            "0 = Not at all accurate … 100 = Extremely accurate.",
        ),
        PageBreak(),
        Text(SLIDER_BLURB),
        slider(
            "trust_pre",
            "How much do you trust climate scientists?",
            "0 = not at all … 100 = very strongly.",
        ),
        PageBreak(),
        Text(
            "Please indicate how much you agree or disagree with the following statements."
        ),
        likert7(
            "alien_inst_1",
            "The prospect of working as a climate scientist has always seemed beyond my reach.",
        ),
        likert7(
            "alien_inst_2",
            "Careers in climate research are accessible only to a privileged few.",
        ),
        likert7(
            "alien_social_1",
            "Climate scientists have a different social background than me.",
        ),
        likert7(
            "alien_social_2",
            "Climate scientists move in different social circles than me.",
        ),
        likert7(
            "alien_spatial_1",
            "Climate science has no positive impact on my local area.",
        ),
        likert7(
            "alien_spatial_2",
            "Very few climate scientists live or work in my local area.",
        ),
        PageBreak(),
        Text(
            "How often do you see or hear information about climate change in the following places?"
        ),
        *[
            IntSlot(
                id=f"alien_info_{index}",
                prompt=label,
                anchors="1 = Never, 2 = Rarely, 3 = Occasionally, 4 = Frequently, 5 = Very frequently.",
                lo=1,
                hi=5,
                max_tokens=4,
            )
            for index, label in enumerate(
                [
                    "Traditional media (e.g., newspapers, TV, radio)",
                    "Online news (e.g., news websites, podcasts, YouTube)",
                    "Social media (e.g., Facebook, TikTok, Instagram)",
                    "Fiction (e.g., films, series, books, comics)",
                    "Personal conversations (e.g., friends/family, text messages, messaging apps)",
                    "In-person events (e.g., museums, public talks)",
                ],
                start=1,
            )
        ],
        PageBreak(),
        Text("You are now moving on to a different section of the study."),
        Text("Please pay close attention to the information you will be provided."),
        Text("Thank you."),
    ],
)

TRANSITION_TO_OUTCOMES = Block(
    key="transition_outcomes",
    title="Transition",
    elements=[
        PageBreak(),
        Text("You are now moving on to the final section of the study."),
        Text("Please answer the following questions to the best of your ability."),
        Text("Thank you."),
    ],
)

# --------------------------------------------------------------------------- #
# post-treatment outcome blocks
# --------------------------------------------------------------------------- #

PRIMARY = Block(
    key="trust_multidimensional",
    title="Trust in climate scientists (primary outcome)",
    elements=[
        PageBreak(),
        Text(
            "Please answer the following questions on how you perceive climate scientists."
        ),
        Text(SLIDER_BLURB),
        slider(
            "trust_competent_1",
            "How incompetent or competent are most climate scientists?",
            "0 = Very incompetent … 100 = Very competent.",
        ),
        slider(
            "trust_intelligent_1",
            "How unintelligent or intelligent are most climate scientists?",
            "0 = Very unintelligent … 100 = Very intelligent.",
        ),
        slider(
            "trust_qualified_1",
            "How unqualified or qualified are most climate scientists?",
            "0 = Very unqualified … 100 = Very qualified.",
        ),
        slider(
            "trust_honest_1",
            "How dishonest or honest are most climate scientists?",
            "0 = Very dishonest … 100 = Very honest.",
        ),
        slider(
            "trust_ethical_1",
            "How unethical or ethical are most climate scientists?",
            "0 = Very unethical … 100 = Very ethical.",
        ),
        slider(
            "trust_sincere_1",
            "How insincere or sincere are most climate scientists?",
            "0 = Very insincere … 100 = Very sincere.",
        ),
        slider(
            "trust_concerned_1",
            "How unconcerned or concerned are most climate scientists about people's wellbeing?",
            "0 = Very unconcerned … 100 = Very concerned.",
        ),
        slider(
            "trust_improve_1",
            "How uneager or eager are most climate scientists to improve others' lives?",
            "0 = Very uneager … 100 = Very eager.",
        ),
        slider(
            "trust_considerate_1",
            "How inconsiderate or considerate are most climate scientists of others' interests?",
            "0 = Very inconsiderate … 100 = Very considerate.",
        ),
        slider(
            "trust_feedback_1",
            "How open, if at all, are most climate scientists to feedback?",
            "0 = Not open at all … 100 = Very open.",
        ),
        slider(
            "trust_transparent_1",
            "How unwilling or willing are most climate scientists to be transparent?",
            "0 = Very unwilling … 100 = Very willing.",
        ),
        slider(
            "trust_attention_1",
            "How much or how little attention do climate scientists pay to other people's views?",
            "0 = Very little attention … 100 = A great deal of attention.",
        ),
    ],
)

FUNDING_PERCEPTIONS = Block(
    key="funding_perceptions",
    title="Funding for research on climate change",
    elements=[
        PageBreak(),
        Text(SLIDER_BLURB),
        slider(
            "funding_5",
            "Do you think the federal government is spending too much, too little or about the right amount of money on climate change research?",
            "0 = far too little, 50 = about the right amount, 100 = far too much.",
        ),
    ],
)

INSTITUTIONAL_TRUST = Block(
    key="inst_trust",
    title="Trust in climate scientific institutions and the government",
    elements=[
        PageBreak(),
        Text("How much do you trust the following institutions?"),
        Text(SLIDER_BLURB),
        slider(
            "inst_trust_epa_1",
            "Environmental Protection Agency (EPA)",
            "0 = not at all … 100 = very strongly.",
        ),
        slider(
            "inst_trust_nasa_1",
            "National Aeronautics and Space Administration (NASA)",
            "0 = not at all … 100 = very strongly.",
        ),
        slider(
            "inst_trust_noaa_1",
            "National Oceanic and Atmospheric Administration (NOAA)",
            "0 = not at all … 100 = very strongly.",
        ),
        slider(
            "inst_trust_uni_1",
            "Universities and colleges",
            "0 = not at all … 100 = very strongly.",
        ),
        slider(
            "inst_trust_gov_1",
            "Federal government",
            "0 = not at all … 100 = very strongly.",
        ),
    ],
)

POLICY_ROLE = Block(
    key="policy_role",
    title="Scientists' role in policy making",
    elements=[
        PageBreak(),
        Text("To what extent do you agree or disagree with the following statements?"),
        Text(SLIDER_BLURB),
        slider(
            "policy_1_1",
            "Climate scientists should work closely with policy makers to integrate scientific results into policy-making.",
            "0 = Strongly disagree … 100 = Strongly agree.",
        ),
        slider(
            "policy_2_1",
            "Climate scientists should actively advocate for specific policies.",
            "0 = Strongly disagree … 100 = Strongly agree.",
        ),
        slider(
            "policy_3_1",
            "Climate scientists should communicate their findings to policy makers.",
            "0 = Strongly disagree … 100 = Strongly agree.",
        ),
        slider(
            "policy_4_1",
            "Climate scientists should be more involved in the policy-making process.",
            "0 = Strongly disagree … 100 = Strongly agree.",
        ),
    ],
)

TRUST_SINGLE = Block(
    key="trust_post",
    title="Single-item trust in climate scientists",
    elements=[
        PageBreak(),
        Text(SLIDER_BLURB),
        slider(
            "trust_post_1",
            "How much do you trust climate scientists?",
            "0 = not at all … 100 = very strongly.",
        ),
    ],
)

DISTRUST_SINGLE = Block(
    key="distrust_post",
    title="Single-item distrust in climate scientists",
    elements=[
        PageBreak(),
        Text(SLIDER_BLURB),
        slider(
            "distrust_1",
            "How much do you distrust climate scientists?",
            "0 = not at all … 100 = very strongly.",
        ),
    ],
)

DONATION = Block(
    key="donation",
    title="Donation behavior",
    elements=[
        PageBreak(),
        Text(
            "On the following page, you will have the opportunity to allocate real money between yourself and a non-profit organization."
        ),
        Text(
            "After data collection is complete, we will randomly select 100 participants from this study to receive a $10 bonus payment."
        ),
        Text(
            "If you are selected, the amount you allocate to yourself will be paid to you as a bonus, and the amount "
            "you allocate to the organization will be donated on your behalf."
        ),
        PageBreak(),
        Text(
            "The organization you can choose to allocate real money to is the American Meteorological Society (AMS), a "
            "non-profit, non-partisan society of 12,000 scientists and other professionals that supports climate change "
            "research. With your donation, you help AMS to advance science for the benefit of society."
        ),
        Text(
            "You may allocate the $10 in any way you like: keep all $10 for yourself, donate all $10 to AMS, or choose any split in between."
        ),
        IntSlot(
            id="donation",
            prompt="Of the $10, how much would you like to donate to the AMS?",
            anchors="Whole dollars, $0 to $10.",
            lo=0,
            hi=10,
            allow_dollar=True,
            max_tokens=6,
        ),
    ],
)

NEWSLETTER = Block(
    key="newsletter",
    title="Subscription climate science newsletter",
    elements=[
        PageBreak(),
        Text("Learn more about climate science", style="head"),
        Text(
            "If you'd like to learn more about climate science and solutions, you can subscribe to the newsletter by "
            'climate scientist Katharine Hayhoe. Her newsletter "Talking Climate" provides short, accessible updates '
            "on climate science and climate solutions for a general audience."
        ),
        Text(
            "Signing up takes less than a minute. Please select the free subscription option — there is no need to choose a paid version."
        ),
        Text(
            "The link below will open the newsletter in a new tab. You can switch back to the current tab and continue the survey right away."
        ),
        Text("[ Open Talking Climate newsletter (opens in a new tab) ]"),
        Text("Note: Subscribing to this newsletter is optional."),
        PageBreak(),
        ChoiceSlot(
            id="newsletter",
            prompt='Did you subscribe to the "Talking Climate" newsletter on the previous page?',
            options=("Yes", "No"),
            max_tokens=4,
        ),
    ],
)

BELIEF_POST = Block(
    key="belief_post",
    title="Belief in climate change",
    elements=[
        PageBreak(),
        Text(SLIDER_BLURB),
        slider(
            "belief_post_1",
            'How accurate do you think this statement is? "Human activities are causing climate change."',
            "0 = not at all accurate … 100 = extremely accurate.",
        ),
    ],
)

CONCERN = Block(
    key="concern",
    title="Climate change concern",
    elements=[
        PageBreak(),
        Text("Please indicate your views on the following questions."),
        Text(SLIDER_BLURB),
        slider(
            "concern_1_1",
            "How concerned are you about climate change?",
            "0 = Not at all … 100 = Extremely.",
        ),
        slider(
            "concern_2_1",
            "How serious a problem is climate change?",
            "0 = Not at all … 100 = Extremely.",
        ),
        slider(
            "concern_3_1",
            "Relative to other issues facing the U.S., how important is climate change?",
            "0 = The least important issue … 100 = The most important issue.",
        ),
    ],
)

BEHAVIOR = Block(
    key="behavior",
    title="Individual-level climate mitigation behavior",
    elements=[
        PageBreak(),
        Text(
            "How likely are you to engage in the following activities in the next twelve months?"
        ),
        Text(SLIDER_BLURB),
        slider(
            "individual_meat_1",
            "Eat less meat",
            "0 = Not likely at all … 100 = Extremely likely.",
        ),
        slider(
            "individual_transport_1",
            "Walk, bicycle, carpool, or take public transportation more often instead of driving by yourself",
            "0 = Not likely at all … 100 = Extremely likely.",
        ),
        slider(
            "individual_solar_1",
            "Install a solar panel",
            "0 = Not likely at all … 100 = Extremely likely.",
        ),
        slider(
            "individual_fly_1",
            "Go on less personal (non-business) air travel",
            "0 = Not likely at all … 100 = Extremely likely.",
        ),
        slider(
            "individual_talk_1",
            "Talk to friends and family about the importance of climate change",
            "0 = Not likely at all … 100 = Extremely likely.",
        ),
        slider(
            "individual_donate_1",
            "Donate to an environmental NGO",
            "0 = Not likely at all … 100 = Extremely likely.",
        ),
    ],
)

POLICY_GENERAL = Block(
    key="policy_general",
    title="Support for climate policies (general)",
    elements=[
        PageBreak(),
        Text(SLIDER_BLURB),
        slider(
            "policy_general_1",
            'How much do you oppose or support the following statement? "The U.S. government should do more to reduce global warming"',
            "0 = Strongly oppose … 100 = Strongly support.",
        ),
    ],
)

POLICY_SPECIFIC = Block(
    key="policy_specific",
    title="Support for climate policies (specific)",
    elements=[
        PageBreak(),
        Text("How much do you support or oppose the following policies?"),
        Text(SLIDER_BLURB),
        slider(
            "policy_specific_1_1",
            "Raising taxes on fossil fuels (e.g., gas, oil, coal)",
            "0 = Strongly oppose … 100 = Strongly support.",
        ),
        slider(
            "policy_specific_2_1",
            "Expanding infrastructure for public transportation",
            "0 = Strongly oppose … 100 = Strongly support.",
        ),
        slider(
            "policy_specific_3_1",
            "Increasing the use of sustainable energy such as wind and solar energy",
            "0 = Strongly oppose … 100 = Strongly support.",
        ),
        slider(
            "policy_specific_4_1",
            "Protecting forested and land areas",
            "0 = Strongly oppose … 100 = Strongly support.",
        ),
        slider(
            "policy_specific_5_1",
            "Increasing taxes on carbon-intensive foods (e.g., beef and dairy products)",
            "0 = Strongly oppose … 100 = Strongly support.",
        ),
        slider(
            "policy_specific_6_1",
            "Investing more in green jobs and businesses",
            "0 = Strongly oppose … 100 = Strongly support.",
        ),
        slider(
            "policy_specific_7_1",
            "Introducing laws to keep waterways and oceans clean",
            "0 = Strongly oppose … 100 = Strongly support.",
        ),
    ],
)

#: Shown first, always.
POST_PRIMARY = PRIMARY

#: Randomised per respondent: the instrument presents the secondary and tertiary
#: outcome blocks in random order.
POST_RANDOMISED = (
    FUNDING_PERCEPTIONS,
    INSTITUTIONAL_TRUST,
    POLICY_ROLE,
    TRUST_SINGLE,
    DISTRUST_SINGLE,
    DONATION,
    NEWSLETTER,
    BELIEF_POST,
    CONCERN,
    BEHAVIOR,
    POLICY_GENERAL,
    POLICY_SPECIFIC,
)

END_OF_SURVEY = Block(
    key="end",
    title="End of survey",
    elements=[
        PageBreak(),
        Text("Thank you for your participation."),
        Text(
            "Please let us know if you encountered any problems with today's study or if you have any thoughts, "
            "questions or comments about this study."
        ),
        FreeTextSlot(
            id="comments",
            prompt="Comments",
            hint="Free text; may be left blank.",
            max_tokens=80,
            max_chars=600,
        ),
        Text(
            "Please proceed to the next page to be redirected to your panel for your payment."
        ),
    ],
)

#: Everything before the condition, in display order.
PRE_CONDITION = (CONSENT, DEMOGRAPHICS, IDENTITY, PRE_TREATMENT)
