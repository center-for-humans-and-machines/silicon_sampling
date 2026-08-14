"""Country-variable parts of the ManyLabs Climate survey.

Collaborators translated the master survey and adapted a small, enumerated set of
strings and images to their own country (see
``data/Vlasceanu/intervention_adaptation_manual.pdf``).  Everything that varied is
a template field here, so the master (United States / English) version renders
verbatim while other national versions stay expressible.

Only the US profile is shipped: it is the one the master survey documents.  The
per-country values for the remaining 62 samples live in the collaborator lookup
tables referenced by the adaptation manual (People's Climate Vote percentages,
regional groupings, local climate-disaster examples, national imagery) and are
not reproduced in ``master_survey.pdf``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CountryProfile:
    """The adaptable slots of the survey for one data-collection country."""

    #: Dataset ``country`` value.
    key: str
    #: Used mid-sentence, e.g. "climate change in {country}".
    country: str
    #: Bare country name, e.g. "the United States remains the United States".
    country_bare: str
    #: Nationality adjective, e.g. "the {adjective} way of life".
    adjective: str
    #: Plural demonym, e.g. "together with other {people}".
    people: str
    #: Region named in the psychological-distance feedback screens.
    region: str

    # -- working-together norms ------------------------------------------- #
    #: Flyer artwork is language-specific; the wording is fixed.
    flyer_language: str = "English"

    # -- correcting pluralistic ignorance --------------------------------- #
    #: "The survey included people from {plurig_text1}."
    plurig_text1: str = "the United States"
    #: "Think for a moment about {plurig_text2} and their views on climate change."
    plurig_text2: str = "Americans"
    #: People's Climate Vote agreement percentage for this country/region.
    plurig_percent: int = 65

    # -- system justification --------------------------------------------- #
    #: Two local climate consequences and their reach, as sentences.
    sysjust_consequence_1: str = (
        "For example, floods are becoming more and more frequent, putting a "
        "quarter of Americans at risk of losing their homes."
    )
    sysjust_consequence_2: str = (
        "Similarly, wildfires are becoming more frequent and more intense, "
        "threatening millions of Americans."
    )

    # -- demographics ----------------------------------------------------- #
    #: Income brackets, eight plus a refusal, in local currency.
    income_options: tuple[str, ...] = (
        "Less than $10,000",
        "$10,000 to $14,999",
        "$15,000 to $24,999",
        "$25,000 to $49,999",
        "$50,000 to $99,999",
        "$100,000 to $149,999",
        "$150,000 to $199,999",
        "$200,000 or more",
        "Prefer not to respond",
    )
    #: Education bands, worded for the local schooling system.
    education_options: tuple[str, ...] = (
        "0-6 (up to grade school/elementary school)",
        "7-12 (up to high school)",
        "13-16 (college/undergraduate university/certificate training)",
        "More than 17 years (doctorate degree, medical degree, etc.)",
        "Prefer not to answer",
    )


UNITED_STATES = CountryProfile(
    key="United States",
    country="the United States",
    country_bare="the United States",
    adjective="American",
    people="Americans",
    region="North America",
)
