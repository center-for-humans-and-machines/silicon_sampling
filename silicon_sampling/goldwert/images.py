"""What the stimulus pictures showed, in words.

Nine of the eleven kept arms display something that is not text, and the previous
audit disposed of all of it with three adjectives — "captioned", "redundant",
"decorative".  Checked against the actual files, those adjectives hold for about
half of the pictures and fail for the other half, and they fail in the direction
that matters: the uncaptioned ones are disproportionately the ones carrying the
manipulation.  ``SystemJustification`` calls its twelve photographs captioned; four
are not, and two of those four are a full-frame US flag and the Macy's Thanksgiving
parade, which is the patriotic prime of a patriotism intervention.
``ThreatInjustEfficacy`` has two *whole screens* whose only content is photographs,
one of them the three children's placards that the next question then asks the
respondent to judge the fairness of.  ``BindingMorals`` asks "how impure do the
Great Smoky Mountains look to you in the picture on the right above" and, in a text
transcript, there was no picture on the right and nothing above.

So every picture is described here, from the picture, and the description goes into
the transcript where the bracketed placeholder used to go.

**Where the files came from.**  Forty of the forty-seven ``<img>`` references in the
eighteen exports still resolve on the hosts the survey hot-linked them from — nine
different Qualtrics brand domains plus Pixabay and two Google user-content CDNs —
and are vendored under ``data/Goldwert/Materials/stimuli`` with
``index.json`` recording provenance per file, so this table can be re-checked
without the network.  Seven references — six distinct files — cannot be recovered at
all: ``CollEfficacyEmoBenefit``'s five and ``LetterFuture``'s one were deleted from
their media libraries and answer 404 on every brand host that ever served this
survey, and the ``.docx`` exports carry only Qualtrics' 1.5 kB
"image unavailable" placeholder in their place.  Those seven are marked
:data:`UNRECOVERABLE` and say so in the transcript rather than being invented.  (One
of them, ``IM_cIwXCbvkxMpchqC``, is the same time-capsule graphic the ICPC package
independently found missing — the two studies share that intervention, which is a
small confirmation that the file really is gone rather than that we asked wrongly.)

**Two conventions, inherited from the ICPC and Pfänder templates.**  Text set
*inside* a picture is transcribed verbatim when the picture carries the
manipulation, because it is on-screen text that happens to have been typeset as an
image.  A decorative photograph gets a plain, flat description and nothing more:
writing "a distressing image of ..." would put the experimenter's reading of the
stimulus into the stimulus.

Keys are the string the ``.qsf`` uses — the Qualtrics graphic id for a
``Graphic.php?IM=...`` link or a hot-linked file's URL — so a key this table has not
seen is a hard, visible failure rather than a silently blank screen.
"""

from __future__ import annotations

#: Pictures whose files are gone from every host that ever served them, with what
#: the surrounding copy says they showed.  Kept apart from :data:`IMAGE_ALT` so that
#: "we looked and it is missing" cannot be confused with "we looked at it".
UNRECOVERABLE: dict[str, str] = {
    # -- 11. collective efficacy and emotional benefit --------------------- #
    # Shown on both branches of the video-sharing feedback screen.
    "IM_9pLdrbSUrk1mtcH": (
        "illustrated the claim that 80% of Americans, not the 60% people assume, "
        "think we should adopt climate-friendly behaviours"
    ),
    "IM_h7x4kGA36EfKkww": "illustrated the Sunrise Movement's campaigning",
    "IM_d5Wtk2iF1PHKKp0": (
        "illustrated climate activism working through policy, on the "
        '"Yes. Indeed!" branch'
    ),
    "IM_b7LLGNKGrxjKxiC": (
        'illustrated the same claim on the "Wrong!" branch of the same question'
    ),
    "IM_y0Deg0XO2pucZNM": (
        "illustrated the closing claim that taking climate action makes people "
        "happier and builds friendships"
    ),
    # -- 17. a letter to future generations ------------------------------- #
    "IM_cIwXCbvkxMpchqC": (
        "illustrated a family on a walk finding a time capsule containing a letter"
    ),
}

#: Qualtrics graphic id, or hot-linked URL, -> what the picture shows.
IMAGE_ALT: dict[str, str] = {
    # -- 2. misperception correction: risks and solutions ----------------- #
    # One news photograph per correction topic, piped onto the writing screen by
    # the respondent's own choice of the issue most disruptive to their life.
    "IM_DYpXMnBecfr2OeR": (
        "A photograph of a long queue of people in office clothes waiting outdoors, "
        "many holding folders and envelopes."
    ),
    "IM_dmKPO7nSWMUWsCD": (
        "A photograph taken from a road running into a wall of brown wildfire smoke, "
        "with fire trucks and a pickup on the carriageway ahead."
    ),
    "IM_HjxNVd2OFSqq18S": (
        "A photograph of a shopper with a full trolley in front of the almost empty "
        'refrigerated meat shelves of a supermarket, under a "natural & organic" '
        "chicken sign."
    ),
    "IM_RtZpMzM5bHN0lNo": (
        "A photograph of a filling-station price board reading, on three lines, "
        "Regular 7.25 9/10, Plus 7.45 9/10, Supreme 7.75 9/10, with a man refuelling "
        "a car behind it."
    ),
    "IM_oIzsGkDMoi2h3wr": (
        "A photograph of an empty classroom: rows of vacant desks, a wall clock, and "
        "no people."
    ),
    # -- 6. system justification ------------------------------------------ #
    "IM_cOx1Lml02s3zzXU": (
        "A photograph of the Chicago River between downtown towers, with a large "
        "United States flag flying in the right foreground."
    ),
    "IM_71zLI3GR0Jx7RLU": (
        "A photograph of the Grand Canyon in bright sun, layered red rock ridges "
        "running to the horizon."
    ),
    "IM_ac0kd3Ld6uR8COy": (
        "A photograph looking down the main street of Telluride, Colorado, towards "
        "the snow-streaked mountains that close off the valley."
    ),
    "https://cdn.pixabay.com/photo/2017/11/02/10/53/field-2910710_960_720.jpg": (
        "A photograph of round straw bales scattered across a cut cornfield, hills "
        "behind."
    ),
    # Google user-content URLs are opaque and long; keyed by the whole string
    # because that is what the export holds.
    "https://lh4.googleusercontent.com/lWvfTDegyBWz6aL4Sk10SZ2fj-h6zTmIzh3j-NLgFluOmZiRV5R87ZcerKDyAzxXJ1tBPyY2GiYNkqnBpsKGAU29vTHfoYNo-wAbz05XLWrkUdUqQRHwDGx5cu_TPZyljwooZ9Nr": (
        "A photograph of a meadow at sunrise with a bicycle leaning against a tree."
    ),
    "IM_7QVH6DEo70xjt1Y": (
        "A photograph of two surfers standing on the sand watching a large breaking "
        "wave."
    ),
    "IM_d0hOzAKRONzOAJM": (
        "A photograph of the flag of the United States filling the frame, flying "
        "from a pole against a blue sky."
    ),
    "IM_8celYrIM4QFYqay": (
        "A photograph of the Macy's Thanksgiving Day Parade: a giant turkey float in "
        "a pilgrim hat, marchers in autumn-leaf costumes, and balloons lettered "
        '"macy\'s".'
    ),
    "IM_5ubDkEZD65re9me": (
        "An aerial photograph of a flooded Houston at dusk: brown water covering "
        "parkland and streets up to the walls of apartment blocks, the downtown "
        "skyline behind."
    ),
    "IM_ehCVnfmEmMpvOGq": (
        "A photograph of a suspension bridge silhouetted against an orange sky, with "
        "a hillside of burning trees glowing behind it."
    ),
    "IM_0AErbjvw9XtREai": (
        "A photograph of Yosemite Valley from the river: boulders and a fallen log in "
        "the foreground, pines, and the valley walls beyond."
    ),
    "IM_7UiRqF4drUNbQ9M": (
        "A photograph of two adults and three children standing in a line with arms "
        "outstretched, fingertip to fingertip, to span the trunk of a giant sequoia."
    ),
    # -- 7. connecting to ecological disruptions --------------------------- #
    "IM_dmPP9U6M625vIto": (
        "A photograph of eight small dead songbirds laid out side by side on a pale "
        "surface, one of them yellow-breasted."
    ),
    "IM_51MO8bRvmwvWYui": (
        "A photograph of a large flock of birds in flight over reeds at sunrise."
    ),
    # The reasoning task: the respondent is asked to read this chart, then to pick
    # which of four contributor charts matches its shape.
    "IM_dba7ygULwYwqv78": (
        "A line chart titled by the surrounding text as global surface temperature. "
        'The y axis is labelled "Anomaly (°C)" and runs from about -0.8 to +0.8; the '
        "x axis runs 1870 to about 2010. A noisy black annual series with a smooth "
        "red trend through it sits near -0.4 until about 1910, crosses the dashed "
        "0.0 baseline around 1940, is flat to about 1975, then climbs steadily to "
        "about +0.5 at the right-hand edge. The years after roughly 1985 are shaded "
        "pink and the earlier years yellow."
    ),
    "IM_d4EH2kiMmmUzro2": (
        "Five line charts on a common 1870-2010 x axis, each with a °C anomaly y "
        'axis and the post-1985 years shaded pink. Top, spanning the width: "Global '
        'Surface Temperature", the noisy series described above, rising to about '
        '+0.5. Below it four smaller panels: "A) Solar Activity", a red curve '
        "oscillating in a regular eleven-year cycle between 0.0 and 0.1 with no "
        'trend; "B) Volcanic Activity", a blue curve at 0.0 punctuated by downward '
        'spikes to about -0.15 in the 1880s, 1960s and 1990s; "C) Human Activity", a '
        "smooth violet curve rising monotonically from 0.0 to about +0.85, steepening "
        'after 1980; and "D) Internal Variability", a green series oscillating '
        "randomly between about -0.2 and +0.2 with no trend."
    ),
    # -- 9. linking individual and structural change ----------------------- #
    "https://lh7-us.googleusercontent.com/KhRWXM_7Lpt1dYF6nI29F76eGCb-8zU3j1m49bruauDx6SdFVtSRWrVh-IwCsuwfjdRuu0xSTf7EyaKB9CnOmbk9Rnfjl3a5u_nFmlTPRmx-K9tEJy0hMmjWkSQluUTAg7YBYramar27gqPM_yCjwpo": (
        'A scatter-and-line chart titled "Women Less Suited for Politics", y axis '
        '"% of Americans" from 0% to 60%, x axis 1974 to about 2021. The points '
        "start near 48%, peak just above 50% in the mid-1970s, fall steeply through "
        "the 1980s to about 22% by 1994, drift sideways in the mid-20s to 2010, and "
        "end at 13%."
    ),
    "IM_eLNJGb7pVzJZX94": (
        "Two choropleth maps of the United States side by side under the heading "
        '"National average of American adults who:". The left map, shaded in oranges '
        'and yellows, is headed "Are worried about climate change: 64%"; the right '
        'map, shaded in blues, "Who discuss climate change at least occasionally: '
        '36%". A shared 0-100% colour scale sits between them and major cities are '
        "marked."
    ),
    "https://lh7-us.googleusercontent.com/ixE-3cOWSQYC1FAajtRr0cJwJ9t33LpeSuOzJMcYe6jWyItZVmHqv6b9tgp45X2xG_LKtQ0spCiRx9Lxe4oRn6fOceV8zzz22G4ss4BAL2uPjBRaKldD736kTfD6IEUtPia7XW2VSbSoAwBTzBS7nrg": (
        "A pictogram of spreading influence: one green human figure at the left, an "
        "arrow to a column of five figures, and an arrow from each of those to a row "
        "of five more, twenty-five in all."
    ),
    "IM_0cAx1CfHwBSypjU": (
        "A two-panel diagram. On the left, a crowd of mostly black figures with a few "
        "green ones, and arrows running up from the crowd to a government building "
        'and to an office tower marked "$$$", both with a black figure on them. An '
        "arrow leads to the right-hand panel, where the building and the tower now "
        "carry green figures, the arrows point back down from them to the crowd, and "
        "most of the crowd has turned green."
    ),
    # -- 10. binding moral foundations ------------------------------------- #
    # The four items on these screens ask the respondent to rate the picture, two of
    # them by pointing at "the picture on the right above".
    "IM_S10HtCTIm7KZBcB": (
        "Two photographs of the same view of the Great Smoky Mountains, side by side "
        "from a fixed camera. Left: ridge behind ridge in clear blue air, the far "
        "skyline sharp. Right: the same ridges washed out in white haze, the far "
        "skyline barely visible."
    ),
    "IM_ezjlteZe9a6zMKW": (
        "A photograph of the Old Faithful geyser at full height, a tall white column "
        "of water and steam above the pale sinter mound, under a cloudy sky."
    ),
    "IM_0fwlB39wuFKTxyu": (
        "A photograph of two adults and three children standing at the foot of a "
        "living giant sequoia, arms outstretched and not reaching round it."
    ),
    "IM_dnUYQdFrBZU2ZZs": (
        "A photograph of a giant sequoia burnt hollow: the trunk is a blackened "
        "shell open to head height, with a firefighter in yellow standing inside it, "
        "and the ground around is bare ash."
    ),
    # -- 13. threat, injustice and efficacy -------------------------------- #
    # These three are a whole screen on their own, immediately before the item
    # asking whether it is fair that the vulnerable suffer most.
    "IM_6RVF0fBqnHKVcjA": (
        "A photograph of a small boy at a demonstration holding a hand-drawn "
        'placard: a crying Earth with a thought bubble lettered "HELP".'
    ),
    "IM_3ws2CKvdgFXx0Xk": (
        "A photograph of a small girl in a pink coat, carried above a crowd, holding "
        'a hand-lettered placard reading "Save Our Earth".'
    ),
    "IM_6XS0zM8aWVJIiSW": (
        "A photograph of a small girl on an adult's shoulders holding a placard "
        'lettered "My future is in your hands" over two red painted handprints and a '
        "drawing of the Earth."
    ),
    "IM_9Rlo2MiiXNHoL0a": (
        "A photograph of three workers in harnesses and tool belts fitting solar "
        "panels to a house roof in bright sun."
    ),
    # -- 14. dynamic anger norm -------------------------------------------- #
    # The trend chart is the dynamic norm itself: the prose gives the 57% endpoint
    # and says "more and more", but only the picture gives the trajectory.
    "IM_d68HgE6XdAphr2S": (
        'A line chart titled "% of Americans who feel angry about US inaction on '
        'Climate Change". The y axis is marked only at 20% and 70%; the x axis only '
        "at 2016 and 2023. A single dark blue line starts a little above 35%, dips "
        "slightly, then rises through the middle years and steepens at the end to a "
        "labelled 57%."
    ),
    "IM_bw35QyYCRE3focC": (
        "A photograph of a child in a face mask cycling along a road in thick orange "
        "haze, with motorcyclists and a truck dimly visible behind."
    ),
    "IM_e4gt7KV6wtq77XU": (
        "A photograph of a heavy industrial plant seen from above, a dozen chimneys "
        "and cooling towers pouring white and yellow smoke across apartment blocks, "
        "mountains behind."
    ),
    "IM_0jKm6EpHj5B6SEu": (
        'A semicircular gauge filled about two-thirds in blue, labelled "61%" in the '
        'centre, captioned "% of Americans who believe the US government should be '
        'doing more to address climate change."'
    ),
    "IM_bKQugdhkQb4F0nY": (
        'A semicircular gauge filled about two-thirds in blue, labelled "68%" in the '
        'centre, captioned "% of Americans who believe corporations and industry '
        'should be doing more to address climate change."'
    ),
    # -- dropped arms, described because the modality audit rests on them --- #
    # This one never reaches a transcript; it is here so that the audit's claim
    # about its arm can be checked against the picture it is about. The four other
    # dropped-arm assets are in EXPORT_LABEL, because nobody has seen them.
    "IM_cwoPQEwiY2j6vWu": (
        "A chart of climate-related health risks accompanying the Lancet global "
        "health frame."
    ),
    # -- the shared battery, shown to every arm ---------------------------- #
    # The MacArthur ladder. Also the picture ICPC's own table describes, from the
    # same NYU asset id, because both instruments use the same SES item.
    "IM_cHZBVvBtL6PePS6": (
        "A drawing of a grey ten-runged ladder standing upright and leaning slightly "
        "to the left, with no labels on it."
    ),
}

#: Assets for which there is no file and never was a URL: Qualtrics ``Graphics``
#: questions whose media library is on a host the export does not name, so the
#: fetcher recorded ``no-url (Graphics question; host unknown)`` and stopped.
#:
#: Kept out of :data:`IMAGE_ALT` for the same reason :data:`UNRECOVERABLE` is kept
#: out of it, and the reason is worth restating because this table got it wrong
#: once.  These four entries used to sit in ``IMAGE_ALT`` phrased as descriptions
#: — "Screenshot 2 of the same New York Times article." — which made
#: ``modality_audit.csv`` count them ``described`` and made the audit's summary
#: sentence, "five assets remain undescribed and all five are videos", false in
#: both halves.  Nobody had opened a file, and the true count with these four
#: moved out is six, still all videos: one Qualtrics-hosted clip and five YouTube
#: ids, every one of them in an arm the audit drops precisely because a recorded
#: talk is its stimulus.  What is actually known is the export's own
#: ``GraphicsDescription`` field, which is the filename the author uploaded:
#: ``"NYT Article final_1.png"``, ``"NYT Article final_2.png"``,
#: ``"NYT Article final_3_4_merged.png"`` and ``"Intervention comb2"``.  Those
#: filenames are evidence and they are recorded as such; the earlier entry for the
#: Co-Benefits asset went further and gave its pixel dimensions and visual style,
#: neither of which is in the export, the ``.docx`` (whose embedded copies are
#: zero-byte placeholders) or anywhere else in the materials.
#:
#: Both arms are dropped, so nothing rendered depends on these; the cost of the
#: error was to the audit's own bookkeeping, which is the thing the audit exists
#: to make checkable.
EXPORT_LABEL: dict[str, str] = {
    "IM_4VMaSOsNCCFfwmW": (
        'the export names it "Intervention comb2" and the arm\'s 73 words of '
        "surrounding text say only to read it, so the infographic is the whole "
        "stimulus"
    ),
    "IM_k3qYGs04M7HJAxb": 'the export names it "NYT Article final_1.png"',
    "IM_sizaDz3eLHlk4cR": 'the export names it "NYT Article final_2.png"',
    "IM_nN9nC8TzJLBAjpD": 'the export names it "NYT Article final_3_4_merged.png"',
}

#: Video and audio assets, keyed by the id or file token the export carries.
#:
#: The control's clip is here for a reason that outranks the rest of this table.
#: The previous audit reasoned that a knot-tying video is "semantically null, so a
#: content-free control screen is a faithful rendering rather than a gap".  The
#: premise is true and the conclusion does not follow: a real control participant
#: spent five minutes being asked to concentrate on something before answering the
#: outcome battery, and a synthetic one who is handed the battery straight away is
#: not a null-treatment respondent, they are a differently-treated one.  Every
#: treatment effect in this study is a contrast against this arm, so a
#: mis-specified control biases all ten of them in the same direction at once.
#: What is faithful is to say what the screen was.
MEDIA_ALT: dict[str, str] = {
    # -- 0. neutral control ------------------------------------------------ #
    # Everything in this description is in the sources. The .qsf gives the two live
    # questions of the arm — the instruction "Please carefully watch the following
    # video (you may be asked about it in the following pages). You will be able to
    # advance the page once this video is over." and a 560x315 YouTube iframe for
    # SeqXZLQFtqQ — and a page timer with MinSeconds 240, parked in that export's
    # trash block. The title is that video's own on YouTube. The five minutes is the
    # paper's Methods ("a 5-min, thematically unrelated video on how to tie knots"),
    # and the control arm's median condDuration of 308 s agrees with it. There are
    # no comprehension items: the arm's only other questions are an empty
    # "Click to write the question text" stub and that timer, both in the trash, so
    # "you may be asked about it" was an inducement to attend and nothing more.
    "SeqXZLQFtqQ": (
        'A five-minute instructional video titled "4 Easy Knots - Knots you can tie '
        "when you don't know how to tie knots!\", demonstrating four knots and how to "
        "tie each one. It says nothing about climate change or about anything else in "
        "this survey. The participant watched it through: the page would not advance "
        "until it had finished."
    ),
    # -- 11. collective efficacy and emotional benefit --------------------- #
    "F_7QHa8eflRUzcOt8": (
        "A video of a climate march, shown on both branches of the demonstration "
        'question under the instruction "watch to feel the zeal and power of climate '
        'marches". The file is Climate_March_V3.mp4.'
    ),
    # -- the shared battery, shown to every arm ---------------------------- #
    # This one is not decoration: it is the stimulus of the `video` outcome, which
    # is a quarter of the `public_awareness` composite. The next screen asks whether
    # the respondent is willing to share "this information (above)" on social media,
    # so leaving the video undescribed asked them to share nothing in particular.
    "NvNjz1dnwqQ": (
        'A short video from the UN Environment Programme titled "Broken record: '
        "UNEP's #EmissionsGap Report 2023\", summarising that report's finding that "
        "current national pledges leave the world far off track for the Paris "
        "temperature goals."
    ),
}


def describe(key: str) -> str | None:
    """The words for one picture, or ``None`` when this table has never seen it.

    ``None`` rather than a raise, so that a template can still be produced from a
    newly downloaded ``.qsf``; the caller renders a deliberately ugly marker,
    because an undescribed stimulus image is a fidelity hole and not a detail.
    """
    described = IMAGE_ALT.get(key)
    if described is not None:
        return " ".join(described.split())
    missing = UNRECOVERABLE.get(key)
    if missing is not None:
        return (
            "picture not recoverable: the file has been deleted from the survey's "
            "media library and answers 404 on every host the export names. From the "
            f"surrounding copy it {' '.join(missing.split())}"
        )
    labelled = EXPORT_LABEL.get(key)
    if labelled is not None:
        return (
            "picture never seen: the export gives no URL for it and no copy of the "
            f"file exists in the materials. All that is known is that "
            f"{' '.join(labelled.split())}"
        )
    return None
