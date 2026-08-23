"""What the stimulus pictures showed, in words.

Ten of the twelve arms carry static images and two of them — the working-together
flyer and the two pie charts — put the *manipulation itself* inside the picture.
A text transcript that silently dropped them would show a respondent in arm 2
fifteen slider items about a flyer they were never shown, which is not a weaker
version of the study, it is a different study.

Every entry below was written from the picture.  Fifty-one of the fifty-seven
files came straight off the Qualtrics graphic CDN, which still serves them
publicly; the CDN answered 404 or 400 for the remaining six, five of which were
then recovered out of the embedded images of ``master_survey.pdf``; the sixth —
the time capsule on the third screen of *A Letter to Future Generations* — is gone
from both and is marked as such rather than invented.  So fifty-six of the
fifty-seven are vendored under ``data/ICPC/Materials/stimuli``, where
``index.json`` records for each one whether it came from the CDN, from the PDF, or
not at all, and this table can be re-checked against them without the network.

Two conventions, both inherited from the Pfänder and Voelkel templates:

* text *inside* a picture that carries the manipulation is transcribed verbatim,
  because it is on-screen text that happens to have been set in an image;
* a purely decorative photograph gets a plain description and no more, because
  writing an evaluative caption ("a distressing image of ...") would put the
  experimenter's framing into the stimulus.

Keys are the Qualtrics graphic ids (``IM_...``) with two remote URLs for the
images the survey hot-linked from Pixabay and Google's CDN, so the table is keyed
by the same string the ``.qsf`` uses and a missing entry is a hard failure rather
than a silently blank screen.
"""

from __future__ import annotations

FLYER = """A round black-and-white flyer. Along the top, in large bold type,
"Let's Do It Together". Below it a row of silhouetted figures alternating with
globes showing the Americas, then, in bold capitals, "WE NEED TO REDUCE OUR
CARBON FOOTPRINT." Below that, in bold: "A majority of people are taking steps
to reduce their carbon footprint." Then "Here's what you can do to make a
difference:" followed by three italic bullets: "support policies aimed at
reducing greenhouse gasses", "share climate change information on social media",
"donate to tree planting organizations". At the foot, in small bold type, "Please
reduce your carbon footprint." and, largest of all, "Join in!\""""

#: Qualtrics graphic id -> description.  Every ``<img>`` in the US instrument.
IMAGE_ALT: dict[str, str] = {
    # -- 2. working-together identity / social norms --------------------- #
    # One flyer, redisplayed above all fifteen slider items.
    "IM_5iPrjjoRKOswfCS": FLYER,
    # -- 3. negative emotions -------------------------------------------- #
    "IM_0CmRY0YTfTNN3pA": (
        "A photograph of an emaciated polar bear, ribs and hip bones visible "
        "through its fur, walking across a thin slab of sea ice surrounded by "
        "open water. Credited in the corner to Kerstin Langenberger, "
        "www.arctic-dreams.com."
    ),
    "IM_cUc5zOM1z9PrgLc": (
        "Three photographs side by side: a firefighter silhouetted against a "
        "wall of flame in a burning field; a dried lake bed cracked into plates "
        "under a pink dawn sky; and a satellite view of a hurricane's eye off a "
        "green coastline."
    ),
    "IM_2nRZPo0HfYTTwtE": (
        "A wall clock whose face is a satellite photograph of the Earth, with "
        "white numerals 1 to 12 around the rim and gold hands."
    ),
    "IM_3xQetjaSyGNeRD0": (
        "An illustration in flat turquoise on cream: a giant hourglass whose "
        "upper bulb is a world map draining through the waist into the lower "
        "bulb, with a small woman standing at the left shading her eyes to look "
        "up at it."
    ),
    # -- 4. scientific consensus ----------------------------------------- #
    "IM_0TXaGFXfwBpwGgK": (
        "A pie chart, almost entirely one blue wedge labelled 99% in white, "
        "with a sliver of grey for the remainder."
    ),
    # -- 5. collective action -------------------------------------------- #
    "IM_6X7hfWRnTmCI2zA": (
        "A composite photograph of one city skyline, split down the middle: the "
        "left half sits under an orange smoke-filled sky above cracked bare "
        "earth, the right half under blue sky above green grass."
    ),
    "IM_083g4vcb0cJqagu": (
        "A photograph of a woman running up a long flight of stone steps beside "
        "the sea in low sunlight."
    ),
    "IM_blT2Ou01IS4dOMC": (
        "A photograph of a person in a hooded sweatshirt sitting on concrete "
        "steps with their head in their hands."
    ),
    "IM_a00HLdeZCao0XwW": (
        "A photograph looking down on a student slumped face-down over a "
        "notebook at a desk strewn with crumpled sheets of paper."
    ),
    "IM_ePUDaHctDuofZ6S": (
        "A black-and-white photograph of a couple on a beach, embracing and "
        "looking at each other."
    ),
    "IM_0lFI5VB88xuPWgS": (
        "A photograph of two young boys facing each other, fists raised, one "
        "shouting at the other."
    ),
    "IM_3JciPqnTZ6WwgOG": (
        "A line chart titled 'Emissions Reductions Starting 2030', with the x "
        "axis running 2020 to 2100 and the right-hand y axis labelled "
        "'Temperature Change °C' from about 1.0 to 2.0. Three shaded "
        "trajectories are keyed at the top left in this order: 'No Change' in "
        "red, rising past 2.0; 'Net Zero Emissions' in green, peaking near 1.4 "
        "around 2040 and falling back towards 1.0; and 'Paris Accord "
        "Commitments' in blue, levelling near 1.5. A red box drawn around the "
        "years before about 2040 has a red arrow pointing into it from the "
        "label 'Window of Opportunity'."
    ),
    "IM_cSUxn0lJpHNlDrU": (
        "A photograph of a large seated crowd at an indoor conference holding a "
        "red banner that reads 'SYSTEM CHANGE not CLIMATE CHANGE', with the "
        "smaller line 'GLOBAL CAMPAIGN TO DEMAND CLIMATE JUSTICE' beneath it."
    ),
    "IM_0Cd3Ve48sYgPjgi": (
        "Two satellite maps of the Antarctic ozone layer side by side in "
        "false colour, the blue-purple hole at the centre much smaller in the "
        "right-hand map. Captioned 'Sept. 1994' and 'Oct. 2019', credited to "
        "c&en's Stereo Chemistry."
    ),
    "IM_ehVV1zAkBpVLfrE": (
        "A black-and-white head-and-shoulders photograph of a smiling woman "
        "with long fair hair, standing in front of a wooden fence."
    ),
    "IM_6LqOyzKWEfGIHtk": (
        "A photograph of a small group of demonstrators holding a yellow banner "
        "reading 'THIS IS AN EMERGENCY - ACT LIKE IT'."
    ),
    "IM_3fu1Oruy2mhUS9g": (
        "A photograph of a street demonstration; a young woman in the "
        "foreground holds up a hand-painted cardboard placard of an hourglass "
        "draining a burning Earth into a flooded one, lettered 'CLIMATE IS "
        "CHANGING'. Other placards behind her are in German."
    ),
    "IM_8e4CeeC9VwiqTC6": (
        "An infographic. At the left, three figures in large orange type: "
        "'2,060+ Governments Declared', '1+ Billion People', '18 National "
        "Governments + The EU'. At the right, a line chart titled 'Governments "
        "That Have Declared a Climate Emergency', rising from near zero in "
        "January 2018 to just over 2,000 by January 2022."
    ),
    "IM_9NUhV2i2V6x0EwS": (
        "A green map of Central America with five forest areas traced on it, "
        "captioned in white capitals: 'THE 5 GREAT FORESTS OF MESOAMERICA "
        "COVER 3X THE SIZE OF SWITZERLAND'."
    ),
    "IM_4PHnBri8hOnx5zM": (
        "A photograph of a young woman in a black mask at an outdoor protest, "
        "holding a hand-lettered sign reading '#Shame on DOOSAN'."
    ),
    "IM_3watB8PWFr9iH2u": (
        "A photograph of demonstrators at a station entrance holding up a "
        "banner reading 'No More Coal', with Korean text beneath."
    ),
    "IM_5pdZjQIjlG4T8sS": (
        "A photograph of a large street march of mostly young people carrying "
        "placards, with a banner reading 'Parade in Climate Action' and Korean "
        "text below it."
    ),
    "IM_3w3a9eZuVmQQ1cq": (
        "A photograph of a dense crowd at an outdoor climate march, hundreds of "
        "hand-made placards raised above it."
    ),
    "IM_9uIUW1HyxyPGY1o": (
        "A photograph of two surfers standing on an empty beach watching a "
        "large breaking wave."
    ),
    # -- 6. system justification ----------------------------------------- #
    "IM_cOx1Lml02s3zzXU": (
        "A photograph of the Chicago river running between glass towers at "
        "dusk, tour boats on the water and a United States flag flying in the "
        "right foreground."
    ),
    "IM_71zLI3GR0Jx7RLU": (
        "A photograph of the Grand Canyon at sunset from the south rim, red "
        "buttes receding into haze."
    ),
    "IM_ac0kd3Ld6uR8COy": (
        "A photograph looking down the main street of a small mountain town, "
        "brick storefronts on both sides and a cliff face of yellow-leaved "
        "aspen and grey rock closing off the end of the valley."
    ),
    "https://cdn.pixabay.com/photo/2017/11/02/10/53/field-2910710_960_720.jpg": (
        "A photograph of a harvested wheat field covered in round straw bales, "
        "with a forested mountain range behind it."
    ),
    (
        "https://lh4.googleusercontent.com/lWvfTDegyBWz6aL4Sk10SZ2fj-h6zTmIzh3j"
        "-NLgFluOmZiRV5R87ZcerKDyAzxXJ1tBPyY2GiYNkqnBpsKGAU29vTHfoYNo-wAbz05XLW"
        "rkUdUqQRHwDGx5cu_TPZyljwooZ9Nr"
    ): (
        "A photograph of a bicycle leaning against a birch tree at the edge of "
        "a meadow, low sun through the leaves."
    ),
    "IM_7QVH6DEo70xjt1Y": (
        "A photograph of two surfers standing on an empty beach watching a "
        "large breaking wave."
    ),
    "IM_d0hOzAKRONzOAJM": (
        "A photograph of the flag of the United States flying from a pole "
        "against a blue sky."
    ),
    "IM_8celYrIM4QFYqay": (
        "A photograph of a Macy's Thanksgiving Day Parade float — a giant "
        "turkey balloon on a gilded wagon lettered 'MACY'S THANKSGIVING DAY "
        "PARADE' — moving down a city street lined with spectators."
    ),
    "IM_5ubDkEZD65re9me": (
        "An aerial photograph of a flooded city at dusk: brown water covering "
        "streets and parkland up to the walls of apartment blocks, a skyline "
        "beyond."
    ),
    "IM_ehCVnfmEmMpvOGq": (
        "A photograph of a suspension bridge silhouetted against an orange sky, "
        "the hillside behind it dotted with burning trees."
    ),
    "IM_0AErbjvw9XtREai": (
        "A photograph of a granite valley at dawn — sheer cliffs above conifers "
        "and a still river in the foreground."
    ),
    "IM_7UiRqF4drUNbQ9M": (
        "A photograph of seven adults and children holding hands in a line "
        "across the base of an enormous redwood trunk that fills the frame."
    ),
    # -- 7. decreasing psychological distance ----------------------------- #
    "IM_0CZb2DXTxXmBV2K": (
        "A photograph of an oil refinery at dusk, a single stack throwing up a "
        "vast black and orange plume across the sky."
    ),
    "IM_1NBwgMOeg3uQOai": (
        "A photograph taken from a road at night of a hillside wildfire, "
        "burning trees running up the slope, a pickup truck silhouetted in the "
        "foreground."
    ),
    "IM_3dWENzCePiSAp4a": (
        "An aerial photograph of a flooded suburb, brown water up to the eaves "
        "of street after street of houses, a few small boats moving through it."
    ),
    "IM_cYFOq6J5NoSyuLY": (
        "A photograph from behind of a young woman at an outdoor demonstration "
        "wearing a large hand-painted placard on her back: a drawing of the "
        "Earth with a lit fuse, lettered 'TIME IS RUNNING OUT!'"
    ),
    "IM_5hCYDX781agpAXk": (
        "A photograph of dry ground cracked into a mosaic of plates, filling "
        "the frame."
    ),
    # -- 8. correcting pluralistic ignorance ------------------------------ #
    "IM_3lzCjxmNHHU8LdA": (
        "A pie chart with a large dark navy wedge labelled '65% Agree' in white "
        "and a smaller pale grey-blue wedge for the remainder."
    ),
    # -- 9. a letter to future generations -------------------------------- #
    "IM_cIwXCbvkxMpchqC": (
        "[picture not recoverable: the Qualtrics graphic for this screen has "
        "been deleted from the survey's media library and does not appear in "
        "master_survey.pdf. From the surrounding copy it illustrated a family "
        "finding a time capsule.]"
    ),
    # -- 10. dynamic social norms ---------------------------------------- #
    "IM_cG4mBFNnd1qHgx0": (
        "A dumbbell chart. For eleven countries and an average — France, Spain, "
        "Mexico, Germany, Kenya, UK, Canada, Australia, US, South Africa, "
        "Poland, Average — a blue dot marks the 2013 value and an orange dot "
        "the 2018 value, joined by an upward grey arrow; every pair rises. The "
        "y axis runs from 30 to 90. Sourced below the chart to 'Pew Research "
        "Center, Spring 2018 & 2013 Global Attitudes Surveys'."
    ),
    # -- 12. binding moral foundations ----------------------------------- #
    "IM_aeJJeNNG1Uz87fU": (
        "A composite image of a silhouetted figure standing on top of a globe "
        "turned to show North America, holding up the flag of the United "
        "States."
    ),
    # -- screens every arm saw (also described in vlasceanu.content_shared,
    # -- which is what the shared blocks are actually rendered from) -------- #
    "IM_5yIefx9rRtlDZAh": (
        "Letterhead of New York University: the NYU torch logo in a box, then "
        "'New York University', the underlined italic line 'A private university "
        "in the public service', and an address block for the Faculty of Arts "
        "and Science, Department of Psychology, 6 Washington Place, Room 550, "
        "New York, NY 10003-6634."
    ),
    "IM_cHZBVvBtL6PePS6": (
        "A drawing of a grey ten-runged ladder standing at a slight angle and "
        "leaning to the left."
    ),
    "IM_b8G7LPNmxmA1kpg": (
        "The logo of Eden Reforestation Projects: a broad-canopied green tree on "
        "sandy ground beside the words 'Eden Reforestation Projects', with a "
        "small badge reading 'Platinum Transparency 2022 - Candid'."
    ),
    "IM_1RNzkUY1MgDoOG2": (
        "A pictogram of eight simple tree outlines arranged four over four; "
        "1 of them are filled in solid green and the remaining 7 are left "
        "as plain outlines, marking progress through the task."
    ),
    "IM_0TAnYAmvVk6kImy": (
        "A pictogram of eight simple tree outlines arranged four over four; "
        "2 of them are filled in solid green and the remaining 6 are left "
        "as plain outlines, marking progress through the task."
    ),
    "IM_cZuXMjNNjqR9ICW": (
        "A pictogram of eight simple tree outlines arranged four over four; "
        "3 of them are filled in solid green and the remaining 5 are left "
        "as plain outlines, marking progress through the task."
    ),
    "IM_3rtDN0GX9knK7jM": (
        "A pictogram of eight simple tree outlines arranged four over four; "
        "4 of them are filled in solid green and the remaining 4 are left "
        "as plain outlines, marking progress through the task."
    ),
    "IM_bjhWCWErt0yYQ5w": (
        "A pictogram of eight simple tree outlines arranged four over four; "
        "5 of them are filled in solid green and the remaining 3 are left "
        "as plain outlines, marking progress through the task."
    ),
    "IM_bEqK8mYQpaU0mua": (
        "A pictogram of eight simple tree outlines arranged four over four; "
        "6 of them are filled in solid green and the remaining 2 are left "
        "as plain outlines, marking progress through the task."
    ),
    "IM_d0hZ4COek1qydXU": (
        "A pictogram of eight simple tree outlines arranged four over four; "
        "7 of them are filled in solid green and the remaining 1 are left "
        "as plain outlines, marking progress through the task."
    ),
    "IM_4HDvrRb6ApPIuNw": (
        "A pictogram of eight simple tree outlines arranged four over four; "
        "8 of them are filled in solid green and the remaining 0 are left "
        "as plain outlines, marking progress through the task."
    ),
}


def describe(key: str) -> str:
    """The words for one image reference, or a loud placeholder.

    A missing key is rendered rather than raised so that a template can still be
    produced from a newly downloaded ``.qsf``; the marker is deliberately ugly
    because an undescribed stimulus image is a fidelity hole, not a detail.
    """
    alt = IMAGE_ALT.get(key)
    if alt is None:
        return f"[picture not described: {key}]"
    return " ".join(alt.split())
