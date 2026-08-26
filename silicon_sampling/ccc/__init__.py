"""Voelkel et al. (2026), the Climate Change Challenge.

A registered-report megastudy on the persuasiveness of the most-cited climate
messages: ten message framings against three placebo controls, 13,821 US
respondents, every primary outcome measured before *and* after the message.

This is a different study from :mod:`silicon_sampling.voelkel`, which is the
Strengthening Democracy Challenge by the same first author.  They share seven
generic columns and nothing else.  Only ``voelkel.qsf`` is reused here, because it
parses this study's Qualtrics export unmodified.

Why it earns a place beside the other four: **Pfänder took its items from this
study.**  All three climate-concern items are verbatim identical on the same
101-point scale, the general-policy item is verbatim identical, and three of six
behavioural-intention items are.  No other reference study shares an instrument
with the target.

See ``docs/reports/ccc_validation/`` for the fidelity verification that ran before
any of this was written, and for the six data traps it caught.
"""
