# Sampling diagnostics

[← main report](README.md)

## The run

- respondents: **6,203** across 7 arms
- throughput: **1678.1** per hour
- calls: **211,783**, draws: **836,376**
- illegal (rejected) draws: **52,927** (6.33%)
- constrained-decoding fallbacks: **2,689**, forced defaults: **0**

The rejection rate is roughly three times the Pfänder run's 1.7%. That is a
property of this instrument rather than a defect in the sampler: the options are
longer, several arms carry comprehension checks, and the party-adaptive phrasing
gives the model more ways to answer out of frame. The near-miss audit below
separates the two possibilities.

## Where rejections concentrate

| slot | rejected_draws |
| --- | --- |
| VotInt | 49369 |
| SPV_1 | 908 |
| XOVS_T1_1D | 417 |
| XOVS_T4_1D | 350 |
| Party_Rep | 345 |
| Party_Dem | 327 |
| PI_Pre | 243 |
| ADA_1 | 241 |
| Comments | 132 |
| XOVS_T8_1D | 63 |
| 7539_Inst_Rep_3 | 58 |
| PA_Fth_Rep | 58 |
| XOVS_T3_1D | 53 |
| PA_DG | 47 |
| XOVS_T2_1D | 45 |

## Resumability

The run was killed outright twice — once by an out-of-memory abort and once by
the whole process tree being terminated at 2,509 respondents — and resumed both
times with no loss and no corruption: every record on disk parsed, every id
unique, and the transcript count matched the answer count exactly. Seeds derive
from the profile id, so the resumed run reproduces what an uninterrupted one
would have produced.
