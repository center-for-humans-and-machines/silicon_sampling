# Effects, ours against theirs

[← main report](README.md)

Every effect is the difference from the shared null control, in percentage points
of the outcome's scale range. All nine outcomes are natively 0-100, so the
conversion is a no-op here and the units are directly comparable.

**Six of the nine outcomes are reverse-scored so that high is bad** — more
animosity, more support for undemocratic practices, more distrust. A treatment
that works produces a *negative* effect.

## Human effects (Human 1), the target

| condition | ADA | BEPF | Composite | OppBip | PA | SPV | SUC | SocDis | SocDistrust |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Harmful_Experiences | 1.10 | 0.93 | 0.37 | -0.36 | -2.31 | 0.61 | 1.98 | 0.24 | 0.56 |
| Misperception_Competition | 1.01 | 0.40 | -0.48 | -0.63 | -3.03 | -0.95 | 1.60 | -0.85 | -1.85 |
| Misperception_Suffering | 2.84 | 0.79 | 0.21 | 0.34 | -4.72 | 0.98 | 3.75 | -0.19 | -1.82 |
| Partisan_Threat | 0.42 | -0.94 | 0.54 | 1.06 | 0.97 | -0.68 | 2.28 | 0.89 | 0.00 |
| Party_Overlap | 0.11 | 1.56 | -0.56 | 1.32 | -4.26 | -0.29 | -0.39 | -0.35 | -2.23 |
| System_Justification | 0.23 | 0.63 | -0.24 | -0.47 | -2.33 | 0.54 | 0.50 | -0.52 | -1.51 |

## Our effects

| condition | ADA | BEPF | Composite | OppBip | PA | SPV | SUC | SocDis | SocDistrust |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Harmful_Experiences | 0.79 | 2.43 | 0.70 | -0.01 | 0.85 | 0.82 | 1.14 | 0.60 | -0.99 |
| Misperception_Competition | 6.12 | 0.26 | -0.13 | -2.43 | 0.63 | 5.35 | 8.20 | -8.09 | -11.03 |
| Misperception_Suffering | 7.11 | -3.40 | 0.52 | -2.64 | -0.88 | 7.08 | 8.90 | -6.78 | -5.23 |
| Partisan_Threat | -2.08 | -3.82 | -0.86 | 2.29 | -0.57 | -2.92 | -1.65 | 0.64 | 1.19 |
| Party_Overlap | 0.16 | -1.68 | -2.72 | -3.94 | -1.37 | -0.05 | 2.01 | -7.05 | -9.85 |
| System_Justification | -2.56 | -1.50 | -0.24 | 0.70 | 0.68 | -1.38 | -1.66 | 2.48 | 1.29 |

## Error, ours minus theirs

| condition | ADA | BEPF | Composite | OppBip | PA | SPV | SUC | SocDis | SocDistrust |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Harmful_Experiences | -0.30 | 1.50 | 0.33 | 0.35 | 3.16 | 0.21 | -0.84 | 0.36 | -1.55 |
| Misperception_Competition | 5.11 | -0.15 | 0.36 | -1.80 | 3.66 | 6.30 | 6.60 | -7.24 | -9.18 |
| Misperception_Suffering | 4.27 | -4.19 | 0.31 | -2.99 | 3.84 | 6.10 | 5.15 | -6.60 | -3.41 |
| Partisan_Threat | -2.51 | -2.88 | -1.41 | 1.24 | -1.54 | -2.24 | -3.94 | -0.25 | 1.19 |
| Party_Overlap | 0.05 | -3.24 | -2.16 | -5.26 | 2.89 | 0.24 | 2.40 | -6.69 | -7.62 |
| System_Justification | -2.79 | -2.13 | -0.01 | 1.17 | 3.01 | -1.93 | -2.17 | 3.00 | 2.80 |

## Weighted check

The study's own estimates use per-outcome survey weights. Ours are unweighted, so
the headline comparison is unweighted on both sides; this is the weighted version
of the human effects, to show whether any conclusion turns on it.

| condition | ADA | BEPF | Composite | OppBip | PA | SPV | SUC | SocDis | SocDistrust |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Harmful_Experiences | 1.12 | 0.97 | 0.40 | -0.37 | -2.28 | 0.59 | 2.03 | 0.23 | 0.57 |
| Misperception_Competition | 0.95 | 0.40 | -0.50 | -0.65 | -3.01 | -0.97 | 1.57 | -0.88 | -1.85 |
| Misperception_Suffering | 2.83 | 0.85 | 0.23 | 0.37 | -4.68 | 0.96 | 3.74 | -0.15 | -1.82 |
| Partisan_Threat | 0.45 | -0.91 | 0.54 | 1.04 | 0.96 | -0.68 | 2.32 | 0.92 | 0.04 |
| Party_Overlap | 0.11 | 1.59 | -0.56 | 1.31 | -4.27 | -0.30 | -0.38 | -0.37 | -2.23 |
| System_Justification | 0.20 | 0.67 | -0.23 | -0.44 | -2.32 | 0.52 | 0.47 | -0.46 | -1.51 |

## All pairs

| outcome | condition | estimate_h | se_h | estimate_l | se_l |
| --- | --- | --- | --- | --- | --- |
| PA | Harmful_Experiences | -2.313 | 0.992 | 0.850 | 1.090 |
| PA | Misperception_Competition | -3.033 | 0.971 | 0.626 | 1.079 |
| PA | Misperception_Suffering | -4.718 | 0.983 | -0.879 | 1.107 |
| PA | Partisan_Threat | 0.971 | 0.917 | -0.572 | 1.094 |
| PA | Party_Overlap | -4.262 | 0.927 | -1.368 | 1.151 |
| PA | System_Justification | -2.330 | 0.904 | 0.685 | 1.127 |
| ADA | Harmful_Experiences | 1.095 | 1.135 | 0.792 | 1.093 |
| ADA | Misperception_Competition | 1.005 | 1.144 | 6.117 | 1.165 |
| ADA | Misperception_Suffering | 2.838 | 1.117 | 7.111 | 1.243 |
| ADA | Partisan_Threat | 0.423 | 1.061 | -2.085 | 1.061 |
| ADA | Party_Overlap | 0.108 | 1.076 | 0.156 | 1.120 |
| ADA | System_Justification | 0.234 | 1.068 | -2.557 | 1.042 |
| SPV | Harmful_Experiences | 0.612 | 1.004 | 0.824 | 0.967 |
| SPV | Misperception_Competition | -0.950 | 0.959 | 5.349 | 1.052 |
| SPV | Misperception_Suffering | 0.978 | 0.988 | 7.076 | 1.145 |
| SPV | Partisan_Threat | -0.682 | 0.857 | -2.918 | 0.864 |
| SPV | Party_Overlap | -0.289 | 0.935 | -0.053 | 0.980 |
| SPV | System_Justification | 0.543 | 0.935 | -1.383 | 0.933 |
| SUC | Harmful_Experiences | 1.980 | 1.152 | 1.144 | 1.162 |
| SUC | Misperception_Competition | 1.600 | 1.154 | 8.199 | 1.243 |
| SUC | Misperception_Suffering | 3.755 | 1.080 | 8.904 | 1.343 |
| SUC | Partisan_Threat | 2.282 | 1.148 | -1.654 | 1.072 |
| SUC | Party_Overlap | -0.388 | 1.085 | 2.012 | 1.207 |
| SUC | System_Justification | 0.504 | 1.107 | -1.662 | 1.124 |
| OppBip | Harmful_Experiences | -0.360 | 1.070 | -0.014 | 1.054 |
| OppBip | Misperception_Competition | -0.633 | 1.012 | -2.432 | 1.018 |
| OppBip | Misperception_Suffering | 0.342 | 1.028 | -2.643 | 1.059 |
| OppBip | Partisan_Threat | 1.058 | 1.019 | 2.294 | 1.015 |
| OppBip | Party_Overlap | 1.318 | 1.065 | -3.942 | 1.190 |
| OppBip | System_Justification | -0.469 | 1.056 | 0.697 | 1.059 |
| SocDistrust | Harmful_Experiences | 0.564 | 1.330 | -0.988 | 1.364 |
| SocDistrust | Misperception_Competition | -1.852 | 1.387 | -11.034 | 1.410 |
| SocDistrust | Misperception_Suffering | -1.817 | 1.290 | -5.226 | 1.392 |
| SocDistrust | Partisan_Threat | 0.004 | 1.282 | 1.195 | 1.327 |
| SocDistrust | Party_Overlap | -2.226 | 1.351 | -9.849 | 1.508 |
| SocDistrust | System_Justification | -1.506 | 1.286 | 1.294 | 1.358 |
| SocDis | Harmful_Experiences | 0.237 | 1.363 | 0.595 | 1.269 |
| SocDis | Misperception_Competition | -0.853 | 1.333 | -8.090 | 1.302 |
| SocDis | Misperception_Suffering | -0.186 | 1.322 | -6.781 | 1.346 |
| SocDis | Partisan_Threat | 0.891 | 1.267 | 0.641 | 1.267 |
| SocDis | Party_Overlap | -0.352 | 1.321 | -7.046 | 1.405 |
| SocDis | System_Justification | -0.517 | 1.298 | 2.478 | 1.224 |
| BEPF | Harmful_Experiences | 0.929 | 1.074 | 2.427 | 1.199 |
| BEPF | Misperception_Competition | 0.404 | 1.022 | 0.257 | 1.145 |
| BEPF | Misperception_Suffering | 0.788 | 1.038 | -3.403 | 1.141 |
| BEPF | Partisan_Threat | -0.940 | 0.980 | -3.818 | 1.132 |
| BEPF | Party_Overlap | 1.563 | 1.016 | -1.677 | 1.159 |
| BEPF | System_Justification | 0.631 | 0.996 | -1.497 | 1.168 |
| Composite | Harmful_Experiences | 0.372 | 0.626 | 0.704 | 0.396 |
| Composite | Misperception_Competition | -0.482 | 0.602 | -0.126 | 0.410 |
| Composite | Misperception_Suffering | 0.210 | 0.604 | 0.520 | 0.437 |
| Composite | Partisan_Threat | 0.541 | 0.594 | -0.865 | 0.372 |
| Composite | Party_Overlap | -0.558 | 0.621 | -2.721 | 0.462 |
| Composite | System_Justification | -0.236 | 0.609 | -0.243 | 0.363 |
