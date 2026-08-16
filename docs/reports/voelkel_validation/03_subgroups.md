# Subgroups

[← main report](README.md)

**Only three of these moderators were ever visible to the model.** The Voelkel
instrument asks gender, race and party on screen; age, education and ideology
came from the panel supplier and appear nowhere a respondent could read them. A
synthetic respondent therefore cannot condition on them, and a subgroup result
over those is measuring something the model was never shown. Both are reported,
labelled.

## Subgroup effect agreement

| moderator | visible_to_model | n_levels | n_pairs | directional_pct | spearman_rho | pearson_r | pearson_adj |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gender | True | 2 | 108 | 52.778 | 0.223 | 0.333 | 0.541 |
| race | True | 5 | 270 | 57.407 | 0.170 | 0.188 | 1.000 |
| party_gen | True | 3 | 162 | 58.642 | 0.183 | 0.250 | 0.405 |
| age_band | False | 4 | 216 | 52.315 | 0.221 | 0.255 | 0.586 |
| education | False | 4 | 216 | 54.630 | 0.184 | 0.218 | 0.447 |

## Demographic parity gap

Worst- minus best-served group per moderator: how unevenly the sample's control
means are wrong across groups.

| moderator | visible_to_model | dpd | worst_abs_err | worst_group | best_group |
| --- | --- | --- | --- | --- | --- |
| age_band | False | 4.582 | 25.871 | 60+ | 18-29 |
| education | False | 1.797 | 24.721 | Postgraduate | HS or less |
| gender | True | 0.695 | 24.058 | Female | Male |
| party_gen | True | 3.728 | 25.680 | Republican | Democrat |
| race | True | 3.257 | 24.265 | White | Asian |

## Control-condition group means

| moderator | visible_to_model | level | outcome | human_mean | synthetic_mean | abs_error |
| --- | --- | --- | --- | --- | --- | --- |
| gender | True | Female | PA | 67.55 | 62.31 | 5.24 |
| gender | True | Female | ADA | 26.11 | 15.64 | 10.47 |
| gender | True | Female | SPV | 8.95 | 12.37 | 3.42 |
| gender | True | Female | SUC | 52.36 | 15.46 | 36.90 |
| gender | True | Female | OppBip | 20.41 | 84.50 | 64.09 |
| gender | True | Female | SocDistrust | 55.75 | 77.80 | 22.04 |
| gender | True | Female | SocDis | 30.35 | 78.34 | 47.99 |
| gender | True | Female | BEPF | 52.98 | 35.01 | 17.97 |
| gender | True | Female | Composite | 39.29 | 47.68 | 8.38 |
| gender | True | Male | PA | 68.61 | 62.77 | 5.83 |
| gender | True | Male | ADA | 27.36 | 16.25 | 11.11 |
| gender | True | Male | SPV | 12.86 | 13.16 | 0.30 |
| gender | True | Male | SUC | 51.32 | 15.86 | 35.45 |
| gender | True | Male | OppBip | 21.52 | 83.13 | 61.61 |
| gender | True | Male | SocDistrust | 51.00 | 77.06 | 26.06 |
| gender | True | Male | SocDis | 30.87 | 77.57 | 46.70 |
| gender | True | Male | BEPF | 49.90 | 35.19 | 14.71 |
| gender | True | Male | Composite | 39.13 | 47.62 | 8.49 |
| race | True | Asian | PA | 62.37 | 61.89 | 0.48 |
| race | True | Asian | ADA | 25.66 | 15.95 | 9.72 |
| race | True | Asian | SPV | 13.50 | 13.87 | 0.37 |
| race | True | Asian | SUC | 47.49 | 16.94 | 30.55 |
| race | True | Asian | OppBip | 23.24 | 85.18 | 61.94 |
| race | True | Asian | SocDistrust | 52.94 | 77.02 | 24.08 |
| race | True | Asian | SocDis | 36.16 | 77.72 | 41.56 |
| race | True | Asian | BEPF | 48.23 | 37.37 | 10.86 |
| race | True | Asian | Composite | 38.72 | 48.24 | 9.52 |
| race | True | Black | PA | 67.03 | 63.00 | 4.03 |
| race | True | Black | ADA | 33.56 | 16.73 | 16.83 |
| race | True | Black | SPV | 17.58 | 12.92 | 4.66 |
| race | True | Black | SUC | 57.00 | 15.73 | 41.27 |
| race | True | Black | OppBip | 21.90 | 83.54 | 61.64 |
| race | True | Black | SocDistrust | 57.84 | 78.93 | 21.09 |
| race | True | Black | SocDis | 37.69 | 81.57 | 43.89 |
| race | True | Black | BEPF | 53.25 | 36.22 | 17.03 |
| race | True | Black | Composite | 43.30 | 48.58 | 5.29 |
| race | True | Hispanic | PA | 65.38 | 61.40 | 3.98 |
| race | True | Hispanic | ADA | 34.97 | 15.79 | 19.18 |
| race | True | Hispanic | SPV | 21.63 | 11.04 | 10.59 |
| race | True | Hispanic | SUC | 55.11 | 14.43 | 40.68 |
| race | True | Hispanic | OppBip | 24.75 | 84.43 | 59.68 |
| race | True | Hispanic | SocDistrust | 55.37 | 76.60 | 21.23 |
| race | True | Hispanic | SocDis | 38.25 | 77.90 | 39.66 |
| race | True | Hispanic | BEPF | 47.34 | 41.04 | 6.30 |
| race | True | Hispanic | Composite | 42.69 | 47.83 | 5.14 |
| race | True | Other | PA | 66.96 | 61.55 | 5.40 |
| race | True | Other | ADA | 30.65 | 19.29 | 11.35 |
| race | True | Other | SPV | 11.34 | 14.08 | 2.74 |
| race | True | Other | SUC | 53.73 | 19.75 | 33.98 |
| race | True | Other | OppBip | 22.15 | 80.32 | 58.17 |
| race | True | Other | SocDistrust | 56.43 | 73.24 | 16.81 |
| race | True | Other | SocDis | 29.85 | 72.67 | 42.81 |
| race | True | Other | BEPF | 50.09 | 33.84 | 16.25 |
| race | True | Other | Composite | 40.17 | 46.84 | 6.67 |
| race | True | White | PA | 68.66 | 62.65 | 6.01 |
| race | True | White | ADA | 25.06 | 15.57 | 9.50 |
| race | True | White | SPV | 9.11 | 12.67 | 3.56 |
| race | True | White | SUC | 51.11 | 15.33 | 35.79 |
| race | True | White | OppBip | 20.44 | 84.05 | 63.62 |
| race | True | White | SocDistrust | 52.75 | 77.65 | 24.90 |
