# Applying the silicon sampling pipeline to Voelkel
So we're trying to do silicon sampling trying to replicate experiments on the participant level.
Now there is already a plan for the pfander study [](/docs/plans/2026-08-14-pfander-silicon-sampling.md) and we now basically want to reproduce this plan for the Voelkel study.

First challenge is to produce the text templates from the Voelkel study. Please check all the data in [](/data/Voelkel/) and try to reproduce textual templates that look the way the participants see the questionnaire and mark the spots where the participant answers (like for pfander). If there are non-textual interventions, drop them and only focus on the textual ones.

Then do silicon sampling with Qwen2.5 7B just like in the [](/docs/plans/2026-08-14-pfander-silicon-sampling.md) plan.

Then for data analysis: The goal of this whole exercise is to estimate how good our silicon sampling approach will be for the jan pfander megastudy. However, the pfander megastudy has no data published yet (by design), so we're trying on similar studies like Voelkel to evaluate our approach.

Voelkel has participant level responses published [](/data/Voelkel/). So we want to apply the data analysis from pfander at [](/opt/llm_predictions_megastudy) to the Voelkel data as well as possible. Their repo contains R code. Don't use that directly. Translate what is needed to python. Then produce a report on how well our silicon sampling does for the Voelkel data based on the pfanders evaluation metrics.