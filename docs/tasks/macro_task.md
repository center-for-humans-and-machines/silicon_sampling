# Optimize the Pfander Megastudy Prediction
So the approach is to use base model LLMs to fill out quesitionnaires the way they are presented to participants. The final goal is to produce predictions that optimize the scoring metrics (tier 1) from /opt/silicon-sample-submission and /opt/llm_predictions_megastudy for the pfander megastudy (no human data available yet by design).

We will:
1. Silicon sample the outcomes of various related studies with published data using base-model LLMs.
2. Compute calibrations for the existing base models using the published data from related studies.
3. Adjust Pfander predictions using calibrations
4. Predict which model should do best on Pfander for the scoring metrics from /opt/silicon-sample-submission and /opt/llm_predictions_megastudy.

## Target Models
1. Qwen/Qwen2.5-7B (local)
2. Qwen/Qwen2.5-72B (on DAIS)
3. deepseek-ai/DeepSeek-V4-Flash-Base (On DAIS)

## Already Existing predictions (previous work)
* Qwen/Qwen2.5-7B on Voelkel et al 2024
* Qwen/Qwen2.5-7B on the Pfander Megastudy
* deepseek-ai/DeepSeek-V4-Flash-Base on Voelkel et al 2024
* deepseek-ai/DeepSeek-V4-Flash-Base on the Pfander Megastudy

We probably want to eventually re-sample those using the demographic prefilling

## Demographic Silicon Sampling / Calibration
* In the existing work, we also sampled most of the pre-treatment demographics using the LLMs. This is an unnecessary source of error. Given that Pfander and some of the other studies we consider use a representative US sample, we can algorithmically fill fields like income or party alignment. Use the data from the studies with partipant answer data claiming to use a representative US sample to compute the statistics of income / party alignment etc. and their covariance. For all studies, including Pfander, use this to automatically fill out the demographics question. Exception: For Pfander, we know exact statistics on Gender, Age and Race (check documents and prev work), so for these demographic items make sure to match those statistics.


## Steps To Do Now (don't have to be done in exact order)
1. Check the papers, codebooks, datasets in [](/data/calibration/). Which studies are suitable for calibration and model evaluation?
    * Can the questionnaire presented to the human participants be reconstructed from the present material?
    * Are there textual interventions?
    * Is the questionnaire response data available on the participant level?
    * Only if the three items above are true, we should use the study / dataset. There are additional desirable properties for the study/dataset that make it more useful. Use the following items to order what datasets to work on first:
        - The dataset has a representative US sample. Filter to only US participants if US and non-US is available.
        - The dataset has similar post-treatment items as the pfander megastudy
        - The dataset has a similar topic / treatment to the pfander megastudy
2. For the usable studies, reproduce textual templates of the questionnaire to be filled out by LLMs like the participants would see it.
    * See for example: /data/Voelkel/text_templates which was produced in previous work.
3. Compute the demographics of a representative US sample and pre-fill templates with demographics.
4. Silicon sample the other items using the three models.
5. Do proper data analysis focussed around detecting consistent biases and calibration errors of the base LLM models. Also, compute the metrics as defined in /opt/silicon-sample-submission and /opt/llm_predictions_megastudy.
6. Test and evaluate calibration approaches using item-level and study level leave-one-out cross-validation.
    * These should be calibrations directly on the data, without resampling using the base LLM
    * For example, we already know that base-LLMs overestimate effect. Compute overestimation factor and rescale post-treatment outcomes.
    * Use cross-validation using the studies with published participant responses and evaluate each calibration using the pfander metrics /opt/silicon-sample-submission and /opt/llm_predictions_megastudy.
    * Based on what you find in data analysis, come up with possible calibrations and corrections yourself.
7. Sample the Pfander megastudy using the three models and apply the corrections and calibrations you decided on to be helpful using the cross-validation.
8. Write a final report (one summary with sub-reports on details) in which you say which calibrations you included, what the biggest remaining issues are you suspect for the silicon sample faithfulness, what core results you got from data analysis and which model you think will do best on the Pfander megastudy.

## Work Parallization and "what is expensive"
* The by far biggest time cost for this project will be the queuing and sampling time on the DAIS cluster.
    + Handle queues based on the info in the skill [](/.claude/skills/handling_cluster_queues/)
* Qwen2.5 7B can and should be ran locally. Thus, it can be parallized with silicon sampling work on the DAIS cluster.
* Local analysis of silicon sampled results should be considered cheap.
    + It can and should be done on partial results (E.g. if only the silicon sampling from a few of the targeted studies are done).
    + You should produce intermetidate docs/reports on your data analysis results.

## A time constraint
The Pfander megastudy prediction challenge has its submission date pretty soon. Because we want at least a little bit of time to make potential final corrections and bring the submission in the right format, we need output calibrated Pfander predicitons for this task by 29.08.2026 3PM. For now, earlier than 26.08.2026, don't worry about this. But then, depending on queue times on DAIS, we might need to prioritize predicting the Pfander study above silicon sampling all of the calibration studies.

## Git handling
Commit and push partial results. I.e. if a big new silicon sampling job finished and you did data analysis on it.