# LLM Social Deduction Games
## General Remarks
### Purpose
This is a project with the goal of achieving top evaluation performance on the silicon sampling social science megastudy outcome prediction competition at https://janpfander.github.io/llm_predictions_megastudy/.

### Environment
You are working inside a Docker container specified in [Dockerfile.gpu](/container/Dockerfile.gpu) and [docker-compose](/container/docker-compose.yml).

You have a local Nvidia RTX 4090 available. VLLM and required packages are all installed. Test and small model runs should be done locally using this GPU + VLLM as inference engine.

### Implementation Location
Everything reusable and relevant for reproduction should be structured into the silicon_sampling folder / package. This package is pre-installed editable in your Docker container.

### Commands
* **Format:** `black .`
* **Lint:** Use exactly `flake8 . --max-line-length=200 --extend-ignore=E203,W503`

Note: Lint and format your implementations when finishing a plan.

### Tasks, Plan and Report workflow
The typical workflow for implementing a feature involves the user specifying a task either directly in the prompt or a markdown file in docs/tasks. For real features not minor edits, unless specified otherwise you should first try to understand the task, gather context and create an implementation plan in `docs/plans/`. If things are unclear about the task, this should be communicated clearly with the user and in the draft before a plan is implemented.

The status is tracked in the first heading: 

* `# [DRAFT] Title -- plan is being written`
* `# [ACTIVE] Title -- plan is approved and in progress`
* `# [DONE] Title -- plan is fully implemented`
* `# [PAUSED] Title -- plan is on hold`
* `# [ABANDONED] Title -- plan was dropped, kept for reference`

Completed plans ([DONE], [ABANDONED]) are moved to `docs/plans/archive/`. Finished tasks are moved to `docs/tasks/archive`

If your executed task produces results deserving more than just a few sentences of output, you should report the results in a markdown file in `docs/reports`. Depending on the size of the report, you may structure it into sub-reports in subfolders of `docs/reports` and you may include plots.

### Using git
When you finish a plan in which you implemented a feature in the silicon sampling repositiory, push the result to github. Feel free to either push directly to main or manage / merge branches yourself. Never rebase, force-push, cherry-pick or do anything overly fancy with git though.