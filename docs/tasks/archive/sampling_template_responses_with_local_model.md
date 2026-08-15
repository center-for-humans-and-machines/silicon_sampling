So at this point, you should have an RTX4090 available locally, have Qwen2.5-7B pre-downloaded with huggingface and a local VLLM installation that can do inference with this model. You should also have text templates available from the questionnaire of Jan Pfanders Study [](/data/pfander/text_templates).

So our goal will be to silicon sample the responses of the participants from Jan Pfanders Study using this local LLM. This is in the context of the currently running silicon sampling competition.

Please check the [preregistration](/data/pfander/preregistration.html). It contains the information on how many participants and has gender, age, race statistics. The first few questions in our text templates are about race, gender, age. They should be pre-filled according to these statistics.

Then, we want to sample the rest of the questionnaire outcomes using Qwen2.5-7B. This is a base model. So it won't be manually prompted. Instead, it should just literally get the questionnaire with all the previous answers up to the points where a participant response is required and then sample continuation tokens. Most template fields have a limited set of legal responses. Use an illegal -> resampling scheme and make sure to not overgenerate too many tokens (the model won't stop on it's own, we'll want low max_new_tokens and then truncate to just the legal answer). Sample using temperature 1.0, no repetition penalty and top_k / top_p disabled (I.e. sample faithfully to the learned text distribution).

Sample one questionnaire per participant, pre-filled with age / gender / race according to preregisterd statistics.

Save the literal sampled templates (~18000 text files) in [](/data/pfander/silicon_sampling/qwen25_7b/raw) and produce an additional single CSV file with just the answers at [](/data/pfander/silicon_sampling/qwen25_7b/samples.csv) (~18000 rows with all answers from a single sample each (including race/gender/age and intervention)).