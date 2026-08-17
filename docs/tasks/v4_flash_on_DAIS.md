# Redoing silicon sampling with bigger model
In this task, you will rerun the voelkel_pipeline.md task and jan_pfanders_silcion_sampling.md task with a bigger model.

You will have to run inference remotely on DAIS. There, the model DeepSeek-V4-Flash-Base should be already pre-downloaded and available (check).

Please use VLLM to resample the silicon sampling participants on DAIS using DeepSeek-V4-Flash-Base just like you did with Qwen 2.5 7B locally. Pull completions down and add them to the analysis reports you wrote for pfanders and voelkel (I.e. redo the analysis with the new sampled partcipants and compare to humans and smaller model). The main goal of this exercise is to determine if bigger base models improve the silicon sampling faithfulness.