# Translate Qualtrics survey to text-based fill-out template
So for this task, we'll work with the study at [Vlasceanu](/data/Vlasceanu). We got the [](/data/Vlasceanu/master_survey.pdf) which tells us the details of the Qualtrics study. Also, the original paper [](/data/Vlasceanu/paper.pdf) tells us exactly how the study was created.

## Long-term Goal
We want base-model LLMs to fill out the survey similar to the way humans would. I.e. we want to silicon sample participants. The base-model LLMs will answer the questionnaire in a way as close as possible to how the humans were presented the task. I.e. each time a decision is made by the participant, the LLM gets the document until that point and exactly predicts the continuation tokens for that descision from the context of the rest of the document.

## Task right now
Take the qualtrics survey and format it into text documents that correspond to what a participant would see when running the study. Do one text document per intervention and make sure to only include what the participant sees. For example, the participant does not actually see the IRB number or that 1. is a Control Distracter.

Mark the situations in which the participant is doing a response and indicate what are legal responses. Come up with a format that lets us easily identify the positions where an LLM response generation will be required and what the legal options are. For example, after "Yes, I am at least 18 years old and want to participate", we could add a dash and then have the two legal options to generate be Yes or No. We also translate a 0-100 slider to text. I.e. "Answer from 0 (Not at all) to 100 (Extemely) - " and then allow any number from 0-100 as legal options. Do something similar for similar answer types.

There are also some images in the survey. Look at them an translate them into an appropiate "alt-text".

## Note on the pdf
The [master_survey](/data/Vlasceanu/master_survey.pdf) pdf has its text mapping corrupted. You will have to do all your inferences via visual recognition of the contents.

## Remarks
Do a best effort approach of translating non-textual elements into text. Try to make the output look like a fully textual transcript of the study one might find uploaded as supplementary material of the study somewhere in the internet.

In the end write a report on the most problematic elements regarding the textual transformation. Produce the output texts in [](/data/Vlasceanu/text_survey).