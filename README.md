# Fine-Tuning LLM
Repo on fine-tuning generative AI models with `openai` and `langchain` packages.

## License
This Repo is under the [Apache 2.0](/LICENSE) license, but be mindful about the underlying packages and datas license when using the content. 

## Introduction
Large Language Models generalize well to a wide range of tasks, and with the right tools (such as prompt engineering) can lead to custom solutions for specialized tasks. There are scenarios when these options are not enough to provide the dedicated solution and fine tuning is required. 

Fine-tuning an Small Language Model (SLM) could be the next logical choice. Why not fine-tune the LLM? Although, fine tuning an LLM is an option swapping it for a SLM would be beneficial when it comes to computation resources and cost considerations. Additionally, if the use case does not require a large general knowledge base then finetuning and SLM for a specific task is ideal.

## Synthetic Data
This repo currently supports the following type of dataset generation:

### SynthPrompt
Generation of synthetic data from prompting a Large Language Model based on its pre-trained knowledge of the subject.
LLM -> Synthetic Data -> SLM (Fine Tuning)

### CorpusGeneration (TODO)
Generation of synthetic data by providing a Large Langugage Model with contextual, potentially domain specific, knowledge to understand and produce a training set.

### Data Structure Ouputs
This notebook focuses on fine-tuning the small language model(s `gpt-4o-mini-*`. The training data must follow the structure below:

```json
{"messages": 
    [
        {"role": "system", "content": <content>}, 
        {"role": "user", "content": <content>}, 
        {"role": "assistant", "content": <content>}
    ]
}
```
To ensure that this output structure is enforced when generating the training data via an LLM there are two options:

1. Prompt the model to provide the desired structure as part of the system message (Current use in this repo)
2. Utilize the native kwarg `response_format` in the cilent object when creating the call. We can utilize pydantic classes with the wrapper method `client.beta.chat.completions.parse()` [[1]](#reference)

## Fine Tuning Task
Some examples of specific SLM finetuned for a particular domain or task.

### Insurance Agent
Step 1: Generate Sytntheic Data: Use the main run script of `synthetic.chatcompletion.py` to generate the training data as a jsonl format
Step 2: FineTuning: See `train.py --run True` for application of `openai_finetuning.finetune.openAIFinetuning()` method for insurance agent.
Step 3: Test: Run `train.py --run False` for an example query to the finetuned insurance agent.

## Considerations
This repository is follows part of the tutorial provided by OpenAI [2], with modifications to the base scripts and logic.

# Reference
[1]: OpenAI. "Introducing Structured Outputs in the API." OpenAI, https://openai.com/index/introducing-structured-outputs-in-the-api/ 06Aug2024.
[2]: OpenAI. "Fine-Tuning" OpenAI, https://platform.openai.com/docs/guides/fine-tuning/