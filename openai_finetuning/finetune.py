from openai import OpenAI
from openai.types.fine_tuning.fine_tuning_job import Hyperparameters

import time
from datetime import datetime

from typing import Dict, Union

run_date = datetime.today().strftime('%Y-%m-%d')

def openAIFinetuning(
        training_data_path:str, 
        model:str = 'gpt-3.5-turbo',
        training_config :Dict[str, Union[str, bool, int]] = {
            'wait': True,
            'verbose': True,
            'patience': 30,
            'n_epochs': 10
        } 
        ) -> str:
  "Free fine-tuning until Sept-23 : https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset "
  if run_date > '2024-09-23':
    assert 'gpt-4o' not in _args.model, "Unable to finetune gpt-4o unless Tier4"

  # Create client and upload training data
  client = OpenAI()

  file_name = client.files.create(
    file=open(training_data_path, "rb"),
    purpose="fine-tune"
  )

  # Train model for 10 Epoch, gpt-4o-mini requires Tier4 previleages
  finetune_model = client.fine_tuning.jobs.create(
    training_file=f"{file_name.id}", 
    model=model,
    hyperparameters=Hyperparameters(n_epochs= training_config.get('n_epochs'))
  )

  job = client.fine_tuning.jobs.retrieve(finetune_model.id)

  if training_config.get('wait'):
    while job.status != 'succeeded':
      job = client.fine_tuning.jobs.retrieve(finetune_model.id)
      if training_config.get('verbose'):
        print(f"Job status is {job.status}")
        print(f"Job Estimated Finish time is {job.estimated_finish}")

    # Wait before retrieve job status
    time.sleep(training_config.get('patience'))

  client.close()
  return job


def testmodel(client, modelname:str, question:str = 'Whats the captital of France!'):
  """
  model name can be accessed from the retrieved job via the client through the fine_tuned_model attribute.

  Example:
    ```
    client = OpenAI()
    finetune_model = FineTuningJob() # open ai object
    job = client.fine_tuning.jobs.retrieve(finetune_model.id)
    job.fine_tuned_model
    ```

  Args:
    client:
    modelname
  """
  completion = client.chat.completions.create(
  model=modelname,
  messages=[
      {"role": "user", "content": f"{question}"}
  ]
  )

  return completion.choices[0].message


if __name__ == "__main__":
    import argparse
    import os.path as osp

    file_dir = osp.dirname(__file__)
    parent_dir = osp.dirname(file_dir)

    def fileParser():
        parser = argparse.ArgumentParser(
            prog='OpenAI Client Finetuning',
            description="Arguments for finetuning with openai client"
        )
        parser.add_argument("--filepath", default=osp.join(parent_dir, 'data', 'mydata.jsonl'), type=str, required=False)
        parser.add_argument("--model", default='gpt-3.5-turbo', type=str, required=False) # gpt-4o-mini-2024-07-18
        parser.add_argument("--run", default=False, type=bool, required=False)

        return parser.parse_args()
    
    _args = fileParser()

    job = openAIFinetuning(
        training_data_path=_args.filepath, 
        model=_args.model,
        training_config={
            'wait': True,
            'verbose': True,
            'patience': 30
        } 
        )
    
    response = testmodel(client=OpenAI(), modelname=job.fine_tuned_model)