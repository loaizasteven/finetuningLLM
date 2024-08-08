from openai import OpenAI
from openai.types.fine_tuning.fine_tuning_job import Hyperparameters

import time

# Create client and upload training data
client = OpenAI()
file_name = client.files.create(
  file=open("mydata.jsonl", "rb"),
  purpose="fine-tune"
)

# Train gpt-3.5-turbo for 10 Epoch, gpt-4o-mini requires Tier4 previleages
finetune_model = client.fine_tuning.jobs.create(
  training_file=f"{file_name.id}", 
  model="gpt-3.5-turbo",
  hyperparameters=Hyperparameters(n_epochs=1)
)
job = client.fine_tuning.jobs.retrieve(finetune_model.id)
while job.status != 'succeeded':
    job = client.fine_tuning.jobs.retrieve(finetune_model.id)

    print(f"Job status is {job.status}")
    print(f"Job Estimated Finish time is {job.estimated_finish}")
    time.sleep(30)

completion = client.chat.completions.create(
model=job.fine_tuned_model,
messages=[
    {"role": "user", "content": "Whats the captital of France!"}
]
)

print(completion.choices[0].message)