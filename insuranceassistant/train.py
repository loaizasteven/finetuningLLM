import os.path as osp
import sys

import pprint

file_dir = osp.dirname(__file__)
parent_dir = osp.dirname(file_dir)

sys.path.insert(0, parent_dir)

from openai_finetuning.finetune import testmodel, openAIFinetuning
from openai import OpenAI

if __name__ == "__main__":
    import argparse

    def fileParser():
        parser = argparse.ArgumentParser(
            prog='OpenAI Client Finetuning',
            description="Arguments for finetuning with openai client"
        )
        parser.add_argument("--modelname", default=None, type=str, required=False)
        parser.add_argument("--run", default=False, type=bool, required=False)

        return parser.parse_args()
    
    _args = fileParser()
    
    if _args.run:
        job = openAIFinetuning(
            training_data_path=osp.join(parent_dir, 'data/syntheticdata.jsonl'), 
            model='gpt-4o-mini-2024-07-18',
            training_config={
                'wait': False,
                'verbose': True,
                'patience': 30,
                'n_epochs': 25
            } 
            )
    else:
        print('Test Mode Only \n')

        # create client to openai api
        client = OpenAI()

        # get finetuned modelname
        finetuned_model_ls = client.models.list() # filter to finetuned_model_ls.data entries owned_by not contains 'openai'
        modelname = _args.modelname or finetuned_model_ls.data[-1].id # if wait=True then use `job.fine_tuned_model`
        query = "Whats the captital of NY!"
        response = testmodel(client=client, modelname=modelname)

        pprint.pprint(f"Running Inference on finetuned model: {modelname}")
        pprint.pprint({'userInput': query})
        pprint.pprint(response)
