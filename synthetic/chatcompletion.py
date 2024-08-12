from openai import OpenAI

from openai.types.chat.chat_completion import ChatCompletion, ChatCompletionMessage
from pydantic import BaseModel
from typing import Dict, Any, Union
import jsonlines

import sys
import os.path as osp
import json 

file_dir = osp.dirname(__file__)
parent_dir = osp.dirname(file_dir)

sys.path.insert(0, parent_dir)
from structuredoutput.classes import OpenAIResponse, TrainingClass


class CreateData(BaseModel):
    """ Class to generate sample finetuning data from LLM to be used for task specific SLM

    Attributes:
        systemprompt: system message prompt
        userinput: input query
        modelname: model name, see OpenAi for available models
        client: OpenAI client
        maxtokens: The maximum number of tokens to generate includes both the prompt and completion, has a ceiling based
                    on model choice.
        completion: ChatCompletion type generate within the __call__() method
    """
    systemprompt: str
    userinput: str
    modelname: str
    client: Any | None = None
    maxtokens: int = 16000
    completion: ChatCompletion | None = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Client
        self.client = OpenAI()
    
    def __call__(self) -> ChatCompletionMessage:
        self.completion = self.client.beta.chat.completions.parse(
            model = self.modelname,
            max_tokens= self.maxtokens,
            response_format=TrainingClass,
            messages=[
                {"role": "system", "content": f"{self.systemprompt}"},
                {"role": "user", "content": f"{self.userinput}"}
            ]
        )

        return self.completion.choices[0].message

    def parseobj(self, classobj:bool =False) -> Union[TrainingClass, Dict]:
        msg = self.completion.choices[0].message
        return msg.parsed if classobj else json.loads(msg.content)

    def jsondump(self, outputfile:str) -> str:
        try:
            trainingdata = self.parseobj()
            with jsonlines.open(outputfile, 'w') as writer:
                writer.write_all(trainingdata.get('data'))

            return f"Synthetic data completed and dumped to -> {outputfile}"
        except (AttributeError, BaseException) as e:
            return f"Error: Unable to dump content -> {e} \n"
    
    def __del__(self):
        self.client.close()

    def close(self):
        self.__del__()

        print('Message: closing client connection \n')


if __name__ == "__main__":
    import pprint
    import yaml

    with open(file=osp.join(parent_dir, 'prompt/syntheticdata.yml'), mode='rb') as _readstream:
        synthconfig = yaml.load(stream=_readstream, Loader=yaml.FullLoader)

    # Synth Config Metadata
    pprint.pprint(synthconfig.get('metadata'))

    # Generate Call
    synthdata = CreateData(
        systemprompt = synthconfig.get('system_prompt'),
        userinput = "Provide 100 trianing examples",
        modelname = "gpt-4o-2024-08-06"
    )

    synthdata()

    # Dump Content
    status = synthdata.jsondump(outputfile = osp.join(parent_dir, 'data', './syntheticinsurancedata.jsonl'))
    print(status)

    # close client
    synthdata.close()