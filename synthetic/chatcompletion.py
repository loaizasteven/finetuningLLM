from openai import OpenAI

from openai.types.chat.chat_completion import ChatCompletion, ChatCompletionMessage
from pydantic import BaseModel
from typing import Dict, Any


class createData(BaseModel):
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
    maxtokens: int = 3500
    completion: ChatCompletion | None = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Client
        self.client = OpenAI()
    
    def __call__(self) -> ChatCompletionMessage:
        self.completion = self.client.chat.completions.create(
            model = self.modelname,
            max_tokens= self.maxtokens,
            
            messages=[
                {"role": "system", "content": f"{self.systemprompt}"},
                {"role": "user", "content": f"{self.userinput}"}
            ]
        )

        return self.completion.choices[0].message

    def jsondump(self, outputfile:str) -> str:
        try:
            trainingdata = self.completion.choices[0].message.content

            with open(outputfile, 'w') as outfile:
                outfile.write(trainingdata)

            return f"Synthetic data completed and dumped to -> {outputfile}"
        except (AttributeError, BaseException) as e:
            return f"Error: Unable to dump content -> {e} \n"
    
    def __del__(self):
        self.client.close()

    def close(self):
        self.__del__()

        print('Message: closing client connection \n')

if __name__ == "__main__":
    import os 
    import os.path as osp

    import pprint
    import yaml

    file_dir = osp.dirname(__file__)
    parent_dir = osp.dirname(file_dir)

    with open(file=osp.join(parent_dir, 'prompt/syntheticdata.yml'), mode='rb') as _readstream:
        synthconfig = yaml.load(stream=_readstream, Loader=yaml.FullLoader)

    # Synth Config Metadata
    pprint.pprint(synthconfig.get('metadata'))

    # Generate Call
    synthdata = createData(
        systemprompt = synthconfig.get('system_prompt'),
        userinput = "Provide 5 trianing examples",
        modelname = "gpt-3.5-turbo-16k"
    )

    # synthdata()

    # Dump Content
    status = synthdata.jsondump(outputfile = osp.join(parent_dir, 'data', './syntheticdata.jsonl'))
    print(status)

    # close client
    synthdata.close()