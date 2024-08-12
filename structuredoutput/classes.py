from pydantic import BaseModel
from typing import Dict, List


class Message(BaseModel):
    role: str
    content: str

class OpenAIResponse(BaseModel):
    messages: List[Message]
    
class TrainingClass(BaseModel):
    data: List[OpenAIResponse]
