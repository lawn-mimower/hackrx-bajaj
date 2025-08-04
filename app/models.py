from pydantic import BaseModel
from typing import List

class HackRequest(BaseModel):
    documents: str
    questions: List[str]

class HackResponse(BaseModel):
    success: bool
    processing_info: str
    answers: List[str]
