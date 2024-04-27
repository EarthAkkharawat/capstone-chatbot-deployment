from typing import List, Optional
from datetime import datetime

from pydantic import BaseModel


class Question(BaseModel):
    id: Optional[int] = 0
    question: str


class ResponseIR(BaseModel):
    id: Optional[int] = 0
    time: str
    question: str
    source_docs: str
