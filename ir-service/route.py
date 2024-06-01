from fastapi import APIRouter, HTTPException, status
from retriever import main
from models import Question, ResponseIR
import logging

ir_service = APIRouter()


@ir_service.post("/createdocs", status_code=201)
async def create_task(payload: Question):
    if not payload or payload.question is None:
        raise HTTPException(status_code=400, detail="Question is required")
    question = payload.question
    logging.info(question)
    response = main(question)
    logging.info("Finished retrieving documents")
    time = response["time"]
    question = response["question"]
    source_docs = response["reranked_docs"]
    return {"time": time, "question": question, "reranked_docs": source_docs}

