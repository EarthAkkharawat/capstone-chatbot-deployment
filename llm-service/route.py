from fastapi import APIRouter, HTTPException, status
from models import Question
from inference_finetuned import main
import requests
import json
import logging

llm_service = APIRouter()


def call_api(question, url):
    headers = {"Content-Type": "application/json"}
    payload = {"question": question}
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    return response.json()


# Create a new question
@llm_service.post("/createresponse", status_code=201)
async def create_task(payload: Question):
    print("Getting into create_task function")
    logging.info("Getting into create_task function")
    if not payload or payload.question is None:
        raise HTTPException(status_code=400, detail="Question is required")
    question = payload.question
    url = "http://ir-service:8000/createdocs"
    response_ir = call_api(question, url)
    print("Finished call ir-service")
    logging.info("Finished call ir-service")
    time_ir = response_ir["time"]
    source_docs = response_ir["reranked_docs"]
    if source_docs == "":
        return {"answer": "ขออภัยครับไม่สามารถตอบคำถามนี้ได้"}
    response_llm, time_llm = main(question, source_docs)
    print("Got response from LLM")
    logging.info("Got response from LLM")
    if len(response_llm) < 5 or response_llm == "ขออภัยครับ ไม่สามารถตอบคำถามนี้ได้":
        return {"answer": "ขออภัยครับไม่สามารถตอบคำถามนี้ได้"}

    response_llm = "จาก" + source_docs.split("\n\n")[0] + "\n" +  response_llm
    del response_ir, source_docs
    return {"answer": response_llm}
