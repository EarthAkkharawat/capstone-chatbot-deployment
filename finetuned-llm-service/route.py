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
@llm_service.post("/createresponse_tensorrt", status_code=201)
async def create_task(payload: Question):
    logging.info("Getting into create_task function")
    if not payload or payload.question is None:
        raise HTTPException(status_code=400, detail="Question is required")
    question = payload.question
    print("Question  ")
    print(question)
    logging.info(question)
    url = "http://ir-service:8007/createdocs"
    response_ir = call_api(question, url)
    logging.info("Finished call ir-service")
    time_ir = response_ir["time"]
    source_docs = response_ir["reranked_docs"]
    print("Source doc  ")
    print(source_docs)
    if source_docs == "":
        return {"answer": "ขออภัยครับไม่สามารถตอบคำถามนี้ได้"}
    response_llm = main(question, source_docs)
    if response_llm is None: return {"answer": "ขออภัยครับ ไม่สามารถตอบคำถามนี้ได้"}
    logging.info("Got response from LLM")
    if len(response_llm) < 5 or response_llm == "ขออภัยครับ ไม่สามารถตอบคำถามนี้ได้":
        return {"answer": "ขออภัยครับไม่สามารถตอบคำถามนี้ได้"}

    #response_llm = "จาก" + source_docs.split("\n\n")[0] + "\n" +  response_llm
    logging.info("Answer after post-processing: ")
    logging.info(response_llm)
    del response_ir, source_docs
    return {"answer": response_llm}