from fastapi import APIRouter, HTTPException, status
from models import Question
from inference_finetuned import main
import requests
import json

llm_service = APIRouter()

def call_api(question, url):
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "question": question
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    return response.json()
        
# Create a new question
@llm_service.post("/createresponse", status_code=201)
async def create_task(payload: Question):
    if not payload or payload.question is None:
        raise HTTPException(status_code=400, detail="Question is required")
    question = payload.question
    url = "http://127.0.0.1:8000/createdocs"
    response_ir = call_api(question, url)
    time_ir = response_ir["time"]
    source_docs = response_ir["reranked_docs"]
    
    response_llm, time_llm = main(question, source_docs)
    
    return {"time": time_ir + time_llm, "question": question, "reranked_docs": source_docs, "answer": response_llm}



# from fastapi.testclient import TestClient

# client = TestClient(llm_service)

# # Example test
# response = client.post("/askq", json={"question": "ขอดูอย่างคดีที่มีการพิพากษาของศาลฎีกาต่างจากศาลอุทธรณ์หน่อยได้ไหมครับ"})
# print(response.status_code)
# print(response)
