from fastapi import APIRouter, HTTPException, status
from retriever import main
from models import Question, ResponseIR

ir_service = APIRouter()

# Create a new question
@ir_service.post("/askq", status_code=201)
async def create_task(payload: Question):
    if not payload or payload.question is None:
        raise HTTPException(status_code=400, detail="Question is required")
    question = payload.question
    print(question)
    response = main(question)
    time = response["time"]
    question = response["question"]
    source_docs = response["reranked_docs"]
    return {"time": time, "question": question, "reranked_docs": source_docs}



# from fastapi.testclient import TestClient

# client = TestClient(ir_service)

# # Example test
# response = client.post("/askq", json={"question": "ขอดูอย่างคดีที่มีการพิพากษาของศาลฎีกาต่างจากศาลอุทธรณ์หน่อยได้ไหมครับ"})
# print(response.status_code)
# print(response)
