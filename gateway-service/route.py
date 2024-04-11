from fastapi import APIRouter, HTTPException, status
from models import Question, ResponseIR
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,TemplateSendMessage,ImageSendMessage, StickerSendMessage, AudioSendMessage
)
from linebot.models.template import *
from linebot import (
    LineBotApi, WebhookHandler
)
api_service = APIRouter()

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
@api_service.post("/webhook", status_code=201)
def callback():
    json_line = request.get_json(force=False,cache=False)
    json_line = json.dumps(json_line)
    decoded = json.loads(json_line)
    no_event = len(decoded['events'])
    for i in range(no_event):
        event = decoded['events'][i]
        event_handle(event)
    return '',200

def event_handle(event):
    print(event)
    try:
        userId = event['source']['userId']
    except:
        print('error cannot get userId')
        return ''

    try:
        rtoken = event['replyToken']
    except:
        print('error cannot get rtoken')
        return ''
    try:
        msgId = event["message"]["id"]
        msgType = event["message"]["type"]
    except:
        print('error cannot get msgID, and msgType')
        sk_id = np.random.randint(1,17)
        replyObj = StickerSendMessage(package_id=str(1),sticker_id=str(sk_id))
        line_bot_api.reply_message(rtoken, replyObj)
        return ''

    if msgType == "text":
        msg = str(event["message"]["text"])
        # data = {'message': msg}
        # ans = request.post(url = api_endpoint, data = data)

        # replyObj = TextSendMessage(text='รับทราบ โปรดรอสักครู่')
        # line_bot_api.reply_message(rtoken, replyObj)
        time.sleep(10)

        url = "http://0.0.0.0:8001/createresponse"

        response_llm = call_api(msg,url)
        replyObj = TextSendMessage(text=response_llm)
        line_bot_api.reply_message(rtoken, replyObj)
        # replyObj = TextSendMessage(text=ans)
        # line_bot_api.reply_message(rtoken, replyObj)
        

    else:
        sk_id = np.random.randint(1,17)
        replyObj = StickerSendMessage(package_id=str(1),sticker_id=str(sk_id))
        line_bot_api.reply_message(rtoken, replyObj)
    return ''


# async def create_task(payload: Question):
#     if not payload or payload.question is None:
#         raise HTTPException(status_code=400, detail="Question is required")
#     question = payload.question
#     print(question)
#     response = main(question)
#     time = response["time"]
#     question = response["question"]
#     source_docs = response["reranked_docs"]
#     return {"time": time, "question": question, "reranked_docs": source_docs}


# from fastapi.testclient import TestClient

# client = TestClient(ir_service)

# # Example test
# response = client.post("/askq", json={"question": "ขอดูอย่างคดีที่มีการพิพากษาของศาลฎีกาต่างจากศาลอุทธรณ์หน่อยได้ไหมครับ"})
# print(response.status_code)
# print(response)
