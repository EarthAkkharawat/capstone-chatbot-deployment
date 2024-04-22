from fastapi import APIRouter, HTTPException, status, Request
import requests
import json
from starlette.responses import Response
import os
from linebot.v3 import (
    WebhookHandler
)
from linebot.v3.exceptions import (
    InvalidSignatureError
)
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage
)
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent
)
from dotenv import load_dotenv

load_dotenv()
api_service = APIRouter()

lineaccesstoken = os.getenv('LINEACCESSTOKEN')
channel_secret = os.getenv('CHANNEL_SECRET')
configuration = Configuration(access_token=lineaccesstoken)
handler = WebhookHandler(channel_secret)

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
async def callback(request: Request):
    # get X-Line-Signature header value
    signature = request.headers.get('X-Line-Signature')

    # get request body as text
    body = await request.body()
    # api_service.logger.info("Request body: " + body.decode())

    # handle webhook body
    try:
        handler.handle(body.decode(), signature)
    except InvalidSignatureError:
        api_service.logger.info("Invalid signature. Please check your channel access token/channel secret.")
        raise HTTPException(status_code=400, detail="Invalid signature")

    return Response(content="OK")

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    url = "http://llm-service:8001/createresponse"
    question = event.message.text
    print("question = ", question)
    response = call_api(question, url)
    print("answer from llm-service: ", response["answer"])
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=response["answer"])]

            )
        )