from fastapi import APIRouter, HTTPException, status, Request
import requests
import json
import os
from starlette.responses import Response
from models import Question
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage,
)
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent,
    ImageMessageContent,
    VideoMessageContent,
    AudioMessageContent,
    FileMessageContent,
    LocationMessageContent,
    StickerMessageContent,
)
from dotenv import load_dotenv

load_dotenv()
api_service = APIRouter()

lineaccesstoken = os.getenv("LINEACCESSTOKEN")
channel_secret = os.getenv("CHANNEL_SECRET")
configuration = Configuration(access_token=lineaccesstoken)
handler = WebhookHandler(channel_secret)


def is_first_time_api_call(model = "finetuned"):
    flag_file = f"api_call_flag_{model}.txt"

    if os.path.exists(flag_file):
        return False
    else:
        with open(flag_file, "w") as f:
            f.write(f"API_call_flag_{model}")
        return True


def call_api(question, url, model = "finetuned", timeout=300):
    print("Enter call_api function")
    if is_first_time_api_call(model):
        print("This is the first time the API call is being made.")
        timeout = 999999
    else:
        print("This is not the first time the API call is being made.")

    try:
        headers = {"Content-Type": "application/json"}
        payload = {"question": question}
        print(payload)
        response = requests.post(
            url, headers=headers, data=json.dumps(payload), timeout=timeout
        )
        print(response)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()  
    except requests.exceptions.Timeout:
        print("Timeout error: The request took too long to respond.")
    return "Timeout"


# api for base
@api_service.post("/base", status_code=201)
async def callback(request: Question):
    print("Using base SeaLLM-7B-v2 model")
    try:
        question = request.question

        url = "http://llm_base-service:8005/createresponse_base"
        response = call_api(question, url, "base")
        if response == "Timeout":
            answer = "ขออภัยครับ ไม่สามารถตอบคำถามนี้ได้"
        else: answer = response["answer"]
        del response
        return {"answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# api for finetuned chatbot arena
@api_service.post("/finetuned", status_code=201)
async def callback(request: Question):
    print("Using finetuned SeaLLM-7B-v2 model")
    try:
        question = request.question 
        url = "http://llm-service:8001/createresponse"
        response = call_api(question, url, "finetuned")
        if response == "Timeout":
            answer = "ขออภัยครับ ไม่สามารถตอบคำถามนี้ได้"
        else: answer = response["answer"]
        del response
        return {"answer": answer}

    except Exception as e:
        # If an error occurs, return an HTTPException with status code 500 (Internal Server Error)
        raise HTTPException(status_code=500, detail=str(e))


# Create a new question
@api_service.post("/webhook", status_code=201)
async def callback(request: Request):
    print("Answering LineOA using finetuned SeaLLM-7B-v2 model")
    # get X-Line-Signature header value
    signature = request.headers.get("X-Line-Signature")

    body = await request.body()
    # api_service.logger.info("Request body: " + body.decode())

    # handle webhook body
    try:
        handler.handle(body.decode(), signature)
    except InvalidSignatureError:
        api_service.logger.info(
            "Invalid signature. Please check your channel access token/channel secret."
        )
        raise HTTPException(status_code=400, detail="Invalid signature")

    return Response(content="OK")


@handler.add(
    MessageEvent,
    message=(
        TextMessageContent,
        ImageMessageContent,
        VideoMessageContent,
        AudioMessageContent,
        FileMessageContent,
        LocationMessageContent,
        StickerMessageContent,
    ),
)
def handle_message(event):
    url = "http://llm-service:8001/createresponse"

    if event.message.type == "text":
        question = event.message.text
        greeting = [
            "สวัสดี",
            "สวัสดีครับ",
            "สวัสดีค่ะ",
            "สวัสดีคับ",
            "สวัสดีค่า",
            "hello",
            "Hello",
            "HELLO",
        ]
        thank = [
            "ขอบคุณ",
            "ขอบคุณครับ",
            "ขอบคุณค่ะ",
            "ขอบคุณนะครับ",
            "ขอบคุณคับ",
            "ขอบคุนคับ",
            "thank",
            "thank you",
            "แต้งกิ้ว",
            "แต๊งกิ้ว",
            "แต้ง",
            "แต๊ง",
            "ขอบคุน",
        ]
        if question not in greeting and question not in thank:
            response = call_api(question, url, "finetuned")
            answer = ""
            if response == "Timeout":
                answer = "ขออภัยครับ ไม่สามารถตอบคำถามนี้ได้"
            else:
                answer = response["answer"]
            print("Answer from llm-service: ", answer)
            with ApiClient(configuration) as api_client:
                line_bot_api = MessagingApi(api_client)
                line_bot_api.reply_message_with_http_info(
                    ReplyMessageRequest(
                        reply_token=event.reply_token,
                        messages=[TextMessage(text=answer)],
                    )
                )
    else:
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            line_bot_api.reply_message_with_http_info(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text="สวัสดีครับ มีอะไรให้ช่วยไหมครับ ถามได้เลยครับ")],
                )
            )
