from fastapi import APIRouter, HTTPException, status, Request
from pydantic import BaseModel
import requests
import logging
import json
import os
import aiohttp
from starlette.responses import Response
from models import Question
from linebot.v3 import WebhookHandler
from linebot.v3.messaging.models.push_message_request import PushMessageRequest
from linebot.v3.messaging.models.push_message_response import PushMessageResponse
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
async def is_first_time_api_call(model="finetuned"):
    flag_file = f"api_call_flag_{model}.txt"
    if os.path.exists(flag_file):
        return False
    else:
        with open(flag_file, "w") as f:
            f.write(f"API_call_flag_{model}")
        return True
def is_first_time_api_call_line(model="finetuned"):
    flag_file = f"api_call_flag_{model}.txt"
    if os.path.exists(flag_file):
        return False
    else:
        with open(flag_file, "w") as f:
            f.write(f"API_call_flag_{model}")
        return True

async def call_api(question, url, model="finetuned", timeout=300):
    logging.info("Enter call_api function")
    first_time = await is_first_time_api_call(model)
    if first_time:
        logging.info("This is the first time the API call is being made.")
        timeout = 999999
    else:
        logging.info("This is not the first time the API call is being made.")

    try:
        headers = {"Content-Type": "application/json"}
        payload = {"question": question}
        logging.info(payload)
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=timeout) as response:
                logging.info(response)
                response.raise_for_status()
                return await response.json()
    except aiohttp.ClientError as e:
        logging.info(f"Client error: {str(e)}")
        print(e)
        return "Timeout"
    return "Timeout"

def call_api_line(question, url, model="finetuned", timeout=300):
    logging.info("Enter call_api function")
    first_time = is_first_time_api_call_line(model)
    if first_time:
        logging.info("This is the first time the API call is being made.")
        timeout = 999999
    else:
        logging.info("This is not the first time the API call is being made.")

    try:
        headers = {"Content-Type": "application/json"}
        payload = {"question": question}
        logging.info(payload)
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.info(f"Request error: {str(e)}")
        print(e)
        return "Timeout"
    return "Timeout"


# api for base
@api_service.post("/base", status_code=201)
async def callback(request: Question):
    logging.info("Using base SeaLLM-7B-v2 model")
    try:
        question = request.question

        url = "http://basellm-service:8005/createresponse_base"
        response = await call_api(question, url, "base")
        if response == "Timeout":
            answer = "ขออภัยครับ ไม่สามารถตอบคำถามนี้ได้"
        else:
            answer = response["answer"]
        del response
        return {"answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# api for finetuned chatbot arena
@api_service.post("/finetuned", status_code=201)
async def callback(request: Question):
    logging.info("Using finetuned SeaLLM-7B-v2 model")
    try:
        question = request.question
        url = "http://trtllm-service:8009/createresponse_tensorrt"
        response = await call_api(question, url, "finetuned")
        if response == "Timeout":
            answer = "ขออภัยครับ ไม่สามารถตอบคำถามนี้ได้"
        else:
            answer = response["answer"]
        del response
        return {"answer": answer}

    except Exception as e:
        # If an error occurs, return an HTTPException with status code 500 (Internal Server Error)
        raise HTTPException(status_code=500, detail=str(e))

# api for finetuned chatbot arena
@api_service.post("/tensorrt", status_code=201)
async def callback(request: Question):
    logging.info("Using SeaLLM-7B-v2 with tensorrt  model")
    try:
        question = request.question
        url = "http://llm-service:8009/createresponse_tensorrt"
        response = await call_api(question, url, "tensorrt")
        if response == "Timeout":
            answer = "ขออภัยครับ ไม่สามารถตอบคำถามนี้ได้"
        else:
            answer = response["answer"]
        del response
        return {"answer": answer}

    except Exception as e:
        # If an error occurs, return an HTTPException with status code 500 (Internal Server Error)
        raise HTTPException(status_code=500, detail=str(e))


# Create a new question
@api_service.post("/webhook", status_code=201)
async def callback(request: Request):
    logging.info("Answering LineOA using finetuned SeaLLM-7B-v2 model")
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
    url = "http://trtllm-service:8009/createresponse_tensorrt"

    greeting = [
        "สวัสดี",
        "สวัสดีครับ",
        "สวัสดีค่ะ",
        "สวัสดีคับ",
        "สวัสดีค่า",
        "hello",
        "Hello",
        "HELLO",
        "hi"
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

    if event.message.type == "text":
        question = event.message.text
        greeting_present = any(word in question for word in greeting)
        thank_present = any(word in question for word in thank)
        if not greeting_present and not thank_present:
            with ApiClient(configuration) as api_client:
                line_bot_api = MessagingApi(api_client)
                push_message_request = PushMessageRequest(to=event.source.user_id, messages=[TextMessage(type="text", text="รับทราบครับ โปรดรอสักครู่")])                # push_message_request.to = event.source.user_id                push_message_request = PushMessageRequest(to=event.source.user_id, messages=[TextMessage(type="text", text="รับทราบครับ โปรดรอสักครู่")])                # push_message_request.to = event.source.user_id
                api_response = line_bot_api.push_message(push_message_request)
            response = call_api_line(question, url, "finetuned")
            answer = ""
            if response == "Timeout":
                answer = "ขออภัยครับ ไม่สามารถตอบคำถามนี้ได้"
            else:
                answer = response["answer"]
            logging.info("Answer from llm-service: ", answer)

            with ApiClient(configuration) as api_client:
                line_bot_api = MessagingApi(api_client)
                line_bot_api.reply_message_with_http_info(
                    ReplyMessageRequest(
                        reply_token=event.reply_token,
                        messages=[TextMessage(text=answer)],
                    )
                )
        elif greeting_present and not thank_present:
            with ApiClient(configuration) as api_client:
                line_bot_api = MessagingApi(api_client)
                line_bot_api.reply_message_with_http_info(
                    ReplyMessageRequest(
                        reply_token=event.reply_token,
                        messages=[TextMessage(text="สวัสดีครับ มีอะไรให้ช่วยไหมครับ ถามได้เลยครับ")],
                    )
            )
        elif thank_present and not greeting_present:
            with ApiClient(configuration) as api_client:
                line_bot_api = MessagingApi(api_client)
                line_bot_api.reply_message_with_http_info(
                    ReplyMessageRequest(
                        reply_token=event.reply_token,
                        messages=[TextMessage(text="ยินดีมากครับ")],
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