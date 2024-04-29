from fastapi import APIRouter, HTTPException, status, Request
import requests
import json
from starlette.responses import Response
import os
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


def call_api(question, url, timeout=180):
    try:
        headers = {"Content-Type": "application/json"}
        payload = {"question": question}
        response = requests.post(
            url, headers=headers, data=json.dumps(payload), timeout=timeout
        )
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()  # Assuming the API returns JSON data
    except requests.exceptions.Timeout:
        print("Timeout error: The request took too long to respond.")
    except requests.exceptions.RequestException as e:
        print("Method error:", e)
        # You can handle other types of errors here, like connection errors, etc.
        # For simplicity, just printing the error message for now
    return "Timeout"

#api for base-line
@api_service.post("/baseline", status_code=201)
async def callback(request: Request):
    try:
        # Parse JSON request body
        body = await request.json()

        # Extract necessary data from the JSON body
        question = body.get("question")  # Assuming your JSON request has a "data" field

        # Perform some processing on the data, if needed
        url = "http://llm-service:8005/createresponse"
        response =  call_api(url,question)
        answer = response["answer"]

        # Return a response
        return {"answer": answer}

    except Exception as e:
        # If an error occurs, return an HTTPException with status code 500 (Internal Server Error)
        raise HTTPException(status_code=500, detail=str(e))
#api for finetuned chatbot arena
@api_service.post("/baseline", status_code=201)
async def callback(request: Request):
    try:
        # Parse JSON request body
        body = await request.json()

        # Extract necessary data from the JSON body
        question = body.get("question")  # Assuming your JSON request has a "data" field

        # Perform some processing on the data, if needed
        url = "http://llm-service:8001/createresponse"
        response =  call_api(url,question)
        answer = response["answer"]

        # Return a response
        return {"answer": answer}

    except Exception as e:
        # If an error occurs, return an HTTPException with status code 500 (Internal Server Error)
        raise HTTPException(status_code=500, detail=str(e))



# Create a new question
@api_service.post("/webhook", status_code=201)
async def callback(request: Request):
    # get X-Line-Signature header value
    signature = request.headers.get("X-Line-Signature")

    # get request body as text
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
        response = call_api(question, url)
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
