import os
import re
import sys
import json
import time
import torch
import numpy as np
import onnx

from torch.nn import DataParallel
from transformers import (
    AutoTokenizer,
)
import onnxruntime as rt

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {DEVICE} device")


# Load both LLM model and tokenizer
def load_LLM_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        "/Users/earthakkharawat/Documents/senior/capstone/capstone-chatbot-deployment/llm-service/tokenizer"
    )
    sessOptions = rt.SessionOptions()
    # sessOptions.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    model = rt.InferenceSession(
        "/Users/earthakkharawat/Documents/senior/capstone/capstone-chatbot-deployment/llm-service/model.onnx",
        sessOptions,
    )
    return model, tokenizer, sessOptions


model, tokenizer, sessOptions = load_LLM_and_tokenizer()
# model.config.use_cache = False

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = DataParallel(model).module


INFERENCE_SYSTEM_PROMPT = (
    "คุณคือนักกฎหมายที่จะตอบคำถามเกี่ยวกับกฎหมาย จงตอบคำถามโดยใช้ความรู้ที่ให้ดังต่อไปนี้"
)


def generate_inference_prompt(
    question: str, knowledge: str, system_prompt: str = INFERENCE_SYSTEM_PROMPT
) -> str:
    return f"""<s>[INST] คำสั่ง:
	{system_prompt.strip()}

	คำถาม:
	{question.strip()}

	ข้อมูลที่ให้:
	{knowledge.strip()}

	คำตอบ:[/INST]</s>""".strip()


max_new_tokens = 256  # @param {type: "integer"}
temperature = 0.7  # @param {type: "number"}


def generate_answer_with_timer(text: str):
    start_time = time.time()
    batch = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=4096,
    )
    # print(batch["input_ids"])
    # print("\n\n")
    # print(batch["attention_mask"])
    with torch.cuda.amp.autocast():
        output_tokens = model.run(
            [],
            {
                "input": batch["input_ids"].int().numpy(),
                "attention_mask": batch["attention_mask"].int().numpy(),
            },
        )

    # print(output_tokens[0].shape)
    token_ids = np.argmax(output_tokens, axis=-1)
    response = tokenizer.decode(token_ids[0][0], skip_special_tokens=True)
    response_time = time.time() - start_time
    # print(response)
    return response, response_time


def main(question, source_docs):
    # question = "ขอดูอย่างคดีที่มีการพิพากษาของศาลฎีกาต่างจากศาลอุทธรณ์หน่อยได้ไหมครับ"
    # knowledge = """คดีหมายเลข 934/2566\nคดี 934/2566\nโจทก์ฟ้องและแก้ฟ้องขอให้ลงโทษตามพระราชบัญญัติว่าด้วยความ\nผิดของพนักงานในองค์การหรือหน่วยงานของรัฐ พ.ศ. 2502 มาตรา 4, 8, 11\nประมวลกฎหมายอาญา มาตรา 90, 91, 334, 335 (11) ศาลชั้นต้นไต่สวนมูลฟ้องแล้ว เห็นว่าคดีมีมูล ให้ประทับฟ้อง จำเลยให้การปฏิเสธ ศาลชั้นต้นพิพากษาว่า จำเลยมีความผิดตามพระราชบัญญัติว่าด้วย\nความผิดของพนักงานในองค์การหรือหน่วยงานของรัฐ"""
    text = generate_inference_prompt(question, source_docs)
    answer, response_time = generate_answer_with_timer(text)
    substring = "คำตอบ:"
    start_index = text.find(substring)
    if start_index != -1:
        answer = answer[start_index + len(substring) :]
        print(answer)
    print("\nFinished inference with finetuned typhoon-7b\n")
    # print(answer)
    return answer, response_time
