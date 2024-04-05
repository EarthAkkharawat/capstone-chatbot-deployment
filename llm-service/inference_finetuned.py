import os
import re
import sys
import json
import time
import torch
import pandas as pd
import bitsandbytes as bnb

from torch.nn import DataParallel
from peft import AutoPeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    PeftModel,
)

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {DEVICE} device")


model_name_or_path = "SeaLLMs/SeaLLM-7B-v2"
finetuned_weight = "./seallms_experiments"


# Load both LLM model and tokenizer
def load_LLM_and_tokenizer():
    bnb_config = BitsAndBytesConfig(
        # load_in_8bit=True,
        # llm_int8_has_fp16_weight=True,
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
        # local_files_only=True,
        device_map="auto",  # NOTE use gpu
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )
    # tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        finetuned_weight,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


model, tokenizer = load_LLM_and_tokenizer()
model.config.use_cache = False
model = PeftModel.from_pretrained(model, finetuned_weight)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = DataParallel(model).module


INFERENCE_SYSTEM_PROMPT = """คุณคือนักกฎหมายที่จะตอบคำถามเกี่ยวกับกฎหมาย จงตอบคำถามโดยใช้ความรู้ที่ให้ดังต่อไปนี้
ถ้าหากคุณไม่รู้คำตอบ ให้ตอบว่าไม่รู้ อย่าสร้างคำตอบขึ้นมาเอง"""


def generate_inference_prompt(
    question: str, knowledge: str, system_prompt: str = INFERENCE_SYSTEM_PROMPT
) -> str:
    return f"""<s><|im_start|>system
{system_prompt.strip()}\nความรู้ที่ให้:
{knowledge.strip()}</s><|im_start|>user
{question.strip()}</s><|im_start|>assistant
"""


max_new_tokens = 512  # @param {type: "integer"}
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
    with torch.cuda.amp.autocast():
        output_tokens = model.generate(
            input_ids=batch["input_ids"].to(
                DEVICE
            ),  # NOTE if gpu is unavailable DELETE ".to(DEVICE)"
            attention_mask=batch["attention_mask"].to(DEVICE),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
            repetition_penalty=1.05,
            num_return_sequences=1,
            do_sample=True,
            use_cache=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(
        output_tokens[0][len(batch["input_ids"][0]) :], skip_special_tokens=True
    )
    response_time = time.time() - start_time
    return response, response_time


def main(question, knowledge):
    # question = "เมื่อไหร่ที่สมาคมนายจ้างจะถือว่าเลิก"
    # knowledge = """พระราชบัญญัติแรงงานสัมพันธ์ (ฉบับที่ 3) พ.ศ. 2544 - หมวด 6 (สมาคมนายจ้าง)

    # มาตรา 82  สมาคมนายจ้างย่อมเลิกด้วยเหตุใดเหตุหนึ่ง ดังต่อไปนี้
    # (1) ถ้ามีข้อบังคับของสมาคมนายจ้างกำหนดให้เลิกในกรณีใด เมื่อมีกรณีนั้น
    # (2) เมื่อที่ประชุมใหญ่มีมติให้เลิก
    # (3) เมื่อนายทะเบียนมีคำสั่งให้เลิก
    # (4) เมื่อล้มละลาย"""
    text = generate_inference_prompt(question, knowledge)
    answer, response_time = generate_answer_with_timer(text)
    print("\nFinished inference with finetuned seallms-7b-v2\n")
    # print(answer)
    # print(response_time)
    return answer, response_time
