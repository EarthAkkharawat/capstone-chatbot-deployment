import os
import re
import sys
import json
import time
import torch
import pandas as pd
import bitsandbytes as bnb

from torch.nn import DataParallel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
#from optimum.nvidia import AutoModelForCausalLM
from optimum.nvidia.quantization import AutoQuantizationConfig, Float8QuantizationConfig
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
    tokenizer = AutoTokenizer.from_pretrained(
        finetuned_weight,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        # load_in_8bit=True,
        # llm_int8_has_fp16_weight=True,
        load_in_4bit=True,
        # bnb_4bit_use_double_quant=True,
        # bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    # qconfig = AutoQuantizationConfig.from_description(
    #     weight="float8",
    #     activation="float8",
    #     tokenizer=tokenizer,
    #     dataset="c4-new",
    #     max_sequence_length=4096,
    #     # device="cuda"
    # )
    print("Finished quantization")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
        # local_files_only=True,
        device_map="auto",  # NOTE use gpu
        torch_dtype=torch.bfloat16,
        use_cache=False,
        # use_fp8=True,
    )
    print("Finished load base model")
    model.config.use_cache = False
    model = PeftModel.from_pretrained(model, finetuned_weight)
    return model, tokenizer


model, tokenizer = None, None

print("Let's use", torch.cuda.device_count(), "GPUs!")


INFERENCE_SYSTEM_PROMPT = """คุณคือนักกฎหมายที่จะตอบคำถามเกี่ยวกับกฎหมาย จงตอบคำถามโดยใช้ความรู้ที่ให้ดังต่อไปนี้
ถ้าหากคุณไม่รู้คำตอบ ให้ตอบว่าไม่รู้ อย่าสร้างคำตอบขึ้นมาเอง"""


def generate_inference_prompt(
    question: str, knowledge: str, system_prompt: str = INFERENCE_SYSTEM_PROMPT
) -> str:
    global tokenizer
    if tokenizer is None:
        _, tokenizer = load_LLM_and_tokenizer()
    batch = tokenizer.encode(
        knowledge,
        return_tensors="pt",
        add_special_tokens=False,
        max_length=2800,
        truncation=True,
        padding="max_length",
    )
    knowledge = tokenizer.decode(batch[0], skip_special_tokens=True)
    return f"""<s><|im_start|>system
{system_prompt.strip()}\nความรู้ที่ให้:
{knowledge.strip()}</s><|im_start|>user
{question.strip()}</s><|im_start|>assistant
"""


def generate_answer_with_timer(text: str):
    start_time = time.time()
    global model
    global tokenizer
    if model is None or tokenizer is None:
        model, tokenizer = load_LLM_and_tokenizer()
    print("\nUsing model and tokenizer\n")
    batch = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    print("\nFinished tokenize input\n")

    with torch.cuda.amp.autocast():
        output_tokens = model.generate(
            input_ids=batch["input_ids"].to(
                DEVICE
            ),  # NOTE if gpu is unavailable DELETE ".to(DEVICE)"
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            num_return_sequences=1,
            do_sample=True,
            use_cache=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    print("\nFinished generate answer\n")
    print("Batch length:", batch["input_ids"].shape)
    response = tokenizer.decode(
        output_tokens[0][len(batch["input_ids"][0]) :], skip_special_tokens=True
    )
    del output_tokens
    del batch
    response_time = time.time() - start_time
    return response, response_time

def detect_foreign_characters(text):
    # Regular expression pattern to match Thai, English letters, digits, and special characters
    pattern = r'[^\u0E00-\u0E7F\u0041-\u005A\u0061-\u007A\u0030-\u0039\u0020-\u007E\“”]'
    # Find all characters that do not match the pattern
    foreign_chars = re.findall(pattern, text)
    return foreign_chars

def main(question, knowledge):
    # question = "เมื่อไหร่ที่สมาคมนายจ้างจะถือว่าเลิก"
    # knowledge = """พระราชบัญญัติแรงงานสัมพันธ์ (ฉบับที่ 3) พ.ศ. 2544 - หมวด 6 (สมาคมนายจ้าง)

    # มาตรา 82  สมาคมนายจ้างย่อมเลิกด้วยเหตุใดเหตุหนึ่ง ดังต่อไปนี้
    # (1) ถ้ามีข้อบังคับของสมาคมนายจ้างกำหนดให้เลิกในกรณีใด เมื่อมีกรณีนั้น
    # (2) เมื่อที่ประชุมใหญ่มีมติให้เลิก
    # (3) เมื่อนายทะเบียนมีคำสั่งให้เลิก
    # (4) เมื่อล้มละลาย"""
    if knowledge == "":
        return "ขออภัยครับ ไม่สามารถตอบคำถามได้", 0.0
    text = generate_inference_prompt(question, knowledge)
    answer, response_time = generate_answer_with_timer(text)
    print("\nFinished inference with finetuned seallms-7b-v2\n")
    print("Answer before post-processing: \n ", answer)

    if "<" in answer and ">" in answer:
        start_index = answer.find("<")
        end_index = answer.find(">") + 1
        answer = answer.replace(answer[start_index:end_index], "").strip()

    if "\n" in answer:
        answer = answer.replace("\n", " ")

    non_standard_chars = detect_foreign_characters(answer)
    if non_standard_chars:
        answer = "ขออภัยครับ ไม่สามารถตอบคำถามนี้ได้"
        print("Foreign characters found: ", non_standard_chars)
    else:
        print("No Foreign characters found.")

    del text, non_standard_chars
    print("Answer after post-processing: \n")
    print(answer)
    print(response_time)
    torch.cuda.empty_cache()
    return answer, response_time