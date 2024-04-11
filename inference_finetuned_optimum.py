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
    BitsAndBytesConfig,
)
import onnxruntime as rt
from optimum.nvidia import AutoModelForCausalLM

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {DEVICE} device")

model_name_or_path = "SeaLLMs/SeaLLM-7B-v2"


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
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        # quantization_config=bnb_config,
        trust_remote_code=True,
        # local_files_only=True,
        use_fp8=True,
        device_map="auto",  # NOTE use gpu
    )
    tokenizer.pad_token = tokenizer.eos_token  # </s>
    # tokenizer.add_special_tokens = False
    # tokenizer.padding_side = "right"
    return model, tokenizer


model, tokenizer = load_LLM_and_tokenizer()
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
        )
    response = tokenizer.decode(
        output_tokens[0][len(batch["input_ids"][0]) :], skip_special_tokens=True
    )
    response_time = time.time() - start_time
    return response, response_time


def main(question, source_docs):
    # question = "ขอดูอย่างคดีที่มีการพิพากษาของศาลฎีกาต่างจากศาลอุทธรณ์หน่อยได้ไหมครับ"
    # knowledge = """คดีหมายเลข 934/2566\nคดี 934/2566\nโจทก์ฟ้องและแก้ฟ้องขอให้ลงโทษตามพระราชบัญญัติว่าด้วยความ\nผิดของพนักงานในองค์การหรือหน่วยงานของรัฐ พ.ศ. 2502 มาตรา 4, 8, 11\nประมวลกฎหมายอาญา มาตรา 90, 91, 334, 335 (11) ศาลชั้นต้นไต่สวนมูลฟ้องแล้ว เห็นว่าคดีมีมูล ให้ประทับฟ้อง จำเลยให้การปฏิเสธ ศาลชั้นต้นพิพากษาว่า จำเลยมีความผิดตามพระราชบัญญัติว่าด้วย\nความผิดของพนักงานในองค์การหรือหน่วยงานของรัฐ"""
    text = generate_inference_prompt(question, source_docs)
    answer, response_time = generate_answer_with_timer(text)
    print(answer)
    print(response_time)
    print("\nFinished inference with finetuned typhoon-7b\n")
    # print(answer)
    return answer, response_time
