import os
import re
import sys
import json
import time
import torch
import pandas as pd
import bitsandbytes as bnb

from torch.nn import DataParallel
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import torch.onnx
from pprint import pprint
from trl import SFTTrainer
from peft import LoraConfig, PeftConfig, PeftModel, AutoPeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from transformers.utils.import_utils import (
    is_accelerate_available,
    is_bitsandbytes_available,
)

print("accelerate = ", is_accelerate_available())
print("bits and bytes = ", is_bitsandbytes_available())

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {DEVICE} device")

model_name_or_path = (
    "/tarafs/data/project/proj0183-ATS/finetune/lanta-finetune/typhoon-7b"
)


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
        load_in_4bit = True,
        device_map = 'auto',
        use_cache = False,
        # torch_dtype = torch.bfloat16,
        temperature = 0.7,
        top_p=0.9,
        repetition_penalty = 1.1,
        do_sample = True,
        pad_token_id=tokenizer.eos_token_id,
    )
    tokenizer.pad_token = tokenizer.eos_token  # </s>
    # tokenizer.add_special_tokens = False
    # tokenizer.padding_side = "right"
    return model, tokenizer


WEIGHT_DIR = "/tarafs/data/project/proj0183-ATS/finetune/lanta-finetune/experiments"  # @param {type:"string"}
model, tokenizer = load_LLM_and_tokenizer()
model.config.use_cache = False
model = PeftModel.from_pretrained(model, WEIGHT_DIR)

model.eval()

# Export the PyTorch model to ONNX
max_sequence_length = 4096
max_new_tokens = 256
temperature = 0.7
# no_repeat_ngram_size=2,
# typical_p=1.,
top_p = 0.95
repetition_penalty = 1.05
num_return_sequences = 1

dummy_input = torch.randint(
    0, 100, (1, max_sequence_length), dtype=torch.int, device="cuda"
)
attention_mask = torch.ones(1, max_sequence_length, dtype=torch.int, device="cuda")
model_input = (
    dummy_input,
    attention_mask,
    max_new_tokens,
    temperature,
    top_p,
    repetition_penalty,
    num_return_sequences,
)
torch.onnx.export(
    model,  # PyTorch model
    (dummy_input, attention_mask),  # Dummy input tensor (adjust shape as needed)
    "./onnx_file/model.onnx",  # Output ONNX file path
    export_params=True,
    verbose=False,
    opset_version=16,
    input_names=["input", "attention_mask"],
)

tokenizer.save_pretrained(
    "/tarafs/data/project/proj0183-ATS/finetune/lanta-finetune/llm-service/tokenizer"
)

model.push_to_hub("seallms-7b-v2")
