from huggingface_hub import HfApi
from enum import Enum
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
from pprint import pprint
from trl import SFTTrainer
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    AutoPeftModelForCausalLM
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)

api = HfApi()

class Model(Enum):
    TYPHOON = "typhoon-7b"
    SEALLMS = "seallms-7b-v2"


dir_name = "seallms_experiments"
weight_dir = f"/tarafs/data/project/proj0183-ATS/finetune/lanta-finetune/{dir_name}"
model_name_or_path = "/tarafs/data/project/proj0183-ATS/finetune/lanta-finetune/typhoon-7b"

username = "capstone-chatbot"
model_name = Model.SEALLMS.value
api.upload_folder(
    folder_path=weight_dir,
    repo_id=f"{username}/{model_name}",
    token="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",
    repo_type="model",
)
