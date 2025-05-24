import torch
from transformers import LlamaModel
import pandas as pd
import random
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, required=True)
parser.add_argument('--ckpt_path', type=str, default=None)
parser.add_argument('--use_lora', action="store_true")
parser.add_argument('--llama', action="store_true")
parser.add_argument('--infer_file', type=str, required=True)
parser.add_argument('--predict_file', type=str, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--seed', type=int, required=True)
args = parser.parse_args()
print("Begin to get get embeddings.....")

def seed_everything(seed=23):
    print("seed",seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(123)

infer_data = pd.read_csv(args.infer_file)
instruction_list = infer_data.apply(
    lambda row: pd.Series(
        {'instruction': f"Human: \n" + row['Instruction'] + "\n\nAssistant:\n"}
    ), axis=1
)['instruction'].to_list()


tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
model = LlamaModel.from_pretrained(args.model_name_or_path, device_map='auto')
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"

if torch.cuda.is_available():
    device = torch.device(0)
else:
    device = torch.device('cpu')



embings = []
for prompt in instruction_list:
    # print("tokening.....")
    print(prompt)
    inputs = tokenizer(prompt, padding=True, return_tensors="pt")
    # print("decoding.....")
    with torch.no_grad():
        outputs = model(input_ids = inputs["input_ids"].to(device),  attention_mask = inputs["attention_mask"].to(device), output_hidden_states=True)
    last_hiddens = torch.mean(outputs.last_hidden_state, dim=1)
    last_hiddens2 = torch.squeeze(last_hiddens)
    embings.append(last_hiddens2)

stacked_tensor = torch.stack(embings)
ndarray_data = stacked_tensor.cpu().numpy()
torch.save(ndarray_data, args.predict_file)
