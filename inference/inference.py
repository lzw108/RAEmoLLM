import torch
from tqdm import tqdm
from transformers import LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import  PeftModel
import argparse
import pandas as pd
import json
import os
import random
import numpy as np

def seed_everything(seed=23):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, required=True)
parser.add_argument('--use_lora', action="store_true")
parser.add_argument('--llama', action="store_true")
parser.add_argument('--infer_file', type=str, required=True)
parser.add_argument('--predict_file', type=str, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--seed', type=int, required=True)
args = parser.parse_args()

seed_everything(args.seed)

infer_data = pd.read_json(args.infer_file, lines=True)
if "mistral" in args.model_name_or_path.lower():
    instruction_list = infer_data.apply(
        lambda row: pd.Series(
            {'instruction': f"<s>[INST] " + row['instruction'] + "[/INST]"}
        ), axis=1
    )['instruction'].to_list()

elif "llama" in args.model_name_or_path.lower() or "vicuna" in args.model_name_or_path.lower():
    instruction_list = infer_data.apply(
        lambda row: pd.Series(
            {'instruction': f"Human: \n" + row['instruction'] + "\n\nAssistant:\n"}
        ), axis=1
    )['instruction'].to_list()

elif "gemma" in args.model_name_or_path.lower():
    instruction_list = infer_data['instruction'].to_list()

else:
    print("please adjust the format according to different LLMs")

if __name__ == '__main__':
    load_type = torch.float16
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    model_config = AutoConfig.from_pretrained(args.model_name_or_path)

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=load_type, config=model_config, device_map='auto')

    if device==torch.device('cpu'):
        model.float()

    model.eval()
    print("Load model successfully")
    batch_size = args.batch_size
    responses = []
    with open(args.predict_file, 'w', encoding="utf-8") as write_f:
        for i in range(0, len(instruction_list), batch_size):
            batch_data = instruction_list[i: min(i + batch_size, len(instruction_list))]
            inputs = tokenizer(batch_data, return_tensors="pt",padding=True)
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)
            generation_output = model.generate(
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    max_new_tokens=256,
                    do_sample=False
                )
            for j in range(generation_output.shape[0]):
                response = tokenizer.decode(generation_output[j], skip_special_tokens=True, spaces_between_special_tokens=False)
                data_one = {}
                data_one["output"] = response
                write_f.write(json.dumps(data_one, indent=None, ensure_ascii=False) + "\n")
                responses.append(response)
                print(response)
                print("j",j)
            print("i",i)
