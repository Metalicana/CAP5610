# eval_fast.py
import os
import gc
import re
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# --------- CONFIG ---------
MODEL = "meta-llama/Llama-2-7b-hf"  # or your local llama model
FT_MODEL = "./llama2-edu-qlora/lora_adapter"
DATA_DIR = "./EduInstruct"
BATCH_SIZE = 8
MAX_NEW_TOKENS = 256
MAX_EVAL_SAMPLES = None  # or None for full eval

# --------- UTILS ---------
def extract_numeric_answer(text):
    match = re.search(r'####\s*([+-]?\d+\.?\d*)', text)
    if match:
        return match.group(1)
    match = re.search(r'[Aa]nswer:\s*([+-]?\d+\.?\d*)', text)
    if match:
        return match.group(1)
    numbers = re.findall(r'([+-]?\d+\.?\d*)', text)
    return numbers[-1] if numbers else None

def extract_choice_answer(text):
    match = re.search(r'[Aa]nswer:\s*([A-Z])', text)
    if match:
        return match.group(1).upper()
    match = re.search(r'\b([A-Z])\b(?!\w)', text[-50:])
    return match.group(1).upper() if match else None

def extract_ground_truth(example):
    output = example.get('output', '')
    if output:
        if (val := extract_numeric_answer(output)):
            return ('numeric', val)
        if (val := extract_choice_answer(output)):
            return ('choice', val)
        return ('string', output.strip())
    return (None, None)

def parse_model_answer(text, answer_type):
    if answer_type == 'numeric':
        return extract_numeric_answer(text)
    elif answer_type == 'choice':
        return extract_choice_answer(text)
    else:
        return re.split(r'[.!?]\s+', text)[0].strip()

def compare(pred, gold, typ):
    if pred is None or gold is None:
        return False
    try:
        if typ == 'numeric':
            return abs(float(pred) - float(gold)) < 1e-5
        if typ == 'choice':
            return pred.strip().upper() == gold.strip().upper()
        return pred.strip().lower() == gold.strip().lower()
    except:
        return False

def build_prompt(ex):
    return f"### Instruction:\n{ex['instruction']}\n\n### Input:\n{ex['input']}\n\n### Response:\n"

# --------- LOAD ---------
print("Loading dataset...")
dataset = load_from_disk(DATA_DIR)
if MAX_EVAL_SAMPLES:
    dataset = dataset.select(range(min(MAX_EVAL_SAMPLES, len(dataset))))
print(f"Loaded {len(dataset)} samples")

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    use_cache=False
)
base_model.eval()
base_model = PeftModel.from_pretrained(base_model, FT_MODEL)
base_model = torch.compile(base_model)

print("Model ready.")

# --------- EVAL ---------
print("Beginning batched evaluation...")
results = {'correct': 0, 'total': 0, 'by_subject': {}}

batched = dataset.to_dict()
total = len(dataset)

for start in tqdm(range(0, total, BATCH_SIZE)):
    end = min(start + BATCH_SIZE, total)
    batch = Dataset.from_dict({k: v[start:end] for k, v in batched.items()})
    prompts = [build_prompt(ex) for ex in batch]
    gt_list = [extract_ground_truth(ex) for ex in batch]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(base_model.device)

    with torch.no_grad():
        outputs = base_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id
        )

    for i in range(len(batch)):
        raw = tokenizer.decode(outputs[i][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        typ, gold = gt_list[i]
        pred = parse_model_answer(raw, typ)
        if compare(pred, gold, typ):
            results['correct'] += 1
        subject = batch[i].get('subject', 'unknown')
        if subject not in results['by_subject']:
            results['by_subject'][subject] = {'correct': 0, 'total': 0}
        results['by_subject'][subject]['total'] += 1
        if compare(pred, gold, typ):
            results['by_subject'][subject]['correct'] += 1

    results['total'] += len(batch)

# --------- REPORT ---------
print("\n=== FINAL RESULTS ===")
overall = results['correct'] / results['total'] * 100
print(f"Overall: {results['correct']}/{results['total']} = {overall:.2f}%")

for subject, res in results['by_subject'].items():
    acc = res['correct'] / res['total'] * 100
    print(f"  {subject.capitalize():12s} {res['correct']:4d}/{res['total']:4d} = {acc:5.2f}%")

print("Done.")
