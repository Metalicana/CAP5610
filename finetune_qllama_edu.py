import os
import evaluate
import bitsandbytes.nn
from datasets import load_from_disk
from sympy import false
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import bitsandbytes as bnb

# ---- CONFIG ----
MODEL_NAME = "meta-llama/Meta-Llama-3-8B"       # or your local base model
#MODEL_NAME = "meta-llama/Llama-2-7b-hf" #for local with no GPU
#MODEL_NAME = "meta-llama/Llama-2-7b"
#MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
DATA_DIR = "./EduInstruct"
OUTPUT_DIR = "./llama3-edu-qlora"
LORA_R = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = ["q_proj","v_proj","k_proj","o_proj","w1","w2"]  # typical targets, adapt if mismatch
BATCH_SIZE = 4
EPOCHS = 3
LR = 2e-4
MAX_LENGTH = 512

# ---- Load tokenizer & dataset ----
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token

dataset = load_from_disk(DATA_DIR)

def make_text(example):
    text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    return {"text": text}

dataset = dataset.map(make_text, remove_columns=dataset.column_names)
# Tokenize
def tokenize_fn(ex):
    return tokenizer(ex["text"], truncation=True, max_length=MAX_LENGTH)

dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

# split
dataset = dataset.train_test_split(test_size=0.05, seed=42)
train_ds = dataset["train"]
eval_ds = dataset["test"]

# ---- Load quantized model and prepare for k-bit training ----
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # bitsandbytes 4-bit quant
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=bnb_config,
    # quantization_config=bnb.nn.quantization.QuantizationConfig(
    #     load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True
    # )
)

# Prepare for k-bit training
model = prepare_model_for_kbit_training(model)

# ---- PEFT LoRA adapter ----
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# ---- TrainingArgs & Trainer ----
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    fp16=True,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    remove_unused_columns=False,
    report_to="wandb"  # optional; set WANDB_PROJECT env var
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# ---- Train ----
trainer.train()
# Save adapter weights only (small)
model.save_pretrained(OUTPUT_DIR + "/lora_adapter")
print("Saved adapters to", OUTPUT_DIR + "/lora_adapter")