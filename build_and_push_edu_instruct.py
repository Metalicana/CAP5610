"""
Build and upload an open K–12 educational fine-tuning dataset: EduInstruct.

Includes:
 - GSM8K (math reasoning)
 - ARC (science)
 - SciQ (science Q&A)
 - RACE (reading comprehension)

Usage:
    1. pip install datasets huggingface_hub tqdm
    2. python build_and_push_edu_instruct.py --username your_hf_username
    3. You’ll be prompted to login with `huggingface-cli login` if not already logged in.
"""

import argparse
from datasets import load_dataset, concatenate_datasets
from huggingface_hub import HfApi
from tqdm import tqdm

# --- Formatting functions ---
def format_gsm8k(example):
    return {
        "instruction": "Solve the following math problem and explain your reasoning step by step.",
        "input": example["question"],
        "output": example["answer"],
        "subject": "math"
    }

def format_arc(example):
    question = example["question"]
    choices = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(example["choices"]["text"]))
    output = f"Correct answer: {example['answerKey']}"
    return {
        "instruction": "Answer this science question and explain your reasoning.",
        "input": f"{question}\n\nChoices:\n{choices}",
        "output": output,
        "subject": "science"
    }

def format_sciq(example):
    return {
        "instruction": "Answer the following science question clearly and accurately.",
        "input": example["question"],
        "output": example["correct_answer"],
        "subject": "science"
    }

def format_race(example):
    passage = example["article"]
    question = example["question"]
    choices = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(example["options"]))
    output = f"Answer: {example['answer']}"
    return {
        "instruction": "Read the passage and answer the question.",
        "input": f"Passage:\n{passage}\n\nQuestion:\n{question}\n\nChoices:\n{choices}",
        "output": output,
        "subject": "reading"
    }

def main(username: str):
    print("🔹 Loading datasets...")
    gsm8k = load_dataset("openai/gsm8k", "main", split="train").map(format_gsm8k)
    arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train").map(format_arc)
    sciq = load_dataset("sciq", split="train").map(format_sciq)
    race = load_dataset("ehovy/race", "all", split="train").map(format_race)

    print("🔹 Merging datasets...")
    edu_dataset = concatenate_datasets([gsm8k, arc, sciq, race])
    edu_dataset = edu_dataset.shuffle(seed=42)
    print(f"Combined dataset size: {len(edu_dataset):,} examples")

    # Save locally
    edu_dataset.save_to_disk("./EduInstruct")
    print("Saved locally as './EduInstruct'")

    # Push to Hugging Face Hub
    repo_id = f"{username}/EduInstruct"
    print(f"Uploading to Hugging Face Hub at: {repo_id}")
    edu_dataset.push_to_hub(repo_id, private=False)
    print(f"Upload complete! View it at: https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", type=str, required=True, help="Your Hugging Face username")
    args = parser.parse_args()

    # Make sure user is logged in
    try:
        from huggingface_hub import login
        print("Checking Hugging Face authentication...")
        login()  # Opens CLI login prompt if not already authenticated
    except Exception:
        print("Please run `huggingface-cli login` manually before running this script.")

    main(args.username)