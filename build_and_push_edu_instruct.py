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
    3. You'll be prompted to login with `huggingface-cli login` if not already authenticated.
"""

import argparse
import sys
from pathlib import Path
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

# --- Formatting functions ---
def format_gsm8k(example):
    """Format GSM8K dataset example."""
    return {
        "instruction": "Solve the following math problem and explain your reasoning step by step.",
        "input": example.get("question", ""),
        "output": example.get("answer", ""),
        "subject": "math"
    }

def format_arc(example):
    """Format ARC dataset example."""
    question = example.get("question", "")
    choices_list = example.get("choices", {}).get("text", [])
    choices = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices_list))
    answer_key = example.get("answerKey", "")
    output = f"Correct answer: {answer_key}"
    return {
        "instruction": "Answer this science question and explain your reasoning.",
        "input": f"{question}\n\nChoices:\n{choices}",
        "output": output,
        "subject": "science"
    }

def format_sciq(example):
    """Format SciQ dataset example."""
    return {
        "instruction": "Answer the following science question clearly and accurately.",
        "input": example.get("question", ""),
        "output": example.get("correct_answer", ""),
        "subject": "science"
    }

def format_race(example):
    """Format RACE dataset example."""
    passage = example.get("article", "")
    question = example.get("question", "")
    options = example.get("options", [])
    choices = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(options))
    answer = example.get("answer", "")
    output = f"Answer: {answer}"
    return {
        "instruction": "Read the passage and answer the question.",
        "input": f"Passage:\n{passage}\n\nQuestion:\n{question}\n\nChoices:\n{choices}",
        "output": output,
        "subject": "reading"
    }

def load_dataset_safe(dataset_name, config=None, split="train", format_func=None, desc=""):
    """Safely load a dataset with error handling."""
    try:
        print(f"  Loading {desc or dataset_name}...")
        if config:
            dataset = load_dataset(dataset_name, config, split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)
        
        if format_func:
            dataset = dataset.map(format_func, desc=f"Formatting {desc or dataset_name}")
        
        print(f"  ✓ Loaded {len(dataset):,} examples from {desc or dataset_name}")
        return dataset
    except Exception as e:
        print(f"  ✗ Error loading {desc or dataset_name}: {e}")
        raise

def validate_dataset(dataset, min_size=1):
    """Validate that dataset has required structure and minimum size."""
    if len(dataset) < min_size:
        raise ValueError(f"Dataset too small: {len(dataset)} < {min_size}")
    
    required_fields = ["instruction", "input", "output", "subject"]
    sample = dataset[0]
    missing_fields = [field for field in required_fields if field not in sample]
    if missing_fields:
        raise ValueError(f"Dataset missing required fields: {missing_fields}")
    
    print(f"  ✓ Dataset validation passed: {len(dataset):,} examples with required fields")

def main(username: str = None, output_dir: str = "./EduInstruct", 
         datasets: list = None, split: str = "train", 
         shuffle_seed: int = 42, push_to_hub: bool = True):
    """
    Main function to build and optionally upload the EduInstruct dataset.
    
    Args:
        username: Hugging Face username (required if push_to_hub=True)
        output_dir: Local directory to save dataset
        datasets: List of dataset names to include (None = all)
        split: Dataset split to use
        shuffle_seed: Random seed for shuffling
        push_to_hub: Whether to upload to Hugging Face Hub
    """
    if push_to_hub and not username:
        raise ValueError("Username is required when push_to_hub=True")
    
    # Default to all datasets if none specified
    if datasets is None:
        datasets = ["gsm8k", "arc", "sciq", "race"]
    
    print("🔹 Loading datasets...")
    dataset_list = []
    
    # Load datasets based on selection
    if "gsm8k" in datasets:
        gsm8k = load_dataset_safe("openai/gsm8k", "main", split, format_gsm8k, "GSM8K")
        dataset_list.append(gsm8k)
    
    if "arc" in datasets:
        arc = load_dataset_safe("allenai/ai2_arc", "ARC-Challenge", split, format_arc, "ARC")
        dataset_list.append(arc)
    
    if "sciq" in datasets:
        sciq = load_dataset_safe("allenai/sciq", None, split, format_sciq, "SciQ")
        dataset_list.append(sciq)
    
    if "race" in datasets:
        race = load_dataset_safe("ehovy/race", "all", split, format_race, "RACE")
        dataset_list.append(race)
    
    if not dataset_list:
        raise ValueError("No datasets selected. Choose from: gsm8k, arc, sciq, race")
    
    print(f"\n🔹 Merging {len(dataset_list)} datasets...")
    edu_dataset = concatenate_datasets(dataset_list)
    edu_dataset = edu_dataset.shuffle(seed=shuffle_seed)
    print(f"Combined dataset size: {len(edu_dataset):,} examples")
    
    # Validate dataset
    print("\n🔹 Validating dataset...")
    validate_dataset(edu_dataset)
    
    # Save locally
    output_path = Path(output_dir)
    print(f"\n🔹 Saving dataset to {output_path}...")
    try:
        edu_dataset.save_to_disk(str(output_path))
        print(f"✓ Saved locally to '{output_path}'")
    except Exception as e:
        print(f"✗ Error saving dataset: {e}")
        raise
    
    # Push to Hugging Face Hub
    if push_to_hub:
        repo_id = f"{username}/EduInstruct"
        print(f"\n🔹 Uploading to Hugging Face Hub at: {repo_id}")
        try:
            edu_dataset.push_to_hub(repo_id, private=False)
            print(f"✓ Upload complete! View it at: https://huggingface.co/datasets/{repo_id}")
        except Exception as e:
            print(f"✗ Error uploading to Hub: {e}")
            print("Dataset was saved locally. You can upload manually later.")
            raise
    else:
        print("\n✓ Dataset saved locally (skipping Hub upload)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build and upload the EduInstruct educational dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--username", 
        type=str, 
        default=None,
        help="Your Hugging Face username (required if --push-to-hub is set)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./EduInstruct",
        help="Local directory to save the dataset (default: ./EduInstruct)"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        choices=["gsm8k", "arc", "sciq", "race"],
        default=None,
        help="Datasets to include (default: all)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)"
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)"
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Skip uploading to Hugging Face Hub (only save locally)"
    )
    parser.add_argument(
        "--skip-login",
        action="store_true",
        help="Skip Hugging Face login check (use if already authenticated)"
    )
    
    args = parser.parse_args()
    
    # Check authentication if pushing to hub
    push_to_hub = not args.no_push
    if push_to_hub:
        if not args.username:
            parser.error("--username is required when pushing to Hub (or use --no-push to skip upload)")
        
        if not args.skip_login:
            try:
                from huggingface_hub import login
                print("Checking Hugging Face authentication...")
                login()  # Opens CLI login prompt if not already authenticated
                print("✓ Authentication successful")
            except Exception as e:
                print(f"⚠ Warning: Could not verify authentication: {e}")
                print("Please run `huggingface-cli login` manually if upload fails.")
    
    try:
        main(
            username=args.username,
            output_dir=args.output_dir,
            datasets=args.datasets,
            split=args.split,
            shuffle_seed=args.shuffle_seed,
            push_to_hub=push_to_hub
        )
        print("\n✓ All done!")
    except KeyboardInterrupt:
        print("\n✗ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)