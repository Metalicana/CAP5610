from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_from_disk
from peft import PeftModel
import numpy as np
import re
from tqdm import tqdm
import torch
import os
import gc

#MODEL = "meta-llama/Llama-3-8b"       # base
MODEL = "meta-llama/Llama-2-7b-hf"
FT_MODEL = "./llama2-edu-qlora/lora_adapter"  # fine-tuned adapters
DATA_DIR = "./EduInstruct"
MAX_EVAL_SAMPLES = None  # Set to None to evaluate all, or a number to limit
FORCE_CPU = False  # Set to True to force CPU-only evaluation (slower but uses less GPU memory)

# Set threading to avoid conflicts
torch.set_num_threads(1)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False, use_cache=False)
tokenizer.pad_token = tokenizer.eos_token

# Determine device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def generate_answer(model, prompt, max_new_tokens=256):
    """Generate answer from model given a prompt."""
    # For quantized models with device_map="auto", inputs are handled automatically
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # For models with device_map="auto", let the model handle device placement
    # Otherwise, move to appropriate device
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        # device_map="auto" - model will handle placement, but we need to move inputs
        # Find the first device from the device map
        first_device = None
        for device_info in model.hf_device_map.values():
            if isinstance(device_info, (list, tuple)) and len(device_info) > 0:
                first_device = device_info[0]
                break
            elif isinstance(device_info, torch.device):
                first_device = device_info
                break
        
        if first_device and first_device.type != 'meta':
            inputs = {k: v.to(first_device) for k, v in inputs.items()}
        elif torch.cuda.is_available():
            inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
    elif hasattr(model, 'device') and model.device.type != 'meta':
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    elif torch.cuda.is_available():
        inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
    
    with torch.no_grad():
        out = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=False, 
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
            num_beams=1  # Disable beam search to avoid threading issues
        )
    # Decode only the generated part (exclude the prompt)
    generated_text = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return generated_text.strip()

def build_eval_prompt(example):
    """Build the same prompt format used in training (without the response)."""
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n"
    return prompt

def extract_numeric_answer(text):
    """Extract numeric answer from text. Handles formats like '#### 97' or 'Answer: 97'."""
    # Look for pattern like "#### 97" (GSM8K format)
    match = re.search(r'####\s*([+-]?\d+\.?\d*)', text)
    if match:
        return match.group(1)
    
    # Look for "Answer: 97" or "answer: 97"
    match = re.search(r'[Aa]nswer:\s*([+-]?\d+\.?\d*)', text)
    if match:
        return match.group(1)
    
    # Extract last number in the text
    numbers = re.findall(r'([+-]?\d+\.?\d*)', text)
    if numbers:
        return numbers[-1]
    
    return None

def extract_choice_answer(text):
    """Extract multiple choice answer (A, B, C, D, etc.) from text."""
    # Look for "Answer: C" or "answer: C" pattern
    match = re.search(r'[Aa]nswer:\s*([A-Z])', text)
    if match:
        return match.group(1).upper()
    
    # Look for standalone letter at the end (possibly after newline)
    match = re.search(r'\b([A-Z])\b(?!\w)', text[-50:])  # Check last 50 chars
    if match:
        return match.group(1).upper()
    
    return None

def extract_ground_truth(example):
    """Extract ground truth answer from example."""
    # Priority: output field, then answer field, then answerKey/correct_answer
    if 'output' in example and example['output']:
        output = example['output']
        # Check if it's a numeric answer (GSM8K format)
        numeric = extract_numeric_answer(output)
        if numeric:
            return ('numeric', numeric)
        # Check if it's a choice answer
        choice = extract_choice_answer(output)
        if choice:
            return ('choice', choice)
        # Otherwise return as string
        return ('string', output.strip())
    
    if 'answer' in example and example['answer']:
        answer = str(example['answer']).strip()
        # Check if it's a single letter (choice)
        if len(answer) == 1 and answer.isalpha():
            return ('choice', answer.upper())
        # Check if it's numeric
        try:
            float(answer)
            return ('numeric', answer)
        except:
            return ('string', answer)
    
    if 'answerKey' in example and example['answerKey']:
        return ('choice', str(example['answerKey']).strip().upper())
    
    if 'correct_answer' in example and example['correct_answer']:
        return ('string', str(example['correct_answer']).strip())
    
    return None

def parse_model_answer(generated_text, answer_type):
    """Parse the model's generated answer based on expected answer type."""
    if answer_type == 'numeric':
        return extract_numeric_answer(generated_text)
    elif answer_type == 'choice':
        return extract_choice_answer(generated_text)
    else:  # string
        # Return the full generated text or first sentence
        sentences = re.split(r'[.!?]\s+', generated_text)
        return sentences[0].strip() if sentences else generated_text.strip()

def compare_answers(predicted, ground_truth, answer_type):
    """Compare predicted answer with ground truth."""
    if predicted is None or ground_truth is None:
        return False
    
    if answer_type == 'numeric':
        try:
            pred_num = float(predicted)
            gt_num = float(ground_truth)
            # Allow small floating point differences
            return abs(pred_num - gt_num) < 1e-5
        except:
            return False
    elif answer_type == 'choice':
        return str(predicted).upper().strip() == str(ground_truth).upper().strip()
    else:  # string
        # Case-insensitive string matching
        return str(predicted).lower().strip() == str(ground_truth).lower().strip()

def evaluate_single_model(dataset, model, model_name, max_samples=None):
    """Evaluate a single model on the dataset."""
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    results = {'correct': 0, 'total': 0, 'by_subject': {}}
    
    print(f"\nEvaluating {model_name} on {len(dataset)} samples...")
    
    # Process in smaller batches to manage memory
    batch_size = 2  # Process and clear memory every N samples
    
    for i, example in enumerate(tqdm(dataset, desc=f"Evaluating {model_name}")):
        # Build prompt
        prompt = build_eval_prompt(example)
        
        # Extract ground truth
        gt_info = extract_ground_truth(example)
        if gt_info is None:
            continue
        
        answer_type, ground_truth = gt_info
        subject = example.get('subject', 'unknown')
        
        # Generate answer
        try:
            answer = generate_answer(model, prompt)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as e:
            print(f"Error generating {model_name} answer: {e}")
            answer = ""
        
        # Parse answer
        parsed = parse_model_answer(answer, answer_type)
        
        # Compare
        correct = compare_answers(parsed, ground_truth, answer_type)
        
        # Update results
        results['total'] += 1
        if correct:
            results['correct'] += 1
        
        # Update by subject
        if subject not in results['by_subject']:
            results['by_subject'][subject] = {'correct': 0, 'total': 0}
        
        results['by_subject'][subject]['total'] += 1
        if correct:
            results['by_subject'][subject]['correct'] += 1
        
        # Periodic memory cleanup after every sample
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    return results

def get_model_loading_config():
    """Get configuration for model loading based on available resources."""
    if FORCE_CPU or not torch.cuda.is_available():
        return {
            'max_memory': None,
            'device_map': None,
            'dtype': torch.float32,
            'device': 'cpu'
        }
    else:
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        max_memory_gb = max(2.0, total_memory * 0.80)  # Use 80% of GPU, minimum 2GB (more conservative)
        max_memory = {0: f"{max_memory_gb:.1f}GiB", "cpu": "30GiB"}
        return {
            'max_memory': max_memory,
            'device_map': "auto",
            'dtype': torch.float16,
            'device': 'cuda:0'
        }

def evaluate_models_separately(dataset, max_samples=None):
    """Evaluate base and fine-tuned models separately to avoid mutex conflicts."""
    results = {
        'base': {'correct': 0, 'total': 0, 'by_subject': {}},
        'ft': {'correct': 0, 'total': 0, 'by_subject': {}}
    }
    
    # Step 1: Evaluate base model
    print("\n" + "="*60)
    print("STEP 1: Evaluating Base Model")
    print("="*60)
    
    print("Loading base model with 4-bit quantization and CPU offloading...")
    # Clear memory before loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Calculate max memory - use most of GPU but leave some buffer
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        max_memory_gb = max(2.0, total_memory * 0.85)  # Use 85% of GPU, minimum 2GB
        max_memory = {0: f"{max_memory_gb:.1f}GiB", "cpu": "30GiB"}
        print(f"GPU memory: {total_memory:.1f}GB, allocating {max_memory_gb:.1f}GB to model")
    else:
        max_memory = None
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # bitsandbytes 4-bit quant
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,  # Enable double quantization to save more memory
        bnb_4bit_compute_dtype=torch.float16 if load_config['device'] != 'cpu' else torch.float32
    )
    
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL, 
            quantization_config=bnb_config,
            device_map=load_config['device_map'],
            max_memory=load_config['max_memory'],
            dtype=load_config['dtype'],
            low_cpu_mem_usage=True,
            use_cache=False
        )
        if load_config['device'] == 'cpu':
            base_model = base_model.to("cpu")
        print("Base model loaded.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying with CPU-only fallback...")
        # Fallback: load to CPU only
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            quantization_config=bnb_config,
            device_map=None,
            max_memory=None,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
            use_cache=False
        ).to("cpu")
        print("Base model loaded on CPU (fallback).")
    
    try:
        results['base'] = evaluate_single_model(dataset, base_model, "Base Model", max_samples)
    finally:
        # Clean up base model
        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("Base model unloaded.")
    
    # Step 2: Evaluate fine-tuned model
    print("\n" + "="*60)
    print("STEP 2: Evaluating Fine-tuned Model")
    print("="*60)
    
    print("Loading fine-tuned model with 4-bit quantization and CPU offloading...")
    # Clear memory before loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Re-get config in case memory situation changed
    load_config = get_model_loading_config()
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # bitsandbytes 4-bit quant
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,  # Enable double quantization to save more memory
        bnb_4bit_compute_dtype=torch.float16 if load_config['device'] != 'cpu' else torch.float32
    )
    
    try:
        base_for_ft = AutoModelForCausalLM.from_pretrained(
            MODEL,
            quantization_config=bnb_config,
            device_map=load_config['device_map'],
            max_memory=load_config['max_memory'],
            dtype=load_config['dtype'],
            low_cpu_mem_usage=True,
            use_cache=False
        )
        if load_config['device'] == 'cpu':
            base_for_ft = base_for_ft.to("cpu")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying with CPU-only fallback...")
        base_for_ft = AutoModelForCausalLM.from_pretrained(
            MODEL,
            quantization_config=bnb_config,
            device_map=None,
            max_memory=None,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
            use_cache=False
        ).to("cpu")
    
    ft_model = PeftModel.from_pretrained(base_for_ft, FT_MODEL)
    print("Fine-tuned model loaded.")
    
    try:
        results['ft'] = evaluate_single_model(dataset, ft_model, "Fine-tuned Model", max_samples)
    finally:
        # Clean up fine-tuned model
        del ft_model, base_for_ft
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("Fine-tuned model unloaded.")
    
    return results

def print_results(results):
    """Print evaluation results in a readable format."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    # Overall accuracy
    base_acc = results['base']['correct'] / results['base']['total'] * 100 if results['base']['total'] > 0 else 0
    ft_acc = results['ft']['correct'] / results['ft']['total'] * 100 if results['ft']['total'] > 0 else 0
    
    print(f"\nOverall Accuracy:")
    print(f"  Base Model:     {results['base']['correct']}/{results['base']['total']} = {base_acc:.2f}%")
    print(f"  Fine-tuned:     {results['ft']['correct']}/{results['ft']['total']} = {ft_acc:.2f}%")
    print(f"  Improvement:    {ft_acc - base_acc:+.2f}%")
    
    # By subject
    print(f"\nAccuracy by Subject:")
    all_subjects = set(results['base']['by_subject'].keys()) | set(results['ft']['by_subject'].keys())
    
    for subject in sorted(all_subjects):
        if subject in results['base']['by_subject']:
            base_subj = results['base']['by_subject'][subject]
            base_subj_acc = base_subj['correct'] / base_subj['total'] * 100 if base_subj['total'] > 0 else 0
        else:
            base_subj_acc = 0
            base_subj = {'correct': 0, 'total': 0}
        
        if subject in results['ft']['by_subject']:
            ft_subj = results['ft']['by_subject'][subject]
            ft_subj_acc = ft_subj['correct'] / ft_subj['total'] * 100 if ft_subj['total'] > 0 else 0
        else:
            ft_subj_acc = 0
            ft_subj = {'correct': 0, 'total': 0}
        
        print(f"  {subject.capitalize():15s} Base: {base_subj['correct']:4d}/{base_subj['total']:4d} ({base_subj_acc:5.2f}%) | "
              f"FT: {ft_subj['correct']:4d}/{ft_subj['total']:4d} ({ft_subj_acc:5.2f}%) | "
              f"Δ: {ft_subj_acc - base_subj_acc:+.2f}%")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    # Load dataset
    print(f"Loading dataset from {DATA_DIR}...")
    dataset = load_from_disk(DATA_DIR)
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Safety check: limit evaluation if MAX_EVAL_SAMPLES is too high
    # if MAX_EVAL_SAMPLES is None:
    #     print("WARNING: MAX_EVAL_SAMPLES is None. This will evaluate all samples and may cause OOM.")
    #     print("Consider setting MAX_EVAL_SAMPLES to a smaller number (e.g., 100-1000) for testing.")
    #     response = input("Continue with full evaluation? (y/n): ")
    #     if response.lower() != 'y':
    #         MAX_EVAL_SAMPLES = 100
    #         print(f"Limiting to {MAX_EVAL_SAMPLES} samples.")
    
    # Clear any CUDA cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Evaluate models separately to avoid mutex conflicts
    results = evaluate_models_separately(dataset, max_samples=MAX_EVAL_SAMPLES)
    
    # Print results
    print_results(results)
    
    # Final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
