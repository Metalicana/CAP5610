## **Build scripts**
1) Build and Push Edu Instruct Dataset
pip install datasets huggingface_hub tqdm
python build_and_push_edu_instruct.py --username viennh2012

2) Install Llama
- pip install llama-stack
- download.sh https://download.llamameta.net/*?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiaXExYTM4bmE2ZnltNGtleXN4M2xiOWRzIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZG93bmxvYWQubGxhbWFtZXRhLm5ldFwvKiIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc2MzYxNzc4Mn19fV19&Signature=GsKLW2wbU3AcVkxzcSFAktVXN88geX3aITCCS4BcR%7EHds3Sx7atPesrDyCGGEQkmEDhS%7ECa5zzJI9Y4MvmskYvjvasnw3JWlx%7Eutq52EmV%7E-4sMTZPbiCuwyD6rTU7f1FIKU-5VTq7KYV35SboewjrnADyHI7g-0hzugXLQd0TcZ7qbF7etXEcTvZU%7E0Km6tUSaRvOJ1XUKJ8jMFvMK8VP%7EHj%7E6mJRMVuJ8tHCI-fvopZzWajWn%7ERIregrvy9iwf9Vl6DtkWDWIqPLWUwna4Gw51qi%7EOziLfvq8rDUgHXJBn0ns0g1P4xG0BiH17dhyWaAi6YGkOZUsk4rnFz8s3xQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1149885780620845
- Install llama-models cli
  - pyenv install 3.14.0
  - pyenv global 3.14.0
  - deactivate
  - source .venv2/bin/activate
  - pip install llama-models
  - hf auth login
    - Token: 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 
    - Select "n" for adding to git credential
  - Make sure you have python3.10 or higher
  - Wandb API key:
    - f6593607fe4fe920d6cd3cfba61a9f3d232752dd
3) Finetune Dataset 
+ pip install -U transformers datasets accelerate peft bitsandbytes einops evaluate rouge_score
+ pip install wandb
+ Notes:
   - load_in_4bit=True requires bitsandbytes v0.39+ and CUDA GPU.
   - target_modules vary by model impl; check your model’s module names.
   - For CPU inference or full model, you can peft load adapters back onto a base model.

+ LoRA
- load_in_4bit=False
- use fp16 or bf16 accordingly


4) Hyperparameters & compute guidance
- Suggested starting hyperparameters (adapt to dataset size & GPU):
  + Epochs: 3 (monitor eval & early-stop)
  + Batch size per device: 2–8 (depending on GPU RAM)
  + Learning rate: 2e-4 (QLoRA), tune 1e-4–5e-4
  + LoRA rank r: 8–32 (start 16)
  + Max length: 256–1024 (Edu prompts likely short — use 512)
  + Optimizer: AdamW (transformers default)
  + Scheduler: linear with warmup 0–300 steps
- Compute rough estimates:
  + QLoRA on 7B/8B: can run on a single 80GB A100 or multi-GPU (NVIDIA 40GB A100 x2) comfortably.
  + 13B+ benefits from multi-GPU or flatten tricks.
- If you don’t have GPUs, consider smaller model (Mistral 7B, LLaMA-2 7B) or cloud instances.

For CPU inference or full model, you can peft load adapters back onto a base model.

## **HuggingFace Datasets**
ARC: https://huggingface.co/datasets/allenai/ai2_arc
RACE: https://huggingface.co/datasets/ehovy/race