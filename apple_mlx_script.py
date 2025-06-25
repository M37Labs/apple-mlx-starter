import json
import os
import random
from typing import Dict, List, Tuple


import numpy as np
import requests
from tqdm import tqdm
from transformers import AutoTokenizer

from colorama import init, Fore, Style

init(autoreset=True)

def info(msg): print(Fore.CYAN + msg)
def success(msg): print(Fore.GREEN + msg)
def warn(msg): print(Fore.YELLOW + msg)
def error(msg): print(Fore.RED + msg)
def bold(msg): print(Style.BRIGHT + msg)
def divider(char='-', length=60): print(Fore.MAGENTA + char * length)


# Constants
SQUAD_URL = "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/train-v1.1.json"
MODEL_NAME = "mlx-community/Mistral-7B-Instruct-v0.2-4bit"
MLX_QUANTIZE_MODEL = 'mlx-community/Mistral-7B-Instruct-v0.2-4bit' # keep emtpy when quantizing new non 4bit model
OUTPUT_DIR = "m37labs-PI1"
TRAIN_DATA_DIR = "train_data"
VALIDATION_SPLIT = 0.1  # 10% for validation
TEST_SPLIT = 0.1  # 10% for testing
# HF_MODEL = "<add_hf_model_repo>" # keep emtpy when quantizing new non 4bit model

def download_squad_data() -> Dict:
    """Download SQuAD v1.1 training data."""
    response = requests.get(SQUAD_URL)
    response.raise_for_status()
    return response.json()


# This function will be change based on the data you want to use
def extract_squad_samples(data: Dict) -> List[Dict[str, str]]:
    """Extract and format SQuAD samples into prompt format."""
    samples = []
    for article in data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                answer = qa['answers'][0]['text']  # Taking first answer only
               
                # Format in the specified prompt template
                formatted_text = {
                    "text": f"### system: {context}\n\n### user: {question}\n\n### assistant: {answer}"
                }
                samples.append(formatted_text)
    return samples


def split_data(samples: List[Dict[str, str]], val_split: float, test_split: float) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """Split samples into training, validation, and test sets."""
    # Shuffle the samples
    random.shuffle(samples)
   
    # Calculate split indices
    total_samples = len(samples)
    val_size = int(total_samples * val_split)
    test_size = int(total_samples * test_split)
   
    # Split the data
    test_samples = samples[:test_size]
    val_samples = samples[test_size:test_size + val_size]
    train_samples = samples[test_size + val_size:]
   
    return train_samples, val_samples, test_samples


def save_samples_as_jsonl(samples: List[Dict[str, str]], output_dir: str, filename: str):
    """Save samples in JSONL format required by MLX-LM."""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, filename)
   
    print(f"Saving {len(samples)} samples to {output_file}")
    with open(output_file, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")


def main():
    random.seed(42)

    divider('=')
    info("Welcome to the SQuAD → MLX-LM Data Preparation Script!")
    print(Fore.WHITE + "Prepares your data for model fine-tuning, even if you're not a machine learning guru.")
    divider()

    info("Step 1: Downloading SQuAD data...")
    try:
        squad_data = download_squad_data()
        success("✓ SQuAD data downloaded successfully.")
    except Exception as e:
        error("Failed to download SQuAD data!")
        error(str(e))
        exit(1)
    divider()

    info("Step 2: Extracting Q&A samples and formatting prompts...")
    samples = extract_squad_samples(squad_data)
    success(f"✓ Extracted {len(samples):,} formatted samples.")
    print("   Each prompt is structured for MLX-LM training, containing context, question, and answer.")
    divider()

    info("Step 3: Splitting into training/validation/test sets...")
    train_samples, val_samples, test_samples = split_data(samples, VALIDATION_SPLIT, TEST_SPLIT)
    print(f"    {Fore.GREEN}Training: {len(train_samples):,} samples")
    print(f"    {Fore.YELLOW}Validation: {len(val_samples):,} samples")
    print(f"    {Fore.CYAN}Test: {len(test_samples):,} samples")
    print(f"{Fore.CYAN}Note: Validation and test sets help evaluate your model honestly. "
          "We're not letting it cheat by seeing all the answers first!")
    divider()

    info("Step 4: Saving datasets as JSONL...")
    save_samples_as_jsonl(train_samples, TRAIN_DATA_DIR, "train.jsonl")
    save_samples_as_jsonl(val_samples, TRAIN_DATA_DIR, "valid.jsonl")
    save_samples_as_jsonl(test_samples, TRAIN_DATA_DIR, "test.jsonl")
    divider('=')
    success("All datasets saved! 🚀")
    divider('=')

bright = Style.BRIGHT
reset = Style.RESET_ALL


print(f"{Fore.GREEN}{bright}🎉 Data preparation complete!{reset}\n")
print(f"{Fore.YELLOW}{bright}To start training with LoRA and 4-bit quantization, follow these steps:{reset}\n")


divider()
print(f"{Fore.YELLOW}⚡ {bright}Notes:{reset}")
print(f"  {Fore.YELLOW}• Skip step 1a if you already have the quantized model")
print(f"  • Skip step 1b if you're using an existing MLX quantized model{reset}")
print(f"  • Ensure model selected is `apple-mlx` compatible{reset}")
print(f"  • Re-quantizing an already 4bit model -> 4bit will throw error and waste resource{reset}")

divider('─')
print(f"{Fore.CYAN}{bright}Step 1: Model preparation{reset}")
print(f"   {Fore.WHITE}a) If you already have a 4-bit quantized MLX model:{reset}")
print(f"      {Fore.YELLOW}Simply ensure it's located at: {Fore.GREEN}MLX_QUANTIZE_MODEL{reset}")
print()
print(f"   {Fore.WHITE}b) If you need to convert/quantize a Hugging Face model:{reset}")
print(f"""      {Fore.CYAN}python3 -m mlx_lm convert \\
        --hf-path {Fore.GREEN}{MODEL_NAME}{reset} \\
        --mlx-path {Fore.GREEN}MLX_QUANTIZE_MODEL{reset} \\
        -q{reset}
""")

divider('─')
print(f"{Fore.CYAN}{bright}Step 2: Training{reset}")
print(f"""      {Fore.CYAN}python3 -m mlx_lm lora \\
        --model {Fore.GREEN}{bright}{MLX_QUANTIZE_MODEL}{reset} \\
        --train \\
        --data {Fore.GREEN}{bright}{TRAIN_DATA_DIR}{reset} \\
        --num-layers 8 \\
        --batch-size 2 \\
        --iters 100 \\
        --learning-rate 5e-5 \\
        --steps-per-report 10 \\
        --adapter-path {Fore.GREEN}{bright}{OUTPUT_DIR}{reset} \\
        --fine-tune-type lora{reset}
""")

divider('─')
print(f"{Fore.CYAN}{bright}Step 3: Testing your fine-tuned model{reset}")
print(f"""      {Fore.CYAN}python3 -m mlx_lm lora \\
        --model {Fore.GREEN}{bright}{MLX_QUANTIZE_MODEL}{reset} \\
        --adapter-path {Fore.GREEN}{bright}{OUTPUT_DIR}{reset} \\
        --test \\
        --data {Fore.GREEN}{bright}{TRAIN_DATA_DIR}{reset} \\
        --test-batches 5{reset}
""")



divider('═')
print(f"{Fore.GREEN}{bright}You're all set to train and test your model like a pro! 🚀{reset}")
divider('═')



if __name__ == "__main__":
    main()



