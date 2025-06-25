import json
import os
import random
from typing import Dict, List, Tuple


import numpy as np
import requests
from tqdm import tqdm
from transformers import AutoTokenizer


# Constants
SQUAD_URL = "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/train-v1.1.json"
MODEL_NAME = "mlx-community/Mistral-7B-Instruct-v0.2-4bit"
MLX_QUANTIZE_MODEL = 'phi2_quantized'
OUTPUT_DIR = "m37labs-PI1"
TRAIN_DATA_DIR = "train_data"
VALIDATION_SPLIT = 0.1  # 10% for validation
TEST_SPLIT = 0.1  # 10% for testing
# HF_MODEL = "<add_hf_model_repo>"

def download_squad_data() -> Dict:
    """Download SQuAD v1.1 training data."""
    response = requests.get(SQUAD_URL)
    response.raise_for_status()
    return response.json()


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
    # Set random seed for reproducibility
    random.seed(42)
   
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TRAIN_DATA_DIR, exist_ok=True)
   
    # Download and process SQuAD data
    print("Downloading SQuAD data...")
    squad_data = download_squad_data()
   
    print("Extracting samples...")
    samples = extract_squad_samples(squad_data)
    print(f"Extracted {len(samples)} samples")
   
    # Split into training, validation, and test sets
    print("\nSplitting data into training, validation, and test sets...")
    train_samples, val_samples, test_samples = split_data(samples, VALIDATION_SPLIT, TEST_SPLIT)
    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")
   
    # Save samples in JSONL format for MLX-LM
    save_samples_as_jsonl(train_samples, TRAIN_DATA_DIR, "train.jsonl")
    save_samples_as_jsonl(val_samples, TRAIN_DATA_DIR, "valid.jsonl")  # Changed to valid.jsonl as expected by MLX-LM
    save_samples_as_jsonl(test_samples, TRAIN_DATA_DIR, "test.jsonl")
   
    print("\nData preparation complete!")
    print("\nTo start training with LoRA and 4-bit quantization, run the following commands:")
    print("\n1. First, convert and quantize the model:")
    print(f"""python3 -m mlx_lm convert \\
        --hf-path {MODEL_NAME} \\
        --mlx-path {MLX_QUANTIZE_MODEL} \\
        -q""")                              # Comment this line when you want to Upload in HF
        # -q                                # Uncomment this line when you want to Upload in HF
        # --upload-repo {HF_MODEL}""")      # Uncomment this line when you want to Upload in HF
   
    print("\n2. Then, run the training:")
    print(f"""python3 -m mlx_lm lora \\
    --model {MLX_QUANTIZE_MODEL} \\
    --train \\
    --data {TRAIN_DATA_DIR} \\
    --num-layers 8 \\
    --batch-size 2 \\
    --iters 100 \\
    --learning-rate 5e-5 \\
    --steps-per-report 10 \\
    --adapter-path {OUTPUT_DIR} \\
    --fine-tune-type lora""")
   
    print("\n3. To test the model after training:")
    print(f"""python3 -m mlx_lm lora \\
    --model {MLX_QUANTIZE_MODEL} \\
    --adapter-path {OUTPUT_DIR} \\
    --test \\
    --data {TRAIN_DATA_DIR} \\
    --test-batches 5""")


if __name__ == "__main__":
    main()



