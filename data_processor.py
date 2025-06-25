import json
import os
import random
from typing import Dict, List, Tuple
import requests
from config import SQUAD_URL, VALIDATION_SPLIT, TEST_SPLIT, TRAIN_DATA_DIR
from utils import info, success, error, divider


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
                        "text": f"### context: {context}\n\n### user: {question}\n\n### assistant: {answer}"
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


def process_data():
    """Main data processing workflow."""
    # Set seed for reproducibility
    random.seed(42)
    
    # Print welcome message
    divider('=')
    info("Welcome to the SQuAD → MLX-LM Data Preparation Script!")
    print("Prepares your data for model fine-tuning, even if you're not a machine learning guru.")
    divider()
    
    # Download SQuAD data
    info("Step 1: Downloading SQuAD data...")
    try:
        squad_data = download_squad_data()
        success("✓ SQuAD data downloaded successfully.")
    except Exception as e:
        error("Failed to download SQuAD data!")
        error(str(e))
        exit(1)
    divider()
    
    # Extract Q&A samples and format prompts
    info("Step 2: Extracting Q&A samples and formatting prompts...")
    samples = extract_squad_samples(squad_data)
    success(f"✓ Extracted {len(samples):,} formatted samples.")
    print("   Each prompt is structured for MLX-LM training, containing context, question, and answer.")
    divider()
    
    # Split data into training/validation/test sets
    info("Step 3: Splitting into training/validation/test sets...")
    train_samples, val_samples, test_samples = split_data(samples, VALIDATION_SPLIT, TEST_SPLIT)
    print(f"    Training: {len(train_samples):,} samples")
    print(f"    Validation: {len(val_samples):,} samples")
    print(f"    Test: {len(test_samples):,} samples")
    print("Note: Validation and test sets help evaluate your model honestly. "
          "We're not letting it cheat by seeing all the answers first!")
    divider()
    
    # Save datasets as JSONL
    info("Step 4: Saving datasets as JSONL...")
    save_samples_as_jsonl(train_samples, TRAIN_DATA_DIR, "train.jsonl")
    save_samples_as_jsonl(val_samples, TRAIN_DATA_DIR, "valid.jsonl")
    save_samples_as_jsonl(test_samples, TRAIN_DATA_DIR, "test.jsonl")
    divider('=')
    success("All datasets saved! ��")
    divider('=') 