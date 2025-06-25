# MLX Fine-tuning with SQuAD Dataset

This repository contains code for fine-tuning a Mistral-7B model using Apple's MLX framework with the SQuAD dataset.

## Requirements

- macOS (Apple Silicon Mac recommended)
- Python 3.9+
- Git

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/M37Labs/apple-mlx.git
cd apple-mlx
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install mlx mlx-lm transformers numpy requests tqdm
```

## Usage

### 1. Prepare the Dataset

Run the data preparation script to download and format the SQuAD dataset:

```bash
python apple_mlx_script.py
```

This script will:

- Download SQuAD v1.1 training data
- Format it for fine-tuning
- Split it into training, validation, and test sets
- Save the data in the required format
- **Output the exact commands to run for the next steps**

### 2-4. Follow the Generated Commands

After running the script, it will output specific commands for:

1. Converting and quantizing the model
2. Running the training with LoRA
3. Testing the fine-tuned model

**Important:** The exact commands are generated dynamically based on the variables defined in the script (like `MODEL_NAME`, `MLX_QUANTIZE_MODEL`, `OUTPUT_DIR`, etc.). Always use the commands provided by the script output rather than copying from this README.

Example of what the generated commands might look like:

```bash
# 1. Convert and quantize the model:
python -m mlx_lm convert \
    --hf-path mlx-community/Mistral-7B-Instruct-v0.2-4bit \
    --mlx-path phi2_quantized \
    -q

# 2. Run the training:
python -m mlx_lm lora \
    --model phi2_quantized \
    --train \
    --data train_data \
    --num-layers 8 \
    --batch-size 2 \
    --iters 100 \
    --learning-rate 5e-5 \
    --steps-per-report 10 \
    --adapter-path m37labs-PI1 \
    --fine-tune-type lora

# 3. Test the model:
python -m mlx_lm lora \
    --model phi2_quantized \
    --adapter-path m37labs-PI1 \
    --test \
    --data train_data \
    --test-batches 5
```

## Project Structure

- `apple_mlx_script.py`: Main script for data preparation
- `train_data/`: Directory containing formatted training data
- `m37labs-PI1/`: Output directory for the fine-tuned model
- `phi2_quantized/`: Directory for the quantized model

## Notes

- This project uses 4-bit quantization to reduce memory usage
- LoRA (Low-Rank Adaptation) is used for efficient fine-tuning
- The model is fine-tuned on the SQuAD question-answering dataset
- You can modify the variables at the top of `apple_mlx_script.py` to customize model names, output directories, etc.

## Optional: Uploading to Hugging Face

To upload your model to Hugging Face, edit the `apple_mlx_script.py` file:

1. Uncomment the relevant lines in the script
2. Add your HF model repository name to the `HF_MODEL` variable
