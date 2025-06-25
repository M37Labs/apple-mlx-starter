# Phi-2 Fine-tuning with MLX and LoRA

This project fine-tunes the Phi-2 model on the SQuAD v1.1 dataset using MLX and LoRA with 4-bit quantization.

## Setup

1. Install required packages:
```bash
pip install mlx-lm transformers requests numpy tqdm
```

## Training Process

1. **Prepare Data**:
```bash
python cursor_apple_mlx.py
```
This will:
- Download SQuAD v1.1 dataset
- Create training (70,081 samples), validation (8,759 samples), and test (8,759 samples) sets
- Save them in JSONL format in the `train_data` directory

2. **Convert and Quantize Model**:
```bash
python -m mlx_lm convert --hf-path microsoft/phi-2 --mlx-path phi2_quantized -q
```
This creates a 4-bit quantized version of Phi-2 (average 4.504 bits per weight)

3. **Train with LoRA**:
```bash
python -m mlx_lm lora \
    --model phi2_quantized \
    --train \
    --data train_data \
    --num-layers 8 \
    --batch-size 2 \
    --iters 100 \
    --learning-rate 5e-5 \
    --steps-per-report 10 \
    --adapter-path mlx_phi2_finetuned \
    --fine-tune-type lora
```

Training metrics:
- Only 0.024% parameters trainable (655K out of 2.78B)
- Validation loss improved from 2.723 to 2.228
- Peak memory usage: 3.661 GB
- Training speed: 81-234 tokens/sec

## Model Files

The trained model consists of two parts:
1. Base quantized model: `phi2_quantized/`
2. LoRA adapter weights: `mlx_phi2_finetuned/adapters.safetensors`

## Known Issues

1. The current version of MLX has some limitations with the Metal backend for text generation. We're working on finding alternative ways to demonstrate the model's capabilities.

2. For testing the model, you can use either:
   ```bash
   # Option 1: Test on validation set
   python -m mlx_lm lora \
       --model phi2_quantized \
       --resume-adapter-file mlx_phi2_finetuned/adapters.safetensors \
       --test \
       --data train_data \
       --test-batches 5

   # Option 2: Interactive generation (may have Metal backend issues)
   python -m mlx_lm generate \
       --model phi2_quantized \
       --adapter-path mlx_phi2_finetuned \
       --prompt "### Question: What is...?\n\n### Context: ...\n\n### Answer:" \
       --max-tokens 50
   ```

## Training Results

The model shows improvement in performance:
- Training loss decreased from 2.488 to 2.330
- Validation loss improved from 2.723 to 2.228
- The model was trained for 100 iterations
- Used 4-bit quantization to reduce memory usage while maintaining performance

## Directory Structure
```
.
├── cursor_apple_mlx.py      # Main training script
├── train_data/             # Training data directory
│   ├── train.jsonl         # Training samples
│   └── validation.jsonl    # Validation samples
├── phi2_quantized/         # Quantized model directory
└── mlx_phi2_finetuned/     # Fine-tuned model outputs
    └── adapters.safetensors # Trained adapter weights
``` 