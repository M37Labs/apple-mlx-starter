from config import MODEL_NAME, MLX_QUANTIZE_MODEL, TRAIN_DATA_DIR, OUTPUT_DIR
from utils import divider, bright, reset, Fore


def print_training_instructions():
    """Print the training instructions with proper formatting."""
    print(f"{Fore.YELLOW}{bright}🎉 To start training with LoRA and 4-bit quantization, follow these steps:{reset}\n")

    divider()
    print(f"{Fore.YELLOW}⚡ {bright}Notes:{reset}")
    print(f"  {Fore.YELLOW}• Skip step 1a if you already have the quantized model")
    print(f"  • Skip step 1b if you're using an existing MLX quantized model{reset}")
    print(f"  • Ensure model selected is `apple-mlx` compatible{reset}")
    print(f"  • Re-quantizing an already 4bit model -> 4bit will throw error and waste resource{reset}")

    divider('─')
    print(f"{Fore.CYAN}{bright}Step 1: Model preparation{reset}")
    print(f"   {Fore.WHITE}a) If you already have a 4-bit quantized MLX model:{reset}")
    print(f"      {Fore.YELLOW}Simply ensure it's located at: {Fore.GREEN}{MLX_QUANTIZE_MODEL}{reset}")
    print()
    print(f"   {Fore.WHITE}b) If you need to convert/quantize a Hugging Face model:{reset}")
    print(f"""      {Fore.CYAN}python3 -m mlx_lm convert \\
        --hf-path {Fore.GREEN}{MODEL_NAME}{reset} \\
        --mlx-path {Fore.GREEN}{MLX_QUANTIZE_MODEL}{reset} \\
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