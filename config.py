# Configuration constants
SQUAD_URL = "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/train-v1.1.json"
MODEL_NAME = "mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit"
MLX_QUANTIZE_MODEL = "mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit" #Handle for when non 4bit ??
OUTPUT_DIR = "m37labs-PI1"
TRAIN_DATA_DIR = "train_data"
VALIDATION_SPLIT = 0.1  # 10% for validation
TEST_SPLIT = 0.1  # 10% for testing
# HF_MODEL = "<add_hf_model_repo>"  # keep empty when quantizing new non 4bit model 