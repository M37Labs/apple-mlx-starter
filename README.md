# Apple MLX Script - Modular Structure

This project has been reorganized into a modular structure for better maintainability and readability.


# Project Overview 

> This project is a simple script that downloads a dataset, loads a model, and trains it on the dataset using the Apple MLX Python API. It is designed to be a starting point for your own projects.

> It enables fine-tuning a model on a dataset and evaluating its performance. Using LoRA or QLoRA automatically for 4bit quantized models. 

> It enables conversion of large language models to smaller models for faster inference using built-in quantization script i.e. part of the Apple MLX Python package.

## Usage

### Use `main.py` as entry point

```bash
git clone https://github.com/M37Labs/apple-mlx-starter.git
```

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

```bash
pip install -r requirements.txt
```

```bash
python main.py
```

### Follow the instructions in terminal output to train a model and run inference. Example output (DO NOT COPY THIS):

![example output](<Screenshot 2025-06-26 at 19.49.10.png>)

### Tinker with `config.py` to try new models and databases

## Migration Notes

- All existing functionality is preserved
- The original `apple_mlx_script.py` is now deprecated
- No breaking changes to the user interface
- All imports and dependencies remain the same 

## File Structure

```
apple-mlx/
├── main.py                 # Main entry point
├── config.py              # Configuration constants
├── utils.py               # Utility functions and color formatting
├── data_processor.py      # Data processing functions
├── instructions.py        # Training instructions and command generation
├── apple_mlx_script.py    # Backward compatibility wrapper
├── requirements.txt       # Dependencies
└── train_data/           # Generated training data
    ├── train.jsonl
    ├── valid.jsonl
    └── test.jsonl
```

## Module Descriptions

### `main.py`
- **Purpose**: Main entry point that orchestrates the entire workflow
- **Function**: Imports and calls the data processing and instruction modules
- **Usage**: Run with `python main.py`

### `config.py`
- **Purpose**: Centralized configuration management
- **Contains**: All constants, URLs, model names, and directory paths
- **Benefits**: Easy to modify settings without touching other files

### `utils.py`
- **Purpose**: Utility functions and formatting helpers
- **Contains**: Color formatting functions, divider functions, and style constants
- **Benefits**: Consistent styling across the application

### `data_processor.py`
- **Purpose**: Core data processing functionality
- **Contains**: 
  - SQuAD data download
  - Data extraction and formatting
  - Data splitting (train/validation/test)
  - JSONL file saving
  - Main data processing workflow
- **Benefits**: Isolated data processing logic

### `instructions.py`
- **Purpose**: Training instruction generation
- **Contains**: Formatted command generation for model preparation, training, and testing
- **Benefits**: Clean separation of instruction logic

## Benefits of This Structure

1. **Modularity**: Each file has a single responsibility
2. **Maintainability**: Easier to modify specific functionality
3. **Readability**: Clear separation of concerns
4. **Reusability**: Individual modules can be imported and used separately
5. **Testing**: Easier to write unit tests for individual components
6. **Backward Compatibility**: Existing scripts continue to work

