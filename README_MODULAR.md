# Apple MLX Script - Modular Structure

This project has been reorganized into a modular structure for better maintainability and readability.

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

### `apple_mlx_script.py`
- **Purpose**: Backward compatibility wrapper
- **Function**: Maintains compatibility with existing scripts
- **Usage**: Can still run with `python apple_mlx_script.py`

## Benefits of This Structure

1. **Modularity**: Each file has a single responsibility
2. **Maintainability**: Easier to modify specific functionality
3. **Readability**: Clear separation of concerns
4. **Reusability**: Individual modules can be imported and used separately
5. **Testing**: Easier to write unit tests for individual components
6. **Backward Compatibility**: Existing scripts continue to work

## Usage

### Option 1: Use the new main entry point
```bash
python main.py
```

### Option 2: Use the backward compatibility wrapper
```bash
python apple_mlx_script.py
```

### Option 3: Import specific modules
```python
from data_processor import process_data
from config import MODEL_NAME
from utils import info, success
```

## Migration Notes

- All existing functionality is preserved
- The original `apple_mlx_script.py` now acts as a simple wrapper
- No breaking changes to the user interface
- All imports and dependencies remain the same 