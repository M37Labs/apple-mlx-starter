#!/usr/bin/env python3
"""
Apple MLX Script - Main Entry Point
A modular script for preparing SQuAD data and training MLX models.
"""

from data_processor import process_data
from instructions import print_training_instructions


def main():
    """Main entry point for the Apple MLX script."""
    # Process the data
    process_data()
    
    # Print training instructions
    print_training_instructions()


if __name__ == "__main__":
    main() 