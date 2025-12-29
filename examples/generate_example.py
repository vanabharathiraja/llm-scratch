"""
Example: Generate text using a trained model.

This example demonstrates how to use a trained GPT model for text generation.
"""

import os

print("Text Generation Example")
print("=" * 80)
print("\nAfter training a model, you can generate text with:")
print("\nBasic generation:")
print("python generate.py --model_path outputs/shakespeare/best_model.pt --prompt 'First Citizen:' --max_tokens 200")

print("\nWith temperature control (higher = more creative):")
print("python generate.py --model_path outputs/shakespeare/best_model.pt --prompt 'To be or not to be' --max_tokens 150 --temperature 0.8")

print("\nWith top-k sampling (more focused):")
print("python generate.py --model_path outputs/shakespeare/best_model.pt --prompt 'All:' --max_tokens 100 --top_k 50")

print("\nSave output to file:")
print("python generate.py --model_path outputs/shakespeare/best_model.pt --prompt 'Second Citizen:' --max_tokens 200 --output_file generated.txt")

print("\n" + "=" * 80)
