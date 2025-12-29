"""
Example: Finetune a pretrained model on question-answering task.

This example demonstrates how to finetune a pretrained GPT model.
"""

import os
import json

# Example finetuning data for question-answering
finetuning_data = [
    {
        "input": "What is the capital of France?",
        "output": "The capital of France is Paris."
    },
    {
        "input": "Who wrote Romeo and Juliet?",
        "output": "William Shakespeare wrote Romeo and Juliet."
    },
    {
        "input": "What is 2 + 2?",
        "output": "2 + 2 equals 4."
    },
    {
        "input": "What is the largest planet in our solar system?",
        "output": "Jupiter is the largest planet in our solar system."
    },
    {
        "input": "Who painted the Mona Lisa?",
        "output": "Leonardo da Vinci painted the Mona Lisa."
    },
    {
        "input": "What is the speed of light?",
        "output": "The speed of light is approximately 299,792,458 meters per second."
    },
    {
        "input": "When did World War II end?",
        "output": "World War II ended in 1945."
    },
    {
        "input": "What is the chemical symbol for water?",
        "output": "The chemical symbol for water is H2O."
    },
    {
        "input": "Who was the first person to walk on the moon?",
        "output": "Neil Armstrong was the first person to walk on the moon."
    },
    {
        "input": "What is the tallest mountain in the world?",
        "output": "Mount Everest is the tallest mountain in the world."
    }
]

# Duplicate data to have more examples
finetuning_data = finetuning_data * 10

# Create data directory
os.makedirs('data', exist_ok=True)

# Save finetuning data
data_path = 'data/qa_finetune.json'
with open(data_path, 'w', encoding='utf-8') as f:
    json.dump(finetuning_data, f, indent=2)

print(f"Finetuning data saved to {data_path}")
print(f"Total examples: {len(finetuning_data)}")
print("\nTo finetune the model, run:")
print(f"python finetune.py --pretrained_path outputs/shakespeare/best_model.pt --data_path {data_path} --output_dir outputs/qa_finetuned --num_epochs 3 --batch_size 8")
