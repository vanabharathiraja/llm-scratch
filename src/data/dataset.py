"""
Data loading utilities for GPT pretraining and finetuning.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class TextDataset(Dataset):
    """Dataset for text data."""
    
    def __init__(self, data, block_size):
        """
        Args:
            data: Tokenized text data (list of integers)
            block_size: Maximum sequence length
        """
        self.data = data
        self.block_size = block_size
        
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        # Get a chunk of data
        chunk = self.data[idx:idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


class TextFileDataset(Dataset):
    """Dataset that loads text from file."""
    
    def __init__(self, file_path, tokenizer, block_size):
        """
        Args:
            file_path: Path to text file
            tokenizer: Tokenizer instance
            block_size: Maximum sequence length
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        self.data = tokenizer.encode(text)
        self.block_size = block_size
        
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


class FinetuningDataset(Dataset):
    """Dataset for finetuning tasks (e.g., question answering, summarization)."""
    
    def __init__(self, examples, tokenizer, block_size):
        """
        Args:
            examples: List of dicts with 'input' and 'output' keys
            tokenizer: Tokenizer instance
            block_size: Maximum sequence length
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.block_size = block_size
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Encode input and output
        input_text = example['input']
        output_text = example['output']
        
        # Create prompt: "Input: ... Output: ..."
        prompt = f"Input: {input_text} Output: {output_text}"
        tokens = self.tokenizer.encode(prompt)
        
        # Truncate or pad to block_size
        if len(tokens) > self.block_size:
            tokens = tokens[:self.block_size]
        
        # Create input and target
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        
        # Pad if necessary
        if len(x) < self.block_size:
            pad_len = self.block_size - len(x)
            x = torch.cat([x, torch.zeros(pad_len, dtype=torch.long)])
            y = torch.cat([y, torch.full((pad_len,), -1, dtype=torch.long)])  # -1 for ignored indices
        
        return x, y


def create_dataloader(dataset, batch_size, shuffle=True, num_workers=0):
    """Create a DataLoader for the dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def load_text_data(file_path):
    """Load text data from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text
