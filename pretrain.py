"""
Pretrain GPT model from scratch.

This script pretrains a GPT model on a text corpus using next-token prediction.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
import argparse
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.model.gpt import GPT
from src.data.tokenizer import CharTokenizer
from src.data.dataset import TextFileDataset, create_dataloader
from src.utils.training import (
    get_device, save_checkpoint, load_checkpoint, 
    count_parameters, save_config, TrainingMetrics
)


def get_config():
    """Get default model configuration."""
    return {
        'vocab_size': 95,  # Will be updated based on tokenizer
        'n_embd': 384,     # Embedding dimension
        'n_head': 6,       # Number of attention heads
        'n_layer': 6,      # Number of transformer blocks
        'block_size': 256, # Maximum sequence length
        'dropout': 0.1,    # Dropout rate
    }


def train_epoch(model, dataloader, optimizer, device, epoch, metrics):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for i, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        logits, loss = model(x, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        metrics.update(loss.item(), i + epoch * len(dataloader), epoch)
        
        # Print progress
        if (i + 1) % 100 == 0:
            avg_loss = total_loss / (i + 1)
            print(f"Epoch {epoch+1}, Step {i+1}/{len(dataloader)}, Loss: {avg_loss:.4f}")
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main(args):
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load or create tokenizer
    print(f"Loading data from {args.data_path}...")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print("Creating tokenizer...")
    tokenizer = CharTokenizer.from_text(text)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Save tokenizer
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer_path = os.path.join(args.output_dir, 'tokenizer.pkl')
    tokenizer.save(tokenizer_path)
    
    # Split data into train and validation
    n = len(text)
    train_text = text[:int(0.9 * n)]
    val_text = text[int(0.9 * n):]
    
    # Create temporary files for datasets
    os.makedirs('/tmp/llm_data', exist_ok=True)
    train_file = '/tmp/llm_data/train.txt'
    val_file = '/tmp/llm_data/val.txt'
    
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write(train_text)
    with open(val_file, 'w', encoding='utf-8') as f:
        f.write(val_text)
    
    # Create datasets
    print("Creating datasets...")
    config = get_config()
    config['vocab_size'] = tokenizer.vocab_size
    
    train_dataset = TextFileDataset(train_file, tokenizer, config['block_size'])
    val_dataset = TextFileDataset(val_file, tokenizer, config['block_size'])
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = create_dataloader(train_dataset, args.batch_size, shuffle=True)
    val_loader = create_dataloader(val_dataset, args.batch_size, shuffle=False)
    
    # Create model
    print("Creating model...")
    model = GPT(config)
    model = model.to(device)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Save config
    config_path = os.path.join(args.output_dir, 'config.json')
    save_config(config, config_path)
    
    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume_from:
        start_epoch, _ = load_checkpoint(model, optimizer, args.resume_from, device)
    
    # Training loop
    print(f"\nStarting training for {args.num_epochs} epochs...")
    metrics = TrainingMetrics()
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, metrics)
        
        # Validate
        val_loss = evaluate(model, val_loader, device)
        
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}\n")
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt')
        save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path, config)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.output_dir, 'best_model.pt')
            save_checkpoint(model, optimizer, epoch, val_loss, best_path, config)
            print(f"New best model saved! Val Loss: {val_loss:.4f}")
    
    # Save final model
    final_path = os.path.join(args.output_dir, 'final_model.pt')
    save_checkpoint(model, optimizer, args.num_epochs - 1, val_loss, final_path, config)
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    metrics.save(metrics_path)
    
    print("\nTraining completed!")
    print(f"Models saved to {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pretrain GPT model')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to training text file')
    parser.add_argument('--output_dir', type=str, default='outputs/pretrain',
                        help='Directory to save model checkpoints')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    main(args)
