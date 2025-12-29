"""
Finetune a pretrained GPT model on a downstream task.

This script finetunes a pretrained GPT model on task-specific data.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
import argparse
import os
import sys
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model.gpt import GPT
from src.data.tokenizer import CharTokenizer
from src.data.dataset import FinetuningDataset, create_dataloader
from src.utils.training import (
    get_device, save_checkpoint, load_checkpoint,
    count_parameters, load_config, save_config, TrainingMetrics
)


def load_finetuning_data(data_path):
    """Load finetuning data from JSON file."""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


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
        if (i + 1) % 50 == 0:
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
    
    # Load pretrained model config
    print(f"Loading pretrained model from {args.pretrained_path}...")
    checkpoint = torch.load(args.pretrained_path, map_location=device)
    config = checkpoint.get('config')
    
    if config is None:
        print("Warning: No config found in checkpoint. Using default config.")
        config_path = os.path.join(os.path.dirname(args.pretrained_path), 'config.json')
        if os.path.exists(config_path):
            config = load_config(config_path)
        else:
            raise ValueError("No config found. Please provide config.json or a checkpoint with config.")
    
    # Load tokenizer
    tokenizer_path = os.path.join(os.path.dirname(args.pretrained_path), 'tokenizer.pkl')
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
    
    print("Loading tokenizer...")
    tokenizer = CharTokenizer.load(tokenizer_path)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Load finetuning data
    print(f"Loading finetuning data from {args.data_path}...")
    data = load_finetuning_data(args.data_path)
    
    # Split into train and validation
    n = len(data)
    train_data = data[:int(0.9 * n)]
    val_data = data[int(0.9 * n):]
    
    print(f"Train examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")
    
    # Create datasets
    train_dataset = FinetuningDataset(train_data, tokenizer, config['block_size'])
    val_dataset = FinetuningDataset(val_data, tokenizer, config['block_size'])
    
    # Create dataloaders
    train_loader = create_dataloader(train_dataset, args.batch_size, shuffle=True)
    val_loader = create_dataloader(val_dataset, args.batch_size, shuffle=False)
    
    # Create model
    print("Creating model...")
    model = GPT(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Optionally freeze some layers
    if args.freeze_layers > 0:
        print(f"Freezing first {args.freeze_layers} transformer blocks...")
        for i in range(args.freeze_layers):
            for param in model.transformer.h[i].parameters():
                param.requires_grad = False
        print(f"Trainable parameters: {count_parameters(model):,}")
    
    # Create optimizer (only for trainable parameters)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save config
    config_path = os.path.join(args.output_dir, 'config.json')
    save_config(config, config_path)
    
    # Training loop
    print(f"\nStarting finetuning for {args.num_epochs} epochs...")
    metrics = TrainingMetrics()
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
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
    
    print("\nFinetuning completed!")
    print(f"Models saved to {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune GPT model')
    parser.add_argument('--pretrained_path', type=str, required=True,
                        help='Path to pretrained model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to finetuning data (JSON file)')
    parser.add_argument('--output_dir', type=str, default='outputs/finetune',
                        help='Directory to save model checkpoints')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--freeze_layers', type=int, default=0,
                        help='Number of transformer blocks to freeze')
    
    args = parser.parse_args()
    main(args)
