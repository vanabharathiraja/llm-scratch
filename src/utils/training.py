"""
Training utilities and helper functions.
"""

import torch
import os
import json
from datetime import datetime


def get_device():
    """Get the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def get_lr(optimizer):
    """Get current learning rate from optimizer (returns first parameter group's lr)."""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save_checkpoint(model, optimizer, epoch, loss, save_path, config=None):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat(),
    }
    
    if config is not None:
        checkpoint['config'] = config
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', 0)
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Resuming from epoch {epoch}, loss: {loss:.4f}")
    
    return epoch, loss


def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_config(config, save_path):
    """Save configuration to JSON file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {save_path}")


def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


class TrainingMetrics:
    """Track and display training metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.losses = []
        self.steps = []
        self.epochs = []
    
    def update(self, loss, step, epoch):
        """Update metrics."""
        self.losses.append(loss)
        self.steps.append(step)
        self.epochs.append(epoch)
    
    def get_average_loss(self, last_n=100):
        """Get average loss over last n steps."""
        if len(self.losses) == 0:
            return 0.0
        return sum(self.losses[-last_n:]) / min(last_n, len(self.losses))
    
    def save(self, save_path):
        """Save metrics to file."""
        data = {
            'losses': self.losses,
            'steps': self.steps,
            'epochs': self.epochs,
        }
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
