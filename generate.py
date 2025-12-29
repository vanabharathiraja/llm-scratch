"""
Generate text using a trained GPT model.
"""

import torch
import argparse
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model.gpt import GPT
from src.data.tokenizer import CharTokenizer
from src.utils.training import get_device, load_config


def generate_text(model, tokenizer, prompt, max_tokens=100, temperature=1.0, top_k=None, device='cpu'):
    """Generate text from a prompt."""
    model.eval()
    
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    
    # Generate
    with torch.no_grad():
        generated = model.generate(tokens, max_tokens, temperature=temperature, top_k=top_k)
    
    # Decode
    generated_tokens = generated[0].tolist()
    generated_text = tokenizer.decode(generated_tokens)
    
    return generated_text


def main(args):
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model config
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    config = checkpoint.get('config')
    
    if config is None:
        config_path = os.path.join(os.path.dirname(args.model_path), 'config.json')
        if os.path.exists(config_path):
            config = load_config(config_path)
        else:
            raise ValueError("No config found.")
    
    # Load tokenizer
    tokenizer_path = os.path.join(os.path.dirname(args.model_path), 'tokenizer.pkl')
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
    
    print("Loading tokenizer...")
    tokenizer = CharTokenizer.load(tokenizer_path)
    
    # Create and load model
    print("Creating model...")
    model = GPT(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print("\nModel loaded successfully!")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Max sequence length: {config['block_size']}")
    print(f"\nGenerating text with prompt: '{args.prompt}'\n")
    
    # Generate text
    generated_text = generate_text(
        model, tokenizer, args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device
    )
    
    print("Generated text:")
    print("-" * 80)
    print(generated_text)
    print("-" * 80)
    
    # Save to file if specified
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(generated_text)
        print(f"\nGenerated text saved to {args.output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate text using GPT model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, default='',
                        help='Text prompt to start generation')
    parser.add_argument('--max_tokens', type=int, default=100,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (higher = more random)')
    parser.add_argument('--top_k', type=int, default=None,
                        help='Sample from top k tokens only')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path to save generated text')
    
    args = parser.parse_args()
    main(args)
