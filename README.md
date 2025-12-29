# LLM from Scratch

A complete implementation of a GPT-like Large Language Model (LLM) built from scratch in PyTorch. This repository contains all the code necessary for developing, pretraining, and finetuning a transformer-based language model.

## Features

- **GPT Architecture**: Full transformer implementation with multi-head self-attention
- **Pretraining**: Train models from scratch on any text corpus
- **Finetuning**: Adapt pretrained models to specific downstream tasks
- **Text Generation**: Generate text with temperature and top-k sampling
- **Modular Design**: Clean, well-documented code structure
- **Easy to Use**: Simple command-line interface for all operations

## Architecture

The model implements a decoder-only transformer architecture similar to GPT, including:

- Multi-head self-attention mechanism
- Position-wise feed-forward networks
- Layer normalization
- Residual connections
- Positional embeddings
- Causal masking for autoregressive generation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/vanabharathiraja/llm-scratch.git
cd llm-scratch
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Pretraining

Train a model from scratch on your text data:

```bash
python pretrain.py \
    --data_path data/your_text.txt \
    --output_dir outputs/pretrain \
    --batch_size 32 \
    --num_epochs 10 \
    --learning_rate 3e-4
```

### 2. Finetuning

Finetune a pretrained model on a specific task:

```bash
python finetune.py \
    --pretrained_path outputs/pretrain/best_model.pt \
    --data_path data/finetune_data.json \
    --output_dir outputs/finetune \
    --batch_size 16 \
    --num_epochs 5 \
    --learning_rate 1e-4
```

The finetuning data should be in JSON format:
```json
[
    {
        "input": "Question or prompt",
        "output": "Expected response"
    }
]
```

### 3. Text Generation

Generate text using a trained model:

```bash
python generate.py \
    --model_path outputs/pretrain/best_model.pt \
    --prompt "Once upon a time" \
    --max_tokens 200 \
    --temperature 0.8 \
    --top_k 50
```

## Project Structure

```
llm-scratch/
├── src/
│   ├── model/
│   │   ├── __init__.py
│   │   └── gpt.py              # GPT model implementation
│   ├── data/
│   │   ├── __init__.py
│   │   ├── tokenizer.py        # Tokenization utilities
│   │   └── dataset.py          # Dataset classes
│   └── utils/
│       ├── __init__.py
│       └── training.py         # Training utilities
├── configs/
│   ├── small_gpt.json          # Small model config
│   └── medium_gpt.json         # Medium model config
├── examples/
│   ├── pretrain_example.py     # Pretraining example
│   ├── finetune_example.py     # Finetuning example
│   └── generate_example.py     # Generation example
├── pretrain.py                 # Pretraining script
├── finetune.py                 # Finetuning script
├── generate.py                 # Text generation script
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Model Configurations

Two default configurations are provided:

### Small GPT (configs/small_gpt.json)
- Embedding dimension: 384
- Attention heads: 6
- Layers: 6
- Context length: 256 tokens
- Parameters: ~10M

### Medium GPT (configs/medium_gpt.json)
- Embedding dimension: 768
- Attention heads: 12
- Layers: 12
- Context length: 1024 tokens
- Parameters: ~117M

You can create custom configurations by modifying these files or creating new ones.

## Examples

See the `examples/` directory for detailed examples:

1. **Pretraining**: `examples/pretrain_example.py` - Train on Shakespeare text
2. **Finetuning**: `examples/finetune_example.py` - Finetune for Q&A
3. **Generation**: `examples/generate_example.py` - Generate text samples

Run any example:
```bash
python examples/pretrain_example.py
```

## Training Tips

1. **Start Small**: Begin with the small config to verify everything works
2. **Learning Rate**: Use 3e-4 for pretraining, 1e-4 for finetuning
3. **Batch Size**: Adjust based on your GPU memory
4. **Checkpointing**: Models are automatically saved after each epoch
5. **Resume Training**: Use `--resume_from` flag to continue from a checkpoint

## Advanced Usage

### Custom Tokenizer

The default implementation uses a character-level tokenizer. For production use, consider implementing a BPE tokenizer:

```python
from src.data.tokenizer import CharTokenizer

# Create tokenizer from corpus
with open('data.txt', 'r') as f:
    text = f.read()
tokenizer = CharTokenizer.from_text(text)

# Save for later use
tokenizer.save('tokenizer.pkl')
```

### Layer Freezing

When finetuning, you can freeze early layers to prevent catastrophic forgetting:

```bash
python finetune.py \
    --pretrained_path outputs/pretrain/best_model.pt \
    --data_path data/finetune.json \
    --freeze_layers 4
```

### Custom Data Loading

For custom datasets, you can create your own dataset class:

```python
from src.data.dataset import TextDataset

# Your custom data loading logic
data = load_your_data()
dataset = TextDataset(data, block_size=256)
```

## Hardware Requirements

- **Minimum**: CPU with 8GB RAM (slow, small models only)
- **Recommended**: GPU with 8GB+ VRAM (NVIDIA, AMD, or Apple Silicon)
- **Optimal**: GPU with 16GB+ VRAM for larger models

The code automatically detects and uses available hardware (CUDA, MPS, or CPU).

## License

This project is open source and available for educational purposes.

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

## Acknowledgments

This implementation is inspired by:
- "Attention is All You Need" (Vaswani et al., 2017)
- GPT architecture (OpenAI)
- Andrej Karpathy's educational materials

## Citation

If you use this code in your research, please cite:

```bibtex
@software{llm_scratch,
  title = {LLM from Scratch: GPT Implementation},
  author = {vanabharathiraja},
  year = {2024},
  url = {https://github.com/vanabharathiraja/llm-scratch}
}
```