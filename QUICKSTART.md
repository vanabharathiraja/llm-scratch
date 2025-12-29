# Quick Start Guide

This guide will help you get started with training your own GPT-like language model.

## Setup

1. Install PyTorch (choose the right version for your system from [pytorch.org](https://pytorch.org)):
```bash
# For CPU only
pip install torch

# For CUDA (NVIDIA GPUs)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For Apple Silicon (M1/M2)
pip install torch
```

2. Verify installation:
```bash
python -c "import torch; print(f'PyTorch {torch.__version__} installed successfully!')"
```

## Step-by-Step Tutorial

### Step 1: Prepare Your Data

Create a text file with your training data. For this example, we'll use Shakespeare:

```bash
python examples/pretrain_example.py
```

This creates `data/shakespeare_sample.txt`.

### Step 2: Pretrain the Model

Train the model from scratch:

```bash
python pretrain.py \
    --data_path data/shakespeare_sample.txt \
    --output_dir outputs/shakespeare \
    --num_epochs 5 \
    --batch_size 16
```

Training progress will be displayed, and checkpoints will be saved in `outputs/shakespeare/`.

### Step 3: Generate Text

Use your trained model to generate text:

```bash
python generate.py \
    --model_path outputs/shakespeare/best_model.pt \
    --prompt "First Citizen:" \
    --max_tokens 200 \
    --temperature 0.8
```

### Step 4: Finetune for a Specific Task

Prepare finetuning data:

```bash
python examples/finetune_example.py
```

This creates `data/qa_finetune.json`.

Finetune the model:

```bash
python finetune.py \
    --pretrained_path outputs/shakespeare/best_model.pt \
    --data_path data/qa_finetune.json \
    --output_dir outputs/qa_finetuned \
    --num_epochs 3 \
    --batch_size 8
```

### Step 5: Test the Finetuned Model

```bash
python generate.py \
    --model_path outputs/qa_finetuned/best_model.pt \
    --prompt "Input: What is 2 + 2? Output:" \
    --max_tokens 50
```

## Understanding the Parameters

### Pretraining Parameters

- `--data_path`: Path to your text file
- `--output_dir`: Where to save model checkpoints
- `--num_epochs`: How many times to iterate over the data (start with 5-10)
- `--batch_size`: Number of examples per batch (reduce if you run out of memory)
- `--learning_rate`: Learning rate (default: 3e-4)

### Finetuning Parameters

- `--pretrained_path`: Path to pretrained model checkpoint
- `--data_path`: Path to JSON file with input/output pairs
- `--freeze_layers`: Number of layers to freeze (0 = train all layers)
- `--learning_rate`: Lower than pretraining (default: 1e-4)

### Generation Parameters

- `--prompt`: Starting text for generation
- `--max_tokens`: How many tokens to generate
- `--temperature`: Randomness (0.7 = focused, 1.5 = creative)
- `--top_k`: Sample from top k tokens (None = all tokens)

## Tips for Better Results

1. **More Data = Better Models**: The example uses minimal data. For real applications, use at least several MB of text.

2. **Adjust Model Size**: Edit `configs/small_gpt.json` to change model architecture:
   - Increase `n_embd` for more capacity
   - Increase `n_layer` for deeper models
   - Increase `block_size` for longer context

3. **Monitor Training**: Watch the loss decrease. If it doesn't improve, try:
   - Lower learning rate
   - More epochs
   - Larger batch size

4. **GPU Acceleration**: Training on GPU is much faster. The code automatically uses GPU if available.

5. **Checkpointing**: Models are saved after each epoch. You can resume training with `--resume_from`.

## Troubleshooting

**Out of Memory?**
- Reduce `batch_size`
- Reduce model size in config
- Use a smaller `block_size`

**Training Too Slow?**
- Increase `batch_size` if you have memory
- Use a GPU
- Train for fewer epochs initially

**Poor Quality Output?**
- Train longer (more epochs)
- Use more training data
- Adjust temperature during generation

**Model Not Learning?**
- Check that your data is loaded correctly
- Verify learning rate isn't too high or low
- Ensure you have enough data

## Next Steps

- Try training on your own text data
- Experiment with different model configurations
- Implement custom tokenizers for better performance
- Add evaluation metrics
- Implement beam search for generation

## Resources

- Original GPT paper: [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- Attention mechanism: [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- PyTorch documentation: [pytorch.org/docs](https://pytorch.org/docs)

Happy training! 🚀
