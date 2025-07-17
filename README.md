# Transformer Language Model from Scratch

Complete educational implementation of a GPT-style transformer built entirely from scratch using PyTorch. This project demonstrates every component of the transformer architecture with working code and clear explanations.

## Quick Start

```bash
pip install -r requirements.txt
python tests/simple_demo.py
```

The simple demo trains a 5,162-parameter transformer that learns to count (0→1→2→3) with perfect accuracy in under 10 seconds.

## What You'll Learn

- **Attention Mechanism**: How transformers focus on relevant parts of sequences
- **Multi-Head Attention**: Parallel attention for different representation spaces  
- **Positional Encoding**: Teaching models about sequence order without RNNs
- **Autoregressive Generation**: Next-token prediction for text generation
- **Training Dynamics**: Loss optimization, checkpointing, and validation

## Project Structure

```
├── tests/
│   ├── simple_demo.py        # Working transformer demo (START HERE)
│   └── test_model.py         # Component validation
├── src/
│   ├── model/
│   │   ├── attention.py      # Multi-head attention
│   │   ├── embeddings.py     # Token and positional embeddings
│   │   ├── feedforward.py    # Feed-forward networks
│   │   └── transformer.py    # Complete model
│   └── data/
│       ├── tokenizer.py      # Text tokenization
│       └── dataset.py        # Data loading
├── scripts/
│   ├── train.py             # Full training pipeline
│   └── generate.py          # Text generation
└── notebooks/               # Educational walkthroughs
    ├── 01_attention_mechanism.ipynb
    ├── 02_transformer_blocks.ipynb
    ├── 03_positional_encoding.ipynb
    ├── 04_training_process.ipynb
    └── 05_text_generation.ipynb
```

## Learning Path

1. **Start Simple**: Run `python tests/simple_demo.py` to see a working transformer
2. **Understand Components**: Explore the notebooks to learn each piece
3. **Test Everything**: Run `python tests/test_model.py` to validate the implementation
4. **Train Models**: Use `python scripts/train.py` for full-scale training

## Model Sizes

| Configuration | Parameters | Use Case | Training Time |
|---------------|------------|----------|---------------|
| Simple Demo   | 5,162      | Learning | 10 seconds |
| Tiny          | 6.8M       | Experiments | Minutes |
| Small         | ~25M       | Better quality | Hours |
| Medium        | ~100M      | Research | Days |

## Key Features

- **From Scratch Implementation**: No black-box dependencies
- **Educational Focus**: Every line explained with mathematical context
- **Multiple Scales**: Start tiny, scale up gradually
- **Working Examples**: See results immediately, not after hours of training
- **Complete Pipeline**: Data processing through text generation

## Training Examples

```bash
# Train tiny model for quick experiments
python scripts/train.py --model_size tiny --epochs 10

# Train with custom data
python scripts/train.py --text_file your_data.txt --model_size small

# Generate text from trained model
python scripts/generate.py --model_path checkpoints/best_model.pt
```

## Why This Implementation?

Most transformer tutorials either:
- Skip crucial implementation details
- Use pre-built components that hide the mechanics
- Require massive datasets and training time to see results

This project gives you immediate feedback with working models you can understand completely. Start with the 5K parameter demo that learns perfectly in seconds, then scale up to understand how real language models work.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Matplotlib (for notebooks)
- Jupyter (for notebooks)

See `requirements.txt` for exact versions.