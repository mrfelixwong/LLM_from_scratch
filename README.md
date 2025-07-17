# Transformer LLM From Scratch

A comprehensive educational project implementing a GPT-style transformer language model from scratch using PyTorch.

## Goal

Build a complete understanding of the transformer architecture by implementing every component from the ground up, with clear, commented code and educational materials.

## Quick Demo

**Start here to see a working transformer in action:**

```bash
# Setup environment
pip install -r requirements.txt

# Run the simple demo (trains in seconds!)
python tests/simple_demo.py
```

This trains a tiny 5,162-parameter transformer that learns to count (0→1→2→3...) with 100% accuracy.

## Project Structure

```
transformer-llm-from-scratch/
├── tests/
│   ├── simple_demo.py        # START HERE - Working transformer demo
│   └── test_model.py         # Component tests
├── src/
│   ├── model/
│   │   ├── attention.py       # Multi-head attention implementation  
│   │   ├── embeddings.py      # Token and positional embeddings
│   │   ├── feedforward.py     # Feed-forward networks
│   │   └── transformer.py     # Complete transformer model
│   └── data/
│       ├── tokenizer.py       # Tokenization logic
│       └── dataset.py         # Dataset handling
├── notebooks/               # Educational Jupyter notebooks
│   ├── 01_attention_mechanism.ipynb
│   ├── 02_transformer_blocks.ipynb
│   ├── 03_positional_encoding.ipynb
│   ├── 04_training_process.ipynb
│   └── 05_text_generation.ipynb
├── scripts/
│   ├── train.py              # Full training script
│   └── generate.py           # Text generation script
└── data/
    ├── shakespeare.txt       # Training data
    └── simple_patterns.txt   # Simple pattern data
```

## Learning Path

### 1. Start with the Simple Demo
```bash
python tests/simple_demo.py
```
- 5K parameter model that actually works
- Learns counting in 100 epochs (~10 seconds)
- Perfect for understanding basic transformer mechanics

### 2. Explore the Full Implementation
```bash
# Test the complete implementation
python tests/test_model.py

# Train on Shakespeare (6.8M parameters)
python scripts/train.py --model_size tiny --epochs 10 --batch_size 4
```

### 3. Educational Notebooks
```bash
jupyter notebook notebooks/01_attention_mechanism.ipynb
```

### 4. Advanced Training
```bash
# Train larger models
python scripts/train.py --model_size small --epochs 20 --text_file data/shakespeare.txt
```

## Features

- **Immediate Results**: Simple demo shows working transformer in seconds
- **Educational Focus**: Every component heavily commented with mathematical explanations
- **Modular Design**: Each component can be studied independently
- **Multiple Scales**: From 5K to 6.8M parameter models
- **Generation Strategies**: Greedy, top-k, top-p, and temperature sampling
- **Visualization**: Attention maps and training curves in notebooks

## Model Configurations

| Model | Parameters | Use Case | Training Time |
|-------|------------|----------|---------------|
| **Simple Demo** | 5,162 | Learning basics | 10 seconds |
| **Tiny** | 6.8M | Quick experiments | Minutes |
| **Small** | ~25M | Better text quality | Hours |
| **Medium** | ~100M | Research projects | Days |

## Key Concepts Demonstrated

1. **Attention Mechanism** - How transformers "focus" on relevant tokens
2. **Multi-Head Attention** - Parallel attention for different representation spaces
3. **Positional Encoding** - Teaching models about sequence order
4. **Autoregressive Generation** - Next-token prediction
5. **Training Dynamics** - Loss curves, optimization, checkpointing

## Why This Implementation?

- **From Scratch**: No black-box libraries - understand every line
- **Right-Sized Examples**: Start tiny, scale up gradually  
- **Immediate Feedback**: See transformers work before diving deep
- **Complete Pipeline**: Data → Training → Generation → Evaluation