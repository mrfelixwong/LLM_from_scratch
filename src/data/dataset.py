"""
Dataset implementations for language modeling.

This module provides dataset classes and utilities for
preparing text data for transformer training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Tuple, Union
import numpy as np
from pathlib import Path

from .tokenizer import SimpleTokenizer, GPT2Tokenizer


class TextDataset(Dataset):
    """
    Dataset for autoregressive language modeling.
    
    This dataset takes tokenized text and creates input-target pairs
    for next-token prediction training.
    
    Args:
        token_ids: List of token IDs
        block_size: Maximum sequence length
        stride: Stride for creating overlapping sequences (default: block_size)
    """
    
    def __init__(
        self,
        token_ids: List[int],
        block_size: int,
        stride: Optional[int] = None
    ):
        self.token_ids = token_ids
        self.block_size = block_size
        self.stride = stride or block_size
        
        # Create sequences
        self.sequences = self._create_sequences()
    
    def _create_sequences(self) -> List[List[int]]:
        """Create overlapping sequences from token_ids."""
        sequences = []
        
        # Create sequences with sliding window
        for i in range(0, len(self.token_ids) - self.block_size, self.stride):
            sequence = self.token_ids[i:i + self.block_size + 1]  # +1 for target
            if len(sequence) == self.block_size + 1:
                sequences.append(sequence)
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get input and target tensors.
        
        Returns:
            input_ids: [block_size] - input tokens
            target_ids: [block_size] - target tokens (shifted by 1)
        """
        sequence = self.sequences[idx]
        
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        target_ids = torch.tensor(sequence[1:], dtype=torch.long)
        
        return input_ids, target_ids


class TextFileDataset(Dataset):
    """
    Dataset that loads text from files and tokenizes on-the-fly.
    
    Args:
        file_paths: List of text file paths
        tokenizer: Tokenizer to use
        block_size: Maximum sequence length
        max_length: Maximum total length to load (None for all)
    """
    
    def __init__(
        self,
        file_paths: List[Union[str, Path]],
        tokenizer: Union[SimpleTokenizer, GPT2Tokenizer],
        block_size: int,
        max_length: Optional[int] = None
    ):
        self.file_paths = [Path(p) for p in file_paths]
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.max_length = max_length
        
        # Load and tokenize all texts
        self.token_ids = self._load_and_tokenize()
        
        # Create text dataset
        self.dataset = TextDataset(self.token_ids, block_size)
    
    def _load_and_tokenize(self) -> List[int]:
        """Load text files and tokenize them."""
        all_token_ids = []
        total_length = 0
        
        for file_path in self.file_paths:
            if not file_path.exists():
                print(f"Warning: File {file_path} does not exist, skipping.")
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # Tokenize text
                token_ids = self.tokenizer.encode(text, add_special_tokens=False)
                all_token_ids.extend(token_ids)
                
                total_length += len(token_ids)
                
                # Check if we've reached max length
                if self.max_length and total_length >= self.max_length:
                    all_token_ids = all_token_ids[:self.max_length]
                    break
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        print(f"Loaded {len(all_token_ids):,} tokens from {len(self.file_paths)} files")
        return all_token_ids
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.dataset[idx]


class SimpleTextDataset(Dataset):
    """
    Simple dataset for small text experiments.
    
    Takes a single text string and creates a dataset from it.
    """
    
    def __init__(
        self,
        text: str,
        tokenizer: Union[SimpleTokenizer, GPT2Tokenizer],
        block_size: int
    ):
        self.text = text
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # Tokenize text
        self.token_ids = tokenizer.encode(text, add_special_tokens=False)
        
        # Create text dataset
        self.dataset = TextDataset(self.token_ids, block_size)
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.dataset[idx]


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False
) -> DataLoader:
    """
    Create a DataLoader for the dataset.
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for GPU transfer
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop incomplete batches for consistent shapes
    )


def create_sample_text() -> str:
    """Create a sample text for testing."""
    return """
The transformer architecture has revolutionized natural language processing.
It was introduced in the paper "Attention Is All You Need" by Vaswani et al.
The key innovation is the attention mechanism, which allows the model to
focus on different parts of the input sequence when making predictions.

Unlike recurrent neural networks, transformers can process all positions
in parallel, making them much more efficient to train. The architecture
consists of an encoder and decoder, each made up of multiple layers.
Each layer contains multi-head attention and feed-forward sub-layers,
with residual connections and layer normalization.

For language modeling, we typically use only the decoder part of the
transformer, which generates text autoregressively by predicting the
next token given the previous tokens. This is how models like GPT work.

The attention mechanism computes a weighted sum of values, where the
weights are determined by the similarity between queries and keys.
This allows the model to attend to relevant information regardless
of its position in the sequence.

Training large language models requires significant computational
resources and careful optimization. The models learn to represent
language by predicting the next word in a sequence, which forces
them to develop an understanding of grammar, semantics, and even
some aspects of reasoning and world knowledge.
""".strip()


if __name__ == "__main__":
    # Test the dataset implementations
    from .tokenizer import create_tokenizer
    
    print("Testing TextDataset...")
    
    # Create sample data
    sample_text = create_sample_text()
    tokenizer = create_tokenizer("simple")
    
    # Test SimpleTextDataset
    dataset = SimpleTextDataset(sample_text, tokenizer, block_size=32)
    
    print(f"Dataset length: {len(dataset)}")
    print(f"Sample text length: {len(sample_text)} chars")
    print(f"Tokenized length: {len(dataset.token_ids)} tokens")
    
    # Test a few samples
    for i in range(min(3, len(dataset))):
        input_ids, target_ids = dataset[i]
        print(f"\nSample {i}:")
        print(f"Input shape: {input_ids.shape}")
        print(f"Target shape: {target_ids.shape}")
        
        # Decode first few tokens
        input_text = tokenizer.decode(input_ids[:10].tolist())
        target_text = tokenizer.decode(target_ids[:10].tolist())
        print(f"Input preview: {repr(input_text)}")
        print(f"Target preview: {repr(target_text)}")
    
    # Test DataLoader
    dataloader = create_dataloader(dataset, batch_size=4, shuffle=False)
    
    print(f"\nDataLoader test:")
    for batch_idx, (input_batch, target_batch) in enumerate(dataloader):
        print(f"Batch {batch_idx}: input {input_batch.shape}, target {target_batch.shape}")
        if batch_idx >= 2:  # Only show first few batches
            break
    
    print("✓ Dataset tests passed!")