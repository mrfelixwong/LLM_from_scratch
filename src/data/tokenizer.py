"""
Tokenizer implementations for text preprocessing.

This module provides both a simple character-level tokenizer
and a wrapper for the GPT-2 tokenizer.
"""

import re
import json
from typing import List, Dict, Optional
from pathlib import Path

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("Warning: tiktoken not available. Only SimpleTokenizer will work.")


class SimpleTokenizer:
    """
    Simple character-level tokenizer.
    
    This tokenizer treats each character as a separate token,
    which is useful for educational purposes and small experiments.
    """
    
    def __init__(self, vocab: Optional[List[str]] = None):
        if vocab is None:
            # Default vocabulary with common characters
            self.vocab = self._create_default_vocab()
        else:
            self.vocab = vocab
        
        # Create mappings
        self.char_to_id = {char: i for i, char in enumerate(self.vocab)}
        self.id_to_char = {i: char for i, char in enumerate(self.vocab)}
        
        # Special tokens
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.bos_token = '<BOS>'
        self.eos_token = '<EOS>'
        
        # Add special tokens if not present
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for token in special_tokens:
            if token not in self.char_to_id:
                self.vocab.append(token)
                self.char_to_id[token] = len(self.vocab) - 1
                self.id_to_char[len(self.vocab) - 1] = token
        
        self.vocab_size = len(self.vocab)
        self.pad_token_id = self.char_to_id[self.pad_token]
        self.unk_token_id = self.char_to_id[self.unk_token]
        self.bos_token_id = self.char_to_id[self.bos_token]
        self.eos_token_id = self.char_to_id[self.eos_token]
    
    def _create_default_vocab(self) -> List[str]:
        """Create a default vocabulary with common characters."""
        vocab = []
        
        # Add ASCII printable characters
        for i in range(32, 127):  # Printable ASCII
            vocab.append(chr(i))
        
        # Add some common special characters
        vocab.extend(['\n', '\t'])
        
        return vocab
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            List of token IDs
        """
        if add_special_tokens:
            tokens = [self.bos_token_id]
        else:
            tokens = []
        
        for char in text:
            token_id = self.char_to_id.get(char, self.unk_token_id)
            tokens.append(token_id)
        
        if add_special_tokens:
            tokens.append(self.eos_token_id)
        
        return tokens
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        chars = []
        special_ids = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        
        for token_id in token_ids:
            if skip_special_tokens and token_id in special_ids:
                continue
            
            char = self.id_to_char.get(token_id, self.unk_token)
            if not (skip_special_tokens and char in [self.pad_token, self.bos_token, self.eos_token]):
                chars.append(char)
        
        return ''.join(chars)
    
    def save(self, path: str):
        """Save tokenizer vocabulary to file."""
        data = {
            'vocab': self.vocab,
            'special_tokens': {
                'pad_token': self.pad_token,
                'unk_token': self.unk_token,
                'bos_token': self.bos_token,
                'eos_token': self.eos_token
            }
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """Load tokenizer from file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls(vocab=data['vocab'])
        
        # Update special tokens if provided
        if 'special_tokens' in data:
            special = data['special_tokens']
            tokenizer.pad_token = special.get('pad_token', tokenizer.pad_token)
            tokenizer.unk_token = special.get('unk_token', tokenizer.unk_token)
            tokenizer.bos_token = special.get('bos_token', tokenizer.bos_token)
            tokenizer.eos_token = special.get('eos_token', tokenizer.eos_token)
        
        return tokenizer


class GPT2Tokenizer:
    """
    Wrapper for the GPT-2 tokenizer using tiktoken.
    
    This provides a more realistic tokenizer for experiments
    with proper subword tokenization.
    """
    
    def __init__(self, model_name: str = "gpt2"):
        if not TIKTOKEN_AVAILABLE:
            raise ImportError("tiktoken is required for GPT2Tokenizer. Install with: pip install tiktoken")
        
        self.encoding = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.encoding.n_vocab
        
        # GPT-2 doesn't have explicit special tokens, but we can define them
        self.pad_token_id = self.encoding.encode("<|endoftext|>")[0]  # Use endoftext as pad
        self.eos_token_id = self.encoding.encode("<|endoftext|>")[0]
        self.bos_token_id = self.encoding.encode("<|endoftext|>")[0]
        self.unk_token_id = self.encoding.encode("<|endoftext|>")[0]
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add special tokens
            
        Returns:
            List of token IDs
        """
        tokens = self.encoding.encode(text)
        
        if add_special_tokens:
            # For autoregressive training, we don't typically add BOS
            # but we might add EOS depending on the training setup
            pass
        
        return tokens
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        return self.encoding.decode(token_ids)
    
    def tokenize(self, text: str) -> List[str]:
        """Get the actual tokens (for inspection)."""
        token_ids = self.encode(text, add_special_tokens=False)
        return [self.encoding.decode([tid]) for tid in token_ids]


def create_tokenizer(tokenizer_type: str = "simple", **kwargs):
    """
    Factory function to create tokenizers.
    
    Args:
        tokenizer_type: Type of tokenizer ("simple" or "gpt2")
        **kwargs: Additional arguments for tokenizer
        
    Returns:
        Tokenizer instance
    """
    if tokenizer_type == "simple":
        return SimpleTokenizer(**kwargs)
    elif tokenizer_type == "gpt2":
        return GPT2Tokenizer(**kwargs)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")


if __name__ == "__main__":
    # Test simple tokenizer
    print("Testing SimpleTokenizer...")
    simple_tokenizer = SimpleTokenizer()
    
    text = "Hello, world! This is a test."
    encoded = simple_tokenizer.encode(text)
    decoded = simple_tokenizer.decode(encoded)
    
    print(f"Original: {text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Vocab size: {simple_tokenizer.vocab_size}")
    
    # Test GPT-2 tokenizer if available
    if TIKTOKEN_AVAILABLE:
        print("\nTesting GPT2Tokenizer...")
        gpt2_tokenizer = GPT2Tokenizer()
        
        encoded_gpt2 = gpt2_tokenizer.encode(text)
        decoded_gpt2 = gpt2_tokenizer.decode(encoded_gpt2)
        tokens = gpt2_tokenizer.tokenize(text)
        
        print(f"Original: {text}")
        print(f"Tokens: {tokens}")
        print(f"Encoded: {encoded_gpt2}")
        print(f"Decoded: {decoded_gpt2}")
        print(f"Vocab size: {gpt2_tokenizer.vocab_size}")
    
    print("✓ Tokenizer tests passed!")