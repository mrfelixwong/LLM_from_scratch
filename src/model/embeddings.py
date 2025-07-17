"""
Embedding Layers Implementation

This module implements token embeddings and positional encodings
that convert discrete tokens into continuous vector representations.
"""

import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    """
    Token embedding layer that converts token IDs to dense vectors.
    
    Args:
        vocab_size: Size of the vocabulary
        d_model: Model dimension (embedding size)
    """
    
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Initialize embeddings with scaled random values
        # Scaling by sqrt(d_model) helps with training stability
        nn.init.normal_(self.embedding.weight, mean=0, std=d_model**-0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert token IDs to embeddings.
        
        Args:
            x: Token IDs [batch_size, seq_len]
            
        Returns:
            embeddings: [batch_size, seq_len, d_model]
        """
        # Scale embeddings by sqrt(d_model) as in the original paper
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Positional encoding using sine and cosine functions.
    
    Since transformers have no inherent notion of position, we add
    positional encodings to give the model information about the
    relative or absolute position of tokens in the sequence.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Args:
        d_model: Model dimension
        max_len: Maximum sequence length
        dropout: Dropout probability
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Create the division term for the sinusoidal pattern
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to embeddings.
        
        Args:
            x: Token embeddings [batch_size, seq_len, d_model]
            
        Returns:
            x + positional encoding [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        
        # Add positional encoding to embeddings
        x = x + self.pe[:, :seq_len, :].detach()
        return self.dropout(x)


class LearnedPositionalEmbedding(nn.Module):
    """
    Alternative learned positional embeddings (used in GPT).
    
    Instead of fixed sinusoidal patterns, this learns position embeddings
    during training, which can sometimes work better for specific tasks.
    
    Args:
        max_len: Maximum sequence length
        d_model: Model dimension
        dropout: Dropout probability
    """
    
    def __init__(self, max_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        # Learnable position embeddings
        self.position_embeddings = nn.Embedding(max_len, d_model)
        
        # Initialize with small random values
        nn.init.normal_(self.position_embeddings.weight, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learned positional embeddings.
        
        Args:
            x: Token embeddings [batch_size, seq_len, d_model]
            
        Returns:
            x + positional embeddings [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        assert seq_len <= self.max_len, f"Sequence length {seq_len} exceeds max length {self.max_len}"
        
        # Create position indices
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Get position embeddings and add to token embeddings
        position_embeds = self.position_embeddings(positions)
        x = x + position_embeds
        
        return self.dropout(x)


class GPTEmbedding(nn.Module):
    """
    Combined token and positional embeddings for GPT-style models.
    
    This combines token embeddings with learned positional embeddings
    and applies dropout, following the GPT architecture.
    
    Args:
        vocab_size: Vocabulary size
        max_len: Maximum sequence length
        d_model: Model dimension
        dropout: Dropout probability
    """
    
    def __init__(self, vocab_size: int, max_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.position_embedding = LearnedPositionalEmbedding(max_len, d_model, dropout=0.0)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert token IDs to embeddings with positional information.
        
        Args:
            token_ids: Token IDs [batch_size, seq_len]
            
        Returns:
            embeddings: [batch_size, seq_len, d_model]
        """
        # Get token embeddings
        token_embeds = self.token_embedding(token_ids)
        
        # Add positional embeddings
        embeddings = self.position_embedding(token_embeds)
        
        return self.dropout(embeddings)


if __name__ == "__main__":
    # Test the embedding layers
    vocab_size, max_len, d_model = 10000, 512, 256
    batch_size, seq_len = 4, 128
    
    # Create random token IDs
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Test TokenEmbedding
    token_emb = TokenEmbedding(vocab_size, d_model)
    token_output = token_emb(token_ids)
    print(f"Token embedding output shape: {token_output.shape}")
    
    # Test PositionalEncoding
    pos_enc = PositionalEncoding(d_model, max_len)
    pos_output = pos_enc(token_output)
    print(f"With positional encoding shape: {pos_output.shape}")
    
    # Test GPTEmbedding
    gpt_emb = GPTEmbedding(vocab_size, max_len, d_model)
    gpt_output = gpt_emb(token_ids)
    print(f"GPT embedding output shape: {gpt_output.shape}")
    
    print("✓ All embedding tests passed!")