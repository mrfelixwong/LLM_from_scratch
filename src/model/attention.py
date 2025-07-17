"""
Multi-Head Attention Implementation

This module implements the core attention mechanism from "Attention Is All You Need"
with detailed comments explaining each step of the computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    
    The attention function maps a query and a set of key-value pairs to an output.
    Multi-head attention allows the model to jointly attend to information from
    different representation subspaces at different positions.
    
    Args:
        d_model: Model dimension (embedding size)
        n_heads: Number of attention heads
        dropout: Dropout probability
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension of each head
        
        # Linear projections for Q, K, V
        # We use a single linear layer for all heads and then split
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights using Xavier uniform initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training stability."""
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Compute scaled dot-product attention.
        
        Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
        
        Args:
            Q: Query tensor [batch_size, n_heads, seq_len, d_k]
            K: Key tensor [batch_size, n_heads, seq_len, d_k]
            V: Value tensor [batch_size, n_heads, seq_len, d_k]
            mask: Optional mask to prevent attention to certain positions
            
        Returns:
            attention_output: [batch_size, n_heads, seq_len, d_k]
            attention_weights: [batch_size, n_heads, seq_len, seq_len]
        """
        batch_size, n_heads, seq_len, d_k = Q.shape
        
        # Step 1: Compute attention scores by matrix multiplication of Q and K^T
        # scores shape: [batch_size, n_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1))
        
        # Step 2: Scale by sqrt(d_k) to prevent softmax from saturating
        scores = scores / math.sqrt(d_k)
        
        # Step 3: Apply mask if provided (for causal attention in decoder)
        if mask is not None:
            # Set masked positions to large negative value so softmax gives ~0
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Step 4: Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Step 5: Apply attention weights to values
        attention_output = torch.matmul(attention_weights, V)
        
        return attention_output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            output: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, n_heads, seq_len, seq_len]
        """
        batch_size, seq_len, d_model = query.shape
        
        # Step 1: Linear projections and reshape for multi-head attention
        # Transform to [batch_size, seq_len, n_heads, d_k] then transpose to [batch_size, n_heads, seq_len, d_k]
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Step 2: Apply scaled dot-product attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Step 3: Concatenate heads and project
        # Transpose back to [batch_size, seq_len, n_heads, d_k] then reshape to [batch_size, seq_len, d_model]
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Step 4: Final linear projection
        output = self.w_o(attention_output)
        
        return output, attention_weights


def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    Create a causal (lower triangular) mask for decoder self-attention.
    
    This mask ensures that position i can only attend to positions j <= i,
    implementing the autoregressive property needed for language modeling.
    
    Args:
        seq_len: Sequence length
        device: Device to create the mask on
        
    Returns:
        mask: [1, 1, seq_len, seq_len] boolean mask
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions


if __name__ == "__main__":
    # Simple test of the attention mechanism
    batch_size, seq_len, d_model, n_heads = 2, 10, 512, 8
    
    # Create random input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Create attention layer
    attention = MultiHeadAttention(d_model, n_heads)
    
    # Create causal mask
    mask = create_causal_mask(seq_len)
    
    # Forward pass
    output, attn_weights = attention(x, x, x, mask)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print("✓ Multi-head attention test passed!")