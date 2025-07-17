"""
Complete Transformer Model Implementation

This module implements the full GPT-style transformer model
by combining all the components we've built.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .attention import MultiHeadAttention, create_causal_mask
from .embeddings import GPTEmbedding
from .feedforward import FeedForward


class TransformerBlock(nn.Module):
    """
    A single transformer block (layer).
    
    Each block consists of:
    1. Multi-head self-attention with residual connection and layer norm
    2. Position-wise feed-forward network with residual connection and layer norm
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout probability
        layer_norm_eps: Epsilon for layer normalization
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Layer normalization (pre-norm style like GPT)
        self.ln1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ln2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of transformer block.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            return_attention: Whether to return attention weights
            
        Returns:
            output: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, n_heads, seq_len, seq_len] if return_attention=True
        """
        # Pre-norm style (layer norm before attention)
        # Self-attention with residual connection
        ln_x = self.ln1(x)
        attn_output, attention_weights = self.attention(ln_x, ln_x, ln_x, mask)
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual connection
        ln_x = self.ln2(x)
        ff_output = self.feed_forward(ln_x)
        x = x + self.dropout(ff_output)
        
        if return_attention:
            return x, attention_weights
        else:
            return x, None


class GPTModel(nn.Module):
    """
    GPT-style transformer language model.
    
    This implements a decoder-only transformer for autoregressive language modeling.
    The model predicts the next token given the previous tokens.
    
    Args:
        vocab_size: Size of vocabulary
        max_len: Maximum sequence length
        n_layers: Number of transformer layers
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout probability
        tie_weights: Whether to tie input and output embeddings
    """
    
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        tie_weights: bool = True
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        
        # Token and position embeddings
        self.embeddings = GPTEmbedding(vocab_size, max_len, d_model, dropout)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(d_model)
        
        # Output projection to vocabulary
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Optionally tie input and output embeddings (reduces parameters)
        if tie_weights:
            self.lm_head.weight = self.embeddings.token_embedding.embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights following GPT-2 initialization."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Forward pass of the GPT model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask
            return_attention: Whether to return attention weights
            
        Returns:
            logits: [batch_size, seq_len, vocab_size]
            attention_weights: List of attention weights for each layer if return_attention=True
        """
        batch_size, seq_len = input_ids.shape
        
        # Create causal mask for autoregressive generation
        if attention_mask is None:
            attention_mask = create_causal_mask(seq_len, input_ids.device)
        
        # Get embeddings
        x = self.embeddings(input_ids)
        
        # Pass through transformer layers
        all_attention_weights = [] if return_attention else None
        
        for layer in self.layers:
            x, attention_weights = layer(x, attention_mask, return_attention)
            if return_attention:
                all_attention_weights.append(attention_weights)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)
        
        return logits, all_attention_weights
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Starting token IDs [batch_size, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            do_sample: Whether to sample or use greedy decoding
            
        Returns:
            generated_ids: [batch_size, seq_len + max_new_tokens]
        """
        self.eval()
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get predictions for the last token
                logits, _ = self.forward(generated)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply sampling strategies
                if top_k is not None:
                    # Top-k sampling
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                if top_p is not None:
                    # Top-p (nucleus) sampling
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                if do_sample:
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if we exceed max length
                if generated.size(1) >= self.max_len:
                    break
        
        return generated
    
    def get_num_params(self) -> int:
        """Get the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self) -> int:
        """Get the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model_config(size: str = "small") -> dict:
    """
    Create model configuration for different sizes.
    
    Args:
        size: Model size ("tiny", "small", "medium", "large")
        
    Returns:
        config: Dictionary with model hyperparameters
    """
    configs = {
        "tiny": {
            "vocab_size": 50257,  # GPT-2 tokenizer
            "max_len": 256,
            "n_layers": 2,
            "d_model": 128,
            "n_heads": 2,
            "d_ff": 512,
            "dropout": 0.1
        },
        "small": {
            "vocab_size": 50257,
            "max_len": 512,
            "n_layers": 6,
            "d_model": 256,
            "n_heads": 4,
            "d_ff": 1024,
            "dropout": 0.1
        },
        "medium": {
            "vocab_size": 50257,
            "max_len": 1024,
            "n_layers": 12,
            "d_model": 512,
            "n_heads": 8,
            "d_ff": 2048,
            "dropout": 0.1
        },
        "large": {
            "vocab_size": 50257,
            "max_len": 1024,
            "n_layers": 24,
            "d_model": 1024,
            "n_heads": 16,
            "d_ff": 4096,
            "dropout": 0.1
        }
    }
    
    if size not in configs:
        raise ValueError(f"Unknown model size: {size}. Choose from {list(configs.keys())}")
    
    return configs[size]


if __name__ == "__main__":
    # Test the complete model
    config = create_model_config("tiny")
    
    model = GPTModel(**config)
    
    # Create random input
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
    
    # Forward pass
    logits, attention_weights = model(input_ids, return_attention=True)
    
    print(f"Model parameters: {model.get_num_params():,}")
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Number of attention weight tensors: {len(attention_weights)}")
    
    # Test generation
    generated = model.generate(input_ids[:1, :10], max_new_tokens=20, temperature=0.8)
    print(f"Generated shape: {generated.shape}")
    
    print("✓ Complete transformer model test passed!")