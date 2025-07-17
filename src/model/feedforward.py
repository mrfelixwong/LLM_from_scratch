"""
Feed-Forward Network Implementation

This module implements the position-wise feed-forward networks
used in transformer blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    This is applied to each position separately and identically.
    It consists of two linear transformations with a ReLU activation in between.
    
    FFN(x) = max(0, xW1 + b1)W2 + b2
    
    Args:
        d_model: Model dimension
        d_ff: Feed-forward dimension (typically 4 * d_model)
        dropout: Dropout probability
        activation: Activation function ('relu', 'gelu', 'swish')
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = 'relu'):
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Two linear layers
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'swish' or activation == 'silu':
            self.activation = F.silu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training stability."""
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feed-forward network.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # First linear layer + activation
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Second linear layer
        x = self.linear2(x)
        
        return x


class GLU(nn.Module):
    """
    Gated Linear Unit (GLU) activation.
    
    GLU(x) = (xW + b) ⊗ σ(xV + c)
    where ⊗ is element-wise multiplication and σ is sigmoid.
    
    This is used in some modern transformer variants.
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Two parallel linear layers
        self.linear_gate = nn.Linear(d_model, d_ff)
        self.linear_value = nn.Linear(d_model, d_ff)
        
        # Output projection
        self.linear_out = nn.Linear(d_ff, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for linear in [self.linear_gate, self.linear_value, self.linear_out]:
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of GLU.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        gate = torch.sigmoid(self.linear_gate(x))
        value = self.linear_value(x)
        
        # Element-wise multiplication
        gated = gate * value
        gated = self.dropout(gated)
        
        output = self.linear_out(gated)
        
        return output


class SwiGLU(nn.Module):
    """
    SwiGLU activation function from PaLM paper.
    
    SwiGLU(x) = Swish(xW + b) ⊗ (xV + c)
    where Swish(x) = x * sigmoid(x)
    
    This has shown good performance in large language models.
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Three linear layers (gate, value, output)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_value = nn.Linear(d_model, d_ff, bias=False)
        self.w_out = nn.Linear(d_ff, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for linear in [self.w_gate, self.w_value, self.w_out]:
            nn.init.xavier_uniform_(linear.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SwiGLU.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        gate = F.silu(self.w_gate(x))  # Swish activation
        value = self.w_value(x)
        
        # Element-wise multiplication
        gated = gate * value
        gated = self.dropout(gated)
        
        output = self.w_out(gated)
        
        return output


if __name__ == "__main__":
    # Test the feed-forward networks
    batch_size, seq_len, d_model, d_ff = 2, 10, 256, 1024
    
    # Create random input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Test standard feed-forward
    ff = FeedForward(d_model, d_ff)
    output = ff(x)
    print(f"FeedForward output shape: {output.shape}")
    
    # Test GLU
    glu = GLU(d_model, d_ff)
    output_glu = glu(x)
    print(f"GLU output shape: {output_glu.shape}")
    
    # Test SwiGLU
    swiglu = SwiGLU(d_model, d_ff)
    output_swiglu = swiglu(x)
    print(f"SwiGLU output shape: {output_swiglu.shape}")
    
    print("✓ All feed-forward tests passed!")