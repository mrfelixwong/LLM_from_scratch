#!/usr/bin/env python3
"""
Ultra-simple transformer demo that actually learns.
Uses a tiny model with minimal parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TinyAttention(nn.Module):
    def __init__(self, dim=32, heads=2):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.head_dim = dim // heads
        
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False) 
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim)
        
    def forward(self, x):
        B, T, C = x.shape
        
        q = self.q(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        
        # Attention
        att = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        
        # Causal mask
        mask = torch.tril(torch.ones(T, T, device=x.device))
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size=10, dim=32, max_len=10):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(max_len, dim)
        self.attention = TinyAttention(dim)
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
        
    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        
        # Embeddings
        tok_emb = self.embed(x)
        pos_emb = self.pos(pos)
        x = tok_emb + pos_emb
        
        # Single attention layer
        x = x + self.attention(self.ln(x))
        
        # Output
        logits = self.head(x)
        return logits

def train_counting():
    """Train model to count: 0->1, 1->2, 2->3, etc."""
    
    # Create simple counting dataset
    data = []
    for i in range(100):  # 100 examples
        seq = [(i + j) % 10 for j in range(5)]  # [0,1,2,3,4], [1,2,3,4,5], etc.
        data.append(seq)
    
    # Convert to tensors
    X = torch.tensor([seq[:-1] for seq in data])  # Input: [0,1,2,3]
    Y = torch.tensor([seq[1:] for seq in data])   # Target: [1,2,3,4]
    
    print("Training data examples:")
    for i in range(5):
        print(f"Input: {X[i].tolist()} -> Target: {Y[i].tolist()}")
    
    # Create tiny model
    model = TinyTransformer(vocab_size=10, dim=32, max_len=10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Training...")
    
    # Train
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        
        logits = model(X)
        loss = criterion(logits.view(-1, 10), Y.view(-1))
        
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    print("\nTesting the trained model:")
    model.eval()
    
    # Test on new sequences
    test_inputs = [
        [0, 1, 2],
        [3, 4, 5], 
        [7, 8, 9],
        [2, 3, 4]
    ]
    
    with torch.no_grad():
        for test_input in test_inputs:
            x = torch.tensor([test_input])
            logits = model(x)
            predictions = torch.argmax(logits, dim=-1)[0]
            
            print(f"Input: {test_input} -> Predicted: {predictions.tolist()}")
            print(f"Expected: {[(i+1) % 10 for i in test_input]}")
            print()

if __name__ == "__main__":
    train_counting()