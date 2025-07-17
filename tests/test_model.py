#!/usr/bin/env python3
"""
Quick test script to verify the transformer implementation works.
"""

import torch
from src.model.transformer import GPTModel, create_model_config
from src.data.tokenizer import create_tokenizer
from src.data.dataset import SimpleTextDataset, create_dataloader

def test_components():
    """Test individual components."""
    print("🧪 Testing Transformer Components")
    print("=" * 50)
    
    # Test attention
    print("✓ Testing attention mechanism...")
    from src.model.attention import MultiHeadAttention, create_causal_mask
    
    d_model, n_heads, seq_len = 128, 4, 16
    attention = MultiHeadAttention(d_model, n_heads)
    
    x = torch.randn(2, seq_len, d_model)
    mask = create_causal_mask(seq_len)
    
    output, attn_weights = attention(x, x, x, mask)
    assert output.shape == (2, seq_len, d_model)
    assert attn_weights.shape == (2, n_heads, seq_len, seq_len)
    print(f"   Attention output shape: {output.shape}")
    
    # Test embeddings
    print("✓ Testing embeddings...")
    from src.model.embeddings import GPTEmbedding
    
    vocab_size, max_len = 1000, 512
    embedding = GPTEmbedding(vocab_size, max_len, d_model)
    
    token_ids = torch.randint(0, vocab_size, (2, seq_len))
    embedded = embedding(token_ids)
    assert embedded.shape == (2, seq_len, d_model)
    print(f"   Embedding output shape: {embedded.shape}")
    
    # Test feed-forward
    print("✓ Testing feed-forward network...")
    from src.model.feedforward import FeedForward
    
    ff = FeedForward(d_model, d_ff=512)
    ff_output = ff(x)
    assert ff_output.shape == (2, seq_len, d_model)
    print(f"   Feed-forward output shape: {ff_output.shape}")
    
    print("✅ All components working correctly!")


def test_full_model():
    """Test the complete model."""
    print("\n🚀 Testing Complete Model")
    print("=" * 50)
    
    # Create tiny model for testing
    config = create_model_config("tiny")
    model = GPTModel(**config)
    
    print(f"✓ Created model with {model.get_num_params():,} parameters")
    
    # Test forward pass
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
    
    logits, attention_weights = model(input_ids, return_attention=True)
    
    expected_logits_shape = (batch_size, seq_len, config["vocab_size"])
    assert logits.shape == expected_logits_shape
    assert len(attention_weights) == config["n_layers"]
    
    print(f"✓ Forward pass successful")
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Logits shape: {logits.shape}")
    print(f"   Number of attention layers: {len(attention_weights)}")
    
    # Test generation
    prompt_ids = input_ids[:1, :10]  # Take first sample, first 10 tokens
    generated = model.generate(prompt_ids, max_new_tokens=20, temperature=0.8)
    
    assert generated.shape[0] == 1
    assert generated.shape[1] >= prompt_ids.shape[1] + 20
    
    print(f"✓ Generation successful")
    print(f"   Prompt length: {prompt_ids.shape[1]}")
    print(f"   Generated length: {generated.shape[1]}")
    
    print("✅ Model working correctly!")


def test_data_pipeline():
    """Test data loading and preprocessing."""
    print("\n📚 Testing Data Pipeline")
    print("=" * 50)
    
    # Test tokenizer
    tokenizer = create_tokenizer("simple")
    test_text = "Hello world! This is a test."
    
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"✓ Tokenizer working")
    print(f"   Original: {test_text}")
    print(f"   Encoded length: {len(encoded)}")
    print(f"   Decoded: {decoded}")
    
    # Test dataset
    sample_text = "The quick brown fox jumps over the lazy dog. " * 10
    dataset = SimpleTextDataset(sample_text, tokenizer, block_size=32)
    
    print(f"✓ Dataset created with {len(dataset)} samples")
    
    # Test dataloader
    dataloader = create_dataloader(dataset, batch_size=4, shuffle=False)
    
    for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
        print(f"✓ Batch {batch_idx}: input {input_ids.shape}, target {target_ids.shape}")
        if batch_idx >= 2:
            break
    
    print("✅ Data pipeline working correctly!")


def test_training_setup():
    """Test that training setup works."""
    print("\n🏋️ Testing Training Setup")
    print("=" * 50)
    
    # Create small model and data
    config = create_model_config("tiny")
    model = GPTModel(**config)
    
    tokenizer = create_tokenizer("simple")
    sample_text = "The transformer architecture revolutionized NLP. " * 20
    dataset = SimpleTextDataset(sample_text, tokenizer, block_size=64)
    dataloader = create_dataloader(dataset, batch_size=2, shuffle=True)
    
    # Setup training components
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Test one training step
    model.train()
    for input_ids, target_ids in dataloader:
        optimizer.zero_grad()
        
        logits, _ = model(input_ids)
        
        # Reshape for loss computation
        batch_size, seq_len, vocab_size = logits.shape
        logits = logits.view(-1, vocab_size)
        target_ids = target_ids.view(-1)
        
        loss = criterion(logits, target_ids)
        loss.backward()
        optimizer.step()
        
        print(f"✓ Training step successful, loss: {loss.item():.4f}")
        break
    
    print("✅ Training setup working correctly!")


def main():
    """Run all tests."""
    print("🧬 Transformer LLM Implementation Test")
    print("=" * 60)
    
    try:
        test_components()
        test_full_model()
        test_data_pipeline()
        test_training_setup()
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed! The implementation is working correctly.")
        print("\nYou can now:")
        print("1. Run training: python scripts/train.py --model_size tiny --epochs 5")
        print("2. Generate text: python scripts/generate.py --model_path checkpoints/best_model.pt")
        print("3. Explore notebooks: jupyter notebook notebooks/")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()