#!/usr/bin/env python3
"""
Text generation script for the trained transformer model.

This script loads a trained model and generates text using various
sampling strategies.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import argparse
from pathlib import Path

from src.model.transformer import GPTModel, create_model_config
from src.data.tokenizer import create_tokenizer


class TextGenerator:
    """
    Text generator using a trained transformer model.
    """
    
    def __init__(self, model_path: str, model_size: str = "tiny", device: str = 'auto'):
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Create model
        config = create_model_config(model_size)
        self.model = GPTModel(**config)
        
        # Load checkpoint
        self.load_model(model_path)
        
        # Create tokenizer
        self.tokenizer = create_tokenizer("simple")
        
        print(f"Model loaded with {self.model.get_num_params():,} parameters")
    
    def load_model(self, model_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Direct state dict
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded model from {model_path}")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
        do_sample: bool = True,
        num_return_sequences: int = 1
    ) -> list:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling (None to disable)
            top_p: Top-p sampling (None to disable)
            do_sample: Whether to sample or use greedy decoding
            num_return_sequences: Number of sequences to generate
            
        Returns:
            List of generated texts
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)  # Add batch dim
        input_ids = input_ids.to(self.device)
        
        # Repeat for multiple sequences
        if num_return_sequences > 1:
            input_ids = input_ids.repeat(num_return_sequences, 1)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample
            )
        
        # Decode generated sequences
        generated_texts = []
        for i in range(generated_ids.shape[0]):
            generated_tokens = generated_ids[i].cpu().tolist()
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            generated_texts.append(generated_text)
        
        return generated_texts
    
    def interactive_generation(self):
        """Interactive text generation loop."""
        print("\n=== Interactive Text Generation ===")
        print("Type your prompts below. Type 'quit' to exit.")
        print("Commands:")
        print("  /temp <float>  - Set temperature (default: 1.0)")
        print("  /topk <int>    - Set top-k (default: None)")
        print("  /topp <float>  - Set top-p (default: None)")
        print("  /tokens <int>  - Set max tokens (default: 100)")
        print()
        
        # Default settings
        temperature = 1.0
        top_k = None
        top_p = None
        max_tokens = 100
        
        while True:
            try:
                user_input = input("Prompt: ").strip()
                
                if user_input.lower() == 'quit':
                    break
                
                # Handle commands
                if user_input.startswith('/'):
                    parts = user_input.split()
                    command = parts[0].lower()
                    
                    if command == '/temp' and len(parts) > 1:
                        temperature = float(parts[1])
                        print(f"Temperature set to {temperature}")
                    elif command == '/topk' and len(parts) > 1:
                        top_k = int(parts[1]) if parts[1].lower() != 'none' else None
                        print(f"Top-k set to {top_k}")
                    elif command == '/topp' and len(parts) > 1:
                        top_p = float(parts[1]) if parts[1].lower() != 'none' else None
                        print(f"Top-p set to {top_p}")
                    elif command == '/tokens' and len(parts) > 1:
                        max_tokens = int(parts[1])
                        print(f"Max tokens set to {max_tokens}")
                    else:
                        print("Unknown command")
                    continue
                
                if not user_input:
                    continue
                
                # Generate text
                print(f"\nGenerating (temp={temperature}, top_k={top_k}, top_p={top_p}, max_tokens={max_tokens})...")
                
                generated_texts = self.generate(
                    prompt=user_input,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=True
                )
                
                print(f"\nGenerated text:")
                print("-" * 50)
                print(generated_texts[0])
                print("-" * 50)
                print()
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Generate text with trained transformer')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model_size', type=str, default='tiny',
                       choices=['tiny', 'small', 'medium', 'large'],
                       help='Model size configuration')
    parser.add_argument('--prompt', type=str, default=None,
                       help='Text prompt for generation')
    parser.add_argument('--max_tokens', type=int, default=100,
                       help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=None,
                       help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=None,
                       help='Top-p (nucleus) sampling')
    parser.add_argument('--num_samples', type=int, default=1,
                       help='Number of samples to generate')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto/cpu/cuda)')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not Path(args.model_path).exists():
        print(f"Error: Model file {args.model_path} not found")
        return
    
    # Create generator
    generator = TextGenerator(
        model_path=args.model_path,
        model_size=args.model_size,
        device=args.device
    )
    
    if args.interactive:
        # Interactive mode
        generator.interactive_generation()
    else:
        # Single generation
        if args.prompt is None:
            args.prompt = "The future of artificial intelligence"
        
        print(f"Prompt: {args.prompt}")
        print(f"Generating {args.num_samples} sample(s)...")
        
        generated_texts = generator.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_return_sequences=args.num_samples
        )
        
        print("\n" + "="*50)
        for i, text in enumerate(generated_texts):
            if args.num_samples > 1:
                print(f"\nSample {i+1}:")
                print("-" * 30)
            print(text)
            if i < len(generated_texts) - 1:
                print()
        print("="*50)


if __name__ == "__main__":
    main()