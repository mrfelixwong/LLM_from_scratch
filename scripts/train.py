#!/usr/bin/env python3
"""
Training script for the transformer language model.

This script trains a GPT-style transformer on text data with
comprehensive logging and checkpointing.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import time
from pathlib import Path
import argparse
from tqdm import tqdm

from src.model.transformer import GPTModel, create_model_config
from src.data.tokenizer import create_tokenizer
from src.data.dataset import SimpleTextDataset, create_dataloader, create_sample_text


class Trainer:
    """
    Trainer class for the transformer model.
    
    Handles training loop, validation, checkpointing, and logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader = None,
        optimizer: optim.Optimizer = None,
        scheduler = None,
        device: str = 'auto',
        checkpoint_dir: str = 'checkpoints',
        log_interval: int = 100,
        save_interval: int = 1000
    ):
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Set up optimizer
        if optimizer is None:
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=1e-4,
                weight_decay=0.01,
                betas=(0.9, 0.95)
            )
        else:
            self.optimizer = optimizer
        
        self.scheduler = scheduler
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Logging
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
    
    def train_step(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> float:
        """
        Perform a single training step.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            target_ids: Target token IDs [batch_size, seq_len]
            
        Returns:
            loss: Training loss for this step
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move to device
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)
        
        # Forward pass
        logits, _ = self.model(input_ids)
        
        # Compute loss
        # Reshape for cross entropy: [batch_size * seq_len, vocab_size]
        batch_size, seq_len, vocab_size = logits.shape
        logits = logits.view(-1, vocab_size)
        target_ids = target_ids.view(-1)
        
        loss = self.criterion(logits, target_ids)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update parameters
        self.optimizer.step()
        
        if self.scheduler is not None:
            self.scheduler.step()
        
        return loss.item()
    
    def validate(self) -> float:
        """
        Perform validation.
        
        Returns:
            Average validation loss
        """
        if self.val_dataloader is None:
            return float('inf')
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for input_ids, target_ids in self.val_dataloader:
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                
                logits, _ = self.model(input_ids)
                
                # Compute loss
                batch_size, seq_len, vocab_size = logits.shape
                logits = logits.view(-1, vocab_size)
                target_ids = target_ids.view(-1)
                
                loss = self.criterion(logits, target_ids)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return avg_loss
    
    def save_checkpoint(self, filename: str = None):
        """Save model checkpoint."""
        if filename is None:
            filename = f"checkpoint_step_{self.step}.pt"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        print(f"Loaded checkpoint from step {self.step}")
    
    def train(self, num_epochs: int):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
        """
        print(f"Starting training for {num_epochs} epochs")
        print(f"Model parameters: {self.model.get_num_params():,}")
        
        start_epoch = self.epoch
        
        for epoch in range(start_epoch, start_epoch + num_epochs):
            self.epoch = epoch
            epoch_start_time = time.time()
            
            # Training loop
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")
            epoch_losses = []
            
            for batch_idx, (input_ids, target_ids) in enumerate(pbar):
                loss = self.train_step(input_ids, target_ids)
                epoch_losses.append(loss)
                self.step += 1
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss:.4f}'})
                
                # Logging
                if self.step % self.log_interval == 0:
                    avg_loss = sum(epoch_losses[-self.log_interval:]) / min(self.log_interval, len(epoch_losses))
                    self.train_losses.append((self.step, avg_loss))
                    
                    print(f"Step {self.step}: train_loss = {avg_loss:.4f}")
                
                # Validation and checkpointing
                if self.step % self.save_interval == 0:
                    val_loss = self.validate()
                    self.val_losses.append((self.step, val_loss))
                    
                    print(f"Step {self.step}: val_loss = {val_loss:.4f}")
                    
                    # Save best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint("best_model.pt")
                    
                    # Save regular checkpoint
                    self.save_checkpoint()
            
            # End of epoch
            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            
            print(f"Epoch {epoch} completed in {epoch_time:.2f}s, avg_loss = {avg_epoch_loss:.4f}")
        
        print("Training completed!")


def create_data_split(dataset, train_ratio: float = 0.9):
    """Split dataset into train and validation sets."""
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    return train_dataset, val_dataset


def main():
    parser = argparse.ArgumentParser(description='Train transformer language model')
    parser.add_argument('--model_size', type=str, default='tiny', 
                       choices=['tiny', 'small', 'medium', 'large'],
                       help='Model size configuration')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto/cpu/cuda)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--text_file', type=str, default=None,
                       help='Path to text file for training (default: use sample text)')
    
    args = parser.parse_args()
    
    # Create model configuration
    config = create_model_config(args.model_size)
    model = GPTModel(**config)
    
    # Create tokenizer
    tokenizer = create_tokenizer("simple")
    
    # Create dataset
    if args.text_file and Path(args.text_file).exists():
        with open(args.text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"Loaded text from {args.text_file}")
    else:
        text = create_sample_text()
        print("Using sample text for training")
    
    # Adjust block size to model's max length
    block_size = min(config['max_len'] - 1, 128)  # Leave room for generation
    
    dataset = SimpleTextDataset(text, tokenizer, block_size=block_size)
    
    # Split into train/val
    train_dataset, val_dataset = create_data_split(dataset, train_ratio=0.9)
    
    # Create data loaders
    train_dataloader = create_dataloader(train_dataset, args.batch_size, shuffle=True)
    val_dataloader = create_dataloader(val_dataset, args.batch_size, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.95)
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Resume from checkpoint if provided
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train(args.epochs)


if __name__ == "__main__":
    main()