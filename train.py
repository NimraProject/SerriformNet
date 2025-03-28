import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
from typing import Optional, Tuple
import time
import os
import sys
import random

# RMSNorm for stability and efficiency
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Calculate RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return self.weight * (x / rms)

# RoPE implementation (Rotary Position Embedding)
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.base = base
        
        # Make sure dim is even for RoPE
        if dim % 2 != 0:
            raise ValueError(f"Dimension must be even for RoPE, got {dim}")
        
        # Generate frequency bands
        half_dim = dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, half_dim).float() / half_dim))
        self.register_buffer("inv_freq", inv_freq)
        
    def _rotate_half(self, x):
        """Rotates half the hidden dims of x."""
        half_d = x.shape[-1] // 2
        x1 = x[..., :half_d]
        x2 = x[..., half_d:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x, seq_len: Optional[int] = None):
        """
        Apply rotary position embeddings to input tensor x.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            seq_len: Optional sequence length if different from x.shape[1]
            
        Returns:
            Tensor with rotary position embeddings applied
        """
        batch_size, seq_length, dim = x.shape
        
        # Generate position indices
        seq_len = seq_length if seq_len is None else seq_len
        seq_idx = torch.arange(seq_len, device=x.device)
        
        # Generate position embeddings
        # [seq_len, half_dim]
        freqs = torch.outer(seq_idx, self.inv_freq)
        
        # Calculate cos and sin embeddings
        # [seq_len, half_dim] -> [seq_len, half_dim*2]
        emb = torch.cat([freqs, freqs], dim=-1)
        cos_emb = torch.cos(emb)
        sin_emb = torch.sin(emb)
        
        # Reshape for batched broadcasting
        # [1, seq_len, dim]
        cos_emb = cos_emb.unsqueeze(0)
        sin_emb = sin_emb.unsqueeze(0)
        
        # Apply position embeddings
        # x_rot = x * cos + self._rotate_half(x) * sin
        return x * cos_emb + self._rotate_half(x) * sin_emb

# Sparse Shift-Gate Recurrence
class SparseShiftGateRecurrence(nn.Module):
    def __init__(self, dim: int, sparse_ratio: float = 0.25):
        super().__init__()
        self.dim = dim
        self.sparse_ratio = sparse_ratio
        self.active_dim = int(dim * sparse_ratio)
        
        # Learnable scalars for recurrence
        self.alpha = nn.Parameter(torch.ones(self.active_dim) * 0.9)  # Init close to 1 for memory
        self.beta = nn.Parameter(torch.ones(self.active_dim) * 0.1)   # Init small for stability
        
        # Which channels participate in recurrence
        active_channels = np.random.choice(dim, self.active_dim, replace=False)
        active_mask = torch.zeros(dim)
        active_mask[active_channels] = 1.0
        self.register_buffer("active_mask", active_mask)

    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        
        # Initialize hidden state
        h = torch.zeros(batch_size, dim, device=x.device)
        outputs = []
        
        # Only compute recurrence for active channels
        for t in range(seq_len):
            x_t = x[:, t]
            
            # Only update active channels, rest are passthrough
            h_active = h[:, :self.active_dim] * self.alpha + x_t[:, :self.active_dim] * self.beta
            
            # Update hidden state selectively using the active mask
            h_new = x_t.clone()
            h_new[:, :self.active_dim] = h_active
            h = h_new
            
            outputs.append(h.unsqueeze(1))
            
        return torch.cat(outputs, dim=1)

# Gated Fusion Module
class GatedFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.W1 = nn.Linear(dim, dim)
        self.W2 = nn.Linear(dim, dim)
        
    def forward(self, x):
        return x + torch.sigmoid(self.W1(x)) * torch.tanh(self.W2(x))

# Low-Rank Feedforward Network
class LowRankFF(nn.Module):
    def __init__(self, dim: int, reduction_factor: int = 4):
        super().__init__()
        self.hidden_dim = dim // reduction_factor
        self.down_proj = nn.Linear(dim, self.hidden_dim)
        self.up_proj = nn.Linear(self.hidden_dim, dim)
        
    def forward(self, x):
        return self.up_proj(F.gelu(self.down_proj(x)))

# Serriform Block
class SerriformBlock(nn.Module):
    def __init__(self, dim: int, dilation: int = 1):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.dilation = dilation
        self.kernel_size = 5
        
        # Calculate effective kernel size with dilation
        self.effective_kernel_size = (self.kernel_size - 1) * dilation + 1
        
        # Calculate padding needed to maintain sequence length
        # For a kernel of size k and dilation d, padding needed is (k-1)*d/2
        self.padding = ((self.kernel_size - 1) * dilation) // 2
        
        # Depthwise separable conv with dilation 
        self.depth_conv = nn.Conv1d(
            dim, dim, kernel_size=self.kernel_size, 
            padding=self.padding,  # Use calculated padding
            groups=dim, dilation=dilation
        )
        self.point_conv = nn.Conv1d(dim, dim, kernel_size=1)
        
        self.sparse_recurrence = SparseShiftGateRecurrence(dim)
        self.fusion = GatedFusion(dim)
        self.ff = LowRankFF(dim)
        
    def forward(self, x):
        # Apply normalization first (pre-activation)
        residual = x
        x = self.norm(x)
        
        # Reshape for convolution [B, L, D] -> [B, D, L]
        B, L, D = x.shape
        x_conv = x.transpose(1, 2)
        
        # Apply convolutions (now with padding to preserve length better)
        x_conv = self.depth_conv(x_conv)
        x_conv = self.point_conv(x_conv)
        
        # Sequence length might still change slightly due to asymmetric padding
        L_new = x_conv.shape[2]
        
        # Reshape back to [B, L, D] for further processing
        x_conv = x_conv.transpose(1, 2)
        
        # Apply sparse recurrence (use the shorter of the two sequence lengths)
        if L_new < L:
            x_rec = self.sparse_recurrence(x[:, :L_new])
            # Add residual connection (with appropriate truncation)
            residual = residual[:, :L_new]
        else:
            x_rec = self.sparse_recurrence(x)
            # If conv output is longer (unlikely), truncate it
            x_conv = x_conv[:, :L, :]
            L_new = L
        
        # Gated fusion of conv and recurrence pathways
        x = x_conv + x_rec
        x = self.fusion(x)
        
        # Low-rank feedforward
        x = x + self.ff(x)
        
        # Add residual connection
        x = x + residual
        
        return x

# SerriformNet Model
class SerriformNet(nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        dim: int = 512, 
        num_layers: int = 12,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Input embedding layer
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.rope = RotaryPositionalEmbedding(dim)
        
        # Serriform blocks with varying dilations to create hierarchical receptive field
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            # Use varying dilation patterns to create serrated receptive fields
            dilation = 2 ** (i % 3)  # Cycles through 1, 2, 4
            self.blocks.append(SerriformBlock(dim, dilation=dilation))
            
        # Output projection
        self.norm = RMSNorm(dim)
        self.output_proj = nn.Linear(dim, vocab_size, bias=False)
        
        # Tie weights with embedding
        self.output_proj.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, x):
        # Get token embeddings
        token_emb = self.token_embedding(x)
        
        # Apply RoPE
        x = self.rope(token_emb)
        
        # Track original sequence length
        orig_seq_len = x.size(1)
        
        # Process through serriform blocks
        for i, block in enumerate(self.blocks):
            x = block(x)
            
            # Log sequence length changes for debugging (first batch only)
            if i == 0 or i == len(self.blocks) - 1:
                curr_seq_len = x.size(1)
                reduction = orig_seq_len - curr_seq_len
                
                # Only print occasionally to avoid flooding logs
                if i == 0 and x.size(0) > 0 and getattr(self, '_print_counter', 0) % 100 == 0:
                    print(f"Block {i+1}/{len(self.blocks)}: Input seq_len={orig_seq_len}, " +
                          f"Output seq_len={curr_seq_len}, Reduction={reduction}")
                    
            # Exit early if sequence gets too short
            if x.size(1) <= 2:
                print(f"Warning: Sequence too short after block {i+1}, stopping early.")
                break
                
        # Update print counter
        self._print_counter = getattr(self, '_print_counter', 0) + 1
                
        # Final normalization and projection
        x = self.norm(x)
        logits = self.output_proj(x)
        
        return logits

    def generate(self, 
                 prompt_ids: torch.Tensor, 
                 max_new_tokens: int = 100, 
                 temperature: float = 1.0,
                 top_k: int = 50):
        """
        Generate text using the model
        """
        self.eval()
        
        # Start with the prompt
        input_ids = prompt_ids.clone().to(next(self.parameters()).device)
        batch_size = input_ids.shape[0]
        
        generated_ids = []
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                outputs = self(input_ids)
                
                # Get next token logits (last position only)
                next_token_logits = outputs[:, -1, :] / temperature
                
                # Apply top-k sampling
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                
                # Sample from top-k
                probs = F.softmax(top_k_logits, dim=-1)
                next_token_idx = torch.multinomial(probs, num_samples=1)
                
                # Convert back to vocabulary indices
                next_token = torch.gather(top_k_indices, -1, next_token_idx)
                
                # Append to generated sequence
                generated_ids.append(next_token)
                
                # Update input_ids for next iteration
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # If we exceed max sequence length, remove first token
                if input_ids.shape[1] > self.max_seq_len:
                    input_ids = input_ids[:, 1:]
        
        # Concatenate all generated tokens
        generated_tokens = torch.cat(generated_ids, dim=1)
        return generated_tokens

class TextDataset(Dataset):
    def __init__(self, texts, seq_len, tokenizer):
        self.texts = texts
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        # Tokenize and truncate/pad to seq_len
        tokens = self.tokenizer.encode(text)[:self.seq_len]
        
        # Create input and target tensors
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        
        return x, y

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    start_time = time.time()
    
    for i, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        logits = model(x)
        
        # Since convolutions reduce sequence length, we need to truncate y to match
        seq_len_out = logits.size(1)
        seq_len_y = y.size(1)
        
        if seq_len_out != seq_len_y:
            # Truncate y to match the output sequence length
            y = y[:, :seq_len_out]
        
        # Compute loss (cross entropy)
        loss = F.cross_entropy(logits.view(-1, model.vocab_size), y.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if i % 50 == 0:
            elapsed = time.time() - start_time
            print(f"Batch {i}, Loss: {loss.item():.4f}, Time: {elapsed:.2f}s")
            print(f"Input shape: {x.shape}, Output shape: {logits.shape}, Target shape: {y.shape}")
            start_time = time.time()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits = model(x)
            
            # Truncate y to match output sequence length
            seq_len_out = logits.size(1)
            seq_len_y = y.size(1)
            
            if seq_len_out != seq_len_y:
                y = y[:, :seq_len_out]
            
            # Compute loss
            loss = F.cross_entropy(logits.view(-1, model.vocab_size), y.view(-1))
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser(description='Train SerriformNet Language Model')
    parser.add_argument('--dim', type=int, default=512, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=12, help='Number of Serriform blocks')
    parser.add_argument('--max_seq_len', type=int, default=1024, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to dataset')
    parser.add_argument('--save_path', type=str, default='./checkpoints', help='Path to save checkpoints')
    parser.add_argument('--log_interval', type=int, default=100, help='Log interval')
    parser.add_argument('--save_interval', type=int, default=1000, help='Save interval')
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples to use')
    parser.add_argument('--tokenizer', type=str, default='gpt2', help='Tokenizer to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_path, exist_ok=True)
    
    # Load data
    try:
        # Try to import the data module
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from data import load_and_prepare_openwebtext

        print("Loading and preparing data...")
        train_dataloader, val_dataloader, tokenizer, vocab_size = load_and_prepare_openwebtext(
            save_path=args.data_path,
            tokenizer_name=args.tokenizer,
            seq_len=args.max_seq_len,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            num_workers=4,
            seed=args.seed,
            use_memory_mapping=True
        )
        
        print(f"Data loaded. Vocab size: {vocab_size}")
        print(f"Train batches: {len(train_dataloader)}, Validation batches: {len(val_dataloader)}")
    
    except (ImportError, ModuleNotFoundError):
        print("Could not import data module. Using dummy data for demonstration.")
        # For demonstration, create dummy data
        vocab_size = 50257  # GPT-2 vocab size
        
        # Create dummy data loaders
        class DummyDataLoader:
            def __init__(self, batch_size, seq_len, vocab_size, n_batches=100):
                self.batch_size = batch_size
                self.seq_len = seq_len
                self.vocab_size = vocab_size
                self.n_batches = n_batches
                
            def __iter__(self):
                for _ in range(self.n_batches):
                    x = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
                    y = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
                    yield x, y
                    
            def __len__(self):
                return self.n_batches
        
        train_dataloader = DummyDataLoader(args.batch_size, args.max_seq_len, vocab_size)
        val_dataloader = DummyDataLoader(args.batch_size, args.max_seq_len, vocab_size, n_batches=20)
    
    # Initialize model
    print(f"Initializing SerriformNet with {args.num_layers} layers, dim={args.dim}...")
    model = SerriformNet(
        vocab_size=vocab_size,
        dim=args.dim,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len
    ).to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"SerriformNet initialized with {total_params:,} parameters")
    
    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_dataloader))
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Training
        train_loss = train_epoch(model, train_dataloader, optimizer, device)
        
        # Validation
        val_loss = evaluate(model, val_dataloader, device)
        
        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.save_path, "serriformnet_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'args': vars(args)
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path} (val_loss: {val_loss:.4f})")
        
        # Regular epoch checkpoint
        checkpoint_path = os.path.join(args.save_path, f"serriformnet_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
            'args': vars(args)
        }, checkpoint_path)
        
        # Update scheduler
        scheduler.step()
    
    print("Training complete!")
    
    # Generate a sample from the trained model
    try:
        print("\nGenerating sample text...")
        if 'tokenizer' in locals():
            prompt = "The future of artificial intelligence is"
            prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            
            model.eval()
            with torch.no_grad():
                output_ids = model.generate(
                    prompt_ids, 
                    max_new_tokens=50,
                    temperature=0.7,
                    top_k=50
                )
            
            generated_text = tokenizer.decode(output_ids[0])
            print(f"Prompt: {prompt}")
            print(f"Generated: {generated_text}")
    except Exception as e:
        print(f"Could not generate sample text: {e}")

if __name__ == "__main__":
    main()
