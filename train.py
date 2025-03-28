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

# Improved recurrence mechanism with learnable memory tokens
class StructuredStateRecurrence(nn.Module):
    def __init__(self, dim: int, memory_dim: int = None, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.memory_dim = memory_dim or dim // 4
        
        # Memory state projection
        self.memory_proj_in = nn.Linear(dim, self.memory_dim)
        self.memory_proj_out = nn.Linear(self.memory_dim, dim)
        
        # State update gates
        self.forget_gate = nn.Linear(dim + self.memory_dim, self.memory_dim)
        self.update_gate = nn.Linear(dim + self.memory_dim, self.memory_dim)
        self.output_gate = nn.Linear(dim + self.memory_dim, dim)
        
        self.act = nn.SiLU()  # SiLU (Swish) activation for smoother gradients
        self.dropout = nn.Dropout(dropout)
        
        # Initialize with special attention to gates
        self._init_weights()
        
    def _init_weights(self):
        # Initialize forget gate bias to 1.0 (remember more by default)
        nn.init.zeros_(self.forget_gate.bias)
        self.forget_gate.bias.data.fill_(1.0)
        
        # Carefully initialize other weights
        for module in [self.memory_proj_in, self.memory_proj_out, self.update_gate, self.output_gate]:
            nn.init.xavier_normal_(module.weight, gain=1.0)
            nn.init.zeros_(module.bias)
    
    def forward(self, x, memory_state=None):
        batch_size, seq_len, dim = x.shape
        
        # Initialize memory state if not provided
        if memory_state is None:
            memory_state = torch.zeros(batch_size, self.memory_dim, device=x.device)
        
        outputs = []
        
        # Process sequence step by step to maintain causality
        for t in range(seq_len):
            # Get current input
            x_t = x[:, t]
            
            # Project input to memory dimension
            x_memory = self.memory_proj_in(x_t)
            
            # Compute gates using concatenated input and memory state
            combined = torch.cat([x_t, memory_state], dim=-1)
            forget_gate = torch.sigmoid(self.forget_gate(combined))
            update_gate = torch.sigmoid(self.update_gate(combined))
            
            # Update memory state
            memory_update = self.act(x_memory)
            memory_state = forget_gate * memory_state + update_gate * memory_update
            
            # Apply dropout for regularization
            memory_state = self.dropout(memory_state)
            
            # Compute output using gate and updated memory
            combined = torch.cat([x_t, memory_state], dim=-1)
            output = x_t + self.output_gate(combined)
            
            outputs.append(output.unsqueeze(1))
        
        # Stack outputs and return
        return torch.cat(outputs, dim=1), memory_state

# Low-Rank Feedforward Network with improved initialization and dropout
class LowRankFF(nn.Module):
    def __init__(self, dim: int, reduction_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = dim // reduction_factor
        self.down_proj = nn.Linear(dim, self.hidden_dim)
        self.up_proj = nn.Linear(self.hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        
        # Improved initialization for better gradient flow
        self._init_weights()
        
    def _init_weights(self):
        # Use Kaiming initialization for GELU activation
        nn.init.kaiming_normal_(self.down_proj.weight, nonlinearity='linear')
        nn.init.zeros_(self.down_proj.bias)
        
        # Use small random values for up projection (output layer)
        nn.init.normal_(self.up_proj.weight, std=0.02)
        nn.init.zeros_(self.up_proj.bias)
        
    def forward(self, x):
        return self.up_proj(self.dropout(self.act(self.down_proj(x))))

# Enhanced Gated Fusion with mixture-of-experts style routing
class EnhancedGatedFusion(nn.Module):
    def __init__(self, dim: int, num_experts: int = 2, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        
        # Router network to assign weights to experts
        self.router = nn.Sequential(
            nn.Linear(dim, num_experts),
            nn.Softmax(dim=-1)
        )
        
        # Expert networks (simple transformations)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(dim, dim)
            ) for _ in range(num_experts)
        ])
        
        # Final layer norm for stabilization
        self.norm = RMSNorm(dim)
        
    def forward(self, x):
        # Get routing weights
        routing_weights = self.router(x)  # [batch, seq_len, num_experts]
        
        # Apply each expert and weight its output
        expert_outputs = 0
        for i, expert in enumerate(self.experts):
            # Extract expert weight and reshape for broadcast
            weight = routing_weights[..., i:i+1]
            
            # Apply expert and weighted sum
            expert_outputs += weight * expert(x)
        
        # Residual connection and normalization
        return self.norm(x + expert_outputs)

# Serriform Block with causal convolutions and improved components
class SerriformBlock(nn.Module):
    def __init__(self, dim: int, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.dilation = dilation
        self.kernel_size = 5
        self.dropout = dropout
        
        # Calculate correct padding for causal convolution
        # For causal: padding = (kernel_size - 1) * dilation
        self.padding = (self.kernel_size - 1) * dilation
        
        # Causal depthwise separable convolution
        self.depth_conv = nn.Conv1d(
            dim, dim, kernel_size=self.kernel_size, 
            padding=self.padding, padding_mode='zeros',
            groups=dim, dilation=dilation
        )
        self.point_conv = nn.Conv1d(dim, dim, kernel_size=1)
        
        # Ensure convolution is causal by masking future inputs
        self.register_buffer('causal_mask', None)  # Will be created on first forward pass
        
        # Replace simple recurrence with a more advanced mechanism
        self.recurrence = StructuredStateRecurrence(dim, dropout=dropout)
        
        # Improved gated fusion with MoE-style routing
        self.fusion = EnhancedGatedFusion(dim, dropout=dropout)
        
        # Low-rank feed-forward network
        self.ff = LowRankFF(dim, dropout=dropout)
        
        # Dropout for regularization
        self.dropout_layer = nn.Dropout(dropout)
        
    def _apply_causal_mask(self, x):
        """Apply causal mask to ensure we don't look at future tokens."""
        batch_size, channels, seq_len = x.shape
        
        # Create or check the causal mask
        if self.causal_mask is None or self.causal_mask.size(0) < seq_len:
            # Create a causal mask (lower triangular)
            mask = torch.ones(seq_len, seq_len, device=x.device).tril_()
            self.register_buffer('causal_mask', mask, persistent=False)
        
        # Apply the mask to the output of the depth convolution
        # We need to reshape x to apply the mask
        x_reshaped = x.transpose(1, 2)  # [batch, seq_len, channels]
        mask = self.causal_mask[:seq_len, :seq_len]
        
        # Apply mask and reshape back
        x_masked = x_reshaped * mask.unsqueeze(-1)
        return x_masked.transpose(1, 2)  # [batch, channels, seq_len]
        
    def forward(self, x, memory_state=None):
        # Apply normalization first (pre-activation)
        residual = x
        x = self.norm(x)
        
        # Reshape for convolution [B, L, D] -> [B, D, L]
        B, L, D = x.shape
        x_conv = x.transpose(1, 2)
        
        # Apply causal depthwise separable convolution
        x_conv = self.depth_conv(x_conv)
        
        # Ensure causality
        x_conv = x_conv[..., :L]  # We only need the first L elements due to padding
        
        # Apply pointwise convolution
        x_conv = self.point_conv(x_conv)
        
        # Reshape back to [B, L, D] for further processing
        x_conv = x_conv.transpose(1, 2)
        
        # Apply advanced recurrence with current memory state
        x_rec, new_memory_state = self.recurrence(x, memory_state)
        
        # Apply dropout for regularization
        x_conv = self.dropout_layer(x_conv)
        x_rec = self.dropout_layer(x_rec)
        
        # Combine conv and recurrence pathways with enhanced fusion
        x = x_conv + x_rec
        x = self.fusion(x)
        
        # Apply feed-forward network
        x = x + self.ff(x)
        
        # Add residual connection
        x = x + residual
        
        return x, new_memory_state

# SerriformNet Model with improved efficiency and generation capability
class SerriformNet(nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        dim: int = 512, 
        num_layers: int = 12,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        use_cache: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.use_cache = use_cache
        
        # Input embedding layer
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.rope = RotaryPositionalEmbedding(dim)
        self.embed_dropout = nn.Dropout(dropout)
        
        # Serriform blocks with varying dilations to create hierarchical receptive field
        self.blocks = nn.ModuleList()
        
        # Use logarithmic spacing for better long-range modeling
        # This creates a more diverse set of receptive fields
        log_factor = np.log(8) / num_layers  # Targeting max dilation ~8
        
        for i in range(num_layers):
            # More sophisticated dilation strategy: logarithmic spacing
            dilation = max(1, int(np.exp(i * log_factor)))
            self.blocks.append(SerriformBlock(dim, dilation=dilation, dropout=dropout))
            
        # Output projection
        self.norm = RMSNorm(dim)
        self.output_proj = nn.Linear(dim, vocab_size, bias=False)
        
        # Tie weights with embedding
        self.output_proj.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Logging for training
        self._global_step = 0
        
    def _init_weights(self, module):
        """Specialized weight initialization for different module types"""
        if isinstance(module, nn.Linear):
            # Use Xavier for linear layers
            nn.init.xavier_uniform_(module.weight, gain=0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Special handling for embedding layers
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm) or isinstance(module, RMSNorm):
            # Init for normalization layers
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.ones_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            # For convolution layers, use Kaiming init
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            
    def get_input_embeddings(self):
        """Get input embedding layer for compatibility with HF-style code"""
        return self.token_embedding
        
    def forward(
        self, 
        x, 
        past_key_values=None, 
        use_cache=None,
        output_hidden_states=False,
        return_dict=True
    ):
        """
        Forward pass supporting cached key-values for efficient generation
        
        Args:
            x: Input tensor of token IDs [batch_size, seq_len]
            past_key_values: Cached memory states from previous forward passes
            use_cache: Whether to use caching
            output_hidden_states: Whether to return hidden states from all layers
            return_dict: Whether to return a dictionary or a tuple
        """
        use_cache = use_cache if use_cache is not None else self.use_cache
        
        # For training without caching
        if past_key_values is None:
            past_key_values = [None] * len(self.blocks)
        
        # Get token embeddings
        token_emb = self.token_embedding(x)
        
        # Apply RoPE
        hidden_states = self.rope(token_emb)
        hidden_states = self.embed_dropout(hidden_states)
        
        # Track all states if needed
        all_hidden_states = () if output_hidden_states else None
        new_memory_states = () if use_cache else None
        
        # Process through serriform blocks
        for i, (block, past_state) in enumerate(zip(self.blocks, past_key_values)):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                
            # Pass memory state through the block if available
            hidden_states, memory_state = block(hidden_states, past_state)
            
            # Store new memory state
            if use_cache:
                new_memory_states += (memory_state,)
                
            # Update step counter occasionally for debugging
            if i == 0 and self.training and self._global_step % 100 == 0:
                print(f"Processing block {i+1}/{len(self.blocks)}, shape: {hidden_states.shape}")
                
        # Final normalization and projection
        hidden_states = self.norm(hidden_states)
        logits = self.output_proj(hidden_states)
        
        # Add final hidden state
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
            
        # Update global step counter
        if self.training:
            self._global_step += 1
            
        # Return based on configuration
        if return_dict:
            return {
                'logits': logits,
                'past_key_values': new_memory_states if use_cache else None,
                'hidden_states': all_hidden_states
            }
        else:
            return logits, new_memory_states if use_cache else None, all_hidden_states

    def generate(
        self, 
        prompt_ids: torch.Tensor, 
        max_new_tokens: int = 100, 
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
        use_cache: bool = True
    ):
        """
        Generate text using the model, with efficient KV caching
        """
        self.eval()
        
        batch_size = prompt_ids.shape[0]
        device = next(self.parameters()).device
        
        # Move input to device
        input_ids = prompt_ids.clone().to(device)
        
        # Use caching for efficient generation
        past_key_values = None
        generated_ids = []
        
        # Initialize token metadata for repetition penalty
        prev_tokens = []
        
        # First forward pass for the prompt
        with torch.no_grad():
            for token_idx in range(input_ids.shape[1] - 1):
                # Process one token at a time to build cache
                token = input_ids[:, token_idx:token_idx+1]
                outputs = self(token, past_key_values=past_key_values, use_cache=use_cache)
                
                # Update past key values
                past_key_values = outputs['past_key_values'] if use_cache else None
                
                # Track tokens for repetition penalty
                prev_tokens.append(input_ids[:, token_idx].tolist())
            
            # Generate new tokens one by one
            for _ in range(max_new_tokens):
                # Get last token
                next_token_idx = input_ids.shape[1] - 1
                current_token = input_ids[:, next_token_idx:next_token_idx+1]
                
                # Forward pass with cached key-values
                outputs = self(
                    current_token, 
                    past_key_values=past_key_values, 
                    use_cache=use_cache
                )
                
                # Extract logits and update cache
                next_token_logits = outputs['logits'][:, -1, :]
                past_key_values = outputs['past_key_values'] if use_cache else None
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply repetition penalty
                if repetition_penalty != 1.0 and len(prev_tokens) > 0:
                    for i in range(batch_size):
                        for token_id in set(prev_tokens):
                            if token_id[i] < next_token_logits.shape[-1]:
                                next_token_logits[i, token_id[i]] /= repetition_penalty
                
                # Sampling: Top-K followed by Top-P
                if do_sample:
                    # Apply top-k filtering
                    if top_k > 0:
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float('Inf'))
                    
                    # Apply top-p (nucleus) filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        
                        # Shift the indices to the right to keep also the first token above the threshold
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        # Scatter sorted tensors to original indexing
                        indices_to_remove = sorted_indices_to_remove.scatter(
                            -1, sorted_indices, sorted_indices_to_remove
                        )
                        next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float('Inf'))
                    
                    # Sample from the filtered distribution
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to generated sequence
                generated_ids.append(next_token)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Track tokens for repetition penalty
                prev_tokens.append(next_token.squeeze().tolist())
                
                # If we exceed max sequence length, remove first token
                if input_ids.shape[1] > self.max_seq_len:
                    input_ids = input_ids[:, 1:]
                    # Note: we maintain the cache for the kept tokens
        
        # Concatenate all generated tokens
        if len(generated_ids) > 0:
            generated_tokens = torch.cat(generated_ids, dim=1)
            return generated_tokens
        else:
            return torch.zeros((batch_size, 0), dtype=torch.long, device=device)

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

def train_epoch(
    model, 
    dataloader, 
    optimizer, 
    scheduler, 
    device, 
    log_interval=50, 
    grad_clip=1.0,
    use_wandb=False
):
    model.train()
    total_loss = 0
    total_tokens = 0
    start_time = time.time()
    
    for i, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        batch_size, seq_len = x.shape
        total_tokens += batch_size * seq_len
        
        # Forward pass with the updated model signature
        outputs = model(x, use_cache=False, return_dict=True)
        logits = outputs['logits']
        
        # Since logits might be shorter due to convolution, truncate y to match
        seq_len_out = logits.size(1)
        if seq_len_out != y.size(1):
            y = y[:, :seq_len_out]
        
        # Compute loss (cross entropy)
        loss = F.cross_entropy(logits.view(-1, model.vocab_size), y.view(-1))
        
        # Backward pass with gradient clipping
        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent explosion
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        # Update metrics
        total_loss += loss.item() * batch_size  # Weight by batch size
        
        # Logging
        if i % log_interval == 0:
            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]['lr']
            
            # Calculate tokens per second
            tokens_per_sec = total_tokens / (elapsed + 1e-8)
            
            # Build log message
            log_info = {
                'step': i,
                'loss': loss.item(),
                'lr': lr,
                'tokens_per_sec': tokens_per_sec,
                'time': elapsed
            }
            
            print(f"Batch {i}, Loss: {loss.item():.4f}, LR: {lr:.8f}, Tokens/s: {tokens_per_sec:.1f}, Time: {elapsed:.2f}s")
            
            # Log to wandb if enabled
            if use_wandb:
                try:
                    import wandb
                    wandb.log(log_info)
                except ImportError:
                    pass
                    
            # Reset counters
            total_tokens = 0
            start_time = time.time()
    
    # Return average loss over all batches
    return total_loss / len(dataloader.dataset)

def evaluate(
    model, 
    dataloader, 
    device, 
    max_eval_batches=None
):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            # Limit evaluation batches if specified
            if max_eval_batches is not None and i >= max_eval_batches:
                break
                
            x, y = x.to(device), y.to(device)
            batch_size = x.shape[0]
            
            # Forward pass
            outputs = model(x, use_cache=False, return_dict=True)
            logits = outputs['logits']
            
            # Truncate y to match output sequence length
            seq_len_out = logits.size(1)
            seq_len_y = y.size(1)
            
            if seq_len_out != seq_len_y:
                y = y[:, :seq_len_out]
            
            # Compute loss
            loss = F.cross_entropy(logits.view(-1, model.vocab_size), y.view(-1))
            
            # Update metrics (weighted by batch size)
            total_loss += loss.item() * batch_size
            total_tokens += batch_size * seq_len_out
    
    # Calculate per-token perplexity
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'total_tokens': total_tokens
    }

def main():
    parser = argparse.ArgumentParser(description='Train SerriformNet Language Model')
    parser.add_argument('--dim', type=int, default=512, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=12, help='Number of Serriform blocks')
    parser.add_argument('--max_seq_len', type=int, default=1024, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='Peak learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='LR warmup steps')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to dataset')
    parser.add_argument('--save_path', type=str, default='./checkpoints', help='Path to save checkpoints')
    parser.add_argument('--log_interval', type=int, default=10, help='Log interval')
    parser.add_argument('--save_interval', type=int, default=1000, help='Save interval')
    parser.add_argument('--eval_interval', type=int, default=500, help='Evaluation interval')
    parser.add_argument('--max_eval_batches', type=int, default=20, help='Maximum evaluation batches')
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples to use')
    parser.add_argument('--tokenizer', type=str, default='gpt2', help='Tokenizer to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='serriformnet', help='WandB project name')
    parser.add_argument('--wandb_name', type=str, default=None, help='WandB run name')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Initialize WandB if requested
    if args.use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                config=vars(args)
            )
            print("Initialized WandB logging")
        except ImportError:
            args.use_wandb = False
            print("WandB not installed. Running without logging.")
    
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
                self.dataset = [None] * (n_batches * batch_size)  # For len(dataset)
                
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
        max_seq_len=args.max_seq_len,
        dropout=args.dropout
    ).to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SerriformNet initialized with {total_params:,} parameters ({trainable_params:,} trainable)")
    
    # Set up optimizer with weight decay on non-bias/norm parameters
    no_decay = ["bias", "norm", "embedding"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() 
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    
    # Calculate total training steps
    total_steps = len(train_dataloader) * args.epochs // args.gradient_accumulation_steps
    
    # Create learning rate scheduler with warmup and cosine decay
    def lr_lambda(current_step):
        if current_step < args.warmup_steps:
            # Linear warmup
            return current_step / max(1, args.warmup_steps)
        else:
            # Cosine decay from args.lr to args.min_lr
            progress = (current_step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
            return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Log configuration
    config_info = {
        "model": {
            "type": "SerriformNet",
            "dim": args.dim,
            "layers": args.num_layers,
            "params": total_params,
            "vocab_size": vocab_size,
            "max_seq_len": args.max_seq_len
        },
        "training": {
            "batch_size": args.batch_size,
            "grad_accum": args.gradient_accumulation_steps,
            "effective_batch": args.batch_size * args.gradient_accumulation_steps,
            "epochs": args.epochs,
            "lr": args.lr,
            "min_lr": args.min_lr,
            "total_steps": total_steps,
            "warmup_steps": args.warmup_steps,
            "weight_decay": args.weight_decay
        },
        "system": {
            "device": args.device,
            "seed": args.seed
        }
    }
    
    print("Configuration:")
    import json
    print(json.dumps(config_info, indent=2))
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs ({total_steps} steps)...")
    
    global_step = 0
    best_val_loss = float('inf')
    step_time_avg = 0
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        # Training
        train_loss = train_epoch(
            model, 
            train_dataloader, 
            optimizer, 
            scheduler, 
            device,
            log_interval=args.log_interval,
            grad_clip=args.grad_clip,
            use_wandb=args.use_wandb
        )
        
        # Epoch complete
        epoch_time = time.time() - epoch_start_time
        
        # Validation
        val_metrics = evaluate(
            model, 
            val_dataloader, 
            device,
            max_eval_batches=args.max_eval_batches
        )
        
        val_loss = val_metrics['loss']
        val_ppl = val_metrics['perplexity']
        
        # Log results
        print(f"Epoch {epoch+1}/{args.epochs} complete in {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}")
        
        if args.use_wandb:
            try:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                    "epoch_time": epoch_time
                })
            except:
                pass
        
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
                'val_ppl': val_ppl,
                'args': vars(args)
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path} (val_loss: {val_loss:.4f}, ppl: {val_ppl:.2f})")
        
        # Regular epoch checkpoint
        checkpoint_path = os.path.join(args.save_path, f"serriformnet_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
            'val_ppl': val_ppl,
            'args': vars(args)
        }, checkpoint_path)
    
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
                    max_new_tokens=100,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    use_cache=True
                )
            
            generated_text = tokenizer.decode(output_ids[0])
            print(f"Prompt: {prompt}")
            print(f"Generated: {generated_text}")
            
            # Log the generation
            if args.use_wandb:
                try:
                    wandb.log({
                        "sample_generation": wandb.Html(f"<p><strong>Prompt:</strong> {prompt}</p><p><strong>Generated:</strong> {generated_text}</p>")
                    })
                except:
                    pass
    except Exception as e:
        print(f"Could not generate sample text: {e}")
        
    # Clean up wandb
    if args.use_wandb:
        try:
            wandb.finish()
        except:
            pass

if __name__ == "__main__":
    main()
