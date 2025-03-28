import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
from typing import Optional, Tuple, List
import time
import os
import sys
import random
import contextlib

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
        
        # Cache for sin/cos values
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)
        self.max_seq_len_cached = 0
        
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
        
        # Use cached sin/cos if available and sequence length is within cached range
        if self.cos_cached is not None and seq_len <= self.max_seq_len_cached:
            cos_emb = self.cos_cached[:seq_len].unsqueeze(0)
            sin_emb = self.sin_cached[:seq_len].unsqueeze(0)
        else:
            # Generate position embeddings
            seq_idx = torch.arange(seq_len, device=x.device)
            # [seq_len, half_dim]
            freqs = torch.outer(seq_idx, self.inv_freq)
            
            # Calculate cos and sin embeddings
            # [seq_len, half_dim] -> [seq_len, half_dim*2]
            emb = torch.cat([freqs, freqs], dim=-1)
            
            # Cache values for future use
            self.cos_cached = torch.cos(emb)
            self.sin_cached = torch.sin(emb)
            self.max_seq_len_cached = seq_len
            
            # Reshape for batched broadcasting [1, seq_len, dim]
            cos_emb = self.cos_cached.unsqueeze(0)
            sin_emb = self.sin_cached.unsqueeze(0)
        
        # Apply position embeddings
        # x_rot = x * cos + self._rotate_half(x) * sin
        return x * cos_emb + self._rotate_half(x) * sin_emb

# Improved parallelized recurrence mechanism (RWKV/Mamba-inspired)
class StructuredStateRecurrence(nn.Module):
    def __init__(self, dim: int, memory_dim: int = None, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.memory_dim = memory_dim or dim // 4
        
        # Memory state projection
        self.memory_proj_in = nn.Linear(dim, self.memory_dim)
        self.memory_proj_out = nn.Linear(self.memory_dim, dim)
        
        # Parallel state update parameters
        self.time_decay = nn.Parameter(torch.randn(self.memory_dim) * 0.01)
        self.time_first = nn.Parameter(torch.randn(self.memory_dim) * 0.01)
        
        # Input projection
        self.key_proj = nn.Linear(dim, self.memory_dim)
        self.value_proj = nn.Linear(dim, self.memory_dim)
        self.output_gate = nn.Linear(dim + self.memory_dim, dim)
        
        self.act = nn.SiLU()  # SiLU (Swish) activation for smoother gradients
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        # Initialize all linear layers
        for module in [self.memory_proj_in, self.memory_proj_out, self.key_proj, self.value_proj, self.output_gate]:
            nn.init.xavier_normal_(module.weight, gain=1.0)
            nn.init.zeros_(module.bias)
        
        # Initialize time parameters with a small value 
        nn.init.normal_(self.time_decay, mean=0.0, std=0.01)
        nn.init.normal_(self.time_first, mean=0.0, std=0.01)
    
    def forward(self, x, memory_state=None):
        batch_size, seq_len, dim = x.shape
        
        # Initialize memory state if not provided
        if memory_state is None:
            memory_state = torch.zeros(batch_size, self.memory_dim, device=x.device)
        
        # Project input to key and value
        k = self.key_proj(x)  # [batch, seq_len, memory_dim]
        v = self.value_proj(x)  # [batch, seq_len, memory_dim]
        
        # Prepare time decay factors - scale for numerical stability
        # and transform to ensure time_mix is between 0 and 1
        time_decay = torch.sigmoid(self.time_decay) * 0.9 + 0.1  # [memory_dim]
        time_first = torch.sigmoid(self.time_first)  # [memory_dim]
        
        # Initialize output and state tensors
        out = torch.zeros((batch_size, seq_len, self.memory_dim), device=x.device)
        next_memory_state = memory_state.clone()
        
        # FIX: Create a scalar decay matrix first
        scalar_decay_mask = torch.zeros((seq_len, seq_len), device=x.device)
        for i in range(seq_len):
            for j in range(i+1):
                # Use a scalar placeholder value for the exponent
                scalar_decay_mask[i, j] = i - j
                
        # FIX: Apply the time_decay vector to create the full decay mask
        # Convert scalar_decay_mask to exponents that will be applied to time_decay
        # Reshape scalar_decay_mask to [1, seq_len, seq_len, 1] for broadcasting
        scalar_decay_mask = scalar_decay_mask.unsqueeze(0).unsqueeze(-1)
        
        # Reshape time_decay to [1, 1, 1, memory_dim] for broadcasting
        time_decay_expanded = time_decay.view(1, 1, 1, -1)
        
        # Compute the decay mask by using broadcasting: time_decay_expanded ^ scalar_decay_mask
        # This computes the decay for each position and memory dimension
        decay_mask = time_decay_expanded ** scalar_decay_mask
                
        # Compute weighted sums of past values for each position
        # For each position t, we weight the values at positions 0..t by the appropriate decay
        v_expanded = v.unsqueeze(1)  # [batch, 1, seq_len, memory_dim]
        k_expanded = k.unsqueeze(1)  # [batch, 1, seq_len, memory_dim]
        
        # Causally mask and apply decay
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(-1)  # [1, seq_len, seq_len, 1]
        
        # k_expanded at each position t: contains keys up to position t
        # Weighted values: each position t has weighted sum of values up to position t
        weighted_values = (v_expanded * causal_mask * decay_mask).sum(dim=2)  # [batch, seq_len, memory_dim]
        
        # Initial contribution from memory state
        if memory_state is not None:
            # Apply memory state effect to first time step with time_first weighting
            memory_contrib = memory_state.unsqueeze(1) * time_first
            weighted_values = weighted_values + memory_contrib
        
        # Update memory state to the last sequence position's state
        if seq_len > 0:
            # Update memory state based on the last position's computation
            next_memory_state = weighted_values[:, -1]
        
        # Apply dropout for regularization
        weighted_values = self.dropout(weighted_values)
        
        # Compute output using gate and updated memory
        x_reshaped = x.reshape(-1, dim)
        memory_reshaped = weighted_values.reshape(-1, self.memory_dim)
        combined = torch.cat([x_reshaped, memory_reshaped], dim=1)
        output = x_reshaped + self.output_gate(combined)
        output = output.reshape(batch_size, seq_len, dim)
        
        return output, next_memory_state

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

# Enhanced Gated Fusion with top-k sparse routing
class EnhancedGatedFusion(nn.Module):
    def __init__(self, dim: int, num_experts: int = 4, top_k: int = 2, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)  # Ensure top_k doesn't exceed num_experts
        
        # Router network to assign weights to experts (no activation - will apply softmax per-token)
        self.router = nn.Linear(dim, num_experts)
        
        # Expert networks (simple transformations)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.SiLU(),
                nn.Dropout(dropout)
            ) for _ in range(num_experts)
        ])
        
        # Final projection (shared across all experts)
        self.output_proj = nn.Linear(dim, dim)
        
        # Final layer norm for stabilization
        self.norm = RMSNorm(dim)
        
    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        
        # Get routing logits
        routing_logits = self.router(x)  # [batch, seq_len, num_experts]
        
        # Compute sparse gating: only keep top-k experts per token
        # First, identify the top-k experts per token
        topk_gate_logits, topk_indices = torch.topk(
            routing_logits, self.top_k, dim=-1
        )  # Both: [batch, seq_len, top_k]
        
        # Normalize the top-k expert weights with softmax
        topk_routing_weights = F.softmax(topk_gate_logits, dim=-1)
        
        # Process input through experts and combine with routing weights
        combined_output = torch.zeros_like(x)
        
        # Create a flattened view for more efficient processing
        x_flat = x.reshape(-1, dim)  # [batch*seq_len, dim]
        
        # For each expert in top-k selection
        for k in range(self.top_k):
            # Get the expert index and weight for each token
            expert_idx = topk_indices[:, :, k]  # [batch, seq_len]
            expert_weight = topk_routing_weights[:, :, k]  # [batch, seq_len]
            
            # Reshape for broadcasting with expert output
            expert_weight = expert_weight.reshape(-1, 1)  # [batch*seq_len, 1]
            
            # Process each batch of tokens with the indices of the k-th expert
            # This is more efficient than looping through all experts
            batched_output = torch.zeros_like(x_flat)
            
            # Group tokens by which expert they route to
            for expert_id in range(self.num_experts):
                # Find tokens that route to this expert
                mask = (expert_idx.reshape(-1) == expert_id)
                if not mask.any():
                    continue
                    
                # Select those tokens
                expert_input = x_flat[mask]
                
                # Apply expert to these tokens
                expert_output = self.experts[expert_id](expert_input)
                
                # Assign back to the right positions
                batched_output[mask] = expert_output
            
            # Weight by the expert's routing weight and add to combined output
            combined_output += (batched_output * expert_weight).reshape(batch_size, seq_len, dim)
        
        # Final projection
        output = self.output_proj(combined_output)
        
        # Residual connection and normalization
        return self.norm(x + output)

# Efficient Depthwise-Pointwise Fused Convolution Implementation
class FusedDWPWConv(nn.Module):
    def __init__(self, dim: int, kernel_size: int, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # Calculate padding for causal convolution
        self.padding = (kernel_size - 1) * dilation
        
        # Depthwise convolution
        self.depth_conv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, 
            padding=self.padding, padding_mode='zeros',
            groups=dim, dilation=dilation
        )
        
        # Pointwise convolution fused with activation and dropout
        self.point_conv = nn.Conv1d(dim, dim, kernel_size=1)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: [batch, seq_len, dim]
        # Reshape for convolution [B, L, D] -> [B, D, L]
        B, L, D = x.shape
        x_conv = x.transpose(1, 2)
        
        # Apply depthwise convolution
        x_conv = self.depth_conv(x_conv)
        
        # Ensure causality by trimming to original length
        x_conv = x_conv[..., :L]
        
        # Apply pointwise convolution with activation
        x_conv = self.act(self.point_conv(x_conv))
        
        # Apply dropout
        x_conv = self.dropout(x_conv)
        
        # Reshape back to [B, L, D]
        return x_conv.transpose(1, 2)

# Efficient Serriform Block with fused convolutions
class SerriformBlock(nn.Module):
    def __init__(self, dim: int, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.dilation = dilation
        self.kernel_size = 5
        self.dropout = dropout
        
        # Use fused depthwise-pointwise convolution for efficiency
        self.conv = FusedDWPWConv(dim, self.kernel_size, dilation, dropout)
        
        # Parallelized recurrence
        self.recurrence = StructuredStateRecurrence(dim, dropout=dropout)
        
        # Enhanced gated fusion with top-k routing
        self.fusion = EnhancedGatedFusion(dim, dropout=dropout)
        
        # Low-rank feed-forward network
        self.ff = LowRankFF(dim, dropout=dropout)
        
        # Optional residual scaling factor (initialized close to 1)
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.9)
        
        # For activation logging
        self.register_forward_hook(self._log_activations)
        self._log_interval = 1000
        self._call_count = 0
        self._activation_stats = {}
        
    def _log_activations(self, module, input, output):
        """Log activation statistics during training"""
        if not module.training:
            return
            
        self._call_count += 1
        if self._call_count % self._log_interval != 0:
            return
            
        # Compute statistics on the output hidden states
        hidden_states, _ = output
        with torch.no_grad():
            self._activation_stats = {
                'mean': hidden_states.mean().item(),
                'std': hidden_states.std().item(),
                'min': hidden_states.min().item(),
                'max': hidden_states.max().item(),
                'norm': hidden_states.norm().item(),
                'dilation': self.dilation,
                'call_count': self._call_count
            }
            print(f"Block stats (dilation={self.dilation}): {self._activation_stats}")
        
    def forward(self, x, memory_state=None):
        # Apply normalization first (pre-activation)
        residual = x
        x = self.norm(x)
        
        # Apply fused convolution operation
        x_conv = self.conv(x)
        
        # Apply the parallelized recurrence
        x_rec, new_memory_state = self.recurrence(x, memory_state)
        
        # Combine conv and recurrence pathways
        x = x_conv + x_rec
        x = self.fusion(x)
        
        # Apply feed-forward network
        x = x + self.ff(x)
        
        # Add residual connection with optional scaling
        x = self.residual_scale * residual + x
        
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
        use_cache: bool = True,
        dilations: List[int] = None,
        gradient_checkpointing: bool = False
    ):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.use_cache = use_cache
        self.gradient_checkpointing = gradient_checkpointing
        
        # Input embedding layer
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.rope = RotaryPositionalEmbedding(dim)
        self.embed_dropout = nn.Dropout(dropout)
        
        # Serriform blocks with varying dilations to create hierarchical receptive field
        self.blocks = nn.ModuleList()
        
        # Use custom dilations if provided, otherwise use logarithmic spacing
        if dilations is None:
            # Logarithmic spacing for better long-range modeling
            log_factor = np.log(8) / num_layers  # Targeting max dilation ~8
            dilations = [max(1, int(np.exp(i * log_factor))) for i in range(num_layers)]
        
        for dilation in dilations:
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
    
    def _gradient_checkpointing_func(self, module, *args, **kwargs):
        """Custom function for gradient checkpointing to save memory"""
        return module(*args, **kwargs)
    
    def get_input_embeddings(self):
        """Get input embedding layer for compatibility with HF-style code"""
        return self.token_embedding
        
    def forward(
        self, 
        x, 
        past_memory_states=None, 
        use_cache=None,
        output_hidden_states=False,
        return_dict=True
    ):
        """
        Forward pass supporting cached memory states for efficient generation
        
        Args:
            x: Input tensor of token IDs [batch_size, seq_len]
            past_memory_states: Cached memory states from previous forward passes
            use_cache: Whether to use caching
            output_hidden_states: Whether to return hidden states from all layers
            return_dict: Whether to return a dictionary or a tuple
        """
        use_cache = use_cache if use_cache is not None else self.use_cache
        
        # For training without caching
        if past_memory_states is None:
            past_memory_states = [None] * len(self.blocks)
        
        # Get token embeddings
        token_emb = self.token_embedding(x)
        
        # Apply RoPE
        hidden_states = self.rope(token_emb)
        hidden_states = self.embed_dropout(hidden_states)
        
        # Track all states if needed
        all_hidden_states = () if output_hidden_states else None
        new_memory_states = () if use_cache else None
        
        # Process through serriform blocks
        for i, (block, past_state) in enumerate(zip(self.blocks, past_memory_states)):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            # Apply gradient checkpointing if enabled (training only)
            if self.gradient_checkpointing and self.training:
                block_output = torch.utils.checkpoint.checkpoint(
                    self._gradient_checkpointing_func,
                    block,
                    hidden_states,
                    past_state
                )
                hidden_states, memory_state = block_output
            else:
                # Standard forward pass
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
                'past_memory_states': new_memory_states if use_cache else None,
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
        Generate text using the model, with efficient memory state caching
        """
        self.eval()
        
        batch_size = prompt_ids.shape[0]
        device = next(self.parameters()).device
        
        # Move input to device
        input_ids = prompt_ids.clone().to(device)
        
        # Use caching for efficient generation
        past_memory_states = None
        generated_ids = []
        
        # Tracks generated tokens for repetition penalty
        prev_tokens_set = set() if repetition_penalty > 1.0 else None
        
        # First forward pass with the full prompt to build cache
        with torch.no_grad():
            # Process the prompt without last token to build cache
            if input_ids.size(1) > 1:
                outputs = self(input_ids[:, :-1], past_memory_states=None, use_cache=use_cache)
                past_memory_states = outputs['past_memory_states'] if use_cache else None
                
                # Track tokens for repetition penalty
                if prev_tokens_set is not None:
                    for token_id in input_ids[:, :-1].reshape(-1).tolist():
                        prev_tokens_set.add(token_id)
            
            # Now generate new tokens one by one
            current_token = input_ids[:, -1:] if input_ids.size(1) > 0 else input_ids
            
            for _ in range(max_new_tokens):
                # Forward pass with cached memory states
                outputs = self(
                    current_token, 
                    past_memory_states=past_memory_states, 
                    use_cache=use_cache
                )
                
                # Extract logits and update cache
                next_token_logits = outputs['logits'][:, -1, :]
                past_memory_states = outputs['past_memory_states'] if use_cache else None
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply repetition penalty - vectorized approach
                if repetition_penalty != 1.0 and prev_tokens_set:
                    # Create tensor from set of previous tokens
                    penalty_tensor = torch.tensor(list(prev_tokens_set), device=device)
                    
                    # Use advanced indexing to apply penalty to all previous tokens at once
                    next_token_logits.index_fill_(
                        dim=-1,
                        index=penalty_tensor,
                        value=-repetition_penalty
                    )
                
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
                
                # For next iteration
                current_token = next_token
                
                # Track token for repetition penalty
                if prev_tokens_set is not None:
                    prev_tokens_set.add(next_token.item())
                
                # If we exceed max sequence length, adjust cache accordingly
                if len(generated_ids) + input_ids.shape[1] > self.max_seq_len:
                    # Note: in a real implementation, we would need to shift the memory states
                    # to "forget" the oldest tokens, but for simplicity we just continue
                    pass
        
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
    use_wandb=False,
    fp16=False,
    log_memory=False
):
    """Train for one epoch with improved efficiency and monitoring."""
    model.train()
    total_loss = 0
    total_tokens = 0
    start_time = time.time()
    
    # Setup for mixed precision training if requested
    scaler = torch.cuda.amp.GradScaler() if fp16 else None
    
    for i, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        batch_size, seq_len = x.shape
        total_tokens += batch_size * seq_len
        
        # Mixed precision context if enabled
        with torch.cuda.amp.autocast() if fp16 else contextlib.nullcontext():
            # Forward pass with the updated model signature
            outputs = model(
                x, 
                use_cache=False, 
                return_dict=True
            )
            logits = outputs['logits']
            
            # Since logits might be shorter due to convolution, truncate y to match
            seq_len_out = logits.size(1)
            if seq_len_out != y.size(1):
                y = y[:, :seq_len_out]
            
            # Compute loss (cross entropy) with optional label smoothing
            loss = F.cross_entropy(logits.view(-1, model.vocab_size), y.view(-1))
        
        # Backward pass with gradient clipping
        optimizer.zero_grad()
        
        if fp16:
            # Mixed precision backward
            scaler.scale(loss).backward()
            
            # Clip gradients (respecting scaling)
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
            # Step with scaler
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard backward
            loss.backward()
            
            # Clip gradients
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
            
            # Add memory stats if requested
            if log_memory and torch.cuda.is_available():
                log_info.update({
                    'gpu_memory_allocated': torch.cuda.memory_allocated() / 1024**2,
                    'gpu_memory_reserved': torch.cuda.memory_reserved() / 1024**2
                })
            
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
    max_eval_batches=None,
    fp16=False,
    log_layer_metrics=False
):
    """Evaluate model with improved metrics and efficiency."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0
    layer_metrics = {}
    
    # Store layer metrics if requested
    if log_layer_metrics:
        # Initialize metrics for each layer
        for i, block in enumerate(model.blocks):
            layer_metrics[f'layer_{i}'] = []
    
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            # Limit evaluation batches if specified
            if max_eval_batches is not None and i >= max_eval_batches:
                break
                
            x, y = x.to(device), y.to(device)
            batch_size = x.shape[0]
            
            # Mixed precision context if enabled
            with torch.cuda.amp.autocast() if fp16 else contextlib.nullcontext():
                # Forward pass with output_hidden_states if logging layer metrics
                outputs = model(
                    x, 
                    use_cache=False, 
                    return_dict=True,
                    output_hidden_states=log_layer_metrics
                )
                logits = outputs['logits']
                
                # Log layer-wise metrics if requested
                if log_layer_metrics and outputs['hidden_states'] is not None:
                    for layer_idx, hidden_state in enumerate(outputs['hidden_states']):
                        # Calculate and store layer stats
                        layer_metrics[f'layer_{layer_idx}'].append({
                            'mean': hidden_state.mean().item(),
                            'std': hidden_state.std().item(),
                            'norm': hidden_state.norm().item(),
                            'batch': i
                        })
                
                # Truncate y to match output sequence length
                seq_len_out = logits.size(1)
                seq_len_y = y.size(1)
                
                if seq_len_out != seq_len_y:
                    y = y[:, :seq_len_out]
                
                # Compute loss
                loss = F.cross_entropy(logits.view(-1, model.vocab_size), y.view(-1))
                
                # Calculate accuracy metrics
                preds = logits.argmax(dim=-1)
                correct = (preds == y).float().sum().item()
                
            # Update metrics (weighted by batch size)
            total_loss += loss.item() * batch_size
            total_tokens += batch_size * seq_len_out
            total_correct += correct
    
    # Calculate per-token perplexity and accuracy
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    accuracy = total_correct / total_tokens
    
    result = {
        'loss': avg_loss,
        'perplexity': perplexity,
        'accuracy': accuracy,
        'total_tokens': total_tokens
    }
    
    # Add layer metrics if collected
    if log_layer_metrics:
        for layer, metrics in layer_metrics.items():
            if metrics:
                # Average metrics across batches
                result[f'{layer}_mean'] = np.mean([m['mean'] for m in metrics])
                result[f'{layer}_std'] = np.mean([m['std'] for m in metrics])
                result[f'{layer}_norm'] = np.mean([m['norm'] for m in metrics])
    
    return result

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
    
    # New arguments for improved functionality
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision training')
    parser.add_argument('--gradient_checkpointing', action='store_true', help='Use gradient checkpointing to save memory')
    parser.add_argument('--log_memory', action='store_true', help='Log GPU memory usage during training')
    parser.add_argument('--log_layer_metrics', action='store_true', help='Log per-layer metrics during evaluation')
    parser.add_argument('--dilations', type=str, default=None, help='Comma-separated list of dilations for blocks')
    parser.add_argument('--eval_generate', action='store_true', help='Generate samples during evaluation')
    parser.add_argument('--eval_generate_tokens', type=int, default=100, help='Number of tokens to generate during eval')
    
    args = parser.parse_args()
    
    # Parse dilations if provided
    custom_dilations = None
    if args.dilations:
        try:
            custom_dilations = [int(d) for d in args.dilations.split(',')]
            # Ensure we have enough dilations for all layers
            if len(custom_dilations) < args.num_layers:
                custom_dilations = custom_dilations + [custom_dilations[-1]] * (args.num_layers - len(custom_dilations))
            # Truncate if we have too many
            custom_dilations = custom_dilations[:args.num_layers]
            print(f"Using custom dilations: {custom_dilations}")
        except:
            print("Error parsing dilations, using default logarithmic spacing.")
            custom_dilations = None
    
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
    
    # Initialize model with new parameters
    print(f"Initializing SerriformNet with {args.num_layers} layers, dim={args.dim}...")
    model = SerriformNet(
        vocab_size=vocab_size,
        dim=args.dim,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        gradient_checkpointing=args.gradient_checkpointing,
        dilations=custom_dilations
    ).to(device)
    
    # Check if model can run in half precision
    if args.fp16 and not torch.cuda.is_available():
        print("Warning: FP16 training requested but CUDA not available. Falling back to full precision.")
        args.fp16 = False
    elif args.fp16:
        # Test model with a small batch in fp16
        try:
            dummy_input = torch.randint(0, vocab_size, (2, 10)).to(device)
            with torch.cuda.amp.autocast():
                _ = model(dummy_input)
            print("Model successfully tested with mixed precision.")
        except Exception as e:
            print(f"Warning: Model failed to run in mixed precision: {e}. Falling back to full precision.")
            args.fp16 = False
    
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
            use_wandb=args.use_wandb,
            fp16=args.fp16,
            log_memory=args.log_memory
        )
        
        # Epoch complete
        epoch_time = time.time() - epoch_start_time
        
        # Validation
        val_metrics = evaluate(
            model, 
            val_dataloader, 
            device,
            max_eval_batches=args.max_eval_batches,
            fp16=args.fp16,
            log_layer_metrics=args.log_layer_metrics
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
    # Import optional modules that may be needed
    import contextlib  # For nullcontext
    try:
        import torchinfo
        # Use torchinfo to print model summary if available
        TORCHINFO_AVAILABLE = True
    except ImportError:
        TORCHINFO_AVAILABLE = False
    
    main()
