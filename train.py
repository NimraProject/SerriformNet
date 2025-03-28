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
    def __init__(self, dim: int, memory_dim: int = None, dropout: float = 0.1, max_seq_len: int = 1024):
        super().__init__()
        self.dim = dim
        self.memory_dim = memory_dim or dim // 4
        self.max_seq_len = max_seq_len
        
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
        
        # Cache for precomputed decay matrices
        self.register_buffer('scalar_decay_masks', None)
        self.register_buffer('causal_masks', None)
        
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
        
    def _precompute_masks(self, seq_len, device):
        """Precompute and cache scalar decay masks and causal masks."""
        # Check if we already have a cached version of sufficient length
        if (self.scalar_decay_masks is not None and 
            self.causal_masks is not None and 
            self.scalar_decay_masks.size(1) >= seq_len):
            # Return existing cached tensors, sliced to current seq_len
            return (
                self.scalar_decay_masks[:, :seq_len, :seq_len, :],
                self.causal_masks[:, :seq_len, :seq_len, :]
            )
        
        # Create full-sized masks for the maximum sequence length
        # We'll create these at max_seq_len size to avoid frequent recomputation
        max_len = min(self.max_seq_len, seq_len * 2)  # At least double the current seq_len
        
        # Create scalar decay mask - stores exponent values (i-j)
        scalar_decay_mask = torch.zeros((max_len, max_len), device=device)
        for i in range(max_len):
            for j in range(i+1):
                scalar_decay_mask[i, j] = i - j
        
        # Create causal mask (lower triangular matrix)
        causal_mask = torch.tril(torch.ones((max_len, max_len), device=device))
        
        # Add batch and memory dimensions for broadcasting
        scalar_decay_mask = scalar_decay_mask.unsqueeze(0).unsqueeze(-1)  # [1, max_len, max_len, 1]
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(-1)  # [1, max_len, max_len, 1]
        
        # Cache for future use
        self.register_buffer('scalar_decay_masks', scalar_decay_mask)
        self.register_buffer('causal_masks', causal_mask)
        
        # Return current needed slice
        return (
            scalar_decay_mask[:, :seq_len, :seq_len, :],
            causal_mask[:, :seq_len, :seq_len, :]
        )
    
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
        
        # Get precomputed masks for current sequence length
        scalar_decay_mask, causal_mask = self._precompute_masks(seq_len, x.device)
        
        # Reshape time_decay to [1, 1, 1, memory_dim] for broadcasting
        time_decay_expanded = time_decay.view(1, 1, 1, -1)
        
        # Compute the decay mask by using broadcasting: time_decay_expanded ^ scalar_decay_mask
        # This computes the decay for each position and memory dimension
        decay_mask = time_decay_expanded ** scalar_decay_mask
                
        # Compute weighted sums of past values for each position
        # For each position t, we weight the values at positions 0..t by the appropriate decay
        v_expanded = v.unsqueeze(1)  # [batch, 1, seq_len, memory_dim]
        
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
    def __init__(
        self, 
        dim: int, 
        num_experts: int = 4, 
        top_k: int = 2, 
        dropout: float = 0.1,
        router_noise: float = 0.01,
        use_load_balancing: bool = True,
        load_balancing_weight: float = 0.01
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)  # Ensure top_k doesn't exceed num_experts
        self.router_noise = router_noise
        self.use_load_balancing = use_load_balancing
        self.load_balancing_weight = load_balancing_weight
        
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
        
        # Track expert usage for monitoring
        self.register_buffer('expert_usage_count', torch.zeros(num_experts))
        self.register_buffer('training_steps', torch.tensor(0, dtype=torch.long))
        
    def _compute_load_balancing_loss(self, routing_weights, expert_counts):
        """
        Compute the GShard-style load balancing auxiliary loss.
        The loss has two components:
        1. Importance loss: ensures all experts are used equally
        2. Load loss: ensures each expert processes similar number of tokens
        """
        # Calculate the fraction of tokens routed to each expert
        # expert_counts: [num_experts]
        # routing_weights: [batch, seq_len, num_experts]
        
        # Flatten batch and sequence dimensions
        flat_weights = routing_weights.reshape(-1, self.num_experts)  # [batch*seq_len, num_experts]
        
        # Calculate the fraction of tokens assigned to each expert
        # Mean of the weights for each expert across all tokens
        router_prob = flat_weights.mean(0)  # [num_experts]
        
        # Calculate the fraction of the router probability for each expert
        # This represents the "importance" of each expert
        router_prob_per_expert = router_prob / router_prob.sum()
        
        # Target: uniform distribution over all experts 
        # Each expert should receive equal weight (1/num_experts)
        target_prob = torch.ones_like(router_prob_per_expert) / self.num_experts
        
        # Importance loss: difference from uniform distribution
        # Use mean squared error to measure this difference
        importance_loss = torch.mean(torch.square(router_prob_per_expert - target_prob))
        
        # Load loss: variance in the number of tokens processed by each expert
        # Higher variance means less balanced load
        if expert_counts.sum() > 0:
            # Normalize counts to fraction of total
            load_fraction = expert_counts.float() / expert_counts.sum()
            # Calculate load variance
            load_variance = torch.mean(torch.square(load_fraction - target_prob))
        else:
            load_variance = torch.tensor(0.0, device=routing_weights.device)
        
        # Combine losses with the same weight (can be adjusted)
        aux_loss = importance_loss + load_variance
        return aux_loss * self.load_balancing_weight
    
    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        
        # Get routing logits
        routing_logits = self.router(x)  # [batch, seq_len, num_experts]
        
        # Add noise to routing logits during training for better exploration
        if self.training and self.router_noise > 0:
            routing_noise = torch.randn_like(routing_logits) * self.router_noise
            routing_logits = routing_logits + routing_noise
        
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
        
        # Keep track of which experts were used for load balancing
        expert_counts = torch.zeros(self.num_experts, device=x.device)
        
        # Full routing weight tensor for load balancing loss
        full_routing_weights = torch.zeros(batch_size, seq_len, self.num_experts, device=x.device)
        
        # For each expert in top-k selection
        for k in range(self.top_k):
            # Get the expert index and weight for each token
            expert_idx = topk_indices[:, :, k]  # [batch, seq_len]
            expert_weight = topk_routing_weights[:, :, k]  # [batch, seq_len]
            
            # Update full routing weights for load balancing loss
            for i in range(self.num_experts):
                mask = (expert_idx == i)
                full_routing_weights[:, :, i] += mask.float() * expert_weight
            
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
                    
                # Count tokens routed to this expert
                expert_counts[expert_id] += mask.sum().item()
                    
                # Select those tokens
                expert_input = x_flat[mask]
                
                # Apply expert to these tokens
                expert_output = self.experts[expert_id](expert_input)
                
                # Assign back to the right positions
                batched_output[mask] = expert_output
            
            # Weight by the expert's routing weight and add to combined output
            combined_output += (batched_output * expert_weight).reshape(batch_size, seq_len, dim)
        
        # Update expert usage statistics during training
        if self.training:
            self.expert_usage_count += expert_counts.detach()
            self.training_steps += 1
        
        # Final projection
        output = self.output_proj(combined_output)
        
        # Residual connection and normalization
        output = self.norm(x + output)
        
        # Compute load balancing auxiliary loss if needed
        if self.training and self.use_load_balancing:
            aux_loss = self._compute_load_balancing_loss(full_routing_weights, expert_counts)
            # Store the loss where it can be accessed
            self._aux_loss = aux_loss
        else:
            self._aux_loss = None
        
        return output
        
    def get_aux_loss(self):
        """Return the most recently computed auxiliary loss, if any."""
        return self._aux_loss
        
    def get_expert_usage(self):
        """Return the average expert usage distribution."""
        if self.training_steps > 0:
            return self.expert_usage_count / self.training_steps
        return self.expert_usage_count

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

# Core block of the SerriformNet architecture
class SerriformBlock(nn.Module):
    def __init__(
        self, 
        dim: int, 
        kernel_size: int = 3,
        dilation: int = 1, 
        memory_dim: int = None,
        dropout: float = 0.1,
        num_experts: int = 4,
        top_k: int = 2,
        ff_reduction_factor: int = 4,
        use_conv: bool = True,
        use_recurrence: bool = True,
        use_moe: bool = True,
        use_lowrank_ff: bool = True,
        max_seq_len: int = 1024,
        router_noise: float = 0.01,
        use_load_balancing: bool = True
    ):
        """
        Serriform Block: The core building block of SerriformNet
        
        Args:
            dim: Hidden dimension
            kernel_size: Kernel size for the convolutional layer
            dilation: Dilation factor for convolution
            memory_dim: Dimension of memory state for recurrence (default: dim // 4)
            dropout: Dropout rate
            num_experts: Number of experts in MoE
            top_k: Number of experts to route to in MoE
            ff_reduction_factor: Reduction factor for low-rank FF
            use_conv: Whether to use convolution
            use_recurrence: Whether to use recurrence
            use_moe: Whether to use MoE
            use_lowrank_ff: Whether to use low-rank FF
            max_seq_len: Maximum sequence length
            router_noise: Noise for MoE router
            use_load_balancing: Whether to use load balancing in MoE
        """
        super().__init__()
        self.dim = dim
        self.use_conv = use_conv
        self.use_recurrence = use_recurrence
        self.use_moe = use_moe
        self.use_lowrank_ff = use_lowrank_ff
        
        # Track aux losses for MoE components
        self.aux_losses = {}
        
        # Input normalization for each component (pre-activation norm)
        self.norm_conv = RMSNorm(dim) if use_conv else None
        self.norm_recurrence = RMSNorm(dim) if use_recurrence else None
        self.norm_moe = RMSNorm(dim) if use_moe else None
        self.norm_ff = RMSNorm(dim) if use_lowrank_ff else None
        
        # 1. Convolutional module (periodic kernel with dilation)
        self.conv = FusedDWPWConv(
            dim=dim, 
            kernel_size=kernel_size, 
            dilation=dilation, 
            dropout=dropout
        ) if use_conv else None
            
        # 2. Recurrence module (RWKV/SSM-style)
        self.recurrence = StructuredStateRecurrence(
            dim=dim,
            memory_dim=memory_dim,
            dropout=dropout,
            max_seq_len=max_seq_len
        ) if use_recurrence else None
            
        # 3. Routing module (Mixture of Experts)
        self.moe = EnhancedGatedFusion(
            dim=dim,
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout,
            router_noise=router_noise,
            use_load_balancing=use_load_balancing
        ) if use_moe else None
            
        # 4. Low-rank feedforward
        self.ff = LowRankFF(
            dim=dim,
            reduction_factor=ff_reduction_factor,
            dropout=dropout
        ) if use_lowrank_ff else None
        
        # Weights for combining outputs if multiple components are active
        self.combining_weights = nn.Parameter(
            torch.ones(sum([use_conv, use_recurrence, use_moe, use_lowrank_ff])) / 
            max(1, sum([use_conv, use_recurrence, use_moe, use_lowrank_ff]))
        )
        
        # Apply skip connection always
        self.skip_scale = nn.Parameter(torch.ones(1) * 0.5)
        
    def forward(self, x, memory_state=None):
        # B, L, D = x.shape
        outputs = []
        
        # Apply convolution if enabled
        if self.use_conv:
            conv_out = self.norm_conv(x)
            conv_out = self.conv(conv_out)
            outputs.append(conv_out)
        
        # Apply recurrence if enabled
        next_memory_state = None
        if self.use_recurrence:
            recurrence_out = self.norm_recurrence(x)
            recurrence_out, next_memory_state = self.recurrence(recurrence_out, memory_state)
            outputs.append(recurrence_out)
            
        # Apply MoE if enabled
        if self.use_moe:
            moe_out = self.norm_moe(x)
            moe_out = self.moe(moe_out)
            outputs.append(moe_out)
            
            # Track auxiliary loss if available
            if hasattr(self.moe, 'get_aux_loss') and self.moe.get_aux_loss() is not None:
                self.aux_losses['moe'] = self.moe.get_aux_loss()
                
        # Apply low-rank FF if enabled 
        if self.use_lowrank_ff:
            ff_out = self.norm_ff(x)
            ff_out = self.ff(ff_out)
            outputs.append(ff_out)
            
        # Combine outputs if multiple components are active
        if len(outputs) > 1:
            # Apply softmax to get normalized weights
            normalized_weights = F.softmax(self.combining_weights, dim=0)
            combined_output = torch.zeros_like(x)
            
            for i, out in enumerate(outputs):
                combined_output = combined_output + normalized_weights[i] * out
                
            # Apply residual connection with learned scale
            output = x + combined_output * self.skip_scale
        elif len(outputs) == 1:
            # Single component, just add residual
            output = x + outputs[0] * self.skip_scale
        else:
            # No components - identity function
            output = x
            
        return output, next_memory_state
    
    def get_aux_losses(self):
        """Return any auxiliary losses from sub-components."""
        return self.aux_losses

# Main SerriformNet model
class SerriformNet(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        dim: int = 512, 
        num_layers: int = 12, 
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        use_cache: bool = True,
        gradient_checkpointing: bool = False,
        arch_config: dict = None
    ):
        """
        SerriformNet: A linear scaling sequence model without attention mechanisms
        
        Args:
            vocab_size: Size of vocabulary
            dim: Hidden dimension size
            num_layers: Number of Serriform blocks
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            use_cache: Whether to use caching during generation
            gradient_checkpointing: Whether to use gradient checkpointing
            arch_config: Architecture configuration dictionary
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.use_cache = use_cache
        self.gradient_checkpointing = gradient_checkpointing
        
        # Set default architecture configuration if not provided
        self.arch_config = arch_config or {
            "use_conv": True,            # Use convolutional layers
            "use_recurrence": True,      # Use recurrent components
            "use_moe": True,             # Use mixture of experts
            "use_lowrank_ff": True,      # Use low-rank feedforward networks
            "conv_kernel_size": 3,       # Kernel size for convolutions
            "conv_dilation_growth": 2,   # Growth factor for dilation in deeper layers
            "num_experts": 4,            # Number of experts in MoE
            "top_k": 2,                  # Number of experts to route to
            "memory_dim_factor": 4,      # Memory dimension as fraction of model dim (dim/factor)
            "ff_reduction_factor": 4,    # Reduction factor for low-rank FF
            "load_balancing": True,      # Use load balancing for MoE
            "router_noise": 0.01,        # Noise added to router during training
            "activation": "silu"         # Activation function (silu, gelu, relu)
        }
        
        # Choose activation function based on config
        if self.arch_config.get("activation", "silu") == "silu":
            self.activation = nn.SiLU()
        elif self.arch_config.get("activation") == "gelu":
            self.activation = nn.GELU()
        elif self.arch_config.get("activation") == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.SiLU()  # Default
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, dim)
        
        # RoPE position embedding
        self.rope = RotaryPositionalEmbedding(dim, max_seq_len)
        
        # Dropout
        self.embed_dropout = nn.Dropout(dropout)
        
        # Build the Serriform blocks based on architecture config
        self.blocks = nn.ModuleList()
        
        for i in range(num_layers):
            # Calculate dilation growth for this layer
            dilation = 1
            if self.arch_config["use_conv"] and self.arch_config.get("conv_dilation_growth", 1) > 1:
                dilation = int(self.arch_config["conv_dilation_growth"] ** (i % 3))
                
            # Configure memory dimension for recurrence
            memory_dim = None
            if self.arch_config["use_recurrence"]:
                memory_dim = dim // self.arch_config.get("memory_dim_factor", 4)
                
            # Add Serriform block with configurable components
            self.blocks.append(
                SerriformBlock(
                    dim=dim,
                    kernel_size=self.arch_config.get("conv_kernel_size", 3),
                    dilation=dilation,
                    memory_dim=memory_dim,
                    dropout=dropout,
                    num_experts=self.arch_config.get("num_experts", 4),
                    top_k=self.arch_config.get("top_k", 2),
                    ff_reduction_factor=self.arch_config.get("ff_reduction_factor", 4),
                    use_conv=self.arch_config["use_conv"],
                    use_recurrence=self.arch_config["use_recurrence"],
                    use_moe=self.arch_config["use_moe"],
                    use_lowrank_ff=self.arch_config["use_lowrank_ff"],
                    max_seq_len=max_seq_len,
                    router_noise=self.arch_config.get("router_noise", 0.01),
                    use_load_balancing=self.arch_config.get("load_balancing", True)
                )
            )
            
        # Final normalization and projection
        self.norm = RMSNorm(dim)
        self.output_proj = nn.Linear(dim, vocab_size)
        
        # Apply weight initialization
        self._init_weights()
        
        # Global step counter for training tracking
        self._global_step = 0
        
    def _init_weights(self):
        """Initialize model weights with appropriate distribution."""
        std = 0.02
        
        # Initialize embedding with small normal distribution
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=std)
        
        # Initialize output projection
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=std)
        nn.init.zeros_(self.output_proj.bias)
        
    def _gradient_checkpointing_func(self, module, *args):
        """Helper function for gradient checkpointing."""
        return module(*args)
        
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
                'hidden_states': all_hidden_states,
                # For compatibility with Hugging Face's generation code
                'past_key_values': new_memory_states if use_cache else None
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
        use_cache: bool = True,
        streaming: bool = False
    ):
        """
        Generate text using the model, with efficient memory state caching
        
        Args:
            prompt_ids: Input token IDs [batch_size, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            top_k: Number of highest probability tokens to keep for top-k sampling
            top_p: Cumulative probability for nucleus sampling
            do_sample: Whether to sample or use greedy decoding
            repetition_penalty: Penalty for repeating tokens
            use_cache: Whether to use memory state caching
            streaming: If True, yield tokens as they're generated
        
        Returns:
            If streaming=False: tensor of shape [batch_size, max_new_tokens]
            If streaming=True: yields each new token as it's generated
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
        
        # Define a generator function for streaming mode
        def token_generator():
            nonlocal past_memory_states, generated_ids
            
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
                    
                    # Update memory states IN PLACE when possible to reduce allocations
                    if use_cache:
                        new_states = outputs['past_memory_states']
                        if past_memory_states is None:
                            past_memory_states = new_states
                        else:
                            # Convert tuple to list for in-place modification
                            if isinstance(past_memory_states, tuple):
                                past_memory_states = list(past_memory_states)
                            
                            # In-place update of existing memory states
                            for i, (old_state, new_state) in enumerate(zip(past_memory_states, new_states)):
                                # Clone with .detach() to ensure we don't keep computation graph
                                past_memory_states[i] = new_state.detach().clone()
                    
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
                    
                    # For streaming mode
                    yield next_token
                    
                    # For next iteration
                    current_token = next_token
                    
                    # Track token for repetition penalty
                    if prev_tokens_set is not None:
                        # Handle batched generation correctly - add all tokens in the batch
                        if next_token.dim() > 1 and next_token.size(0) > 1:
                            # For multi-batch case, add all tokens
                            for token in next_token.view(-1):
                                prev_tokens_set.add(token.item())
                        else:
                            # Single token case
                            prev_tokens_set.add(next_token.item())
        
        # Create the token generator
        gen = token_generator()
        
        # For streaming mode, return the generator directly
        if streaming:
            return gen
        
        # For non-streaming mode, consume the generator and return a tensor
        else:
            # Collect all tokens from the generator
            all_tokens = []
            for token in gen:
                all_tokens.append(token)
            
            # No tokens generated
            if not all_tokens:
                return torch.zeros((batch_size, 0), dtype=torch.long, device=device)
                
            # Concatenate all tokens
            return torch.cat(all_tokens, dim=1)

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
    parser = argparse.ArgumentParser(description="Train SerriformNet")
    
    # Data parameters
    parser.add_argument("--data_path", type=str, default="./data", help="Path to data directory")
    parser.add_argument("--tokenizer", type=str, default="tiktoken", help="Tokenizer to use (tiktoken)")
    parser.add_argument("--vocab_size", type=int, default=50257, help="Vocabulary size")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--block_size", type=int, default=1024, help="Block size for training")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to use")
    
    # Model parameters
    parser.add_argument("--dim", type=int, default=512, help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # Architecture configuration
    parser.add_argument("--no_conv", action="store_true", help="Disable convolution modules")
    parser.add_argument("--no_recurrence", action="store_true", help="Disable recurrence modules")
    parser.add_argument("--no_moe", action="store_true", help="Disable mixture of experts")
    parser.add_argument("--no_lowrank_ff", action="store_true", help="Disable low-rank feedforward")
    parser.add_argument("--conv_kernel_size", type=int, default=3, help="Kernel size for convolution")
    parser.add_argument("--dilation_growth", type=float, default=2.0, help="Growth factor for dilation")
    parser.add_argument("--num_experts", type=int, default=4, help="Number of experts in MoE")
    parser.add_argument("--top_k_experts", type=int, default=2, help="Number of experts to route to in MoE")
    parser.add_argument("--memory_dim_factor", type=int, default=4, help="Memory dimension fraction (dim/factor)")
    parser.add_argument("--ff_reduction_factor", type=int, default=4, help="Reduction factor for low-rank FF")
    parser.add_argument("--no_load_balancing", action="store_true", help="Disable MoE load balancing")
    parser.add_argument("--router_noise", type=float, default=0.01, help="Noise for MoE router")
    parser.add_argument("--activation", type=str, default="silu", choices=["silu", "gelu", "relu"], help="Activation function")
    parser.add_argument("--entropy_weight", type=float, default=0.01, help="Weight for entropy maximization loss")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for AdamW")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 for AdamW")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--lr_schedule", type=str, default="cosine", choices=["cosine", "linear", "none"], help="LR schedule")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Use gradient checkpointing")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--log_interval", type=int, default=10, help="Log interval")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    
    args = parser.parse_args()
    
    # Train the model
    train(args)
