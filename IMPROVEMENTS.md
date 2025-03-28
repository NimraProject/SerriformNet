# SerriformNet Improvements

## Implemented Improvements

### Recurrence Enhancements
- ✅ Precomputed decay masks once per max_seq_len, cached for reuse
- ✅ Fixed tensor dimension handling in the exponential decay computation
- ✅ Added max_seq_len parameter for better context control

### Mixture of Experts Enhancements
- ✅ Added router noise for improved training stability
- ✅ Implemented GShard-style load balancing loss
- ✅ Added expert usage tracking
- ✅ Support for entropy-based loss term to encourage diversity
- ✅ Configurable router noise and load balancing weight

### Generation Optimizations
- ✅ Added support for streaming token generation
- ✅ Implemented in-place cache updates to reduce memory allocations
- ✅ Added HuggingFace-compatible past_key_values support

### Architecture Flexibility
- ✅ Added comprehensive architecture configuration
- ✅ Support for ablation studies (conv-only, recurrence-only, moe-only)
- ✅ Modular component toggling
- ✅ Configurable model parameters (kernel size, dilation, experts count)
- ✅ Combining weights for multi-component fusion

### Training and Monitoring
- ✅ Added gradient norm tracking per layer group
- ✅ Support for auxiliary loss term visualization
- ✅ Expert usage entropy monitoring

## Future Roadmap

### Recurrence Improvements
- 🔲 Rewrite recurrence with implicit recurrence (Mamba SSM-style filtering)
- 🔲 Implement state-space kernel convolution

### MoE Improvements
- 🔲 Implement soft routing + sparse dropout
- 🔲 Add dynamic expert pruning
- 🔲 Support conditional computation with adaptive routing

### Serriform Block Optimizations
- ✅ Support for ablations (conv-only, recurrence-only, moe-only)
- ✅ Support for selective component fusion (e.g., conv + moe)
- 🔲 Monitor and optimize FLOPs per block
- 🔲 Deeper ablation studies on component combinations

### Performance Optimizations
- 🔲 CUDA kernels for custom fused operations
- 🔲 Add support for 8-bit quantization
- 🔲 Implement fused optimizers (AdamW, Lion, etc.)

### Training Enhancements
- ✅ Track gradient norms per-layer
- ✅ Add entropy-based loss term for expert diversity
- 🔲 Implement fused optimizers for large model training

### Architecture Experiments
- 🔲 Add conditional gating between components
- 🔲 Implement multi-resolution convolution paths
- 🔲 Try Gaussian Error Linear Unit variations

## Tracking Progress from Original TODO List

### 🔧 Recurrence
- ✅ Precompute decay masks once per max_seq_len, cache them
- 🔲 Rewrite recurrence with implicit recurrence ala Mamba (SSM-style filtering)
- 🔲 Use state-space kernel convolution

### 🔧 MoE
- ✅ Add router noise + load balancing loss (GShard-style aux loss)
- 🔲 Optionally use soft routing + sparse dropout
- ✅ Track expert usage, penalize overuse

### 🔧 Serriform Block
- ✅ Support for ablations (conv-only, recurrence-only, moe-only, fusion of just two)
- 🔲 Monitor FLOPs and runtime per block

### 🔧 Generation
- ✅ Support past_key_values-style streaming for faster autoregressive use
- ✅ Shift cache update to be in-place to reduce allocations

### 🔧 Training
- ✅ Track gradient norms per-layer (for exploding grads)
- ✅ Plug in entropy-based loss term to encourage diversity in expert usage
- 🔲 Consider fused optimizer for large models

### 🔧 Modularity
- ✅ Split serriform block into submodules that can be toggled
- ✅ Added arch_config.json style configuration 