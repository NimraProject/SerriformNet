# SerriformNet

A novel sequence learning model architecture that combines hierarchical token memory via periodic convolution, sparse shift-gate recurrence, and low-rank feedforward compression without using attention mechanisms.

## Key Features

- **No Attention Mechanism**: Completely avoids KV-caching, scales linearly with context
- **Serrated Receptive Field**: Due to periodic convolution dilation and recurrence, allowing later tokens to "see" far back with fewer parameters
- **Competitive Inference Speed**: Especially on embedded devices and for sequences < 4K tokens
- **Training Stability**: Recurrent + convolution fusion avoids vanishing gradient issues seen in deep RNNs

## Development Status

✅ **Stable**: All unit tests passing
- Fixed tensor dimension handling in the structured state recurrence module
- Memory-efficient implementation with forward/backward compatibility
- Ready for research and experimentation

## Improvements & Features

### Modular Architecture
- Fully configurable model components via architecture config:
  - Enable/disable convolution, recurrence, MoE, or low-rank feedforward layers
  - Configure kernel sizes, dilation growth, experts count, etc.
  - Create leaner or specialized variants with focused components

### Enhanced Recurrence
- Memory-efficient precomputed decay masks cached for reuse
- Support for maximum sequence length specification
- Compatible with Hugging Face style `past_key_values` for integration

### Mixture of Experts
- Router noise for improved exploration during training
- GShard-style load balancing auxiliary loss for even expert utilization
- Expert usage tracking and entropy-based regularization

### Streaming Generation
- Supports token-by-token streaming output via generator API
- In-place cache updates to reduce memory allocations
- Optimized autoregressive performance

### Monitoring & Analysis
- Per-layer gradient norm tracking for training stability
- Expert usage visualization
- MoE auxiliary loss tracking

## Architecture Overview

![SerriformNet Architecture](https://github.com/NimraProject/SerriformNet/blob/28fff94bbe213222ba3c8acb6137e150e8a38d7b/serriform_architecture.png)

1. **Input Embedding Layer**
   - Standard embedding + linear projection
   - Token position fused via RoPE but without attention
   - Modulated into feature maps via multiplicative RoPE gates

2. **Serriform Blocks** (repeated N times)
   ```
   x -> conv_branch -> sparse recurrence -> gated fusion -> lowrank ff -> residual add
   ```
   - **Conv Branch**: Depthwise separable Conv1D with varying dilation
   - **Sparse Recurrence**: `h[t] = α·h[t-1] + β·x[t]` with learned scalars, applied to only 1/4 channels
   - **Gated Fusion**: `y = x + sigmoid(W1x) ⊙ tanh(W2x)` (GLU-style)
   - **LowRank FF**: `x -> W_down (d→r) -> gelu -> W_up (r→d)` where r << d

3. **Normalization**
   - RMSNorm (not LayerNorm) for better performance/memory
   - Applied pre-activation only

## Usage

### Installation

```bash
git clone https://github.com/NimraProject/SerriformNet.git
cd SerriformNet
pip install -r requirements.txt
```

### Training

```bash
python train.py --data_path /path/to/your/data --dim 512 --num_layers 12 --max_seq_len 1024
```

#### Architecture Ablation Studies

Study the contribution of different components:

```bash
# Convolution-only variant
python train.py --no_recurrence --no_moe --no_lowrank_ff

# Recurrence-only variant
python train.py --no_conv --no_moe --no_lowrank_ff

# MoE-only variant
python train.py --no_conv --no_recurrence --no_lowrank_ff
```

#### Additional Configuration Options

```bash
# Adjust MoE configuration
python train.py --num_experts 8 --top_k_experts 2 --router_noise 0.02 --entropy_weight 0.01

# Adjust recurrence for longer context
python train.py --max_seq_len 4096 --memory_dim_factor 8

# Change activation function
python train.py --activation gelu 
```

### Inference with Streaming

```python
import torch
from train import SerriformNet

# Load model
model = SerriformNet(vocab_size=50257, dim=512, num_layers=12)
model.load_state_dict(torch.load('serriformnet_checkpoint.pt'))
model.eval()

# Standard generation
prompt_ids = torch.tensor([[1, 2, 3, 4]]) # Your tokenized prompt
generated = model.generate(
    prompt_ids, 
    max_new_tokens=100, 
    temperature=0.7, 
    top_k=50
)

# Streaming generation
prompt_ids = torch.tensor([[1, 2, 3, 4]])
for token in model.generate(
    prompt_ids,
    max_new_tokens=100,
    temperature=0.7,
    streaming=True
):
    print(token.item())  # Process tokens as they're generated
```

## Custom Model Variants

Create configuration files for specialized model variants:

```python
# Convolutional-focused model
conv_config = {
    "use_conv": True,
    "use_recurrence": False,
    "use_moe": False,
    "use_lowrank_ff": True,
    "conv_kernel_size": 5,
    "conv_dilation_growth": 2.5,
    "ff_reduction_factor": A 
}

# Recurrence-focused model
recurrence_config = {
    "use_conv": False,
    "use_recurrence": True,
    "use_moe": False,
    "use_lowrank_ff": True,
    "memory_dim_factor": 2  # Larger memory dimension
}

# Load with custom configuration
model = SerriformNet(
    vocab_size=50257, 
    dim=512, 
    num_layers=12,
    arch_config=conv_config  # or recurrence_config
)
```

## License

MIT

## Citation

If you use SerriformNet in your research, please cite:

```
@misc{serriformnet2025,
  author = {NimraProject},
  title = {SerriformNet: A Linear-Scaling Sequence Model without Attention},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/NimraProject/SerriformNet}
}
```
