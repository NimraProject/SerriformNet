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

Additional arguments:
- `--dim`: Hidden dimension size (default: 512)
- `--num_layers`: Number of Serriform blocks (default: 12)
- `--max_seq_len`: Maximum sequence length (default: 1024)
- `--batch_size`: Training batch size (default: 32)
- `--epochs`: Number of training epochs (default: 10)
- `--lr`: Learning rate (default: 5e-5)
- `--device`: Device to use (default: cuda if available, else cpu)

### Inference

```python
import torch
from train import SerriformNet

# Load model
model = SerriformNet(vocab_size=50257, dim=512, num_layers=12)
model.load_state_dict(torch.load('serriformnet_checkpoint.pt'))
model.eval()

# Generate text
prompt_ids = torch.tensor([[1, 2, 3, 4]]) # Your tokenized prompt
generated = model.generate(
    prompt_ids, 
    max_new_tokens=100, 
    temperature=0.7, 
    top_k=50
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
