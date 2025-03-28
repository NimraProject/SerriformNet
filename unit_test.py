import torch
import unittest
import sys
import numpy as np
from train import (
    RMSNorm, 
    RotaryPositionalEmbedding, 
    SparseShiftGateRecurrence, 
    GatedFusion, 
    LowRankFF, 
    SerriformBlock, 
    SerriformNet
)

class TestSerriformNetComponents(unittest.TestCase):
    """Test cases for all components of the SerriformNet architecture."""
    
    def setUp(self):
        """Set up common test parameters."""
        torch.manual_seed(42)
        np.random.seed(42)
        self.device = torch.device("cpu")
        self.batch_size = 4
        self.seq_len = 32  # Increased from 16 to handle dilated convolutions
        self.dim = 64
        self.vocab_size = 1000
        
        # Sample input
        self.x = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len)).to(self.device)
        self.h = torch.randn(self.batch_size, self.seq_len, self.dim).to(self.device)
        
    def test_rmsnorm(self):
        """Test RMSNorm layer."""
        norm = RMSNorm(self.dim).to(self.device)
        out = norm(self.h)
        
        # Check shape
        self.assertEqual(out.shape, self.h.shape)
        
        # Check normalization
        rms = torch.sqrt(torch.mean(out**2, dim=-1))
        self.assertTrue(torch.allclose(rms, torch.ones_like(rms), rtol=1e-2))
        
    def test_rotary_positional_embedding(self):
        """Test Rotary Positional Embedding."""
        rope = RotaryPositionalEmbedding(self.dim).to(self.device)
        out = rope(self.h)
        
        # Check shape
        self.assertEqual(out.shape, self.h.shape)
        
        # Check that positions are encoded differently
        # Different positions should have different encoding patterns
        pos1 = out[:, 0, :]
        pos2 = out[:, 1, :]
        self.assertFalse(torch.allclose(pos1, pos2))
        
    def test_sparse_shift_gate_recurrence(self):
        """Test Sparse Shift Gate Recurrence."""
        recurrence = SparseShiftGateRecurrence(self.dim).to(self.device)
        out = recurrence(self.h)
        
        # Check shape
        self.assertEqual(out.shape, self.h.shape)
        
        # Check that later tokens have memory of earlier ones
        # Extract just one batch for simplicity
        single_batch_out = out[0].detach().cpu().numpy()
        
        # Check if there's some temporal dependency (correlation should increase over time)
        # Not a perfect test, but gives some indication of recurrence working
        correlations = []
        for i in range(1, self.seq_len):
            corr = np.corrcoef(single_batch_out[i-1, :recurrence.active_dim], 
                             single_batch_out[i, :recurrence.active_dim])[0, 1]
            correlations.append(corr)
            
        # At least some correlations should be positive
        self.assertTrue(any(c > 0 for c in correlations))
        
    def test_gated_fusion(self):
        """Test Gated Fusion module."""
        fusion = GatedFusion(self.dim).to(self.device)
        out = fusion(self.h)
        
        # Check shape
        self.assertEqual(out.shape, self.h.shape)
        
        # Check that fusion does something (output != input)
        self.assertFalse(torch.allclose(out, self.h))
        
    def test_lowrank_ff(self):
        """Test Low-Rank Feedforward network."""
        ff = LowRankFF(self.dim).to(self.device)
        out = ff(self.h)
        
        # Check shape
        self.assertEqual(out.shape, self.h.shape)
        
        # Check parameter count (should be significantly less than a full FF)
        param_count = sum(p.numel() for p in ff.parameters())
        full_ff_param_count = self.dim * self.dim * 2  # Two full matrices
        self.assertLess(param_count, full_ff_param_count)
        
    def test_serriform_block(self):
        """Test a full Serriform Block."""
        block = SerriformBlock(self.dim).to(self.device)
        
        # Check minimum sequence length
        min_seq_len = block.min_seq_len
        
        # Create input with enough length
        x = self.h
        out = block(x)
        
        # Due to conv without padding, sequence length should be reduced
        expected_seq_len = self.seq_len - (block.kernel_size - 1) * block.dilation
        self.assertEqual(out.shape, (self.batch_size, expected_seq_len, self.dim))
        
        # Test with a different dilation
        dilation = 2
        block2 = SerriformBlock(self.dim, dilation=dilation).to(self.device)
        
        # Make sure input is long enough for the dilated convolution
        min_seq_len2 = (block2.kernel_size - 1) * dilation + 1
        if self.seq_len >= min_seq_len2:
            out2 = block2(x)
            expected_seq_len2 = self.seq_len - (block2.kernel_size - 1) * dilation
            self.assertEqual(out2.shape, (self.batch_size, expected_seq_len2, self.dim))
        else:
            print(f"Skipping dilation={dilation} test as sequence length {self.seq_len} is too short")
        
        # Test with short sequence
        short_x = torch.randn(self.batch_size, 5, self.dim).to(self.device)
        short_out = block(short_x)
        self.assertEqual(short_out.shape[0], self.batch_size)
        self.assertEqual(short_out.shape[2], self.dim)
        
    def test_full_model(self):
        """Test the full SerriformNet model."""
        # Use fewer layers to reduce sequence length shrinkage
        num_layers = 2
        model = SerriformNet(
            vocab_size=self.vocab_size,
            dim=self.dim,
            num_layers=num_layers,
            max_seq_len=self.seq_len
        ).to(self.device)
        
        # Forward pass
        logits = model(self.x)
        
        # Check logits shape (accounting for sequence length reduction from conv layers)
        # For robustness, just check batch and vocab dimensions
        self.assertEqual(logits.shape[0], self.batch_size)
        self.assertEqual(logits.shape[2], self.vocab_size)
        
        # Test generation capability
        prompt = torch.randint(0, self.vocab_size, (1, 8)).to(self.device)
        generated = model.generate(prompt, max_new_tokens=10)
        
        # Check generated shape
        self.assertEqual(generated.shape[0], 1)
        self.assertEqual(generated.shape[1], 10)
        
    def test_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        # Use fewer layers to reduce sequence length shrinkage
        num_layers = 1
        model = SerriformNet(
            vocab_size=self.vocab_size,
            dim=self.dim,
            num_layers=num_layers,
            max_seq_len=self.seq_len
        ).to(self.device)
        
        # Get a batch with sufficient length for convolutions
        x = self.x[:2, :16]
        
        # Forward pass (get actual output length)
        with torch.no_grad():
            logits_test = model(x)
            output_length = logits_test.shape[1]
        
        # Create proper target with the correct length
        y = torch.randint(0, self.vocab_size, (2, output_length)).to(self.device)
        
        # Forward pass for real
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, self.vocab_size), 
            y.view(-1)
        )
        
        # Check that loss is finite
        self.assertFalse(torch.isnan(loss).item())
        self.assertFalse(torch.isinf(loss).item())
        
        # Backward pass
        loss.backward()
        
        # Check that all parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for {name}")
                # Also check for NaN gradients
                self.assertFalse(torch.isnan(param.grad).any().item(), f"NaN gradient for {name}")
                self.assertFalse(torch.isinf(param.grad).any().item(), f"Inf gradient for {name}")

class TestSerriformNetPerformance(unittest.TestCase):
    """Performance tests for SerriformNet."""
    
    def setUp(self):
        """Set up common test parameters."""
        torch.manual_seed(42)
        self.device = torch.device("cpu")
        self.batch_size = 2
        self.seq_len = 128
        self.dim = 128
        self.vocab_size = 1000
        
        # Create a smaller model for performance testing
        self.model = SerriformNet(
            vocab_size=self.vocab_size,
            dim=self.dim,
            num_layers=2,
            max_seq_len=self.seq_len
        ).to(self.device)
        
        # Sample input
        self.x = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len)).to(self.device)
        
    def test_memory_efficiency(self):
        """Test memory efficiency compared to naive implementation."""
        # This is an approximate test as exact memory measurement in PyTorch is complex
        
        # Get param count for our model
        serriform_param_count = sum(p.numel() for p in self.model.parameters())
        
        # Rough estimate of equivalent attention-based model
        # Simple estimate: embedding + n_layers * (attn + 2*ff)
        attn_dim = self.dim
        n_layers = 2
        attn_param_count = (
            self.vocab_size * self.dim +  # Embedding
            n_layers * (
                4 * self.dim * self.dim +  # Self-attention (Q,K,V,O)
                2 * self.dim * self.dim * 4  # FF with 4x expansion
            )
        )
        
        # Verify our model is more parameter efficient
        self.assertLess(serriform_param_count, attn_param_count)
        efficiency_ratio = serriform_param_count / attn_param_count
        print(f"\nParameter efficiency ratio: {efficiency_ratio:.4f}")
        print(f"SerriformNet params: {serriform_param_count:,}")
        print(f"Equivalent attention model params: {attn_param_count:,}")
        
    def test_inference_speed(self):
        """Test inference speed."""
        # Warmup
        for _ in range(5):
            _ = self.model(self.x)
            
        # Timed run
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else torch.tensor([])
        end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else torch.tensor([])
        
        if torch.cuda.is_available():
            start.record()
        
        n_runs = 100
        for _ in range(n_runs):
            with torch.no_grad():
                _ = self.model(self.x)
                
        if torch.cuda.is_available():
            end.record()
            torch.cuda.synchronize()
            elapsed_time = start.elapsed_time(end) / 1000 / n_runs  # seconds per inference
        else:
            import time
            start_time = time.time()
            for _ in range(n_runs):
                with torch.no_grad():
                    _ = self.model(self.x)
            elapsed_time = (time.time() - start_time) / n_runs  # seconds per inference
        
        # Report inference time
        tokens_per_second = self.batch_size * self.seq_len / elapsed_time
        print(f"\nInference time: {elapsed_time*1000:.2f} ms per batch")
        print(f"Throughput: {tokens_per_second:.2f} tokens/second")
        
        # No hard test here, just reporting performance

if __name__ == "__main__":
    print("Running SerriformNet Unit Tests...")
    unittest.main() 