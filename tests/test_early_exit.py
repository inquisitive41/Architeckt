import torch
import pytest
from inference.early_exit import DepthAwareEarlyExit

def test_depth_aware_early_exit_initialization():
    d_model = 64
    vocab_size = 1000
    exit_layers = [8, 16]
    
    daee = DepthAwareEarlyExit(
        d_model=d_model,
        vocab_size=vocab_size,
        exit_layers=exit_layers
    )
    
    assert str(8) in daee.exit_heads
    assert str(16) in daee.exit_heads
    assert str(10) not in daee.exit_heads
    
    assert daee.exit_heads[str(8)].proj.weight.shape == (vocab_size, d_model)

def test_depth_aware_early_exit_forward():
    d_model = 64
    vocab_size = 1000
    batch = 2
    seq_len = 10
    exit_layers = [4]
    
    daee = DepthAwareEarlyExit(d_model, vocab_size, exit_layers)
    x = torch.randn(batch, seq_len, d_model)
    
    # Layer 2 is NOT an exit layer
    logits_none = daee(x, layer_idx=2)
    assert logits_none is None
    
    # Layer 4 IS an exit layer
    logits = daee(x, layer_idx=4)
    assert logits is not None
    assert logits.shape == (batch, seq_len, vocab_size)

def test_depth_aware_early_exit_entropy():
    daee = DepthAwareEarlyExit(64, 10, [1], entropy_alpha=0.5, entropy_margin=0.8)
    
    # Very uniform distribution -> high entropy -> low confidence
    logits_uniform = torch.zeros(1, 10)  # softmax will be 0.1 for all
    should_exit, entropy_high = daee.should_exit(logits_uniform)
    
    assert not should_exit
    assert entropy_high.item() > 2.0  # ln(10) ~ 2.3
    
    # Very sharp distribution -> low entropy -> high confidence
    logits_sharp = torch.zeros(1, 10)
    logits_sharp[0, 0] = 100.0  # softmax will be 1.0 for index 0, 0.0 for others
    should_exit, entropy_low = daee.should_exit(logits_sharp)
    
    assert should_exit
    assert entropy_low.item() < 0.1
    
    # Check EMA update in train mode
    daee.train()
    daee.should_exit(logits_sharp)
    assert daee.ema_entropy.item() < 10.0  # Should have decreased from initial 10.0
