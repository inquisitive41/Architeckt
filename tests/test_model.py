import torch
import pytest
from models.config import ArchitecktConfig
from models.architeckt import ArchitecktModel

@pytest.fixture
def mini_config():
    """Returns a very small config for fast testing."""
    return ArchitecktConfig(
        vocab_size=1000,
        d_model=64,
        d_ff=128,
        n_blocks=4,  # Just 4 blocks
        n_experts=4,
        n_active_experts=2,
        max_heads=4,
        n_scales=3,
        exit_layers=[2]  # Early exit at layer 2
    )

def test_architeckt_model_initialization(mini_config):
    model = ArchitecktModel(mini_config)
    assert len(model.blocks) == 4
    
    # Check weight tying
    assert torch.equal(model.lm_head.weight, model.embedding.weight)
    
    params = model.get_num_params(trainable_only=True)
    assert params > 10000

def test_architeckt_model_forward(mini_config):
    model = ArchitecktModel(mini_config)
    batch = 2
    seq_len = 16
    
    input_ids = torch.randint(0, mini_config.vocab_size, (batch, seq_len))
    labels = torch.randint(0, mini_config.vocab_size, (batch, seq_len))
    
    logits, loss, metrics = model(input_ids, labels=labels)
    
    assert logits.shape == (batch, seq_len, mini_config.vocab_size)
    assert loss.item() > 0.0
    assert "aux_loss" in metrics
    assert "mean_confidence" in metrics

def test_architeckt_model_generate(mini_config):
    model = ArchitecktModel(mini_config)
    model.eval()  # Set to eval mode for generation
    
    batch = 1
    prompt_len = 5
    max_new_tokens = 10
    
    input_ids = torch.randint(0, mini_config.vocab_size, (batch, prompt_len))
    
    # Generate without early exit
    generated_ids, stats = model.generate(
        input_ids, 
        max_new_tokens=max_new_tokens,
        use_early_exit=False,
        use_confidence_verification=True
    )
    
    assert generated_ids.shape == (batch, prompt_len + max_new_tokens)
    assert stats["tokens_generated"] == max_new_tokens
    
    # Generate WITH early exit (force it to exit by lowering margin to something absurd or using a mock)
    # Since it's untrained, it might not trigger natively, but we ensure the code runs without error
    generated_ids_ee, stats_ee = model.generate(
        input_ids, 
        max_new_tokens=max_new_tokens,
        use_early_exit=True,
    )
    
    assert generated_ids_ee.shape == (batch, prompt_len + max_new_tokens)
    assert "early_exit_rate" in stats_ee
