import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import math
import time
import csv
import os
import sys

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.config import ArchitecktConfig
from models.architeckt import ArchitecktModel

# -----------------------------------------------------------------------------
# 1. Baseline Transformer Model
# -----------------------------------------------------------------------------
class BaselineTransformer(nn.Module):
    """Standard Transformer Decoder for Baseline comparison."""
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Embedding(2048, d_model)  # Simple learned pos
        
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, 
            dropout=0.0, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=n_layers)
        
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        
    def forward(self, x, labels=None):
        b, seq_len = x.size()
        pos = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        
        x = self.embedding(x) + self.pos_encoder(pos).unsqueeze(0)
        
        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        x = self.transformer(x, mask=mask, is_causal=True)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = labels[..., 1:].contiguous().view(-1)
            loss = nn.functional.cross_entropy(shift_logits, shift_labels)
            
        return logits, loss

# -----------------------------------------------------------------------------
# 2. Dummy Data Generator
# -----------------------------------------------------------------------------
def get_batch(batch_size, seq_len, vocab_size, device):
    """Generates simple patterned sequences (e.g., repeating sub-sequences) so it's learnable."""
    # Let's create a predictable pattern so the loss can actually converge quickly
    # Pattern: 0, 1, 2, ..., K, 0, 1, ...
    pattern_len = 20
    base_pattern = torch.arange(pattern_len, device=device)
    
    batch = []
    for _ in range(batch_size):
        start_val = torch.randint(0, vocab_size - pattern_len, (1,), device=device).item()
        # Create a sequence of repeating pattern offset by start_val
        repeats = (seq_len // pattern_len) + 2
        seq = (base_pattern + start_val).repeat(repeats)[:seq_len]
        batch.append(seq)
        
    data = torch.stack(batch)
    return data, data.clone()

# -----------------------------------------------------------------------------
# 3. Training Loop
# -----------------------------------------------------------------------------
def run_poc():
    print("="*60)
    print("Architeckt Proof-of-Concept Training Run")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs("results", exist_ok=True)
    csv_file = "results/training_log.csv"
    
    # ------------------
    # Configurations
    # ------------------
    vocab_size = 500
    seq_len = 128
    batch_size = 16
    max_steps = 50
    
    # Architeckt Nano Config
    # ~3.5M parameters
    arch_config = ArchitecktConfig(
        vocab_size=vocab_size,
        d_model=128,
        d_ff=256,
        n_blocks=4,
        n_experts=4,
        n_active_experts=2,
        max_heads=4,
        n_scales=3,
        scale_windows=(32, 64, 128),
        exit_layers=[2]
    )
    
    # ------------------
    # Init Models
    # ------------------
    architeckt = ArchitecktModel(arch_config).to(device)
    baseline = BaselineTransformer(
        vocab_size=vocab_size, d_model=128, n_heads=4, n_layers=4, d_ff=256
    ).to(device)
    
    print(f"Architeckt Params: {architeckt.get_num_params()/1e6:.2f}M")
    print(f"Baseline Params: {sum(p.numel() for p in baseline.parameters())/1e6:.2f}M")
    
    # Optimizers
    opt_arch = AdamW(architeckt.parameters(), lr=1e-3)
    opt_base = AdamW(baseline.parameters(), lr=1e-3)
    
    # ------------------
    # Training Loop
    # ------------------
    history = []
    
    print("\nStarting Training...")
    print(f"{'Step':>6} | {'Arch Loss':>10} | {'Base Loss':>10} | {'Arch Time':>10} | {'Base Time':>10}")
    print("-" * 65)
    
    for step in range(1, max_steps + 1):
        x, y = get_batch(batch_size, seq_len, vocab_size, device)
        
        # --- Train Baseline ---
        t0 = time.perf_counter()
        opt_base.zero_grad()
        _, loss_base = baseline(x, labels=y)
        loss_base.backward()
        opt_base.step()
        t_base = (time.perf_counter() - t0) * 1000  # ms
        
        # --- Train Architeckt ---
        t0 = time.perf_counter()
        opt_arch.zero_grad()
        _, loss_arch, metrics = architeckt(x, labels=y)
        
        # Total loss includes auxiliary routing loss
        total_arch_loss = loss_arch + metrics.get("aux_loss", 0.0)
        total_arch_loss.backward()
        opt_arch.step()
        t_arch = (time.perf_counter() - t0) * 1000  # ms
        
        record = {
            "step": step,
            "arch_loss": loss_arch.item(),
            "base_loss": loss_base.item(),
            "arch_time_ms": t_arch,
            "base_time_ms": t_base,
            "arch_confidence": metrics.get("mean_confidence", 0.0)
        }
        history.append(record)
        
        if step % 20 == 0 or step == 1:
            print(f"{step:>6} | {record['arch_loss']:>10.4f} | {record['base_loss']:>10.4f} | {t_arch:>8.1f}ms | {t_base:>8.1f}ms")
            
    # Save to CSV
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
        
    print(f"\nTraining Complete. Logs saved to {csv_file}")

if __name__ == "__main__":
    run_poc()
