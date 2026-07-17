import os
import sys
import time
import urllib.request
import torch
import torch.nn as nn
from torch.optim import AdamW

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from models.config import ArchitecktConfig
from models.architeckt import ArchitecktModel

# -----------------------------------------------------------------------------
# 1. Dataset Downloading and Tokenization
# -----------------------------------------------------------------------------
DATA_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'tinyshakespeare.txt')

def get_data():
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    if not os.path.exists(DATA_PATH):
        print("Downloading Tiny Shakespeare dataset...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
    
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"Dataset length: {len(text):,} characters. Vocab size: {vocab_size}")
    
    # Simple character-level tokenizer
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    return train_data, val_data, vocab_size, decode

# -----------------------------------------------------------------------------
# 2. Training Script
# -----------------------------------------------------------------------------
def get_batch(data, batch_size, seq_len, device):
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x.to(device), y.to(device)

def main():
    print("="*60)
    print("Training Architeckt on Tiny Shakespeare")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_data, val_data, vocab_size, decode = get_data()
    
    # Config
    batch_size = 16
    seq_len = 64
    max_steps = 100  # Keep it small for CPU
    
    config = ArchitecktConfig(
        vocab_size=vocab_size,
        d_model=128,
        d_ff=256,
        n_blocks=4,
        n_experts=4,
        n_active_experts=2,
        max_heads=4,
        n_scales=3,
        scale_windows=(16, 32, 64),
        exit_layers=[2]
    )
    
    model = ArchitecktModel(config).to(device)
    print(f"Model parameters: {model.get_num_params()/1e6:.2f}M")
    
    optimizer = AdamW(model.parameters(), lr=1e-3)
    
    model.train()
    for step in range(1, max_steps + 1):
        x, y = get_batch(train_data, batch_size, seq_len, device)
        
        t0 = time.perf_counter()
        optimizer.zero_grad()
        logits, loss, metrics = model(x, labels=y)
        
        total_loss = loss + metrics.get("aux_loss", 0.0)
        total_loss.backward()
        optimizer.step()
        t1 = time.perf_counter()
        
        if step % 10 == 0 or step == 1:
            print(f"Step {step:>3} | Loss: {loss.item():.4f} | Aux Loss: {metrics.get('aux_loss', 0.0):.4f} | Time: {(t1-t0)*1000:.1f}ms")
            
    print("\n--- Generating Text ---")
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    with torch.no_grad():
        generated_ids, _ = model.generate(context, max_new_tokens=200, use_early_exit=False)
    
    generated_text = decode(generated_ids[0].tolist())
    print("\nGenerated Text:\n")
    print(generated_text)
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
