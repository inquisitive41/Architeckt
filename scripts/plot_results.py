import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_results():
    csv_file = "results/training_log.csv"
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Run scripts/run_poc_training.py first.")
        return
        
    df = pd.read_csv(csv_file)
    os.makedirs("results/plots", exist_ok=True)
    
    # 1. Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(df['step'], df['arch_loss'], label='Architeckt Loss', color='#2ecc71', linewidth=2)
    plt.plot(df['step'], df['base_loss'], label='Baseline Transformer Loss', color='#e74c3c', linewidth=2, alpha=0.7)
    plt.title('Training Loss Convergence: Architeckt vs Baseline', fontsize=14)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Cross-Entropy Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('results/plots/loss_curve.png', dpi=300)
    print("Saved loss_curve.png")
    
    # 2. Time/Latency
    plt.figure(figsize=(10, 6))
    plt.plot(df['step'], df['arch_time_ms'], label='Architeckt Time/Batch (ms)', color='#3498db', linewidth=2)
    plt.plot(df['step'], df['base_time_ms'], label='Baseline Time/Batch (ms)', color='#9b59b6', linewidth=2)
    plt.title('Training Speed (Throughput)', fontsize=14)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Time per Batch (ms)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('results/plots/throughput.png', dpi=300)
    print("Saved throughput.png")
    
    # 3. Confidence (Architeckt Specific)
    plt.figure(figsize=(10, 6))
    plt.plot(df['step'], df['arch_confidence'], label='Mean Confidence (TLCG)', color='#f39c12', linewidth=2)
    plt.title('Architeckt Internal Confidence Over Time', fontsize=14)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Confidence Score [0, 1]', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('results/plots/confidence.png', dpi=300)
    print("Saved confidence.png")

if __name__ == "__main__":
    plot_results()
