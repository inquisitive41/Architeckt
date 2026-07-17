"""
Architeckt Training Module

Handles the full training loop with:
- Data loading from text corpora
- Optimization with AdamW + cosine schedule
- Gradient accumulation for large effective batch sizes
- Mixed precision (bfloat16) training
- Checkpointing and logging

Training budget analysis (Medium config, ~1.5B params):
    8 × A100 80GB × 7 days ≈ 1,450,000 GPU-hours
    At ~310 TFLOPS/A100 (bf16): ~1.45 × 10^18 total FLOPs
    
    Target: 300B tokens (Chinchilla-optimal for 1.5B params)
    Batch size: 512 sequences × 4096 tokens = 2M tokens/batch
    Steps: 150,000
    Step time target: ~4 seconds → ~6000 steps/day → 7 days for 42K steps
    With gradient accumulation 4×: ~150K effective steps

Memory estimation (per GPU, bf16):
    Model weights:    3.0 GB (1.5B × 2 bytes)
    Optimizer states: 18.0 GB (AdamW: 12 bytes/param for fp32 states)
    Gradients:         3.0 GB
    Activations:      20-30 GB (batch × seq × d_model × n_blocks × overhead)
    Total:            ~45-55 GB — fits in 80GB with activation checkpointing
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.cuda.amp import GradScaler, autocast
from typing import Optional, List, Dict, Any, Iterator
from dataclasses import dataclass, field
import math
import time
import json
import os


@dataclass
class TrainingConfig:
    """Hyperparameters for Architeckt training."""
    # Data
    train_data_path: str = ""
    val_data_path: str = ""
    seq_len: int = 4096
    batch_size_per_gpu: int = 2
    gradient_accumulation_steps: int = 4

    # Optimization
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    max_steps: int = 150_000
    warmup_steps: int = 2_000

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "bfloat16"  # "float16" or "bfloat16"

    # Checkpointing
    save_dir: str = "./checkpoints"
    save_every_steps: int = 5_000
    log_every_steps: int = 100

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Activation checkpointing (trades compute for memory)
    activation_checkpointing: bool = True

    # Distributed Training
    use_fsdp: bool = False

    # Data
    prefetch_factor: int = 2
    num_workers: int = 4


class TextDataset(IterableDataset):
    """Streaming text dataset for large corpora.

    Reads pre-tokenized binary files (.bin) with uint16 token IDs.
    Produces contiguous sequences of length seq_len.
    """

    def __init__(
        self,
        data_path: str,
        seq_len: int,
        seed: int = 42,
        world_size: int = 1,
        rank: int = 0,
    ):
        self.data_path = data_path
        self.seq_len = seq_len
        self.seed = seed
        self.world_size = world_size
        self.rank = rank

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        # Load data
        data = torch.from_file(
            self.data_path,
            shared=True,
            size=os.path.getsize(self.data_path) // 2,
            dtype=torch.int64,
        )

        # Per-worker offset
        tokens_per_worker = len(data) // self.world_size
        start = self.rank * tokens_per_worker

        # Random offset for each epoch
        rng = torch.Generator()
        rng.manual_seed(self.seed)

        offset = start + torch.randint(0, self.seq_len, (1,), generator=rng).item()
        pos = offset

        while pos + self.seq_len + 1 < start + tokens_per_worker:
            chunk = data[pos : pos + self.seq_len + 1]
            yield {
                "input_ids": chunk[:-1],
                "labels": chunk[1:],
            }
            pos += self.seq_len

    @staticmethod
    def from_text_files(
        text_paths: List[str],
        output_path: str,
        tokenizer,
    ):
        """Convert text files to pre-tokenized binary format.

        Args:
            text_paths: list of .txt or .jsonl file paths
            output_path: path for the output .bin file
            tokenizer: HuggingFace tokenizer or callable tokenizing function
        """
        tokens = []
        for path in text_paths:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        text = json.loads(line)["text"]
                    except (json.JSONDecodeError, KeyError):
                        text = line
                    if text.strip():
                        tokens.extend(tokenizer(text))

        tokens_array = torch.tensor(tokens, dtype=torch.int64)
        tokens_array.numpy().tofile(output_path)
        print(f"Saved {len(tokens):,} tokens to {output_path}")


class ArchitecktTrainer:
    """Full training loop for Architeckt.

    Handles distributed training, mixed precision, gradient accumulation,
    checkpointing, and logging.

    Usage:
        model = ArchitecktModel(config)
        train_config = TrainingConfig(...)
        trainer = ArchitecktTrainer(model, train_config)
        trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.config.use_fsdp:
            import torch.distributed as dist
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
            import functools

            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            
            my_auto_wrap_policy = functools.partial(
                size_based_auto_wrap_policy, min_num_params=100000
            )
            device_id = torch.cuda.current_device() if torch.cuda.is_available() else None
            self.model = FSDP(
                self.model, 
                auto_wrap_policy=my_auto_wrap_policy, 
                device_id=device_id
            )
        else:
            self.model = self.model.to(self.device)

        # Create optimizer
        self.optimizer = self._create_optimizer()

        # Create learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision
        self.amp_dtype = torch.bfloat16 if config.amp_dtype == "bfloat16" else torch.float16
        self.scaler = GradScaler(enabled=config.use_amp and config.amp_dtype == "float16")

        # Training state
        self.global_step = 0
        self.best_val_loss = float("inf")

        # Create save directory
        os.makedirs(config.save_dir, exist_ok=True)

    def _create_optimizer(self):
        """Create AdamW optimizer with parameter groups for weight decay."""
        # Separate parameters that should have weight decay from those that shouldn't
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Don't apply weight decay to biases, norms, and embeddings
            if (
                param.ndim < 2
                or "norm" in name.lower()
                or "bias" in name
                or "embedding" in name
                or "threshold" in name  # SwiGLU-T thresholds
                or "log_beta" in name
                or "log_temperature" in name
            ):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        print(f"  Decay params: {sum(p.numel() for p in decay_params):,}")
        print(f"  No-decay params: {sum(p.numel() for p in no_decay_params):,}")

        return AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps,
        )

    def _create_scheduler(self):
        """Cosine schedule with linear warmup."""
        warmup = LinearLR(
            self.optimizer,
            start_factor=0.001,
            end_factor=1.0,
            total_iters=self.config.warmup_steps,
        )
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.max_steps - self.config.warmup_steps,
            eta_min=self.config.min_lr,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[self.config.warmup_steps],
        )

    def _get_dataloader(self, split: str = "train") -> DataLoader:
        """Create a dataloader for training or validation."""
        data_path = self.config.train_data_path if split == "train" else self.config.val_data_path

        if not data_path or not os.path.exists(data_path):
            raise FileNotFoundError(f"Data not found: {data_path}")

        dataset = TextDataset(
            data_path=data_path,
            seq_len=self.config.seq_len,
        )

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size_per_gpu,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            pin_memory=True,
        )

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with gradient accumulation."""
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Forward with optional mixed precision
        with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.config.use_amp):
            logits, loss, metrics = self.model(input_ids=input_ids, labels=labels)

            # Auxiliary loss from routing
            total_loss = loss + metrics["aux_loss"] * self.config.gradient_accumulation_steps

        # Scale loss for gradient accumulation
        total_loss = total_loss / self.config.gradient_accumulation_steps

        # Backward
        if self.config.use_amp and self.config.amp_dtype == "float16":
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        return {
            "loss": loss.item(),
            "aux_loss": metrics["aux_loss"],
            "total_loss": total_loss.item() * self.config.gradient_accumulation_steps,
            "mean_confidence": metrics["mean_confidence"],
        }

    def _optimizer_step(self):
        """Apply gradient clipping and step optimizer."""
        if self.config.use_amp and self.config.amp_dtype == "float16":
            self.scaler.unscale_(self.optimizer)

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
        )

        if self.config.use_amp and self.config.amp_dtype == "float16":
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()

        return grad_norm.item()

    def train(self):
        """Main training loop."""
        print(f"\n{'='*60}")
        print(f"Architeckt Training")
        print(f"{'='*60}")
        print(f"  Model params: {self.model.get_num_params()/1e9:.2f}B")
        print(f"  Device: {self.device}")
        print(f"  Max steps: {self.config.max_steps:,}")
        print(f"  Batch size: {self.config.batch_size_per_gpu} × {self.config.gradient_accumulation_steps} accum")
        print(f"  LR: {self.config.learning_rate} → {self.config.min_lr}")
        print(f"{'='*60}\n")

        train_loader = self._get_dataloader("train")

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        step_metrics = []
        start_time = time.time()

        data_iter = iter(train_loader)

        while self.global_step < self.config.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            # Accumulate gradients
            is_accum_step = (self.global_step + 1) % self.config.gradient_accumulation_steps == 0

            # Optional activation checkpointing
            if self.config.activation_checkpointing and is_accum_step:
                with torch.utils.checkpoint.checkpoint_activations(self.model):
                    metrics = self._train_step(batch)
            else:
                metrics = self._train_step(batch)

            step_metrics.append(metrics)

            if is_accum_step:
                grad_norm = self._optimizer_step()

                if self.global_step % self.config.log_every_steps == 0:
                    elapsed = time.time() - start_time
                    avg_metrics = {
                        k: sum(m[k] for m in step_metrics) / len(step_metrics)
                        for k in step_metrics[0]
                    }
                    lr = self.scheduler.get_last_lr()[0]

                    print(
                        f"Step {self.global_step:>6d}/{self.config.max_steps} | "
                        f"Loss: {avg_metrics['loss']:.4f} | "
                        f"Aux: {avg_metrics['aux_loss']:.4f} | "
                        f"Conf: {avg_metrics['mean_confidence']:.3f} | "
                        f"Grad: {grad_norm:.3f} | "
                        f"LR: {lr:.2e} | "
                        f"Elapsed: {elapsed:.1f}s"
                    )
                    step_metrics = []
                    start_time = time.time()

                # Checkpointing
                if self.global_step % self.config.save_every_steps == 0:
                    self.save_checkpoint()

            self.global_step += 1

        # Final save
        self.save_checkpoint("final")
        print("\nTraining complete!")

    def save_checkpoint(self, name: str = ""):
        """Save model checkpoint."""
        if not name:
            name = f"step_{self.global_step}"

        checkpoint_path = os.path.join(self.config.save_dir, f"architeckt_{name}.pt")

        checkpoint = {
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.model.config,
            "best_val_loss": self.best_val_loss,
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")

    @staticmethod
    def load_checkpoint(
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[object] = None,
        map_location: str = "cpu",
    ) -> int:
        """Load a checkpoint and return the step number."""
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return checkpoint.get("global_step", 0)
