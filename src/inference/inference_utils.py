"""
Inference utilities for Architeckt.

Provides helpers for:
- 8-bit quantization (bitsandbytes) to fit 11B models in 24GB VRAM
- Safe loading of checkpoints
"""

import torch
import torch.nn as nn
from typing import Optional

def load_for_inference(model: nn.Module, checkpoint_path: str, use_8bit: bool = True) -> nn.Module:
    """
    Load an Architeckt model for inference, optionally using 8-bit quantization.
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
        
    if use_8bit:
        try:
            import bitsandbytes as bnb
            print("Converting linear layers to 8-bit precision (bitsandbytes)...")
            
            # Helper to recursively replace Linear layers with Linear8bitLt
            def replace_8bit_linear(module: nn.Module):
                for name, child in module.named_children():
                    if isinstance(child, nn.Linear):
                        has_bias = child.bias is not None
                        new_layer = bnb.nn.Linear8bitLt(
                            child.in_features, 
                            child.out_features, 
                            bias=has_bias, 
                            has_fp16_weights=False,
                            threshold=6.0  # Optional outlier threshold
                        )
                        setattr(module, name, new_layer)
                    else:
                        replace_8bit_linear(child)
                        
            # Replace linear layers with 8-bit versions BEFORE loading weights
            replace_8bit_linear(model)
            
        except ImportError:
            print("Warning: bitsandbytes not found. Falling back to standard precision.")
            use_8bit = False
            
    # Load weights into the model
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if unexpected_keys:
        print(f"Warning: Unexpected keys found in state_dict: {unexpected_keys[:5]}...")
        
    # Move to GPU if not 8-bit (8-bit linear might require specific handling or is already on GPU)
    if torch.cuda.is_available():
        model = model.cuda()
        
    model.eval()
    print("Model loaded successfully for inference.")
    return model
