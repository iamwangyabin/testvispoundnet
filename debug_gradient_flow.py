"""
Debug script to diagnose gradient flow issues in PoundNet GradCAM.
"""

import torch
import torch.nn as nn
import numpy as np
from networks.poundnet_detector import PoundNet
from utils.network_factory import get_model
import yaml

def check_gradient_flow(model, input_tensor):
    """Check gradient flow through the model."""
    print("=== GRADIENT FLOW DIAGNOSTIC ===")
    
    # Enable gradients for all parameters
    for name, param in model.named_parameters():
        param.requires_grad_(True)
        print(f"Parameter {name}: requires_grad={param.requires_grad}, shape={param.shape}")
    
    # Forward pass
    model.eval()
    input_tensor.requires_grad_(True)
    
    print(f"\nInput tensor requires_grad: {input_tensor.requires_grad}")
    
    # Hook to capture gradients at each layer
    gradient_info = {}
    
    def gradient_hook(name):
        def hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                grad_norm = grad_output[0].norm().item()
                gradient_info[name] = {
                    'grad_norm': grad_norm,
                    'grad_shape': grad_output[0].shape,
                    'grad_mean': grad_output[0].mean().item(),
                    'grad_std': grad_output[0].std().item()
                }
                print(f"Gradient at {name}: norm={grad_norm:.6f}, mean={grad_output[0].mean().item():.6f}")
            else:
                gradient_info[name] = {'grad_norm': 0.0, 'grad_output': None}
                print(f"Gradient at {name}: None")
        return hook
    
    # Register hooks on key layers
    hooks = []
    for name, module in model.named_modules():
        if 'image_encoder' in name and ('ln_2' in name or 'attn' in name or 'mlp' in name):
            hook = module.register_backward_hook(gradient_hook(name))
            hooks.append(hook)
    
    # Forward pass
    print("\n=== FORWARD PASS ===")
    output = model(input_tensor)
    
    if isinstance(output, dict):
        logits = output['logits']
    else:
        logits = output
    
    print(f"Logits shape: {logits.shape}")
    print(f"Logits requires_grad: {logits.requires_grad}")
    print(f"Logits values: {logits}")
    
    # Check intermediate activations
    print("\n=== CHECKING INTERMEDIATE ACTIVATIONS ===")
    activation_info = {}
    
    def activation_hook(name):
        def hook(module, input, output):
            if output is not None:
                activation_info[name] = {
                    'shape': output.shape,
                    'requires_grad': output.requires_grad,
                    'mean': output.mean().item(),
                    'std': output.std().item(),
                    'norm': output.norm().item()
                }
                print(f"Activation at {name}: requires_grad={output.requires_grad}, norm={output.norm().item():.6f}")
        return hook
    
    # Register activation hooks
    activation_hooks = []
    for name, module in model.named_modules():
        if 'image_encoder' in name and 'ln_2' in name:
            hook = module.register_forward_hook(activation_hook(name))
            activation_hooks.append(hook)
    
    # Re-run forward pass to capture activations
    model.zero_grad()
    output = model(input_tensor)
    if isinstance(output, dict):
        logits = output['logits']
    else:
        logits = output
    
    # Backward pass
    print("\n=== BACKWARD PASS ===")
    target_class = 0  # Real class
    class_score = logits[0, target_class]
    print(f"Class score: {class_score.item():.6f}")
    print(f"Class score requires_grad: {class_score.requires_grad}")
    
    # Check if class_score has grad_fn
    print(f"Class score grad_fn: {class_score.grad_fn}")
    
    # Backward pass
    try:
        class_score.backward(retain_graph=True)
        print("Backward pass completed successfully")
    except Exception as e:
        print(f"Backward pass failed: {e}")
        return
    
    # Check parameter gradients
    print("\n=== PARAMETER GRADIENTS ===")
    grad_params = 0
    zero_grad_params = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 1e-8:
                grad_params += 1
                print(f"Parameter {name}: grad_norm={grad_norm:.6f}")
            else:
                zero_grad_params += 1
        else:
            print(f"Parameter {name}: grad is None")
    
    print(f"\nSummary: {grad_params} parameters with non-zero gradients, {zero_grad_params} with zero gradients")
    
    # Clean up hooks
    for hook in hooks + activation_hooks:
        hook.remove()
    
    return gradient_info, activation_info

def main():
    """Main diagnostic function."""
    # Load configuration
    with open('cfgs/poundnet.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Convert to namespace-like object
    class Config:
        def __init__(self, d):
            for k, v in d.items():
                if isinstance(v, dict):
                    setattr(self, k, Config(v))
                else:
                    setattr(self, k, v)
    
    cfg = Config(cfg)
    
    # Create model
    print("Creating PoundNet model...")
    model = get_model(cfg)
    
    # Load checkpoint
    checkpoint_path = './weights/poundnet_ViTL_Progan_20240506_23_30_25.ckpt'
    print(f"Loading checkpoint: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("Checkpoint loaded successfully")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Model moved to {device}")
    
    # Create dummy input
    input_tensor = torch.randn(1, 3, 224, 224, device=device)
    
    # Run diagnostic
    gradient_info, activation_info = check_gradient_flow(model, input_tensor)
    
    print("\n=== DIAGNOSTIC COMPLETE ===")

if __name__ == "__main__":
    main()