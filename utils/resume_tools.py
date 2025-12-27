import torch
import timm
import torchvision
import logging

def resume_lightning(model, weight_path):
    print(f"[DEBUG] Loading checkpoint from: {weight_path}")
    print(f"[DEBUG] PyTorch version: {torch.__version__}")
    
    try:
        # First attempt with default settings (will fail in PyTorch 2.6+)
        print("[DEBUG] Attempting to load with default torch.load settings...")
        state_dict = torch.load(weight_path, map_location='cpu')['state_dict']
        print("[DEBUG] Successfully loaded with default settings")
    except Exception as e:
        print(f"[DEBUG] Default loading failed: {e}")
        print("[DEBUG] Attempting to load with weights_only=False...")
        
        # Fallback: use weights_only=False for compatibility
        state_dict = torch.load(weight_path, map_location='cpu', weights_only=False)['state_dict']
        print("[DEBUG] Successfully loaded with weights_only=False")
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.'):
            new_key = key[6:]  # remove `model.` from key
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    model.load_state_dict(new_state_dict)




