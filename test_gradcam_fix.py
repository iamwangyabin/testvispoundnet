#!/usr/bin/env python3
"""
Test script to verify the GradCAM gradient fix.
"""

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import sys
import os

# Add the project root to the path
sys.path.append('.')

from networks.poundnet_detector import PoundNet
from visualization.gradcam.poundnet_gradcam import PoundNetGradCAM
from utils.network_factory import get_model
from utils.resume_tools import resume_lightning
from utils.util import load_config

def create_test_image():
    """Create a test image tensor."""
    # Create a simple test image (224x224x3)
    test_image = np.random.rand(224, 224, 3) * 255
    test_image = test_image.astype(np.uint8)
    
    # Convert to PIL and then to tensor
    pil_image = Image.fromarray(test_image)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tensor = transform(pil_image).unsqueeze(0)  # Add batch dimension
    return tensor

def test_gradcam_fix():
    """Test the GradCAM fix with a simple example."""
    print("=== Testing GradCAM Fix ===")
    
    # Create test image
    print("Creating test image...")
    test_tensor = create_test_image()
    print(f"Test tensor shape: {test_tensor.shape}")
    
    # Load model
    print("Loading PoundNet model...")
    try:
        # Load config
        config = load_config('cfgs/poundnet.yaml')
        
        # Create model
        model = get_model(config)
        
        # Load checkpoint
        checkpoint_path = './weights/poundnet_ViTL_Progan_20240506_23_30_25.ckpt'
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint: {checkpoint_path}")
            resume_lightning(model, checkpoint_path)
        else:
            print("Warning: Checkpoint not found, using random weights")
        
        model.eval()
        print("Model loaded successfully")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Create GradCAM
    print("Creating PoundNetGradCAM...")
    try:
        gradcam = PoundNetGradCAM(model, use_cuda=torch.cuda.is_available())
        print("GradCAM created successfully")
        
    except Exception as e:
        print(f"Error creating GradCAM: {e}")
        return False
    
    # Test GradCAM generation
    print("Testing GradCAM generation...")
    try:
        # Generate CAM for 'real' class
        print("Generating CAM for 'real' class...")
        real_cam = gradcam.generate_cam(test_tensor, target_class='real')
        print(f"Real CAM shape: {real_cam.shape}")
        print(f"Real CAM stats - min: {real_cam.min():.6f}, max: {real_cam.max():.6f}, mean: {real_cam.mean():.6f}")
        
        # Generate CAM for 'fake' class
        print("Generating CAM for 'fake' class...")
        fake_cam = gradcam.generate_cam(test_tensor, target_class='fake')
        print(f"Fake CAM shape: {fake_cam.shape}")
        print(f"Fake CAM stats - min: {fake_cam.min():.6f}, max: {fake_cam.max():.6f}, mean: {fake_cam.mean():.6f}")
        
        # Check if gradients are non-zero
        if real_cam.max() > 1e-6 and fake_cam.max() > 1e-6:
            print("✅ SUCCESS: GradCAM is generating non-zero activation maps!")
            print("✅ The gradient flow issue has been fixed!")
            return True
        else:
            print("❌ FAILURE: GradCAM is still generating zero or near-zero activation maps")
            print("❌ The gradient flow issue persists")
            return False
            
    except Exception as e:
        print(f"Error during GradCAM generation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gradcam_fix()
    if success:
        print("\n🎉 GradCAM fix verification PASSED!")
    else:
        print("\n💥 GradCAM fix verification FAILED!")
    
    sys.exit(0 if success else 1)