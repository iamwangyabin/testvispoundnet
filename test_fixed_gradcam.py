"""
Test script for the fixed PoundNet GradCAM implementation.
This tests the attention-based visualization approach.
"""

import torch
import numpy as np
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from visualization.gradcam.poundnet_gradcam_fixed import PoundNetGradCAM
from networks.poundnet_detector import PoundNet
from omegaconf import OmegaConf

def create_test_image():
    """Create a test image tensor."""
    # Create a simple test pattern
    image = torch.randn(1, 3, 224, 224)
    return image

def load_poundnet_model():
    """Load PoundNet model for testing."""
    print("Loading PoundNet model...")
    
    # Load configuration
    cfg = OmegaConf.load('./cfgs/poundnet.yaml')
    
    # Create model
    model = PoundNet(cfg)
    model.eval()
    
    print("Model loaded successfully")
    return model

def test_fixed_gradcam():
    """Test the fixed GradCAM implementation."""
    print("=== Testing Fixed GradCAM Implementation ===")
    
    # Create test image
    print("Creating test image...")
    test_tensor = create_test_image()
    print(f"Test tensor shape: {test_tensor.shape}")
    
    # Load model
    model = load_poundnet_model()
    
    # Create GradCAM instance
    print("Creating PoundNetGradCAM...")
    gradcam = PoundNetGradCAM(model)
    print("GradCAM created successfully")
    
    # Test CAM generation
    print("Testing CAM generation...")
    
    print("Generating CAM for 'real' class...")
    real_cam = gradcam.generate_cam(test_tensor, target_class='real')
    print(f"Real CAM shape: {real_cam.shape}")
    print(f"Real CAM stats - min: {real_cam.min():.6f}, max: {real_cam.max():.6f}, mean: {real_cam.mean():.6f}")
    
    print("Generating CAM for 'fake' class...")
    fake_cam = gradcam.generate_cam(test_tensor, target_class='fake')
    print(f"Fake CAM shape: {fake_cam.shape}")
    print(f"Fake CAM stats - min: {fake_cam.min():.6f}, max: {fake_cam.max():.6f}, mean: {fake_cam.mean():.6f}")
    
    # Check if CAMs are meaningful (not all zeros)
    real_has_variation = real_cam.max() - real_cam.min() > 1e-6
    fake_has_variation = fake_cam.max() - fake_cam.min() > 1e-6
    
    print(f"\n=== Results ===")
    print(f"Real CAM has variation: {real_has_variation}")
    print(f"Fake CAM has variation: {fake_has_variation}")
    
    if real_has_variation and fake_has_variation:
        print("✅ SUCCESS: Fixed GradCAM generates meaningful activation maps!")
        return True
    else:
        print("❌ FAILURE: GradCAM still generates zero or uniform activation maps")
        return False

if __name__ == "__main__":
    try:
        success = test_fixed_gradcam()
        if success:
            print("\n💥 Fixed GradCAM implementation works correctly!")
        else:
            print("\n💥 Fixed GradCAM implementation still has issues!")
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()