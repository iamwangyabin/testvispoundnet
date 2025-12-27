#!/usr/bin/env python3
"""
Test script to verify that the fixed GradCAM implementation generates
different activations for real and fake classes.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from visualization.gradcam.poundnet_gradcam_fixed import PoundNetGradCAM
from utils.util import load_config_with_cli
from utils.network_factory import get_model
from utils.resume_tools import resume_lightning
import hydra


def test_gradcam_class_differentiation():
    """Test that GradCAM generates different activations for real vs fake classes."""
    
    print("Testing GradCAM class differentiation...")
    
    # Load model (you'll need to adjust paths as needed)
    config_path = "cfgs/poundnet.yaml"
    
    try:
        print(f"Loading configuration from: {config_path}")
        conf = load_config_with_cli(config_path)
        conf = hydra.utils.instantiate(conf)
        
        print("Creating model...")
        model = get_model(conf)
        
        # Load checkpoint if available
        if hasattr(conf, 'resume') and hasattr(conf.resume, 'path'):
            print(f"Loading checkpoint from: {conf.resume.path}")
            resume_lightning(model, conf.resume.path)
        
        model.eval()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            print("Model moved to GPU")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Creating a dummy model for testing...")
        
        # Create a simple dummy model for testing the logic
        class DummyPoundNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.image_encoder = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 64, 3, padding=1),
                    torch.nn.AdaptiveAvgPool2d(1),
                    torch.nn.Flatten()
                )
                self.classifier = torch.nn.Linear(64, 2)
                
                # Add dummy config
                class DummyConfig:
                    class DummyModel:
                        N_CTX_VISION = 16
                    model = DummyModel()
                self.cfg = DummyConfig()
            
            def forward(self, x):
                features = self.image_encoder(x)
                logits = self.classifier(features)
                return {'logits': logits}
            
            def named_modules(self):
                # Return some dummy modules for hook registration
                modules = []
                for name, module in super().named_modules():
                    if 'image_encoder' in name:
                        modules.append((f"image_encoder.transformer.resblocks.23.ln_2", module))
                return modules
        
        model = DummyPoundNet()
        if torch.cuda.is_available():
            model = model.cuda()
    
    # Create test input
    batch_size = 1
    channels = 3
    height = width = 224
    
    # Create a test image with some structure
    test_input = torch.randn(batch_size, channels, height, width)
    if torch.cuda.is_available():
        test_input = test_input.cuda()
    
    print(f"Test input shape: {test_input.shape}")
    
    # Initialize GradCAM
    try:
        gradcam = PoundNetGradCAM(
            model=model,
            target_layers=None,  # Use default layers
            use_cuda=torch.cuda.is_available()
        )
        
        print("GradCAM initialized successfully")
        
        # Test model forward pass
        with torch.no_grad():
            output = model(test_input)
            if isinstance(output, dict):
                logits = output['logits']
            else:
                logits = output
            
            probabilities = torch.softmax(logits, dim=1)
            print(f"Model output logits: {logits}")
            print(f"Model probabilities: {probabilities}")
        
        # Generate CAMs for both classes
        print("\nGenerating CAMs...")
        
        real_cam = gradcam.generate_cam(test_input, target_class='real')
        fake_cam = gradcam.generate_cam(test_input, target_class='fake')
        
        print(f"Real CAM shape: {real_cam.shape}")
        print(f"Fake CAM shape: {fake_cam.shape}")
        
        # Analyze differences
        real_stats = {
            'mean': np.mean(real_cam),
            'std': np.std(real_cam),
            'min': np.min(real_cam),
            'max': np.max(real_cam),
            'sum': np.sum(real_cam)
        }
        
        fake_stats = {
            'mean': np.mean(fake_cam),
            'std': np.std(fake_cam),
            'min': np.min(fake_cam),
            'max': np.max(fake_cam),
            'sum': np.sum(fake_cam)
        }
        
        print(f"\nReal CAM statistics:")
        for key, value in real_stats.items():
            print(f"  {key}: {value:.6f}")
        
        print(f"\nFake CAM statistics:")
        for key, value in fake_stats.items():
            print(f"  {key}: {value:.6f}")
        
        # Check if CAMs are different
        cam_difference = np.abs(real_cam - fake_cam)
        max_difference = np.max(cam_difference)
        mean_difference = np.mean(cam_difference)
        
        print(f"\nCAM Differences:")
        print(f"  Max absolute difference: {max_difference:.6f}")
        print(f"  Mean absolute difference: {mean_difference:.6f}")
        
        # Test comparative CAM generation
        print(f"\nTesting comparative CAM generation...")
        comparative_cams = gradcam.generate_comparative_cam(test_input)
        
        comp_real_cam = comparative_cams['real']
        comp_fake_cam = comparative_cams['fake']
        
        comp_difference = np.abs(comp_real_cam - comp_fake_cam)
        comp_max_diff = np.max(comp_difference)
        comp_mean_diff = np.mean(comp_difference)
        
        print(f"Comparative CAM differences:")
        print(f"  Max absolute difference: {comp_max_diff:.6f}")
        print(f"  Mean absolute difference: {comp_mean_diff:.6f}")
        
        # Determine if fix is working
        threshold = 1e-6  # Minimum difference threshold
        
        if max_difference > threshold and comp_max_diff > threshold:
            print(f"\n✅ SUCCESS: GradCAM generates different activations for real and fake classes!")
            print(f"   Individual CAMs differ by max: {max_difference:.6f}")
            print(f"   Comparative CAMs differ by max: {comp_max_diff:.6f}")
            return True
        else:
            print(f"\n❌ ISSUE: GradCAM still generates identical or nearly identical activations")
            print(f"   Individual CAMs differ by max: {max_difference:.6f}")
            print(f"   Comparative CAMs differ by max: {comp_max_diff:.6f}")
            print(f"   Threshold: {threshold}")
            return False
            
    except Exception as e:
        print(f"Error during GradCAM testing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_gradcam_class_differentiation()
    if success:
        print("\n🎉 Test passed! The GradCAM fix is working correctly.")
    else:
        print("\n⚠️  Test failed! The GradCAM issue may still exist.")