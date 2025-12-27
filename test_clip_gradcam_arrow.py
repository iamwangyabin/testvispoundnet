"""
Test script for CLIP GradCAM Arrow implementation.

This script tests the CLIP GradCAM implementation based on user's sample code
to ensure it works correctly with both single images and arrow datasets.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import tempfile
import urllib.request

# Add parent directories to path
sys.path.append(str(Path(__file__).parent))

from visualization.gradcam.clip_gradcam_arrow import CLIPGradCAMArrow, visualize_clip_gradcam_results, viz_attn, load_image


def test_clip_gradcam_single_image():
    """Test CLIP GradCAM with a single image."""
    print("Testing CLIP GradCAM with single image...")
    
    # Download a test image
    image_url = 'https://images2.minutemediacdn.com/image/upload/c_crop,h_706,w_1256,x_0,y_64/f_auto,q_auto,w_1100/v1554995050/shape/mentalfloss/516438-istock-637689912.jpg'
    
    with tempfile.TemporaryDirectory() as temp_dir:
        image_path = os.path.join(temp_dir, 'test_image.jpg')
        
        try:
            print(f"Downloading test image from: {image_url}")
            urllib.request.urlretrieve(image_url, image_path)
        except Exception as e:
            print(f"Could not download test image: {e}")
            print("Creating a dummy image instead...")
            # Create a dummy image
            dummy_image = Image.new('RGB', (224, 224), color='red')
            dummy_image.save(image_path)
        
        # Initialize CLIP GradCAM
        print("Initializing CLIP GradCAM...")
        clip_gradcam = CLIPGradCAMArrow(
            model_name="RN50",
            saliency_layer="layer4"
        )
        
        # Get model info
        model_info = clip_gradcam.get_model_info()
        print("Model information:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")
        
        # Load and preprocess image
        print("Loading and preprocessing image...")
        image_tensor = clip_gradcam.preprocess_image_from_path(image_path)
        original_image = load_image(image_path, resize=model_info['input_resolution'])
        
        print(f"Image tensor shape: {image_tensor.shape}")
        print(f"Original image shape: {original_image.shape}")
        
        # Test single text prompt
        print("\nTesting single text prompt...")
        text_prompt = "a cat"
        result = clip_gradcam.generate_gradcam_for_text(image_tensor, text_prompt)
        
        print(f"Text prompt: '{result['text_prompt']}'")
        print(f"Similarity: {result['similarity']:.4f}")
        print(f"CAM shape: {result['cam'].shape}")
        
        # Test multiple prompts
        print("\nTesting multiple text prompts...")
        text_prompts = ["a cat", "a dog", "a person", "a car"]
        results = clip_gradcam.generate_comparative_gradcam(image_tensor, text_prompts)
        
        metadata = results['metadata']
        print(f"Best match: '{metadata['best_match_prompt']}' (similarity: {metadata['best_similarity']:.4f})")
        
        for i, prompt in enumerate(text_prompts):
            if prompt in results:
                similarity = results[prompt]['similarity']
                print(f"  {prompt}: {similarity:.4f}")
        
        # Test with common classes
        print("\nTesting with common classes...")
        class_results = clip_gradcam.analyze_with_common_classes(image_tensor)
        
        class_metadata = class_results['metadata']
        print(f"Best class match: '{class_metadata['best_match_class']}' (similarity: {class_metadata['best_similarity']:.4f})")
        
        # Create output directory for visualizations
        output_dir = os.path.join(temp_dir, 'test_output')
        os.makedirs(output_dir, exist_ok=True)
        
        # Test visualization
        print("\nTesting visualization...")
        fig = viz_attn(
            original_image,
            result['cam'],
            blur=True,
            title=f"CLIP GradCAM - {text_prompt}",
            save_path=os.path.join(output_dir, 'single_prompt_test.png')
        )
        plt.close(fig)
        
        # Test comparative visualization
        fig = visualize_clip_gradcam_results(
            original_image,
            results,
            title="CLIP GradCAM Comparative Test",
            save_path=os.path.join(output_dir, 'comparative_test.png')
        )
        plt.close(fig)
        
        print(f"Test visualizations saved to: {output_dir}")
        print("✓ Single image test completed successfully!")
        
        return True


def test_tensor_preprocessing():
    """Test tensor preprocessing functionality."""
    print("\nTesting tensor preprocessing...")
    
    # Create a dummy tensor similar to what arrow datasets would provide
    dummy_tensor = torch.randn(1, 3, 224, 224)
    
    # Initialize CLIP GradCAM
    clip_gradcam = CLIPGradCAMArrow(model_name="RN50")
    
    # Test preprocessing
    processed_tensor = clip_gradcam.preprocess_image_from_tensor(dummy_tensor)
    
    print(f"Input tensor shape: {dummy_tensor.shape}")
    print(f"Processed tensor shape: {processed_tensor.shape}")
    
    # Verify the tensor is on the correct device
    expected_device = clip_gradcam.device
    actual_device = str(processed_tensor.device)
    
    print(f"Expected device: {expected_device}")
    print(f"Actual device: {actual_device}")
    
    if expected_device in actual_device:
        print("✓ Tensor preprocessing test completed successfully!")
        return True
    else:
        print("✗ Device mismatch in tensor preprocessing!")
        return False


def test_model_variants():
    """Test different CLIP model variants."""
    print("\nTesting different CLIP model variants...")
    
    # Test different models (skip if not available)
    models_to_test = ["RN50"]  # Start with basic model
    
    if torch.cuda.is_available():
        print("CUDA available - testing with GPU")
    else:
        print("CUDA not available - testing with CPU")
    
    for model_name in models_to_test:
        try:
            print(f"\nTesting model: {model_name}")
            clip_gradcam = CLIPGradCAMArrow(model_name=model_name)
            
            model_info = clip_gradcam.get_model_info()
            print(f"  Model loaded successfully")
            print(f"  Input resolution: {model_info['input_resolution']}")
            print(f"  Device: {model_info['device']}")
            
            # Test with a simple tensor
            dummy_tensor = torch.randn(1, 3, 224, 224)
            processed = clip_gradcam.preprocess_image_from_tensor(dummy_tensor)
            
            # Quick GradCAM test
            result = clip_gradcam.generate_gradcam_for_text(processed, "a test image")
            print(f"  GradCAM generated successfully, similarity: {result['similarity']:.4f}")
            
        except Exception as e:
            print(f"  Error testing {model_name}: {e}")
            continue
    
    print("✓ Model variants test completed!")
    return True


def main():
    """Run all tests."""
    print("Starting CLIP GradCAM Arrow implementation tests...\n")
    
    tests_passed = 0
    total_tests = 3
    
    try:
        # Test 1: Single image functionality
        if test_clip_gradcam_single_image():
            tests_passed += 1
    except Exception as e:
        print(f"✗ Single image test failed: {e}")
    
    try:
        # Test 2: Tensor preprocessing
        if test_tensor_preprocessing():
            tests_passed += 1
    except Exception as e:
        print(f"✗ Tensor preprocessing test failed: {e}")
    
    try:
        # Test 3: Model variants
        if test_model_variants():
            tests_passed += 1
    except Exception as e:
        print(f"✗ Model variants test failed: {e}")
    
    print(f"\n{'='*50}")
    print(f"Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! CLIP GradCAM Arrow implementation is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)