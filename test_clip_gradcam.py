"""
Test script for CLIP GradCAM implementation.

This script tests the CLIP GradCAM functionality with a simple example
to ensure everything works correctly.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from visualization.gradcam.clip_gradcam import CLIPGradCAM


def create_test_image(size=(224, 224)):
    """Create a simple test image with geometric patterns."""
    # Create a test image with some patterns
    img = np.zeros((*size, 3), dtype=np.uint8)
    
    # Add some colored rectangles
    img[50:100, 50:100] = [255, 0, 0]  # Red square
    img[150:200, 150:200] = [0, 255, 0]  # Green square
    img[100:150, 100:150] = [0, 0, 255]  # Blue square
    
    # Add some circles
    center1 = (75, 175)
    center2 = (175, 75)
    
    y, x = np.ogrid[:size[0], :size[1]]
    mask1 = (x - center1[0])**2 + (y - center1[1])**2 <= 25**2
    mask2 = (x - center2[0])**2 + (y - center2[1])**2 <= 25**2
    
    img[mask1] = [255, 255, 0]  # Yellow circle
    img[mask2] = [255, 0, 255]  # Magenta circle
    
    return img


def test_clip_gradcam_basic():
    """Test basic CLIP GradCAM functionality."""
    print("Testing basic CLIP GradCAM functionality...")
    
    try:
        # Initialize CLIP GradCAM
        clip_gradcam = CLIPGradCAM(model_name="ViT-B/32")
        
        print("✓ CLIP GradCAM initialized successfully")
        
        # Print model info
        model_info = clip_gradcam.get_model_info()
        print(f"Model info: {model_info}")
        
        # Create test image
        test_img = create_test_image()
        
        # Convert to PIL Image and preprocess
        pil_img = Image.fromarray(test_img)
        img_tensor = clip_gradcam.preprocess(pil_img).unsqueeze(0)
        
        print("✓ Test image created and preprocessed")
        
        # Test with simple prompts
        test_prompts = ["a red square", "a blue square", "geometric shapes", "colorful pattern"]
        
        print(f"Testing with prompts: {test_prompts}")
        
        # Generate CAM for specific text
        result = clip_gradcam.generate_cam_for_text(
            image=img_tensor,
            text_prompts=test_prompts,
            target_prompt_idx=0  # Target "a red square"
        )
        
        print("✓ CAM generation successful")
        print(f"Target prompt: {result['target_prompt']}")
        print(f"Target similarity: {result['target_similarity']:.4f}")
        print(f"All similarities: {[f'{s:.4f}' for s in result['similarities']]}")
        
        # Test comparative analysis
        comparative_results = clip_gradcam.generate_comparative_cam(
            image=img_tensor,
            text_prompts=test_prompts
        )
        
        print("✓ Comparative CAM generation successful")
        
        metadata = comparative_results['metadata']
        print(f"Best match: {metadata['best_match_prompt']} (similarity: {metadata['similarities'][metadata['best_match_idx']]:.4f})")
        
        # Test with common classes
        class_results = clip_gradcam.analyze_with_common_classes(
            image=img_tensor,
            class_names=["square", "circle", "pattern", "shape", "color"]
        )
        
        print("✓ Common class analysis successful")
        
        class_metadata = class_results['metadata']
        print(f"Best class match: {class_metadata['best_match_class']} (similarity: {class_metadata['similarities'][class_metadata['best_match_idx']]:.4f})")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_clip_gradcam_with_real_image():
    """Test CLIP GradCAM with a real image if available."""
    print("\nTesting CLIP GradCAM with real image...")
    
    # Look for any image files in common locations
    possible_paths = [
        "test_image.jpg",
        "test_image.png",
        "sample.jpg",
        "sample.png",
        "data/test.jpg",
        "data/test.png"
    ]
    
    test_image_path = None
    for path in possible_paths:
        if os.path.exists(path):
            test_image_path = path
            break
    
    if test_image_path is None:
        print("No test image found, skipping real image test")
        return True
    
    try:
        print(f"Using test image: {test_image_path}")
        
        # Initialize CLIP GradCAM
        clip_gradcam = CLIPGradCAM(model_name="ViT-B/32")
        
        # Load and preprocess image
        img_tensor = clip_gradcam.preprocess_image(test_image_path)
        
        # Test with common prompts
        test_prompts = [
            "a person", "a face", "a man", "a woman",
            "a car", "a building", "a tree", "an animal",
            "real photo", "fake photo", "natural image"
        ]
        
        # Analyze with common classes
        results = clip_gradcam.analyze_with_common_classes(
            image=img_tensor,
            class_names=["person", "face", "car", "building", "tree", "animal", "object"]
        )
        
        metadata = results['metadata']
        print(f"✓ Real image analysis successful")
        print(f"Best match: {metadata['best_match_class']} (similarity: {metadata['similarities'][metadata['best_match_idx']]:.4f})")
        
        # Show top 3 matches
        similarities = metadata['similarities']
        sorted_indices = np.argsort(similarities)[::-1][:3]
        print("Top 3 matches:")
        for i, idx in enumerate(sorted_indices):
            class_name = metadata['class_names'][idx]
            similarity = similarities[idx]
            print(f"  {i+1}. {class_name}: {similarity:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Real image test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_clip_gradcam_visualization():
    """Test CLIP GradCAM visualization functionality."""
    print("\nTesting CLIP GradCAM visualization...")
    
    try:
        # Create output directory
        output_dir = "test_clip_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize CLIP GradCAM
        clip_gradcam = CLIPGradCAM(model_name="ViT-B/32")
        
        # Create test image
        test_img = create_test_image()
        pil_img = Image.fromarray(test_img)
        img_tensor = clip_gradcam.preprocess(pil_img).unsqueeze(0)
        
        # Save original test image
        pil_img.save(os.path.join(output_dir, "test_image.png"))
        
        # Generate CAM for a specific prompt
        result = clip_gradcam.generate_cam_for_text(
            image=img_tensor,
            text_prompts=["geometric shapes", "red square", "colorful pattern"],
            target_prompt_idx=0
        )
        
        # Create a simple visualization
        cam = result['cam']
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(test_img)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # CAM heatmap
        im = axes[1].imshow(cam, cmap='jet')
        axes[1].set_title(f"CAM: {result['target_prompt']}")
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        # Overlay
        heatmap = plt.cm.jet(cam)[:, :, :3]
        overlay = test_img * 0.6 + heatmap * 255 * 0.4
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        axes[2].imshow(overlay)
        axes[2].set_title(f"Overlay (Similarity: {result['target_similarity']:.3f})")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "clip_gradcam_test.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Visualization test successful, saved to {output_dir}")
        return True
        
    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all CLIP GradCAM tests."""
    print("=" * 60)
    print("CLIP GradCAM Test Suite")
    print("=" * 60)
    
    tests = [
        ("Basic Functionality", test_clip_gradcam_basic),
        ("Real Image Analysis", test_clip_gradcam_with_real_image),
        ("Visualization", test_clip_gradcam_visualization)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'-' * 40}")
        print(f"Running: {test_name}")
        print(f"{'-' * 40}")
        
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print(f"\n{'=' * 60}")
    print("Test Summary")
    print(f"{'=' * 60}")
    
    passed = 0
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("\n🎉 All tests passed! CLIP GradCAM is working correctly.")
    else:
        print(f"\n⚠️  {len(results) - passed} test(s) failed. Please check the errors above.")
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)