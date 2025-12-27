"""
Example script for generating GradCAM visualizations with PoundNet.

This script demonstrates how to use the PoundNet GradCAM implementation
to analyze deepfake detection decisions and visualize attention patterns.
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from visualization.gradcam import PoundNetGradCAM, visualize_gradcam, visualize_comparative_gradcam
from visualization.gradcam import visualize_multi_layer_analysis, save_gradcam_results
from visualization.gradcam.utils import preprocess_image_for_gradcam, create_attention_summary_plot
from utils.util import load_config_with_cli
from utils.network_factory import get_model
from utils.resume_tools import resume_lightning
import hydra


def load_poundnet_model(config_path: str, checkpoint_path: str = None):
    """
    Load PoundNet model from configuration and checkpoint.
    
    Args:
        config_path: Path to configuration file
        checkpoint_path: Path to model checkpoint (optional, uses config default)
        
    Returns:
        Loaded PoundNet model
    """
    print(f"Loading configuration from: {config_path}")
    conf = load_config_with_cli(config_path)
    conf = hydra.utils.instantiate(conf)
    
    print("Creating model...")
    model = get_model(conf)
    
    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_path = conf.resume.path
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    resume_lightning(model, checkpoint_path)
    
    model.eval()
    return model, conf


def analyze_single_image(
    model,
    image_path: str,
    output_dir: str,
    target_layers: list = None,
    save_individual: bool = True,
    save_comparative: bool = True,
    save_multilayer: bool = True
):
    """
    Analyze a single image with GradCAM.
    
    Args:
        model: PoundNet model
        image_path: Path to input image
        output_dir: Output directory for results
        target_layers: Specific layers to analyze
        save_individual: Whether to save individual class visualizations
        save_comparative: Whether to save comparative analysis
        save_multilayer: Whether to save multi-layer analysis
    """
    print(f"\nAnalyzing image: {image_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Preprocess image
    image_tensor, original_image = preprocess_image_for_gradcam(image_path)
    image_name = os.path.basename(image_path)
    base_name = os.path.splitext(image_name)[0]
    
    # Initialize GradCAM
    gradcam = PoundNetGradCAM(
        model=model,
        target_layers=target_layers,
        use_cuda=torch.cuda.is_available()
    )
    
    print("Model information:")
    model_info = gradcam.get_model_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # Generate prediction with CAM
    print("Generating prediction and GradCAM...")
    prediction_result = gradcam.generate_prediction_with_cam(image_tensor)
    
    print(f"Prediction: {prediction_result['predicted_class_name']}")
    print(f"Confidence: {prediction_result['confidence']:.4f}")
    print(f"Real probability: {prediction_result['real_prob']:.4f}")
    print(f"Fake probability: {prediction_result['fake_prob']:.4f}")
    
    # Save prediction results
    save_gradcam_results(
        prediction_result,
        output_dir,
        f"{base_name}_prediction",
        save_formats=['png', 'npy', 'json']
    )
    
    # Individual class visualizations
    if save_individual:
        print("Generating individual class visualizations...")
        
        # Real class
        real_cam = gradcam.generate_cam(image_tensor, target_class='real')
        fig_real = visualize_gradcam(
            original_image,
            real_cam,
            title=f"GradCAM - Real Class (Prob: {prediction_result['real_prob']:.3f})",
            save_path=os.path.join(output_dir, f"{base_name}_real_gradcam.png")
        )
        
        # Fake class
        fake_cam = gradcam.generate_cam(image_tensor, target_class='fake')
        fig_fake = visualize_gradcam(
            original_image,
            fake_cam,
            title=f"GradCAM - Fake Class (Prob: {prediction_result['fake_prob']:.3f})",
            save_path=os.path.join(output_dir, f"{base_name}_fake_gradcam.png")
        )
    
    # Comparative analysis
    if save_comparative:
        print("Generating comparative analysis...")
        comparative_cams = gradcam.generate_comparative_cam(image_tensor)
        
        fig_comp = visualize_comparative_gradcam(
            original_image,
            comparative_cams['real'],
            comparative_cams['fake'],
            prediction_result,
            title=f"Comparative GradCAM Analysis - {image_name}",
            save_path=os.path.join(output_dir, f"{base_name}_comparative.png")
        )
    
    # Multi-layer analysis
    if save_multilayer:
        print("Generating multi-layer analysis...")
        layer_results = gradcam.analyze_attention_patterns(image_tensor)
        
        if layer_results:
            fig_multi = visualize_multi_layer_analysis(
                original_image,
                layer_results,
                title=f"Multi-Layer Analysis - {image_name}",
                save_path=os.path.join(output_dir, f"{base_name}_multilayer.png")
            )
            
            # Create attention summary
            fig_summary = create_attention_summary_plot(
                layer_results,
                save_path=os.path.join(output_dir, f"{base_name}_attention_summary.png")
            )
    
    print(f"Results saved to: {output_dir}")
    return prediction_result


def batch_analyze_images(
    model,
    input_dir: str,
    output_dir: str,
    target_layers: list = None,
    image_extensions: list = ['.jpg', '.jpeg', '.png', '.bmp'],
    max_images: int = None
):
    """
    Analyze multiple images in batch.
    
    Args:
        model: PoundNet model
        input_dir: Directory containing input images
        output_dir: Output directory for results
        target_layers: Specific layers to analyze
        image_extensions: Valid image file extensions
        max_images: Maximum number of images to process
    """
    print(f"\nBatch analyzing images from: {input_dir}")
    
    # Find all image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(input_dir).glob(f"*{ext}"))
        image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
    
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    results_summary = []
    
    for i, image_path in enumerate(image_files):
        print(f"\nProcessing {i+1}/{len(image_files)}: {image_path.name}")
        
        try:
            # Create individual output directory
            image_output_dir = os.path.join(output_dir, image_path.stem)
            
            # Analyze image
            result = analyze_single_image(
                model,
                str(image_path),
                image_output_dir,
                target_layers=target_layers,
                save_individual=True,
                save_comparative=True,
                save_multilayer=False  # Skip for batch to save time
            )
            
            # Add to summary
            results_summary.append({
                'image_name': image_path.name,
                'predicted_class': result['predicted_class_name'],
                'confidence': result['confidence'],
                'real_prob': result['real_prob'],
                'fake_prob': result['fake_prob']
            })
            
        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")
            continue
    
    # Save batch summary
    import json
    summary_path = os.path.join(output_dir, "batch_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nBatch analysis complete. Summary saved to: {summary_path}")
    return results_summary


def main():
    parser = argparse.ArgumentParser(description="Generate GradCAM visualizations for PoundNet")
    parser.add_argument('--config', type=str, required=True,
                       help='Path to PoundNet configuration file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (optional)')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to single image for analysis')
    parser.add_argument('--input_dir', type=str, default=None,
                       help='Directory containing images for batch analysis')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--target_layers', type=str, nargs='+', default=None,
                       help='Specific layers to analyze (e.g., image_encoder.transformer.resblocks.23.ln_2)')
    parser.add_argument('--max_images', type=int, default=None,
                       help='Maximum number of images to process in batch mode')
    parser.add_argument('--individual', action='store_true',
                       help='Save individual class visualizations')
    parser.add_argument('--comparative', action='store_true',
                       help='Save comparative analysis')
    parser.add_argument('--multilayer', action='store_true',
                       help='Save multi-layer analysis')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.image and not args.input_dir:
        parser.error("Either --image or --input_dir must be specified")
    
    if args.image and args.input_dir:
        parser.error("Cannot specify both --image and --input_dir")
    
    # Load model
    print("Loading PoundNet model...")
    model, config = load_poundnet_model(args.config, args.checkpoint)
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print("Model moved to GPU")
    
    # Single image analysis
    if args.image:
        analyze_single_image(
            model=model,
            image_path=args.image,
            output_dir=args.output_dir,
            target_layers=args.target_layers,
            save_individual=args.individual,
            save_comparative=args.comparative,
            save_multilayer=args.multilayer
        )
    
    # Batch analysis
    elif args.input_dir:
        batch_analyze_images(
            model=model,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            target_layers=args.target_layers,
            max_images=args.max_images
        )
    
    print("\nGradCAM analysis complete!")


if __name__ == "__main__":
    main()