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
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, Any

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from visualization.gradcam.poundnet_gradcam_fixed import PoundNetGradCAM
from visualization.gradcam import visualize_gradcam, visualize_comparative_gradcam
from visualization.gradcam import visualize_multi_layer_analysis, save_gradcam_results
from visualization.gradcam.utils import preprocess_image_for_gradcam, create_attention_summary_plot
from utils.util import load_config_with_cli
from utils.network_factory import get_model
from utils.resume_tools import resume_lightning
import hydra
import utils  # Add this import for eval() to find utils module
import data


def analyze_arrow_dataset(
    model,
    dataset_config: Dict[str, Any],
    output_dir: str,
    target_layers: list = None,
    max_images_per_subset: int = None,
    save_individual: bool = True,
    save_comparative: bool = True,
    save_multilayer: bool = False
):
    """
    Analyze Arrow dataset with full GradCAM visualizations.
    
    Args:
        model: PoundNet model
        dataset_config: Dataset configuration from config file
        output_dir: Output directory for results
        target_layers: Specific layers to analyze
        max_images_per_subset: Maximum images per subset
        save_individual: Whether to save individual class visualizations
        save_comparative: Whether to save comparative analysis
        save_multilayer: Whether to save multi-layer analysis
    """
    print(f"Analyzing Arrow datasets with full visualizations...")
    print(f"Found {len(dataset_config['source'])} dataset sources")
    
    all_results = []
    
    # Process each dataset source (same as batch_process_arrow.py)
    for source_idx, sub_data in enumerate(dataset_config['source']):
        print(f"\nProcessing dataset {source_idx + 1}/{len(dataset_config['source'])}")
        print(f"Target: {sub_data['target']}")
        print(f"Data root: {sub_data['data_root']}")
        print(f"Benchmark: {sub_data['benchmark_name']}")
        
        # Process each subset in the source
        for subset_idx, sub_set in enumerate(sub_data['sub_sets']):
            print(f"\n  Processing subset: {sub_set}")
            
            try:
                # Create dataset using the same method as test.py
                dataset = eval(sub_data['target'])(
                    sub_data['data_root'],
                    dataset_config['trsf'],
                    subset=sub_set,
                    split=sub_data['split']
                )
                
                # Create data loader
                data_loader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=1,  # Process one image at a time for GradCAM
                    num_workers=0,  # Avoid multiprocessing issues with GradCAM
                    shuffle=False
                )
                
                # Create subset output directory
                subset_output_dir = os.path.join(output_dir, sub_data['benchmark_name'], sub_set)
                os.makedirs(subset_output_dir, exist_ok=True)
                
                # Process images with full visualizations
                subset_results = _analyze_arrow_subset_with_visualizations(
                    model=model,
                    data_loader=data_loader,
                    subset_name=f"{sub_data['benchmark_name']}_{sub_set}",
                    output_dir=subset_output_dir,
                    target_layers=target_layers,
                    max_images=max_images_per_subset,
                    save_individual=save_individual,
                    save_comparative=save_comparative,
                    save_multilayer=save_multilayer
                )
                
                all_results.extend(subset_results)
                print(f"    Generated visualizations for {len(subset_results)} images")
                
            except Exception as e:
                print(f"    Error processing subset {sub_set}: {e}")
                continue
    
    # Save batch summary
    summary_path = os.path.join(output_dir, "arrow_gradcam_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nArrow dataset GradCAM analysis complete!")
    print(f"Total visualizations generated: {len(all_results)}")
    print(f"Summary saved to: {summary_path}")
    
    return all_results


def _analyze_arrow_subset_with_visualizations(
    model,
    data_loader,
    subset_name: str,
    output_dir: str,
    target_layers: list = None,
    max_images: int = None,
    save_individual: bool = True,
    save_comparative: bool = True,
    save_multilayer: bool = False
):
    """Process Arrow dataset subset with full GradCAM visualizations."""
    # Initialize GradCAM
    gradcam = PoundNetGradCAM(
        model=model,
        target_layers=target_layers,
        use_cuda=torch.cuda.is_available()
    )
    
    results = []
    
    print(f"    Generating visualizations for {len(data_loader)} images from {subset_name}")
    
    # Process images with progress bar
    for idx, (img_tensor, label) in enumerate(tqdm(data_loader, desc=f"  {subset_name}")):
        if max_images and idx >= max_images:
            break
        
        try:
            # Move to device
            if torch.cuda.is_available():
                img_tensor = img_tensor.cuda()
            
            # Convert tensor back to image for visualization
            original_image = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            # Denormalize (assuming CLIP normalization)
            mean = np.array([0.48145466, 0.4578275, 0.40821073])
            std = np.array([0.26862954, 0.26130258, 0.27577711])
            original_image = original_image * std + mean
            original_image = np.clip(original_image * 255, 0, 255).astype(np.uint8)
            
            # Generate prediction with CAM
            prediction_result = gradcam.generate_prediction_with_cam(img_tensor)
            
            # Create image-specific output directory
            image_output_dir = os.path.join(output_dir, f"image_{idx:06d}")
            os.makedirs(image_output_dir, exist_ok=True)
            
            # Save prediction results
            save_gradcam_results(
                prediction_result,
                image_output_dir,
                f"image_{idx:06d}_prediction",
                save_formats=['png', 'npy', 'json']
            )
            
            # Individual class visualizations
            if save_individual:
                # Real class
                real_cam = gradcam.generate_cam(img_tensor, target_class='real')
                fig_real = visualize_gradcam(
                    original_image,
                    real_cam,
                    title=f"GradCAM - Real Class (Prob: {prediction_result['real_prob']:.3f})",
                    save_path=os.path.join(image_output_dir, f"image_{idx:06d}_real_gradcam.png")
                )
                plt.close(fig_real)
                
                # Fake class
                fake_cam = gradcam.generate_cam(img_tensor, target_class='fake')
                fig_fake = visualize_gradcam(
                    original_image,
                    fake_cam,
                    title=f"GradCAM - Fake Class (Prob: {prediction_result['fake_prob']:.3f})",
                    save_path=os.path.join(image_output_dir, f"image_{idx:06d}_fake_gradcam.png")
                )
                plt.close(fig_fake)
            
            # Comparative analysis
            if save_comparative:
                comparative_cams = gradcam.generate_comparative_cam(img_tensor)
                fig_comp = visualize_comparative_gradcam(
                    original_image,
                    comparative_cams['real'],
                    comparative_cams['fake'],
                    prediction_result,
                    title=f"Comparative GradCAM - Image {idx:06d}",
                    save_path=os.path.join(image_output_dir, f"image_{idx:06d}_comparative.png")
                )
                plt.close(fig_comp)
            
            # Multi-layer analysis
            if save_multilayer:
                layer_results = gradcam.analyze_attention_patterns(img_tensor)
                if layer_results:
                    fig_multi = visualize_multi_layer_analysis(
                        original_image,
                        layer_results,
                        title=f"Multi-Layer Analysis - Image {idx:06d}",
                        save_path=os.path.join(image_output_dir, f"image_{idx:06d}_multilayer.png")
                    )
                    plt.close(fig_multi)
            
            # Add metadata
            result_summary = {
                'image_index': idx,
                'true_label': label.item(),
                'subset_name': subset_name,
                'predicted_class': prediction_result['predicted_class_name'],
                'confidence': prediction_result['confidence'],
                'real_prob': prediction_result['real_prob'],
                'fake_prob': prediction_result['fake_prob'],
                'predicted_correct': (prediction_result['predicted_class_idx'] == label.item()),
                'output_dir': image_output_dir
            }
            
            results.append(result_summary)
            
        except Exception as e:
            print(f"      Error processing image {idx}: {e}")
            continue
    
    return results


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
    parser.add_argument('--arrow_dataset', action='store_true',
                       help='Process Arrow dataset from config file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--target_layers', type=str, nargs='+', default=None,
                       help='Specific layers to analyze (e.g., image_encoder.transformer.resblocks.23.ln_2)')
    parser.add_argument('--max_images', type=int, default=None,
                       help='Maximum number of images to process in batch mode')
    parser.add_argument('--max_images_per_subset', type=int, default=None,
                       help='Maximum images per dataset subset (for Arrow dataset)')
    parser.add_argument('--individual', action='store_true',
                       help='Save individual class visualizations')
    parser.add_argument('--comparative', action='store_true',
                       help='Save comparative analysis')
    parser.add_argument('--multilayer', action='store_true',
                       help='Save multi-layer analysis')
    parser.add_argument('--dataset_filter', type=str, nargs='+', default=None,
                       help='Filter specific datasets (for Arrow dataset)')
    parser.add_argument('--subset_filter', type=str, nargs='+', default=None,
                       help='Filter specific subsets (for Arrow dataset)')
    
    args = parser.parse_args()
    
    # Validate arguments
    mode_count = sum([bool(args.image), bool(args.input_dir), bool(args.arrow_dataset)])
    if mode_count != 1:
        parser.error("Exactly one of --image, --input_dir, or --arrow_dataset must be specified")
    
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
    
    # Arrow dataset analysis
    elif args.arrow_dataset:
        # Load configuration for dataset
        conf = load_config_with_cli(args.config)
        conf = hydra.utils.instantiate(conf)
        
        # Filter datasets if requested
        if args.dataset_filter or args.subset_filter:
            filtered_sources = []
            for source in conf.datasets.source:
                # Filter by dataset name
                if args.dataset_filter and source['benchmark_name'] not in args.dataset_filter:
                    continue
                
                # Filter by subset names
                if args.subset_filter:
                    filtered_subsets = [s for s in source['sub_sets'] if s in args.subset_filter]
                    if not filtered_subsets:
                        continue
                    source = dict(source)  # Make a copy
                    source['sub_sets'] = filtered_subsets
                
                filtered_sources.append(source)
            
            conf.datasets.source = filtered_sources
            print(f"Filtered to {len(filtered_sources)} dataset sources")
        
        analyze_arrow_dataset(
            model=model,
            dataset_config=conf.datasets,
            output_dir=args.output_dir,
            target_layers=args.target_layers,
            max_images_per_subset=args.max_images_per_subset,
            save_individual=args.individual,
            save_comparative=args.comparative,
            save_multilayer=args.multilayer
        )
    
    print("\nGradCAM analysis complete!")


if __name__ == "__main__":
    main()