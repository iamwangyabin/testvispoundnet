"""
Example script for generating CLIP GradCAM visualizations with Arrow datasets.

This script demonstrates how to use the CLIP GradCAM implementation based on
the user's sample code to analyze arrow datasets and visualize attention patterns.
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
from typing import Dict, Any, List
from PIL import Image

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from visualization.gradcam.clip_gradcam_arrow import CLIPGradCAMArrow, visualize_clip_gradcam_results, viz_attn, load_image
from utils.util import load_config_with_cli
import hydra
import utils  # Add this import for eval() to find utils module
import data


def analyze_arrow_dataset_with_clip_gradcam(
    dataset_config: Dict[str, Any],
    output_dir: str,
    model_name: str = "RN50",
    saliency_layer: str = "layer4",
    max_images_per_subset: int = None,
    class_names: List[str] = None,
    custom_prompts: List[str] = None,
    template: str = "a photo of a {}",
    blur: bool = True,
    save_individual: bool = True,
    save_comparative: bool = True
):
    """
    Analyze Arrow dataset with CLIP GradCAM using user's approach.
    
    Args:
        dataset_config: Dataset configuration from config file
        output_dir: Output directory for results
        model_name: CLIP model name (e.g., "RN50", "ViT-B/32")
        saliency_layer: Target layer for GradCAM
        max_images_per_subset: Maximum images per subset
        class_names: Custom class names to test
        custom_prompts: Custom text prompts (overrides class_names)
        template: Template for class name prompts
        blur: Whether to apply Gaussian blur to attention maps
        save_individual: Whether to save individual visualizations
        save_comparative: Whether to save comparative analysis
    """
    print(f"Analyzing Arrow datasets with CLIP GradCAM...")
    print(f"Model: {model_name}, Layer: {saliency_layer}")
    print(f"Found {len(dataset_config['source'])} dataset sources")
    
    # Initialize CLIP GradCAM
    clip_gradcam = CLIPGradCAMArrow(
        model_name=model_name,
        saliency_layer=saliency_layer
    )
    
    print("CLIP model information:")
    model_info = clip_gradcam.get_model_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    all_results = []
    
    # Process each dataset source
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
                
                # Process images
                subset_results = _analyze_arrow_subset_with_clip_gradcam(
                    clip_gradcam=clip_gradcam,
                    data_loader=data_loader,
                    subset_name=f"{sub_data['benchmark_name']}_{sub_set}",
                    output_dir=subset_output_dir,
                    max_images=max_images_per_subset,
                    class_names=class_names,
                    custom_prompts=custom_prompts,
                    template=template,
                    blur=blur,
                    save_individual=save_individual,
                    save_comparative=save_comparative
                )
                
                all_results.extend(subset_results)
                print(f"    Generated CLIP GradCAM visualizations for {len(subset_results)} images")
                
            except Exception as e:
                print(f"    Error processing subset {sub_set}: {e}")
                continue
    
    # Save batch summary
    summary_path = os.path.join(output_dir, "arrow_clip_gradcam_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nArrow dataset CLIP GradCAM analysis complete!")
    print(f"Total visualizations generated: {len(all_results)}")
    print(f"Summary saved to: {summary_path}")
    
    return all_results


def _analyze_arrow_subset_with_clip_gradcam(
    clip_gradcam: CLIPGradCAMArrow,
    data_loader,
    subset_name: str,
    output_dir: str,
    max_images: int = None,
    class_names: List[str] = None,
    custom_prompts: List[str] = None,
    template: str = "a photo of a {}",
    blur: bool = True,
    save_individual: bool = True,
    save_comparative: bool = True
):
    """Process Arrow dataset subset with CLIP GradCAM visualizations."""
    results = []
    
    print(f"    Generating CLIP GradCAM visualizations for {len(data_loader)} images from {subset_name}")
    
    # Process images with progress bar
    for idx, (img_tensor, label) in enumerate(tqdm(data_loader, desc=f"  {subset_name}")):
        if max_images and idx >= max_images:
            break
        
        try:
            # Move to device
            if torch.cuda.is_available():
                img_tensor = img_tensor.cuda()
            
            # Preprocess for CLIP
            clip_tensor = clip_gradcam.preprocess_image_from_tensor(img_tensor)
            
            # Convert tensor back to image for visualization
            original_image = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            # Denormalize (assuming CLIP normalization)
            mean = np.array([0.48145466, 0.4578275, 0.40821073])
            std = np.array([0.26862954, 0.26130258, 0.27577711])
            original_image = original_image * std + mean
            original_image = np.clip(original_image * 255, 0, 255).astype(np.uint8)
            
            # Create image-specific output directory
            image_output_dir = os.path.join(output_dir, f"image_{idx:06d}")
            os.makedirs(image_output_dir, exist_ok=True)
            
            # Analyze with CLIP GradCAM
            if custom_prompts:
                # Use custom prompts
                results_dict = clip_gradcam.generate_comparative_gradcam(
                    clip_tensor, custom_prompts, blur
                )
                analysis_type = "custom_prompts"
            else:
                # Use class names or defaults
                results_dict = clip_gradcam.analyze_with_common_classes(
                    clip_tensor, class_names, template, blur
                )
                analysis_type = "class_analysis"
            
            metadata = results_dict['metadata']
            best_match = metadata['best_match_prompt']
            best_similarity = metadata['best_similarity']
            
            print(f"      Image {idx}: Best match = '{best_match}' (similarity: {best_similarity:.4f})")
            
            # Save metadata
            metadata_copy = metadata.copy()
            metadata_copy['similarities'] = [float(s) for s in metadata_copy['similarities']]
            with open(os.path.join(image_output_dir, f"image_{idx:06d}_clip_metadata.json"), 'w') as f:
                json.dump(metadata_copy, f, indent=2)
            
            # Individual visualizations
            if save_individual:
                # Save top 3 matches
                similarities = metadata['similarities']
                sorted_indices = np.argsort(similarities)[::-1][:3]
                
                for rank, prompt_idx in enumerate(sorted_indices):
                    if analysis_type == "class_analysis":
                        prompt = template.format(metadata['class_names'][prompt_idx])
                        class_name = metadata['class_names'][prompt_idx]
                    else:
                        prompt = metadata['text_prompts'][prompt_idx]
                        class_name = prompt
                    
                    similarity = similarities[prompt_idx]
                    
                    if prompt in results_dict:
                        cam = results_dict[prompt]['cam']
                        
                        # Create individual visualization
                        fig = viz_attn(
                            original_image,
                            cam,
                            blur=blur,
                            title=f"CLIP GradCAM - {class_name}\nSimilarity: {similarity:.3f}",
                            save_path=os.path.join(image_output_dir, f"image_{idx:06d}_rank{rank+1}_{class_name.replace(' ', '_')}.png")
                        )
                        plt.close(fig)
            
            # Comparative analysis
            if save_comparative:
                fig = visualize_clip_gradcam_results(
                    original_image,
                    results_dict,
                    title=f"CLIP GradCAM Analysis - {subset_name} Image {idx:06d}",
                    save_path=os.path.join(image_output_dir, f"image_{idx:06d}_comparative.png"),
                    blur=blur
                )
                plt.close(fig)
            
            # Add to results summary
            result_summary = {
                'image_index': int(idx),
                'true_label': int(label.item()),
                'subset_name': subset_name,
                'best_match': best_match,
                'best_similarity': float(best_similarity),
                'analysis_type': analysis_type,
                'output_dir': image_output_dir
            }
            
            results.append(result_summary)
            
        except Exception as e:
            print(f"      Error processing image {idx}: {e}")
            continue
    
    return results


def analyze_single_image_with_clip_gradcam(
    image_path: str,
    output_dir: str,
    model_name: str = "RN50",
    saliency_layer: str = "layer4",
    class_names: List[str] = None,
    custom_prompts: List[str] = None,
    template: str = "a photo of a {}",
    blur: bool = True
):
    """
    Analyze a single image with CLIP GradCAM using user's approach.
    
    Args:
        image_path: Path to input image
        output_dir: Output directory for results
        model_name: CLIP model name
        saliency_layer: Target layer for GradCAM
        class_names: Custom class names to test
        custom_prompts: Custom text prompts
        template: Template for class name prompts
        blur: Whether to apply Gaussian blur
    """
    print(f"\nAnalyzing image with CLIP GradCAM: {image_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize CLIP GradCAM
    clip_gradcam = CLIPGradCAMArrow(
        model_name=model_name,
        saliency_layer=saliency_layer
    )
    
    print("CLIP model information:")
    model_info = clip_gradcam.get_model_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # Load and preprocess image
    image_tensor = clip_gradcam.preprocess_image_from_path(image_path)
    original_image = load_image(image_path, resize=model_info['input_resolution'])
    
    image_name = os.path.basename(image_path)
    base_name = os.path.splitext(image_name)[0]
    
    # Analyze with CLIP GradCAM
    if custom_prompts:
        print(f"Using custom prompts: {custom_prompts}")
        results = clip_gradcam.generate_comparative_gradcam(image_tensor, custom_prompts, blur)
        analysis_type = "custom_prompts"
    else:
        print(f"Analyzing with class names...")
        results = clip_gradcam.analyze_with_common_classes(image_tensor, class_names, template, blur)
        analysis_type = "class_analysis"
    
    metadata = results['metadata']
    best_match = metadata['best_match_prompt']
    best_similarity = metadata['best_similarity']
    
    print(f"Best match: '{best_match}' (similarity: {best_similarity:.4f})")
    
    # Save metadata
    metadata_copy = metadata.copy()
    metadata_copy['similarities'] = [float(s) for s in metadata_copy['similarities']]
    with open(os.path.join(output_dir, f"{base_name}_clip_metadata.json"), 'w') as f:
        json.dump(metadata_copy, f, indent=2)
    
    # Individual visualizations for top matches
    similarities = metadata['similarities']
    sorted_indices = np.argsort(similarities)[::-1][:5]
    
    for rank, prompt_idx in enumerate(sorted_indices):
        if analysis_type == "class_analysis":
            prompt = template.format(metadata['class_names'][prompt_idx])
            class_name = metadata['class_names'][prompt_idx]
        else:
            prompt = metadata['text_prompts'][prompt_idx]
            class_name = prompt
        
        similarity = similarities[prompt_idx]
        
        if prompt in results:
            cam = results[prompt]['cam']
            
            # Create individual visualization
            fig = viz_attn(
                original_image,
                cam,
                blur=blur,
                title=f"CLIP GradCAM - {class_name}\nSimilarity: {similarity:.3f}",
                save_path=os.path.join(output_dir, f"{base_name}_rank{rank+1}_{class_name.replace(' ', '_')}.png")
            )
            plt.close(fig)
    
    # Comparative analysis
    fig = visualize_clip_gradcam_results(
        original_image,
        results,
        title=f"CLIP GradCAM Analysis - {image_name}",
        save_path=os.path.join(output_dir, f"{base_name}_comparative.png"),
        blur=blur
    )
    plt.close(fig)
    
    print(f"Results saved to: {output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate CLIP GradCAM visualizations for Arrow datasets")
    parser.add_argument('--image', type=str, default=None,
                       help='Path to single image for analysis')
    parser.add_argument('--arrow_dataset', action='store_true',
                       help='Process Arrow dataset from config file')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file (for Arrow dataset)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--model_name', type=str, default="RN50",
                       choices=["RN50", "RN101", "RN50x4", "RN50x16", "ViT-B/32", "ViT-B/16", "ViT-L/14"],
                       help='CLIP model name')
    parser.add_argument('--saliency_layer', type=str, default="layer4",
                       help='Target layer for GradCAM (e.g., layer4, layer3)')
    parser.add_argument('--max_images_per_subset', type=int, default=None,
                       help='Maximum images per dataset subset (for Arrow dataset)')
    parser.add_argument('--class_names', type=str, nargs='+', default=None,
                       help='Custom class names to test')
    parser.add_argument('--custom_prompts', type=str, nargs='+', default=None,
                       help='Custom text prompts (overrides class names)')
    parser.add_argument('--template', type=str, default="a photo of a {}",
                       help='Template for class name prompts')
    parser.add_argument('--blur', action='store_true', default=True,
                       help='Apply Gaussian blur to attention maps')
    parser.add_argument('--no_blur', action='store_true',
                       help='Disable Gaussian blur')
    parser.add_argument('--individual', action='store_true', default=True,
                       help='Save individual class visualizations')
    parser.add_argument('--comparative', action='store_true', default=True,
                       help='Save comparative analysis')
    
    args = parser.parse_args()
    
    # Handle blur setting
    if args.no_blur:
        args.blur = False
    
    # Validate arguments
    mode_count = sum([bool(args.image), bool(args.arrow_dataset)])
    if mode_count != 1:
        parser.error("Exactly one of --image or --arrow_dataset must be specified")
    
    if args.arrow_dataset and not args.config:
        parser.error("--config is required when using --arrow_dataset")
    
    # Single image analysis
    if args.image:
        analyze_single_image_with_clip_gradcam(
            image_path=args.image,
            output_dir=args.output_dir,
            model_name=args.model_name,
            saliency_layer=args.saliency_layer,
            class_names=args.class_names,
            custom_prompts=args.custom_prompts,
            template=args.template,
            blur=args.blur
        )
    
    # Arrow dataset analysis
    elif args.arrow_dataset:
        # Load configuration
        conf = load_config_with_cli(args.config)
        conf = hydra.utils.instantiate(conf)
        
        analyze_arrow_dataset_with_clip_gradcam(
            dataset_config=conf.datasets,
            output_dir=args.output_dir,
            model_name=args.model_name,
            saliency_layer=args.saliency_layer,
            max_images_per_subset=args.max_images_per_subset,
            class_names=args.class_names,
            custom_prompts=args.custom_prompts,
            template=args.template,
            blur=args.blur,
            save_individual=args.individual,
            save_comparative=args.comparative
        )
    
    print("\nCLIP GradCAM analysis complete!")


if __name__ == "__main__":
    main()