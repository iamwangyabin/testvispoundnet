"""
Example script for generating GradCAM visualizations with original CLIP model.

This script demonstrates how to use CLIP GradCAM implementation to analyze
image classification decisions with common class names and visualize attention patterns.
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

from visualization.gradcam.clip_gradcam import CLIPGradCAM
from visualization.gradcam import visualize_gradcam, visualize_comparative_gradcam
from visualization.gradcam.utils import create_attention_summary_plot


def visualize_clip_gradcam(
    original_image: np.ndarray,
    cam: np.ndarray,
    title: str = "CLIP GradCAM",
    save_path: str = None,
    alpha: float = 0.4
) -> plt.Figure:
    """
    Visualize CLIP GradCAM results.
    
    Args:
        original_image: Original image as numpy array
        cam: Class activation map
        title: Plot title
        save_path: Path to save the visualization
        alpha: Transparency for overlay
        
    Returns:
        Matplotlib figure
    """
    return visualize_gradcam(original_image, cam, title, save_path, alpha)


def visualize_clip_comparative_gradcam(
    original_image: np.ndarray,
    results: Dict[str, Any],
    title: str = "CLIP Comparative GradCAM",
    save_path: str = None,
    max_classes: int = 6
) -> plt.Figure:
    """
    Visualize comparative CLIP GradCAM results for multiple classes.
    
    Args:
        original_image: Original image as numpy array
        results: Results from CLIPGradCAM.generate_comparative_cam()
        title: Plot title
        save_path: Path to save the visualization
        max_classes: Maximum number of classes to show
        
    Returns:
        Matplotlib figure
    """
    metadata = results['metadata']
    class_names = metadata['class_names']
    similarities = metadata['similarities']
    
    # Sort by similarity and take top classes
    sorted_indices = np.argsort(similarities)[::-1][:max_classes]
    
    # Create subplot layout
    n_cols = 3
    n_rows = (len(sorted_indices) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, class_idx in enumerate(sorted_indices):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        class_name = class_names[class_idx]
        similarity = similarities[class_idx]
        
        if class_name in results:
            cam = results[class_name]['cam']
            
            # Create overlay
            heatmap = plt.cm.jet(cam)[:, :, :3]
            overlay = original_image * 0.6 + heatmap * 255 * 0.4
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
            
            ax.imshow(overlay)
            ax.set_title(f"{class_name}\nSimilarity: {similarity:.3f}", fontsize=10)
        else:
            ax.imshow(original_image)
            ax.set_title(f"{class_name}\nSimilarity: {similarity:.3f}\n(CAM failed)", fontsize=10)
        
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(len(sorted_indices), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def analyze_single_image_with_clip(
    image_path: str,
    output_dir: str,
    model_name: str = "ViT-B/32",
    class_names: List[str] = None,
    target_layers: List[str] = None,
    custom_prompts: List[str] = None,
    save_individual: bool = True,
    save_comparative: bool = True,
    template: str = "a photo of a {}"
):
    """
    Analyze a single image with CLIP GradCAM.
    
    Args:
        image_path: Path to input image
        output_dir: Output directory for results
        model_name: CLIP model name
        class_names: List of class names to test
        target_layers: Specific layers to analyze
        custom_prompts: Custom text prompts (overrides class_names)
        save_individual: Whether to save individual class visualizations
        save_comparative: Whether to save comparative analysis
        template: Template for class name prompts
    """
    print(f"\nAnalyzing image with CLIP: {image_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize CLIP GradCAM
    clip_gradcam = CLIPGradCAM(
        model_name=model_name,
        target_layers=target_layers
    )
    
    print("CLIP model information:")
    model_info = clip_gradcam.get_model_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # Load and preprocess image
    image_tensor = clip_gradcam.preprocess_image(image_path)
    
    # Load original image for visualization
    original_image = np.array(Image.open(image_path).convert('RGB'))
    
    image_name = os.path.basename(image_path)
    base_name = os.path.splitext(image_name)[0]
    
    # Use custom prompts or class names
    if custom_prompts:
        print(f"Using custom prompts: {custom_prompts}")
        results = clip_gradcam.generate_comparative_cam(
            image=image_tensor,
            text_prompts=custom_prompts
        )
        analysis_type = "custom_prompts"
    else:
        print(f"Analyzing with common class names...")
        results = clip_gradcam.analyze_with_common_classes(
            image=image_tensor,
            class_names=class_names,
            template=template
        )
        analysis_type = "class_analysis"
    
    metadata = results['metadata']
    print(f"Best match: {metadata['best_match_prompt']} (similarity: {metadata['similarities'][metadata['best_match_idx']]:.4f})")
    
    # Save metadata
    metadata_path = os.path.join(output_dir, f"{base_name}_clip_analysis.json")
    with open(metadata_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        metadata_copy = metadata.copy()
        metadata_copy['similarities'] = metadata_copy['similarities'].tolist()
        json.dump(metadata_copy, f, indent=2)
    
    # Individual class visualizations
    if save_individual:
        print("Generating individual class visualizations...")
        
        # Save top 5 classes
        similarities = metadata['similarities']
        sorted_indices = np.argsort(similarities)[::-1][:5]
        
        for i, class_idx in enumerate(sorted_indices):
            if analysis_type == "class_analysis":
                class_name = metadata['class_names'][class_idx]
                prompt = template.format(class_name)
            else:
                prompt = metadata['text_prompts'][class_idx]
                class_name = prompt
            
            similarity = similarities[class_idx]
            
            if prompt in results:
                cam = results[prompt]['cam']
                
                fig = visualize_clip_gradcam(
                    original_image,
                    cam,
                    title=f"CLIP GradCAM - {class_name}\nSimilarity: {similarity:.3f}",
                    save_path=os.path.join(output_dir, f"{base_name}_clip_{i+1}_{class_name.replace(' ', '_')}.png")
                )
                plt.close(fig)
    
    # Comparative analysis
    if save_comparative:
        print("Generating comparative analysis...")
        
        if analysis_type == "class_analysis":
            fig = visualize_clip_comparative_gradcam(
                original_image,
                results,
                title=f"CLIP Comparative Analysis - {image_name}",
                save_path=os.path.join(output_dir, f"{base_name}_clip_comparative.png")
            )
        else:
            # For custom prompts, create a simpler comparative view
            fig = visualize_comparative_gradcam(
                original_image,
                results[custom_prompts[0]]['cam'],
                results[custom_prompts[1]]['cam'] if len(custom_prompts) > 1 else results[custom_prompts[0]]['cam'],
                {'predicted_class_name': metadata['best_match_prompt'], 
                 'confidence': metadata['similarities'][metadata['best_match_idx']]},
                title=f"CLIP Comparative Analysis - {image_name}",
                save_path=os.path.join(output_dir, f"{base_name}_clip_comparative.png")
            )
        
        plt.close(fig)
    
    print(f"Results saved to: {output_dir}")
    return results


def batch_analyze_images_with_clip(
    input_dir: str,
    output_dir: str,
    model_name: str = "ViT-B/32",
    class_names: List[str] = None,
    target_layers: List[str] = None,
    image_extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp'],
    max_images: int = None,
    template: str = "a photo of a {}"
):
    """
    Analyze multiple images in batch with CLIP.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Output directory for results
        model_name: CLIP model name
        class_names: List of class names to test
        target_layers: Specific layers to analyze
        image_extensions: Valid image file extensions
        max_images: Maximum number of images to process
        template: Template for class name prompts
    """
    print(f"\nBatch analyzing images with CLIP from: {input_dir}")
    
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
            result = analyze_single_image_with_clip(
                image_path=str(image_path),
                output_dir=image_output_dir,
                model_name=model_name,
                class_names=class_names,
                target_layers=target_layers,
                save_individual=True,
                save_comparative=True,
                template=template
            )
            
            # Add to summary
            metadata = result['metadata']
            results_summary.append({
                'image_name': image_path.name,
                'best_match': metadata['best_match_prompt'],
                'best_similarity': float(metadata['similarities'][metadata['best_match_idx']]),
                'top_3_matches': [
                    {
                        'prompt': metadata['text_prompts'][idx],
                        'similarity': float(metadata['similarities'][idx])
                    }
                    for idx in np.argsort(metadata['similarities'])[::-1][:3]
                ]
            })
            
        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")
            continue
    
    # Save batch summary
    summary_path = os.path.join(output_dir, "clip_batch_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nBatch analysis complete. Summary saved to: {summary_path}")
    return results_summary


def analyze_arrow_dataset_with_clip(
    dataset_config: Dict[str, Any],
    output_dir: str,
    model_name: str = "ViT-B/32",
    target_layers: List[str] = None,
    max_images_per_subset: int = None,
    class_names: List[str] = None,
    template: str = "a photo of a {}"
):
    """
    Analyze Arrow dataset with CLIP GradCAM.
    
    Args:
        dataset_config: Dataset configuration
        output_dir: Output directory for results
        model_name: CLIP model name
        target_layers: Specific layers to analyze
        max_images_per_subset: Maximum images per subset
        class_names: List of class names to test
        template: Template for class name prompts
    """
    print(f"Analyzing Arrow datasets with CLIP...")
    print(f"Found {len(dataset_config['source'])} dataset sources")
    
    # Initialize CLIP GradCAM
    clip_gradcam = CLIPGradCAM(
        model_name=model_name,
        target_layers=target_layers
    )
    
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
                import data
                import utils
                dataset = eval(sub_data['target'])(
                    sub_data['data_root'],
                    dataset_config['trsf'],
                    subset=sub_set,
                    split=sub_data['split']
                )
                
                # Create data loader
                data_loader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=1,
                    num_workers=0,
                    shuffle=False
                )
                
                # Create subset output directory
                subset_output_dir = os.path.join(output_dir, sub_data['benchmark_name'], sub_set)
                os.makedirs(subset_output_dir, exist_ok=True)
                
                # Process images
                subset_results = []
                
                for idx, (img_tensor, label) in enumerate(tqdm(data_loader, desc=f"  {sub_set}")):
                    if max_images_per_subset and idx >= max_images_per_subset:
                        break
                    
                    try:
                        # Convert tensor to PIL Image for CLIP preprocessing
                        img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                        # Denormalize
                        mean = np.array([0.48145466, 0.4578275, 0.40821073])
                        std = np.array([0.26862954, 0.26130258, 0.27577711])
                        img_np = img_np * std + mean
                        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
                        
                        # Convert to PIL and preprocess for CLIP
                        pil_image = Image.fromarray(img_np)
                        clip_tensor = clip_gradcam.preprocess(pil_image).unsqueeze(0)
                        
                        # Analyze with CLIP
                        results = clip_gradcam.analyze_with_common_classes(
                            image=clip_tensor,
                            class_names=class_names,
                            template=template
                        )
                        
                        # Create image-specific output directory
                        image_output_dir = os.path.join(subset_output_dir, f"image_{idx:06d}")
                        os.makedirs(image_output_dir, exist_ok=True)
                        
                        # Save results
                        metadata = results['metadata']
                        
                        # Save comparative visualization
                        fig = visualize_clip_comparative_gradcam(
                            img_np,
                            results,
                            title=f"CLIP Analysis - {sub_set} Image {idx:06d}",
                            save_path=os.path.join(image_output_dir, f"image_{idx:06d}_clip_analysis.png")
                        )
                        plt.close(fig)
                        
                        # Save metadata
                        metadata_copy = metadata.copy()
                        metadata_copy['similarities'] = metadata_copy['similarities'].tolist()
                        with open(os.path.join(image_output_dir, f"image_{idx:06d}_clip_metadata.json"), 'w') as f:
                            json.dump(metadata_copy, f, indent=2)
                        
                        # Add to results
                        # Debug: Check data types before JSON serialization
                        true_label_val = label.item()
                        best_match_idx = metadata['best_match_idx']
                        best_similarity_val = metadata['similarities'][best_match_idx]
                        
                        print(f"      Debug - true_label type: {type(true_label_val)}, value: {true_label_val}")
                        print(f"      Debug - best_match_idx type: {type(best_match_idx)}, value: {best_match_idx}")
                        print(f"      Debug - best_similarity type: {type(best_similarity_val)}, value: {best_similarity_val}")
                        
                        result_summary = {
                            'image_index': int(idx),  # Ensure native Python int
                            'true_label': int(true_label_val),  # Convert to native Python int
                            'subset_name': f"{sub_data['benchmark_name']}_{sub_set}",
                            'best_match': metadata['best_match_prompt'],
                            'best_similarity': float(best_similarity_val),  # Convert to native Python float
                            'output_dir': image_output_dir
                        }
                        
                        subset_results.append(result_summary)
                        
                    except Exception as e:
                        print(f"      Error processing image {idx}: {e}")
                        continue
                
                all_results.extend(subset_results)
                print(f"    Generated CLIP visualizations for {len(subset_results)} images")
                
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


def main():
    parser = argparse.ArgumentParser(description="Generate GradCAM visualizations for CLIP")
    parser.add_argument('--image', type=str, default=None,
                       help='Path to single image for analysis')
    parser.add_argument('--input_dir', type=str, default=None,
                       help='Directory containing images for batch analysis')
    parser.add_argument('--arrow_dataset', action='store_true',
                       help='Process Arrow dataset from config file')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file (for Arrow dataset)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--model_name', type=str, default="ViT-B/32",
                       choices=["RN50", "RN101", "RN50x4", "RN50x16", "ViT-B/32", "ViT-B/16", "ViT-L/14"],
                       help='CLIP model name')
    parser.add_argument('--target_layers', type=str, nargs='+', default=None,
                       help='Specific layers to analyze')
    parser.add_argument('--max_images', type=int, default=None,
                       help='Maximum number of images to process in batch mode')
    parser.add_argument('--max_images_per_subset', type=int, default=None,
                       help='Maximum images per dataset subset (for Arrow dataset)')
    parser.add_argument('--class_names', type=str, nargs='+', default=None,
                       help='Custom class names to test')
    parser.add_argument('--custom_prompts', type=str, nargs='+', default=None,
                       help='Custom text prompts (overrides class names)')
    parser.add_argument('--template', type=str, default="a photo of a {}",
                       help='Template for class name prompts')
    parser.add_argument('--individual', action='store_true',
                       help='Save individual class visualizations')
    parser.add_argument('--comparative', action='store_true',
                       help='Save comparative analysis')
    
    args = parser.parse_args()
    
    # Validate arguments
    mode_count = sum([bool(args.image), bool(args.input_dir), bool(args.arrow_dataset)])
    if mode_count != 1:
        parser.error("Exactly one of --image, --input_dir, or --arrow_dataset must be specified")
    
    if args.arrow_dataset and not args.config:
        parser.error("--config is required when using --arrow_dataset")
    
    # Single image analysis
    if args.image:
        analyze_single_image_with_clip(
            image_path=args.image,
            output_dir=args.output_dir,
            model_name=args.model_name,
            class_names=args.class_names,
            target_layers=args.target_layers,
            custom_prompts=args.custom_prompts,
            save_individual=args.individual,
            save_comparative=args.comparative,
            template=args.template
        )
    
    # Batch analysis
    elif args.input_dir:
        batch_analyze_images_with_clip(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            model_name=args.model_name,
            class_names=args.class_names,
            target_layers=args.target_layers,
            max_images=args.max_images,
            template=args.template
        )
    
    # Arrow dataset analysis
    elif args.arrow_dataset:
        from utils.util import load_config_with_cli
        import hydra
        
        # Load configuration
        conf = load_config_with_cli(args.config)
        conf = hydra.utils.instantiate(conf)
        
        analyze_arrow_dataset_with_clip(
            dataset_config=conf.datasets,
            output_dir=args.output_dir,
            model_name=args.model_name,
            target_layers=args.target_layers,
            max_images_per_subset=args.max_images_per_subset,
            class_names=args.class_names,
            template=args.template
        )
    
    print("\nCLIP GradCAM analysis complete!")


if __name__ == "__main__":
    main()