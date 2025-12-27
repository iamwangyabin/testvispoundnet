"""
Visualization utilities for GradCAM results.

This module provides functions for creating beautiful visualizations,
overlaying heatmaps on images, and saving results in various formats.
"""

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from PIL import Image
import os
from typing import List, Tuple, Optional, Union, Dict, Any
import json


def normalize_heatmap(heatmap: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize heatmap to [0, 1] range.
    
    Args:
        heatmap: Input heatmap array
        method: Normalization method ('minmax', 'zscore', 'robust')
        
    Returns:
        Normalized heatmap
    """
    if method == 'minmax':
        hmin, hmax = heatmap.min(), heatmap.max()
        if hmax > hmin:
            return (heatmap - hmin) / (hmax - hmin)
        else:
            return heatmap
    elif method == 'zscore':
        mean, std = heatmap.mean(), heatmap.std()
        if std > 0:
            normalized = (heatmap - mean) / std
            # Clip to reasonable range and rescale to [0, 1]
            normalized = np.clip(normalized, -3, 3)
            return (normalized + 3) / 6
        else:
            return heatmap
    elif method == 'robust':
        q25, q75 = np.percentile(heatmap, [25, 75])
        if q75 > q25:
            return np.clip((heatmap - q25) / (q75 - q25), 0, 1)
        else:
            return heatmap
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def create_heatmap_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: str = 'jet',
    normalize_heatmap_flag: bool = True
) -> np.ndarray:
    """
    Create overlay of heatmap on original image.
    
    Args:
        image: Original image array (H, W, 3) in [0, 255] range
        heatmap: Heatmap array (H, W) 
        alpha: Transparency of heatmap overlay
        colormap: Matplotlib colormap name
        normalize_heatmap_flag: Whether to normalize heatmap
        
    Returns:
        Overlaid image array (H, W, 3) in [0, 255] range
    """
    # Ensure image is in correct format
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Normalize heatmap if requested
    if normalize_heatmap_flag:
        heatmap = normalize_heatmap(heatmap)
    
    # Resize heatmap to match image size if needed
    if heatmap.shape != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)[:, :, :3]  # Remove alpha channel
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    # Create overlay
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlay


def visualize_gradcam(
    image: np.ndarray,
    cam: np.ndarray,
    title: str = "GradCAM Visualization",
    save_path: Optional[str] = None,
    show_original: bool = True,
    show_heatmap: bool = True,
    show_overlay: bool = True,
    colormap: str = 'jet',
    alpha: float = 0.4,
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    Create comprehensive GradCAM visualization.
    
    Args:
        image: Original image array (H, W, 3)
        cam: GradCAM heatmap array (H, W)
        title: Title for the visualization
        save_path: Path to save the figure
        show_original: Whether to show original image
        show_heatmap: Whether to show heatmap
        show_overlay: Whether to show overlay
        colormap: Matplotlib colormap name
        alpha: Transparency for overlay
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    # Prepare subplots
    num_plots = sum([show_original, show_heatmap, show_overlay])
    fig, axes = plt.subplots(1, num_plots, figsize=figsize)
    
    if num_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Original image
    if show_original:
        axes[plot_idx].imshow(image)
        axes[plot_idx].set_title('Original Image')
        axes[plot_idx].axis('off')
        plot_idx += 1
    
    # Heatmap
    if show_heatmap:
        im = axes[plot_idx].imshow(cam, cmap=colormap)
        axes[plot_idx].set_title('GradCAM Heatmap')
        axes[plot_idx].axis('off')
        plt.colorbar(im, ax=axes[plot_idx], fraction=0.046, pad=0.04)
        plot_idx += 1
    
    # Overlay
    if show_overlay:
        overlay = create_heatmap_overlay(image, cam, alpha=alpha, colormap=colormap)
        axes[plot_idx].imshow(overlay)
        axes[plot_idx].set_title('GradCAM Overlay')
        axes[plot_idx].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_comparative_gradcam(
    image: np.ndarray,
    real_cam: np.ndarray,
    fake_cam: np.ndarray,
    prediction_result: Dict[str, Any],
    title: str = "Comparative GradCAM Analysis",
    save_path: Optional[str] = None,
    colormap: str = 'jet',
    alpha: float = 0.4,
    figsize: Tuple[int, int] = (20, 8)
) -> plt.Figure:
    """
    Create comparative visualization for Real vs Fake GradCAM.
    
    Args:
        image: Original image array (H, W, 3)
        real_cam: GradCAM for Real class (H, W)
        fake_cam: GradCAM for Fake class (H, W)
        prediction_result: Dictionary with prediction information
        title: Title for the visualization
        save_path: Path to save the figure
        colormap: Matplotlib colormap name
        alpha: Transparency for overlay
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # Row 1: Real class analysis
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    im1 = axes[0, 1].imshow(real_cam, cmap=colormap)
    axes[0, 1].set_title(f'Real Class Heatmap\n(Prob: {prediction_result["real_prob"]:.3f})')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    real_overlay = create_heatmap_overlay(image, real_cam, alpha=alpha, colormap=colormap)
    axes[0, 2].imshow(real_overlay)
    axes[0, 2].set_title('Real Class Overlay')
    axes[0, 2].axis('off')
    
    # Row 2: Fake class analysis
    axes[1, 0].imshow(image)
    axes[1, 0].set_title('Original Image')
    axes[1, 0].axis('off')
    
    im2 = axes[1, 1].imshow(fake_cam, cmap=colormap)
    axes[1, 1].set_title(f'Fake Class Heatmap\n(Prob: {prediction_result["fake_prob"]:.3f})')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    fake_overlay = create_heatmap_overlay(image, fake_cam, alpha=alpha, colormap=colormap)
    axes[1, 2].imshow(fake_overlay)
    axes[1, 2].set_title('Fake Class Overlay')
    axes[1, 2].axis('off')
    
    # Add prediction information
    pred_text = f"""Prediction: {prediction_result['predicted_class_name']}
Confidence: {prediction_result['confidence']:.3f}
Real Prob: {prediction_result['real_prob']:.3f}
Fake Prob: {prediction_result['fake_prob']:.3f}"""
    
    fig.text(0.02, 0.5, pred_text, fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_multi_layer_analysis(
    image: np.ndarray,
    layer_results: Dict[str, Dict[str, Any]],
    title: str = "Multi-Layer GradCAM Analysis",
    save_path: Optional[str] = None,
    colormap: str = 'jet',
    figsize: Tuple[int, int] = (20, 12)
) -> plt.Figure:
    """
    Create multi-layer analysis visualization.
    
    Args:
        image: Original image array (H, W, 3)
        layer_results: Dictionary with results from analyze_attention_patterns
        title: Title for the visualization
        save_path: Path to save the figure
        colormap: Matplotlib colormap name
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    num_layers = len(layer_results)
    fig, axes = plt.subplots(num_layers, 4, figsize=figsize)
    
    if num_layers == 1:
        axes = axes.reshape(1, -1)
    
    for i, (layer_name, results) in enumerate(layer_results.items()):
        # Original image
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f'Layer: {layer_name.split(".")[-2]}')
        axes[i, 0].axis('off')
        
        # Real class heatmap
        im1 = axes[i, 1].imshow(results['real_cam'], cmap=colormap)
        axes[i, 1].set_title(f'Real Class\n(μ={results["real_stats"]["mean"]:.3f})')
        axes[i, 1].axis('off')
        
        # Fake class heatmap
        im2 = axes[i, 2].imshow(results['fake_cam'], cmap=colormap)
        axes[i, 2].set_title(f'Fake Class\n(μ={results["fake_stats"]["mean"]:.3f})')
        axes[i, 2].axis('off')
        
        # Difference heatmap
        im3 = axes[i, 3].imshow(results['difference_cam'], cmap='RdBu_r', vmin=-1, vmax=1)
        axes[i, 3].set_title('Fake - Real')
        axes[i, 3].axis('off')
        
        # Add colorbars for the last row
        if i == num_layers - 1:
            plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)
            plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)
            plt.colorbar(im3, ax=axes[i, 3], fraction=0.046, pad=0.04)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def save_gradcam_results(
    results: Dict[str, Any],
    output_dir: str,
    image_name: str,
    save_formats: List[str] = ['png', 'jpg', 'npy']
) -> Dict[str, str]:
    """
    Save GradCAM results in multiple formats.
    
    Args:
        results: Dictionary containing GradCAM results
        output_dir: Output directory path
        image_name: Base name for saved files
        save_formats: List of formats to save ('png', 'jpg', 'npy', 'json')
        
    Returns:
        Dictionary mapping format to saved file path
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_files = {}
    
    base_name = os.path.splitext(image_name)[0]
    
    # Save heatmaps
    if 'cam' in results:
        cam = results['cam']
        
        if 'png' in save_formats:
            # Save as PNG with colormap
            plt.figure(figsize=(8, 8))
            plt.imshow(cam, cmap='jet')
            plt.axis('off')
            png_path = os.path.join(output_dir, f"{base_name}_gradcam.png")
            plt.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()
            saved_files['png'] = png_path
        
        if 'jpg' in save_formats:
            # Save as JPG
            cam_normalized = (normalize_heatmap(cam) * 255).astype(np.uint8)
            cam_colored = cv2.applyColorMap(cam_normalized, cv2.COLORMAP_JET)
            jpg_path = os.path.join(output_dir, f"{base_name}_gradcam.jpg")
            cv2.imwrite(jpg_path, cam_colored)
            saved_files['jpg'] = jpg_path
        
        if 'npy' in save_formats:
            # Save raw numpy array
            npy_path = os.path.join(output_dir, f"{base_name}_gradcam.npy")
            np.save(npy_path, cam)
            saved_files['npy'] = npy_path
    
    # Save metadata
    if 'json' in save_formats:
        metadata = {k: v for k, v in results.items() if k != 'cam'}
        # Convert numpy arrays to lists for JSON serialization
        for key, value in metadata.items():
            if isinstance(value, np.ndarray):
                metadata[key] = value.tolist()
            elif isinstance(value, torch.Tensor):
                metadata[key] = value.cpu().numpy().tolist()
        
        json_path = os.path.join(output_dir, f"{base_name}_metadata.json")
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        saved_files['json'] = json_path
    
    return saved_files


def create_attention_summary_plot(
    layer_results: Dict[str, Dict[str, Any]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Create summary plot of attention statistics across layers.
    
    Args:
        layer_results: Dictionary with results from analyze_attention_patterns
        save_path: Path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    layer_names = list(layer_results.keys())
    layer_indices = [name.split('.')[-2] for name in layer_names]
    
    # Extract statistics
    real_means = [results['real_stats']['mean'] for results in layer_results.values()]
    fake_means = [results['fake_stats']['mean'] for results in layer_results.values()]
    real_stds = [results['real_stats']['std'] for results in layer_results.values()]
    fake_stds = [results['fake_stats']['std'] for results in layer_results.values()]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # Mean activation comparison
    x = np.arange(len(layer_indices))
    width = 0.35
    
    ax1.bar(x - width/2, real_means, width, label='Real', alpha=0.7, color='blue')
    ax1.bar(x + width/2, fake_means, width, label='Fake', alpha=0.7, color='red')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Mean Activation')
    ax1.set_title('Mean Activation by Layer')
    ax1.set_xticks(x)
    ax1.set_xticklabels(layer_indices)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Standard deviation comparison
    ax2.bar(x - width/2, real_stds, width, label='Real', alpha=0.7, color='blue')
    ax2.bar(x + width/2, fake_stds, width, label='Fake', alpha=0.7, color='red')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Standard Deviation')
    ax2.set_title('Activation Spread by Layer')
    ax2.set_xticks(x)
    ax2.set_xticklabels(layer_indices)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Difference in means
    mean_diffs = [fake - real for fake, real in zip(fake_means, real_means)]
    colors = ['red' if diff > 0 else 'blue' for diff in mean_diffs]
    ax3.bar(x, mean_diffs, color=colors, alpha=0.7)
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Fake - Real (Mean)')
    ax3.set_title('Attention Difference by Layer')
    ax3.set_xticks(x)
    ax3.set_xticklabels(layer_indices)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.grid(True, alpha=0.3)
    
    # Activation evolution
    ax4.plot(x, real_means, 'o-', label='Real', color='blue', linewidth=2)
    ax4.plot(x, fake_means, 's-', label='Fake', color='red', linewidth=2)
    ax4.fill_between(x, 
                     [m - s for m, s in zip(real_means, real_stds)],
                     [m + s for m, s in zip(real_means, real_stds)],
                     alpha=0.2, color='blue')
    ax4.fill_between(x, 
                     [m - s for m, s in zip(fake_means, fake_stds)],
                     [m + s for m, s in zip(fake_means, fake_stds)],
                     alpha=0.2, color='red')
    ax4.set_xlabel('Layer')
    ax4.set_ylabel('Activation')
    ax4.set_title('Activation Evolution Across Layers')
    ax4.set_xticks(x)
    ax4.set_xticklabels(layer_indices)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def preprocess_image_for_gradcam(
    image_path: str,
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Preprocess image for GradCAM analysis.
    
    Args:
        image_path: Path to input image
        target_size: Target size for resizing
        normalize: Whether to apply ImageNet normalization
        
    Returns:
        Tuple of (preprocessed tensor, original image array)
    """
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = image.copy()
    
    # Resize
    image = cv2.resize(image, target_size)
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
    
    # Normalize if requested
    if normalize:
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor, original_image