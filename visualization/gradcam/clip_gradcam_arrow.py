"""
CLIP GradCAM implementation for arrow datasets based on user's sample code.

This module provides GradCAM functionality specifically designed for CLIP models
working with arrow datasets, using the user's preferred GradCAM approach.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
from scipy import ndimage as filters
import urllib.request

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from networks.clip import clip


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalize to [0, 1]."""
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x


def getAttMap(img, attn_map, blur=True):
    """
    Modified from: https://github.com/salesforce/ALBEF/blob/main/visualization.ipynb
    Create attention map overlay on image.
    """
    if blur:
        attn_map = filters.gaussian_filter(attn_map, 0.02*max(img.shape[:2]))
    attn_map = normalize(attn_map)
    cmap = plt.get_cmap('jet')
    attn_map_c = np.delete(cmap(attn_map), 3, 2)
    attn_map = 1*(1-attn_map**0.7).reshape(attn_map.shape + (1,))*img + \
            (attn_map**0.7).reshape(attn_map.shape+(1,)) * attn_map_c
    return attn_map


def viz_attn(img, attn_map, blur=True, title="CLIP GradCAM", save_path=None):
    """Visualize attention map."""
    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    axes[1].imshow(getAttMap(img, attn_map, blur))
    axes[1].set_title(title)
    axes[1].axis("off")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def load_image(img_path, resize=None):
    """Load and preprocess image."""
    image = Image.open(img_path).convert("RGB")
    if resize is not None:
        image = image.resize((resize, resize))
    return np.asarray(image).astype(np.float32) / 255.


class Hook:
    """Attaches to a module and records its activations and gradients."""

    def __init__(self, module: nn.Module):
        self.data = None
        self.hook = module.register_forward_hook(self.save_grad)
        
    def save_grad(self, module, input, output):
        self.data = output
        output.requires_grad_(True)
        output.retain_grad()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.hook.remove()
        
    @property
    def activation(self) -> torch.Tensor:
        return self.data
    
    @property
    def gradient(self) -> torch.Tensor:
        return self.data.grad


def gradCAM(
    model: nn.Module,
    input: torch.Tensor,
    target: torch.Tensor,
    layer: nn.Module
) -> torch.Tensor:
    """
    Reference: https://arxiv.org/abs/1610.02391
    Generate GradCAM for CLIP model.
    """
    # Zero out any gradients at the input.
    if input.grad is not None:
        input.grad.data.zero_()
        
    # Disable gradient settings.
    requires_grad = {}
    for name, param in model.named_parameters():
        requires_grad[name] = param.requires_grad
        param.requires_grad_(False)
        
    # Attach a hook to the model at the desired layer.
    assert isinstance(layer, nn.Module)
    with Hook(layer) as hook:        
        # Do a forward and backward pass.
        output = model(input)
        output.backward(target)

        grad = hook.gradient.float()
        act = hook.activation.float()
    
        # Global average pool gradient across spatial dimension
        # to obtain importance weights.
        alpha = grad.mean(dim=(2, 3), keepdim=True)
        # Weighted combination of activation maps over channel
        # dimension.
        gradcam = torch.sum(act * alpha, dim=1, keepdim=True)
        # We only want neurons with positive influence so we
        # clamp any negative ones.
        gradcam = torch.clamp(gradcam, min=0)

    # Resize gradcam to input resolution.
    gradcam = F.interpolate(
        gradcam,
        input.shape[2:],
        mode='bicubic',
        align_corners=False)
    
    # Restore gradient settings.
    for name, param in model.named_parameters():
        param.requires_grad_(requires_grad[name])
        
    return gradcam


class CLIPGradCAMArrow:
    """
    CLIP GradCAM implementation for arrow datasets using user's approach.
    """
    
    def __init__(
        self,
        model_name: str = "RN50",
        device: str = None,
        saliency_layer: str = "layer4"
    ):
        """
        Initialize CLIP GradCAM for arrow datasets.
        
        Args:
            model_name: CLIP model name (e.g., "RN50", "ViT-B/32")
            device: Device to use ('cuda' or 'cpu')
            saliency_layer: Target layer for GradCAM
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_name = model_name
        self.saliency_layer = saliency_layer
        
        # Load CLIP model
        print(f"Loading CLIP model: {model_name}")
        self.model, self.preprocess = clip.load(model_name, device=device, jit=False)
        self.model.eval()
        
        print(f"CLIP GradCAM initialized with model: {model_name}")
        print(f"Target layer: {saliency_layer}")
    
    def generate_gradcam_for_text(
        self,
        image_tensor: torch.Tensor,
        text_prompt: str,
        blur: bool = True
    ) -> Dict[str, Any]:
        """
        Generate GradCAM for a specific text prompt.
        
        Args:
            image_tensor: Preprocessed image tensor
            text_prompt: Text prompt for analysis
            blur: Whether to apply Gaussian blur to attention map
            
        Returns:
            Dictionary containing CAM and metadata
        """
        # Tokenize text
        text_input = clip.tokenize([text_prompt]).to(self.device)
        
        # Get target layer
        target_layer = getattr(self.model.visual, self.saliency_layer)
        
        # Generate GradCAM
        attn_map = gradCAM(
            self.model.visual,
            image_tensor,
            self.model.encode_text(text_input).float(),
            target_layer
        )
        
        # Convert to numpy
        attn_map_np = attn_map.squeeze().detach().cpu().numpy()
        
        # Get similarity score
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(text_input)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity
            similarity = (image_features @ text_features.T).item()
        
        return {
            'cam': attn_map_np,
            'text_prompt': text_prompt,
            'similarity': similarity,
            'blur': blur
        }
    
    def generate_comparative_gradcam(
        self,
        image_tensor: torch.Tensor,
        text_prompts: List[str],
        blur: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comparative GradCAM for multiple text prompts.
        
        Args:
            image_tensor: Preprocessed image tensor
            text_prompts: List of text prompts to compare
            blur: Whether to apply Gaussian blur
            
        Returns:
            Dictionary with results for each prompt
        """
        results = {}
        similarities = []
        
        for prompt in text_prompts:
            try:
                result = self.generate_gradcam_for_text(image_tensor, prompt, blur)
                results[prompt] = result
                similarities.append(result['similarity'])
            except Exception as e:
                print(f"Warning: Could not generate CAM for prompt '{prompt}': {e}")
                similarities.append(0.0)
                continue
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_prompt = text_prompts[best_idx]
        
        results['metadata'] = {
            'text_prompts': text_prompts,
            'similarities': similarities,
            'best_match_idx': best_idx,
            'best_match_prompt': best_prompt,
            'best_similarity': similarities[best_idx]
        }
        
        return results
    
    def analyze_with_common_classes(
        self,
        image_tensor: torch.Tensor,
        class_names: List[str] = None,
        template: str = "a photo of a {}",
        blur: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze image with common class names.
        
        Args:
            image_tensor: Preprocessed image tensor
            class_names: List of class names (uses defaults if None)
            template: Template for creating prompts
            blur: Whether to apply Gaussian blur
            
        Returns:
            Analysis results with GradCAMs
        """
        if class_names is None:
            class_names = self._get_default_class_names()
        
        # Create prompts from class names
        text_prompts = [template.format(class_name) for class_name in class_names]
        
        # Generate comparative GradCAM
        results = self.generate_comparative_gradcam(image_tensor, text_prompts, blur)
        
        # Add class-specific metadata
        results['metadata']['class_names'] = class_names
        results['metadata']['template'] = template
        results['metadata']['best_match_class'] = class_names[results['metadata']['best_match_idx']]
        
        return results
    
    def _get_default_class_names(self) -> List[str]:
        """Get default class names for analysis."""
        return [
            "person", "face", "man", "woman", "child",
            "car", "truck", "bus", "motorcycle", "bicycle",
            "dog", "cat", "bird", "horse", "cow",
            "airplane", "train", "boat", "ship",
            "building", "house", "tree", "flower", "grass",
            "food", "fruit", "apple", "banana",
            "computer", "phone", "book", "chair", "table",
            "real photo", "fake photo", "synthetic image", "deepfake",
            "natural scene", "indoor scene", "outdoor scene",
            "portrait", "landscape", "close-up"
        ]
    
    def preprocess_image_from_path(self, image_path: str) -> torch.Tensor:
        """Preprocess image from file path."""
        image = Image.open(image_path).convert('RGB')
        return self.preprocess(image).unsqueeze(0).to(self.device)
    
    def preprocess_image_from_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess image tensor for CLIP."""
        # If tensor is from arrow dataset, it might need denormalization and re-preprocessing
        if image_tensor.dim() == 4:
            image_tensor = image_tensor.squeeze(0)
        
        # Convert to PIL Image and re-preprocess for CLIP
        img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        
        # Denormalize (assuming standard normalization)
        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])
        img_np = img_np * std + mean
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        
        # Convert to PIL and preprocess for CLIP
        pil_image = Image.fromarray(img_np)
        return self.preprocess(pil_image).unsqueeze(0).to(self.device)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_name': self.model_name,
            'saliency_layer': self.saliency_layer,
            'device': self.device,
            'input_resolution': self.model.visual.input_resolution
        }


def visualize_clip_gradcam_results(
    original_image: np.ndarray,
    results: Dict[str, Any],
    title: str = "CLIP GradCAM Analysis",
    save_path: str = None,
    max_classes: int = 6,
    blur: bool = True
) -> plt.Figure:
    """
    Visualize CLIP GradCAM results for multiple classes.
    
    Args:
        original_image: Original image as numpy array
        results: Results from CLIPGradCAMArrow
        title: Plot title
        save_path: Path to save visualization
        max_classes: Maximum number of classes to show
        blur: Whether to apply blur to attention maps
        
    Returns:
        Matplotlib figure
    """
    metadata = results['metadata']
    similarities = metadata['similarities']
    text_prompts = metadata['text_prompts']
    
    # Sort by similarity and take top classes
    sorted_indices = np.argsort(similarities)[::-1][:max_classes]
    
    # Create subplot layout
    n_cols = 3
    n_rows = (len(sorted_indices) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, prompt_idx in enumerate(sorted_indices):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        prompt = text_prompts[prompt_idx]
        similarity = similarities[prompt_idx]
        
        if prompt in results:
            cam = results[prompt]['cam']
            
            # Create attention map overlay
            att_map = getAttMap(original_image, cam, blur)
            ax.imshow(att_map)
            ax.set_title(f"{prompt}\nSimilarity: {similarity:.3f}", fontsize=10)
        else:
            ax.imshow(original_image)
            ax.set_title(f"{prompt}\nSimilarity: {similarity:.3f}\n(CAM failed)", fontsize=10)
        
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