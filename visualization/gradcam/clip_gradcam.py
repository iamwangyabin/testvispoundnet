"""
CLIP GradCAM implementation for visualizing attention patterns.

This module provides GradCAM functionality specifically designed for CLIP models,
allowing visualization of how the model attends to different regions when 
classifying images with text prompts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any
import cv2
import sys
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from networks.clip import clip
from visualization.gradcam.core import ViTGradCAM


class CLIPGradCAM:
    """
    GradCAM implementation for CLIP models.
    
    This class handles CLIP-specific aspects:
    - Text-image similarity computation
    - Vision transformer attention visualization
    - Multiple text prompt support
    """
    
    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str = None,
        target_layers: Union[str, List[str]] = None
    ):
        """
        Initialize CLIP GradCAM.
        
        Args:
            model_name: CLIP model name (e.g., "ViT-B/32", "ViT-L/14")
            device: Device to use ('cuda' or 'cpu')
            target_layers: Specific layers to target for GradCAM
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        
        # Load CLIP model with default design details for vanilla CLIP
        print(f"Loading CLIP model: {model_name}")
        
        # Create default design details for vanilla CLIP (no prompting)
        design_details = {
            'trainer': 'CoOp',  # Use basic transformer blocks
            'vision_depth': 0,   # No visual prompting
            'language_depth': 0, # No text prompting
            'vision_ctx': 0,     # No visual context
            'language_ctx': 0    # No language context
        }
        
        # Monkey patch the build_model function to include design_details
        original_build_model = clip.model.build_model
        def patched_build_model(state_dict):
            return original_build_model(state_dict, design_details)
        
        # Temporarily replace build_model
        clip.model.build_model = patched_build_model
        
        try:
            self.model, self.preprocess = clip.load(model_name, device=device)
            self.model.eval()
        finally:
            # Restore original build_model
            clip.model.build_model = original_build_model
        
        # Get model specifications
        self.model_name = model_name
        if "ViT-B/32" in model_name:
            self.patch_size = 32
            self.input_size = 224
        elif "ViT-B/16" in model_name:
            self.patch_size = 16
            self.input_size = 224
        elif "ViT-L/14" in model_name:
            self.patch_size = 14
            self.input_size = 224
        else:
            # Default values
            self.patch_size = 32
            self.input_size = 224
        
        # Set up target layers
        if target_layers is None:
            target_layers = self._get_default_target_layers()
        elif isinstance(target_layers, str):
            target_layers = [target_layers]
        
        self.target_layers = target_layers
        
        # Initialize ViT GradCAM for the visual encoder
        self.vit_gradcam = ViTGradCAM(
            model=self.model.visual,
            target_layers=target_layers,
            patch_size=self.patch_size,
            input_size=self.input_size,
            use_cuda=(device == "cuda")
        )
        
        print(f"CLIP GradCAM initialized with model: {model_name}")
        print(f"Target layers: {target_layers}")
    
    def _get_default_target_layers(self) -> List[str]:
        """Get default target layers for CLIP visual encoder."""
        # CLIP ViT uses transformer.resblocks structure
        layer_names = []
        for name, module in self.model.visual.named_modules():
            if 'transformer.resblocks' in name and name.endswith('.ln_2'):
                layer_names.append(name)
        
        # Return the last few layers
        return layer_names[-3:] if len(layer_names) >= 3 else layer_names
    
    def generate_cam_for_text(
        self,
        image: torch.Tensor,
        text_prompts: Union[str, List[str]],
        target_prompt_idx: int = None,
        layer_name: str = None,
        normalize: bool = True
    ) -> Dict[str, Any]:
        """
        Generate CAM for specific text prompt(s).
        
        Args:
            image: Input image tensor (1, 3, H, W) or preprocessed
            text_prompts: Text prompt(s) to analyze
            target_prompt_idx: Index of target prompt (if None, uses highest similarity)
            layer_name: Specific layer to analyze
            normalize: Whether to normalize the CAM
            
        Returns:
            Dictionary containing CAM, similarities, and metadata
        """
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]
        
        # Tokenize text prompts
        text_tokens = clip.tokenize(text_prompts).to(self.device)
        
        # Ensure image is on correct device and has correct shape
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)
        
        # Create a wrapper model that outputs logits for the text prompts
        class CLIPWrapper(nn.Module):
            def __init__(self, clip_model, text_features):
                super().__init__()
                self.clip_model = clip_model
                self.text_features = text_features
                
            def forward(self, image):
                image_features = self.clip_model.encode_image(image)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Compute similarities (logits)
                logit_scale = self.clip_model.logit_scale.exp()
                logits = logit_scale * image_features @ self.text_features.t()
                return logits
        
        # Get text features
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Create wrapper model
        wrapper_model = CLIPWrapper(self.model, text_features)
        
        # Temporarily replace the visual model in vit_gradcam
        original_model = self.vit_gradcam.model
        self.vit_gradcam.model = wrapper_model
        
        try:
            # Determine target class (prompt index)
            if target_prompt_idx is None:
                with torch.no_grad():
                    logits = wrapper_model(image)
                    target_prompt_idx = logits.argmax(dim=1).item()
            
            # Generate CAM
            cam = self.vit_gradcam.generate_cam(
                input_tensor=image,
                target_class=target_prompt_idx,
                layer_name=layer_name,
                normalize=normalize
            )
            
            # Get similarities for all prompts
            with torch.no_grad():
                logits = wrapper_model(image)
                similarities = F.softmax(logits, dim=1)[0].cpu().numpy()
            
            result = {
                'cam': cam,
                'similarities': similarities,
                'text_prompts': text_prompts,
                'target_prompt_idx': target_prompt_idx,
                'target_prompt': text_prompts[target_prompt_idx],
                'target_similarity': similarities[target_prompt_idx],
                'layer_name': layer_name or self.target_layers[-1]
            }
            
            return result
            
        finally:
            # Restore original model
            self.vit_gradcam.model = original_model
    
    def generate_comparative_cam(
        self,
        image: torch.Tensor,
        text_prompts: List[str],
        layer_name: str = None,
        normalize: bool = True
    ) -> Dict[str, Any]:
        """
        Generate CAMs for multiple text prompts for comparison.
        
        Args:
            image: Input image tensor
            text_prompts: List of text prompts to compare
            layer_name: Specific layer to analyze
            normalize: Whether to normalize CAMs
            
        Returns:
            Dictionary with CAMs for each prompt and comparison data
        """
        results = {}
        all_similarities = []
        
        for i, prompt in enumerate(text_prompts):
            try:
                result = self.generate_cam_for_text(
                    image=image,
                    text_prompts=text_prompts,
                    target_prompt_idx=i,
                    layer_name=layer_name,
                    normalize=normalize
                )
                
                results[prompt] = {
                    'cam': result['cam'],
                    'similarity': result['similarities'][i]
                }
                
                if i == 0:
                    all_similarities = result['similarities']
                    
            except Exception as e:
                print(f"Warning: Could not generate CAM for prompt '{prompt}': {e}")
                continue
        
        # Add comparison metadata
        results['metadata'] = {
            'text_prompts': text_prompts,
            'similarities': all_similarities,
            'best_match_idx': np.argmax(all_similarities),
            'best_match_prompt': text_prompts[np.argmax(all_similarities)],
            'layer_name': layer_name or self.target_layers[-1]
        }
        
        return results
    
    def analyze_with_common_classes(
        self,
        image: torch.Tensor,
        class_names: List[str] = None,
        layer_name: str = None,
        normalize: bool = True,
        template: str = "a photo of a {}"
    ) -> Dict[str, Any]:
        """
        Analyze image with common class names.
        
        Args:
            image: Input image tensor
            class_names: List of class names (if None, uses default common classes)
            layer_name: Specific layer to analyze
            normalize: Whether to normalize CAMs
            template: Text template for prompts
            
        Returns:
            Analysis results with CAMs and classifications
        """
        if class_names is None:
            class_names = self._get_default_class_names()
        
        # Create text prompts from class names
        text_prompts = [template.format(class_name) for class_name in class_names]
        
        # Generate comparative CAMs
        results = self.generate_comparative_cam(
            image=image,
            text_prompts=text_prompts,
            layer_name=layer_name,
            normalize=normalize
        )
        
        # Add class-specific metadata
        results['metadata']['class_names'] = class_names
        results['metadata']['template'] = template
        results['metadata']['best_match_class'] = class_names[results['metadata']['best_match_idx']]
        
        return results
    
    def _get_default_class_names(self) -> List[str]:
        """Get default common class names for analysis."""
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
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the CLIP model."""
        return {
            'model_name': self.model_name,
            'patch_size': self.patch_size,
            'input_size': self.input_size,
            'target_layers': self.target_layers,
            'device': self.device,
            'num_patches': (self.input_size // self.patch_size) ** 2
        }
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess image for CLIP model.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image tensor
        """
        from PIL import Image
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess(image).unsqueeze(0)
        return image_tensor
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'vit_gradcam'):
            del self.vit_gradcam        }
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess image for CLIP model.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image tensor
        """
        from PIL import Image
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess(image).unsqueeze(0)
        return image_tensor
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'vit_gradcam'):
            del self.vit_gradcam
