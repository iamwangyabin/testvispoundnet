"""
PoundNet-specific GradCAM implementation.

This module provides GradCAM functionality specifically adapted for the PoundNet
deepfake detection model, handling CLIP architecture and prompt learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any
import cv2

from .core import ViTGradCAM


class PoundNetGradCAM(ViTGradCAM):
    """
    GradCAM implementation specifically designed for PoundNet architecture.
    
    This class handles PoundNet-specific aspects:
    - CLIP Vision Transformer with learnable prompts
    - Real vs Fake classification
    - Text-image similarity computation
    - Prompt token handling
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layers: Union[str, List[str]] = None,
        patch_size: int = 14,
        input_size: int = 224,
        use_cuda: bool = True
    ):
        """
        Initialize PoundNetGradCAM.
        
        Args:
            model: The PoundNet model
            target_layers: Layer names to hook for gradient computation
            patch_size: Size of patches in the ViT (14 for ViT-L/14)
            input_size: Input image size (224)
            use_cuda: Whether to use CUDA if available
        """
        # Initialize parent class
        super().__init__(model, target_layers, patch_size, input_size, use_cuda)
        
        # PoundNet-specific attributes
        self.num_prompt_tokens = getattr(model.cfg.model, 'N_CTX_VISION', 16)
        self.class_names = ['Real', 'Fake']
        
        # Override target layers for PoundNet architecture
        if target_layers is None:
            self.target_layers = self._get_poundnet_target_layers()
            self._remove_hooks()  # Remove old hooks
            self._register_hooks()  # Register new hooks
    
    def _get_poundnet_target_layers(self) -> List[str]:
        """Get optimal target layers for PoundNet architecture."""
        layer_names = []
        
        # Target the visual encoder transformer blocks
        for name, module in self.model.named_modules():
            if 'image_encoder.transformer.resblocks' in name:
                if name.endswith('.ln_2'):  # Layer norm after MLP
                    layer_names.append(name)
        
        # If no transformer blocks found, try alternative naming
        if not layer_names:
            for name, module in self.model.named_modules():
                if 'visual.transformer.resblocks' in name:
                    if name.endswith('.ln_2'):
                        layer_names.append(name)
        
        # Return last few layers for best results
        if len(layer_names) >= 3:
            return [layer_names[6], layer_names[12], layer_names[-1]]  # Early, middle, late
        else:
            return layer_names[-2:] if len(layer_names) >= 2 else layer_names
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[Union[int, str]] = None,
        layer_name: Optional[str] = None,
        normalize: bool = True,
        return_logits: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, torch.Tensor]]:
        """
        Generate Class Activation Map for PoundNet input.
        
        Args:
            input_tensor: Input tensor of shape (1, 3, H, W)
            target_class: Target class (0/'Real', 1/'Fake', or None for predicted)
            layer_name: Specific layer to use (if None, uses the last target layer)
            normalize: Whether to normalize the CAM
            return_logits: Whether to return logits along with CAM
            
        Returns:
            CAM as numpy array of shape (H, W), optionally with logits
        """
        self.model.eval()
        
        # Clear previous gradients and activations
        self.gradients.clear()
        self.activations.clear()
        
        # Forward pass
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(True)
        
        # PoundNet forward pass
        output = self.model(input_tensor)
        
        # Extract logits from PoundNet output
        if isinstance(output, dict):
            logits = output['logits']
        else:
            logits = output
        
        # Handle target class specification
        if target_class is None:
            target_class_idx = logits.argmax(dim=1).item()
        elif isinstance(target_class, str):
            if target_class.lower() == 'real':
                target_class_idx = 0
            elif target_class.lower() == 'fake':
                target_class_idx = 1
            else:
                raise ValueError(f"Invalid target class string: {target_class}")
        else:
            target_class_idx = target_class
        
        # Backward pass
        self.model.zero_grad()
        class_score = logits[0, target_class_idx]
        class_score.backward(retain_graph=True)
        
        # Select layer for CAM generation
        if layer_name is None:
            layer_name = self.target_layers[-1]
        
        if layer_name not in self.gradients or layer_name not in self.activations:
            raise ValueError(f"Layer {layer_name} not found in gradients or activations")
        
        # Get gradients and activations
        gradients = self.gradients[layer_name]
        activations = self.activations[layer_name]
        
        # Handle different tensor formats
        if gradients.dim() == 3:
            # Format: (seq_len, batch, dim) -> (batch, seq_len, dim)
            if gradients.shape[1] == 1:
                gradients = gradients.permute(1, 0, 2)
                activations = activations.permute(1, 0, 2)
        
        # Extract patch tokens, accounting for PoundNet structure:
        # [class_token, patch_tokens..., prompt_tokens...]
        batch_size = gradients.shape[0]
        seq_len = gradients.shape[1]
        
        # Calculate token positions
        class_token_idx = 0
        patch_start_idx = 1
        patch_end_idx = patch_start_idx + self.num_patches
        prompt_start_idx = patch_end_idx
        
        # Ensure we don't exceed sequence length
        if patch_end_idx > seq_len:
            # Fallback: use all tokens except class token
            patch_end_idx = seq_len
            patch_start_idx = 1
        
        # Extract patch tokens
        patch_gradients = gradients[0, patch_start_idx:patch_end_idx, :]
        patch_activations = activations[0, patch_start_idx:patch_end_idx, :]
        
        # Adjust for actual number of patch tokens
        actual_patch_tokens = patch_gradients.shape[0]
        if actual_patch_tokens != self.num_patches:
            # Recalculate grid size
            grid_size = int(np.sqrt(actual_patch_tokens))
            if grid_size * grid_size != actual_patch_tokens:
                # Handle non-square patch arrangements
                grid_size = self.patch_grid_size
                # Pad or truncate to expected size
                if actual_patch_tokens < self.num_patches:
                    padding = self.num_patches - actual_patch_tokens
                    patch_gradients = F.pad(patch_gradients, (0, 0, 0, padding))
                    patch_activations = F.pad(patch_activations, (0, 0, 0, padding))
                else:
                    patch_gradients = patch_gradients[:self.num_patches]
                    patch_activations = patch_activations[:self.num_patches]
        else:
            grid_size = self.patch_grid_size
        
        # Compute importance weights (global average pooling of gradients)
        weights = torch.mean(patch_gradients, dim=0, keepdim=True)  # (1, dim)
        
        # Generate CAM by weighted combination
        cam = torch.sum(weights * patch_activations, dim=1)  # (num_patches,)
        
        # Reshape to spatial grid
        cam = cam[:grid_size * grid_size].view(grid_size, grid_size)
        
        # Apply ReLU to focus on positive contributions
        cam = F.relu(cam)
        
        # Convert to numpy
        cam = cam.cpu().numpy()
        
        # Normalize if requested
        if normalize:
            cam = self._normalize_cam(cam)
        
        # Upsample to input resolution
        cam = cv2.resize(cam, (self.input_size, self.input_size))
        
        if return_logits:
            return cam, logits
        else:
            return cam
    
    def generate_comparative_cam(
        self,
        input_tensor: torch.Tensor,
        layer_name: Optional[str] = None,
        normalize: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Generate CAMs for both Real and Fake classes for comparison.
        
        Args:
            input_tensor: Input tensor of shape (1, 3, H, W)
            layer_name: Specific layer to use
            normalize: Whether to normalize the CAMs
            
        Returns:
            Dictionary with 'real' and 'fake' CAMs
        """
        real_cam = self.generate_cam(
            input_tensor, 
            target_class='real', 
            layer_name=layer_name,
            normalize=normalize
        )
        
        fake_cam = self.generate_cam(
            input_tensor, 
            target_class='fake', 
            layer_name=layer_name,
            normalize=normalize
        )
        
        return {
            'real': real_cam,
            'fake': fake_cam
        }
    
    def generate_prediction_with_cam(
        self,
        input_tensor: torch.Tensor,
        layer_name: Optional[str] = None,
        normalize: bool = True,
        confidence_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Generate prediction along with corresponding CAM.
        
        Args:
            input_tensor: Input tensor of shape (1, 3, H, W)
            layer_name: Specific layer to use
            normalize: Whether to normalize the CAM
            confidence_threshold: Threshold for binary classification
            
        Returns:
            Dictionary containing prediction results and CAM
        """
        # Generate CAM for predicted class
        cam, logits = self.generate_cam(
            input_tensor,
            target_class=None,  # Use predicted class
            layer_name=layer_name,
            normalize=normalize,
            return_logits=True
        )
        
        # Process prediction
        probabilities = F.softmax(logits, dim=1)[0]
        predicted_class_idx = logits.argmax(dim=1).item()
        predicted_class_name = self.class_names[predicted_class_idx]
        confidence = probabilities[predicted_class_idx].item()
        
        # Determine if prediction is confident
        is_confident = confidence > confidence_threshold
        
        return {
            'cam': cam,
            'logits': logits.cpu().numpy(),
            'probabilities': probabilities.cpu().numpy(),
            'predicted_class_idx': predicted_class_idx,
            'predicted_class_name': predicted_class_name,
            'confidence': confidence,
            'is_confident': is_confident,
            'real_prob': probabilities[0].item(),
            'fake_prob': probabilities[1].item()
        }
    
    def analyze_attention_patterns(
        self,
        input_tensor: torch.Tensor,
        normalize: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze attention patterns across multiple layers.
        
        Args:
            input_tensor: Input tensor of shape (1, 3, H, W)
            normalize: Whether to normalize the CAMs
            
        Returns:
            Dictionary containing multi-layer analysis
        """
        results = {}
        
        # Generate CAMs for all target layers
        for layer_name in self.target_layers:
            try:
                # Get CAMs for both classes
                real_cam = self.generate_cam(
                    input_tensor, 
                    target_class='real', 
                    layer_name=layer_name,
                    normalize=normalize
                )
                
                fake_cam = self.generate_cam(
                    input_tensor, 
                    target_class='fake', 
                    layer_name=layer_name,
                    normalize=normalize
                )
                
                # Compute attention statistics
                real_stats = {
                    'mean': float(np.mean(real_cam)),
                    'std': float(np.std(real_cam)),
                    'max': float(np.max(real_cam)),
                    'min': float(np.min(real_cam))
                }
                
                fake_stats = {
                    'mean': float(np.mean(fake_cam)),
                    'std': float(np.std(fake_cam)),
                    'max': float(np.max(fake_cam)),
                    'min': float(np.min(fake_cam))
                }
                
                results[layer_name] = {
                    'real_cam': real_cam,
                    'fake_cam': fake_cam,
                    'real_stats': real_stats,
                    'fake_stats': fake_stats,
                    'difference_cam': fake_cam - real_cam
                }
                
            except Exception as e:
                print(f"Warning: Could not analyze layer {layer_name}: {e}")
                continue
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the PoundNet model."""
        info = super().get_layer_info()
        info.update({
            'model_type': 'PoundNet',
            'num_prompt_tokens': self.num_prompt_tokens,
            'class_names': self.class_names,
            'architecture': 'CLIP ViT-L/14 with learnable prompts'
        })
        return info