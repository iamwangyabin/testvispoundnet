"""
Core GradCAM implementation for Vision Transformers.

This module provides the fundamental GradCAM functionality adapted for
Vision Transformer architectures, handling patch tokens and spatial conversion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any
import cv2


class ViTGradCAM:
    """
    GradCAM implementation specifically designed for Vision Transformers.
    
    This class handles the unique aspects of ViT architectures:
    - Patch-based input representation
    - Class tokens vs patch tokens
    - Spatial reconstruction from patch gradients
    - Multiple transformer layers
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
        Initialize ViTGradCAM.
        
        Args:
            model: The Vision Transformer model
            target_layers: Layer names to hook for gradient computation
            patch_size: Size of patches in the ViT (e.g., 14 for ViT-L/14)
            input_size: Input image size (e.g., 224)
            use_cuda: Whether to use CUDA if available
        """
        self.model = model
        self.patch_size = patch_size
        self.input_size = input_size
        self.num_patches = (input_size // patch_size) ** 2
        self.patch_grid_size = input_size // patch_size
        
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Storage for gradients and activations
        self.gradients = {}
        self.activations = {}
        self.hooks = []
        
        # Set default target layers if not provided
        if target_layers is None:
            target_layers = self._get_default_target_layers()
        elif isinstance(target_layers, str):
            target_layers = [target_layers]
            
        self.target_layers = target_layers
        self._register_hooks()
    
    def _get_default_target_layers(self) -> List[str]:
        """Get default target layers for the model."""
        # Try to find transformer blocks in common ViT architectures
        layer_names = []
        for name, module in self.model.named_modules():
            if 'transformer' in name and 'resblocks' in name:
                if name.endswith('.ln_2'):  # Target the layer norm after MLP
                    layer_names.append(name)
        
        if not layer_names:
            # Fallback: try to find any transformer-like layers
            for name, module in self.model.named_modules():
                if any(keyword in name.lower() for keyword in ['block', 'layer', 'transformer']):
                    if isinstance(module, nn.LayerNorm):
                        layer_names.append(name)
        
        # Return the last few layers for analysis
        return layer_names[-3:] if len(layer_names) >= 3 else layer_names
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layers."""
        def forward_hook(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return hook
        
        for layer_name in self.target_layers:
            # Navigate to the target layer
            layer = self.model
            for attr in layer_name.split('.'):
                layer = getattr(layer, attr)
            
            # Register hooks
            forward_handle = layer.register_forward_hook(forward_hook(layer_name))
            backward_handle = layer.register_backward_hook(backward_hook(layer_name))
            
            self.hooks.extend([forward_handle, backward_handle])
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        layer_name: Optional[str] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate Class Activation Map for the input.
        
        Args:
            input_tensor: Input tensor of shape (1, 3, H, W)
            target_class: Target class index for gradient computation
            layer_name: Specific layer to use (if None, uses the last target layer)
            normalize: Whether to normalize the CAM
            
        Returns:
            CAM as numpy array of shape (H, W)
        """
        self.model.eval()
        
        # Clear previous gradients and activations
        self.gradients.clear()
        self.activations.clear()
        
        # Forward pass
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(True)
        
        output = self.model(input_tensor)
        
        # Handle different output formats
        if isinstance(output, dict):
            logits = output.get('logits', output.get('prediction', None))
            if logits is None:
                raise ValueError("Could not find logits in model output")
        else:
            logits = output
        
        # Determine target class
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_score = logits[0, target_class]
        class_score.backward(retain_graph=True)
        
        # Select layer for CAM generation
        if layer_name is None:
            layer_name = self.target_layers[-1]
        
        if layer_name not in self.gradients or layer_name not in self.activations:
            raise ValueError(f"Layer {layer_name} not found in gradients or activations")
        
        # Get gradients and activations
        gradients = self.gradients[layer_name]  # Shape: (seq_len, batch, dim)
        activations = self.activations[layer_name]  # Shape: (seq_len, batch, dim)
        
        # Convert to (batch, seq_len, dim) format
        if gradients.dim() == 3 and gradients.shape[1] == 1:
            gradients = gradients.permute(1, 0, 2)  # (batch, seq_len, dim)
            activations = activations.permute(1, 0, 2)  # (batch, seq_len, dim)
        
        # Extract patch tokens (exclude class token and prompt tokens)
        # Assuming: [class_token, patch_tokens..., prompt_tokens...]
        patch_start_idx = 1  # Skip class token
        patch_end_idx = patch_start_idx + self.num_patches
        
        patch_gradients = gradients[0, patch_start_idx:patch_end_idx, :]  # (num_patches, dim)
        patch_activations = activations[0, patch_start_idx:patch_end_idx, :]  # (num_patches, dim)
        
        # Compute importance weights (global average pooling of gradients)
        weights = torch.mean(patch_gradients, dim=0, keepdim=True)  # (1, dim)
        
        # Generate CAM by weighted combination
        cam = torch.sum(weights * patch_activations, dim=1)  # (num_patches,)
        
        # Reshape to spatial grid
        cam = cam.view(self.patch_grid_size, self.patch_grid_size)  # (14, 14)
        
        # Apply ReLU to focus on positive contributions
        cam = F.relu(cam)
        
        # Convert to numpy with proper detachment
        print(f"[DEBUG] Core CAM tensor requires_grad: {cam.requires_grad}")
        cam = cam.detach().cpu().numpy()
        
        # Normalize if requested
        if normalize:
            cam = self._normalize_cam(cam)
        
        # Upsample to input resolution
        cam = cv2.resize(cam, (self.input_size, self.input_size))
        
        return cam
    
    def _normalize_cam(self, cam: np.ndarray) -> np.ndarray:
        """Normalize CAM to [0, 1] range."""
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam
    
    def generate_multi_layer_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        normalize: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Generate CAMs for all target layers.
        
        Args:
            input_tensor: Input tensor of shape (1, 3, H, W)
            target_class: Target class index for gradient computation
            normalize: Whether to normalize the CAMs
            
        Returns:
            Dictionary mapping layer names to CAM arrays
        """
        cams = {}
        for layer_name in self.target_layers:
            try:
                cam = self.generate_cam(
                    input_tensor, 
                    target_class=target_class, 
                    layer_name=layer_name,
                    normalize=normalize
                )
                cams[layer_name] = cam
            except Exception as e:
                print(f"Warning: Could not generate CAM for layer {layer_name}: {e}")
                continue
        
        return cams
    
    def get_layer_info(self) -> Dict[str, Any]:
        """Get information about the target layers."""
        info = {
            'target_layers': self.target_layers,
            'patch_size': self.patch_size,
            'input_size': self.input_size,
            'num_patches': self.num_patches,
            'patch_grid_size': self.patch_grid_size
        }
        return info
    
    def __del__(self):
        """Cleanup hooks when object is destroyed."""
        self._remove_hooks()


class GradCAMPlusPlus(ViTGradCAM):
    """
    GradCAM++ implementation for Vision Transformers.
    
    Extends ViTGradCAM with improved gradient weighting using second-order derivatives.
    """
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        layer_name: Optional[str] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """Generate improved CAM using GradCAM++ methodology."""
        self.model.eval()
        
        # Clear previous gradients and activations
        self.gradients.clear()
        self.activations.clear()
        
        # Forward pass
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(True)
        
        output = self.model(input_tensor)
        
        # Handle different output formats
        if isinstance(output, dict):
            logits = output.get('logits', output.get('prediction', None))
            if logits is None:
                raise ValueError("Could not find logits in model output")
        else:
            logits = output
        
        # Determine target class
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        
        # Compute gradients
        self.model.zero_grad()
        class_score = logits[0, target_class]
        
        # First-order gradients
        first_grads = torch.autograd.grad(
            outputs=class_score,
            inputs=input_tensor,
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Second-order gradients
        second_grads = torch.autograd.grad(
            outputs=first_grads.sum(),
            inputs=input_tensor,
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Third-order gradients
        third_grads = torch.autograd.grad(
            outputs=second_grads.sum(),
            inputs=input_tensor,
            retain_graph=True
        )[0]
        
        # Backward pass for activations
        class_score.backward(retain_graph=True)
        
        # Select layer for CAM generation
        if layer_name is None:
            layer_name = self.target_layers[-1]
        
        if layer_name not in self.gradients or layer_name not in self.activations:
            raise ValueError(f"Layer {layer_name} not found in gradients or activations")
        
        # Get gradients and activations
        gradients = self.gradients[layer_name]
        activations = self.activations[layer_name]
        
        # Convert to (batch, seq_len, dim) format
        if gradients.dim() == 3 and gradients.shape[1] == 1:
            gradients = gradients.permute(1, 0, 2)
            activations = activations.permute(1, 0, 2)
        
        # Extract patch tokens
        patch_start_idx = 1
        patch_end_idx = patch_start_idx + self.num_patches
        
        patch_gradients = gradients[0, patch_start_idx:patch_end_idx, :]
        patch_activations = activations[0, patch_start_idx:patch_end_idx, :]
        
        # Compute GradCAM++ weights
        alpha_num = patch_gradients.pow(2)
        alpha_denom = 2.0 * patch_gradients.pow(2) + \
                     patch_activations.sum(dim=0, keepdim=True) * patch_gradients.pow(3)
        
        alpha = alpha_num / (alpha_denom + 1e-7)
        weights = (alpha * F.relu(patch_gradients)).sum(dim=0, keepdim=True)
        
        # Generate CAM
        cam = torch.sum(weights * patch_activations, dim=1)
        
        # Reshape to spatial grid
        cam = cam.view(self.patch_grid_size, self.patch_grid_size)
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Convert to numpy with proper detachment
        print(f"[DEBUG] GradCAM++ CAM tensor requires_grad: {cam.requires_grad}")
        cam = cam.detach().cpu().numpy()
        
        # Normalize if requested
        if normalize:
            cam = self._normalize_cam(cam)
        
        # Upsample to input resolution
        cam = cv2.resize(cam, (self.input_size, self.input_size))
        
        return cam