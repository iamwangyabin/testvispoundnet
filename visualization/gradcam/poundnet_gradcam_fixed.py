"""
PoundNet-specific GradCAM implementation with attention-based visualization.

This module provides GradCAM functionality specifically adapted for the PoundNet
deepfake detection model, handling CLIP architecture and prompt learning.
Since CLIP ViT only uses class token for classification, we use attention-based
visualization instead of gradient-based methods.
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
    - Attention-based visualization (since gradients don't flow to patch tokens)
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
        # Store original model state for restoration
        self.original_requires_grad = {}
        self.model = model  # Store model reference before enabling gradients
        
        # Enable gradients for GradCAM computation
        self._enable_gradients_for_gradcam()
        
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
    
    def _enable_gradients_for_gradcam(self):
        """Enable gradients for all parameters needed for GradCAM computation."""
        # Store original gradient states
        for name, param in self.model.named_parameters():
            self.original_requires_grad[name] = param.requires_grad
        
        # Enable gradients for image encoder (visual transformer)
        # This must be done AFTER model initialization to override PoundNet's gradient control
        for name, param in self.model.named_parameters():
            if 'image_encoder' in name:
                param.requires_grad_(True)
    
    def _restore_gradients(self):
        """Restore original gradient states."""
        for name, param in self.model.named_parameters():
            if name in self.original_requires_grad:
                param.requires_grad_(self.original_requires_grad[name])
    
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
            selected = [layer_names[6], layer_names[12], layer_names[-1]]  # Early, middle, late
            return selected
        else:
            selected = layer_names[-2:] if len(layer_names) >= 2 else layer_names
            return selected
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[Union[int, str]] = None,
        layer_name: Optional[str] = None,
        normalize: bool = True,
        return_logits: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, torch.Tensor]]:
        """
        Generate Class Activation Map for PoundNet input using gradient-based visualization.
        
        This method computes class-specific gradients to generate different CAMs
        for real and fake classes, ensuring proper class differentiation.
        
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
        
        # Clear previous gradients and activations for fresh computation
        self.gradients.clear()
        self.activations.clear()
        
        # Ensure input tensor requires gradients and is on correct device
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(True)
        
        # Forward pass to get logits
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
        
        # Generate class-specific gradient-based CAM
        cam = self._generate_attention_cam(input_tensor, target_class_idx)
        
        # Enhanced normalization and contrast improvement
        if normalize:
            cam = self._enhanced_normalize_cam(cam)
        
        # Upsample to input resolution with better interpolation
        cam = cv2.resize(cam, (self.input_size, self.input_size), interpolation=cv2.INTER_CUBIC)
        
        if return_logits:
            return cam, logits
        else:
            return cam
    
    def _generate_attention_cam(self, input_tensor: torch.Tensor, target_class_idx: int) -> np.ndarray:
        """
        Generate CLASS-SPECIFIC attention-based CAM using gradient-weighted attention patterns.
        
        This method computes gradients with respect to the target class and uses them
        to weight attention patterns, making the visualization class-specific.
        """
        
        # Clear previous gradients and activations
        self.gradients.clear()
        self.activations.clear()
        
        # Enable gradients for input tensor
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(True)
        
        # Forward pass to get logits and capture activations
        output = self.model(input_tensor)
        
        # Extract logits from PoundNet output
        if isinstance(output, dict):
            logits = output['logits']
        else:
            logits = output
        
        # Compute gradients with respect to target class
        self.model.zero_grad()
        class_score = logits[0, target_class_idx]
        
        if not class_score.requires_grad:
            # Fallback to activation-based method if gradients not available
            return self._generate_activation_cam(input_tensor, target_class_idx)
        
        # Backward pass to compute gradients
        class_score.backward(retain_graph=True)
        
        # Use the last target layer for gradient-weighted attention
        if not self.activations:
            return self._generate_activation_cam(input_tensor, target_class_idx)
        
        layer_name = list(self.activations.keys())[-1]
        activations = self.activations[layer_name]
        
        # Get corresponding gradients
        if layer_name in self.gradients:
            gradients = self.gradients[layer_name]
        else:
            # If no gradients available, use activation magnitudes
            return self._generate_activation_cam(input_tensor, target_class_idx)
        
        # Handle different tensor formats
        if activations.dim() == 3:
            # Format: (seq_len, batch, dim) -> (batch, seq_len, dim)
            if activations.shape[1] == 1:
                activations = activations.permute(1, 0, 2)
                gradients = gradients.permute(1, 0, 2)
        
        # Extract patch tokens (skip class token at position 0)
        batch_size = activations.shape[0]
        seq_len = activations.shape[1]
        
        # Calculate patch token range
        patch_start_idx = 1
        patch_end_idx = min(257, seq_len)  # Up to 256 patches
        
        patch_activations = activations[0, patch_start_idx:patch_end_idx, :]  # (num_patches, dim)
        patch_gradients = gradients[0, patch_start_idx:patch_end_idx, :]  # (num_patches, dim)
        
        # Compute class-specific importance weights using gradients
        # Method 1: Global average pooling of gradients (standard GradCAM)
        weights = torch.mean(patch_gradients, dim=0, keepdim=True)  # (1, dim)
        cam_standard = torch.sum(weights * patch_activations, dim=1)  # (num_patches,)
        
        # Method 2: Use gradient magnitude for stronger class differentiation
        grad_magnitude = torch.norm(patch_gradients, dim=1)  # (num_patches,)
        act_magnitude = torch.norm(patch_activations, dim=1)  # (num_patches,)
        cam_magnitude = grad_magnitude * act_magnitude  # (num_patches,)
        
        # Method 3: Element-wise gradient-activation product
        cam_elementwise = torch.sum(torch.abs(patch_gradients) * torch.abs(patch_activations), dim=1)  # (num_patches,)
        
        # Choose the method with strongest class-specific signal
        cam_methods = {
            'standard': cam_standard,
            'magnitude': cam_magnitude,
            'elementwise': cam_elementwise
        }
        
        # Select method with highest dynamic range (better class differentiation)
        best_method = max(cam_methods.keys(),
                         key=lambda k: (cam_methods[k].max() - cam_methods[k].min()).item())
        cam = cam_methods[best_method]
        
        # Ensure we have exactly 256 patches
        if len(cam) < 256:
            # Pad with zeros
            padding = 256 - len(cam)
            cam = F.pad(cam, (0, padding))
        elif len(cam) > 256:
            # Truncate
            cam = cam[:256]
        
        # Reshape to spatial grid (16x16)
        attention_map = cam.view(16, 16)
        
        # Apply ReLU to focus on positive contributions
        attention_map = F.relu(attention_map)
        
        # Convert to numpy
        attention_map = attention_map.detach().cpu().numpy()
        
        return attention_map
    
    def _generate_activation_cam(self, input_tensor: torch.Tensor, target_class_idx: int) -> np.ndarray:
        """
        Fallback method using activation magnitudes when attention weights are not available.
        """
        
        # Clear previous activations
        self.activations.clear()
        
        # Forward pass to capture activations
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        if not self.activations:
            # Return a uniform attention map as last resort
            return np.ones((16, 16), dtype=np.float32) * 0.5
        
        # Use the last captured layer
        layer_name = list(self.activations.keys())[-1]
        activations = self.activations[layer_name]
        
        # Handle different tensor formats
        if activations.dim() == 3:
            # Format: (seq_len, batch, dim) -> (batch, seq_len, dim)
            if activations.shape[1] == 1:
                activations = activations.permute(1, 0, 2)
        
        # Extract patch tokens (skip class token at position 0)
        batch_size = activations.shape[0]
        seq_len = activations.shape[1]
        
        # Calculate patch token range
        patch_start_idx = 1
        patch_end_idx = min(257, seq_len)  # Up to 256 patches
        
        patch_activations = activations[0, patch_start_idx:patch_end_idx, :]  # (num_patches, dim)
        
        # Compute activation magnitude for each patch
        patch_magnitudes = torch.norm(patch_activations, dim=1)  # (num_patches,)
        
        # Ensure we have exactly 256 patches
        if len(patch_magnitudes) < 256:
            # Pad with zeros
            padding = 256 - len(patch_magnitudes)
            patch_magnitudes = F.pad(patch_magnitudes, (0, padding))
        elif len(patch_magnitudes) > 256:
            # Truncate
            patch_magnitudes = patch_magnitudes[:256]
        
        # Reshape to spatial grid (16x16)
        activation_map = patch_magnitudes.view(16, 16)
        
        # Convert to numpy
        activation_map = activation_map.detach().cpu().numpy()
        
        
        return activation_map
    
    def generate_comparative_cam(
        self,
        input_tensor: torch.Tensor,
        layer_name: Optional[str] = None,
        normalize: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Generate CAMs for both Real and Fake classes for comparison.
        
        This method ensures proper class-specific gradient computation by
        performing separate backward passes for each class.
        
        Args:
            input_tensor: Input tensor of shape (1, 3, H, W)
            layer_name: Specific layer to use
            normalize: Whether to normalize the CAMs
            
        Returns:
            Dictionary with 'real' and 'fake' CAMs
        """
        # Clear any previous gradients
        self.model.zero_grad()
        
        # Generate Real class CAM with fresh computation
        real_cam = self.generate_cam(
            input_tensor.clone(),
            target_class='real',
            layer_name=layer_name,
            normalize=normalize
        )
        
        # Clear gradients between computations to ensure independence
        self.model.zero_grad()
        self.gradients.clear()
        self.activations.clear()
        
        # Generate Fake class CAM with fresh computation
        fake_cam = self.generate_cam(
            input_tensor.clone(),
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
            'logits': logits.detach().cpu().numpy(),
            'probabilities': probabilities.detach().cpu().numpy(),
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
    
    def _enhanced_normalize_cam(self, cam: np.ndarray) -> np.ndarray:
        """Enhanced CAM normalization with better contrast and attention highlighting."""
        # Remove any NaN or infinite values
        cam = np.nan_to_num(cam, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Apply gamma correction to enhance contrast
        gamma = 0.7  # Values < 1 enhance bright regions
        cam_gamma = np.power(cam, gamma)
        
        # Standard min-max normalization
        cam_min, cam_max = cam_gamma.min(), cam_gamma.max()
        if cam_max > cam_min:
            cam_normalized = (cam_gamma - cam_min) / (cam_max - cam_min)
        else:
            cam_normalized = cam_gamma
        
        # Apply histogram equalization for better contrast
        cam_uint8 = (cam_normalized * 255).astype(np.uint8)
        cam_equalized = cv2.equalizeHist(cam_uint8)
        cam_enhanced = cam_equalized.astype(np.float32) / 255.0
        
        # Apply sigmoid enhancement to make attention more prominent
        cam_sigmoid = 1 / (1 + np.exp(-10 * (cam_enhanced - 0.5)))
        
        # Final normalization
        cam_final = (cam_sigmoid - cam_sigmoid.min()) / (cam_sigmoid.max() - cam_sigmoid.min() + 1e-8)
        
        
        return cam_final

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
    
    def __del__(self):
        """Cleanup: restore original gradient states and remove hooks."""
        try:
            self._restore_gradients()
            self._remove_hooks()
        except:
            pass  # Ignore errors during cleanup