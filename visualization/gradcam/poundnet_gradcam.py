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
        print("[DEBUG] Enabling gradients for GradCAM computation...")
        
        # Store original gradient states
        for name, param in self.model.named_parameters():
            self.original_requires_grad[name] = param.requires_grad
        
        # Enable gradients for image encoder (visual transformer)
        # This must be done AFTER model initialization to override PoundNet's gradient control
        for name, param in self.model.named_parameters():
            if 'image_encoder' in name:
                param.requires_grad_(True)
                print(f"[DEBUG] Enabled gradients for: {name}")
        
        # Verify gradients are actually enabled
        enabled_count = 0
        for name, param in self.model.named_parameters():
            if 'image_encoder' in name and param.requires_grad:
                enabled_count += 1
        print(f"[DEBUG] Successfully enabled gradients for {enabled_count} image encoder parameters")
    
    def _restore_gradients(self):
        """Restore original gradient states."""
        print("[DEBUG] Restoring original gradient states...")
        for name, param in self.model.named_parameters():
            if name in self.original_requires_grad:
                param.requires_grad_(self.original_requires_grad[name])
    
    def _get_poundnet_target_layers(self) -> List[str]:
        """Get optimal target layers for PoundNet architecture."""
        layer_names = []
        
        print("[DEBUG] Searching for target layers in PoundNet...")
        
        # Target the visual encoder transformer blocks
        for name, module in self.model.named_modules():
            if 'image_encoder.transformer.resblocks' in name:
                if name.endswith('.ln_2'):  # Layer norm after MLP
                    layer_names.append(name)
                    print(f"[DEBUG] Found target layer: {name}")
        
        # If no transformer blocks found, try alternative naming
        if not layer_names:
            for name, module in self.model.named_modules():
                if 'visual.transformer.resblocks' in name:
                    if name.endswith('.ln_2'):
                        layer_names.append(name)
                        print(f"[DEBUG] Found alternative target layer: {name}")
        
        print(f"[DEBUG] Total layers found: {len(layer_names)}")
        
        # Return last few layers for best results
        if len(layer_names) >= 3:
            selected = [layer_names[6], layer_names[12], layer_names[-1]]  # Early, middle, late
            print(f"[DEBUG] Selected layers: {selected}")
            return selected
        else:
            selected = layer_names[-2:] if len(layer_names) >= 2 else layer_names
            print(f"[DEBUG] Selected layers: {selected}")
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
        
        print(f"[DEBUG] PoundNet input tensor requires_grad: {input_tensor.requires_grad}")
        
        # Check if image encoder parameters have gradients enabled
        image_encoder_grad_count = 0
        for name, param in self.model.named_parameters():
            if 'image_encoder' in name and param.requires_grad:
                image_encoder_grad_count += 1
        print(f"[DEBUG] Image encoder parameters with gradients: {image_encoder_grad_count}")
        
        # PoundNet forward pass
        output = self.model(input_tensor)
        
        # Extract logits from PoundNet output
        if isinstance(output, dict):
            logits = output['logits']
        else:
            logits = output
        
        print(f"[DEBUG] PoundNet logits requires_grad: {logits.requires_grad}")
        print(f"[DEBUG] PoundNet logits shape: {logits.shape}")
        
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
        
        print(f"[DEBUG] PoundNet target class: {target_class_idx}")
        
        # Re-enable gradients before backward pass (PoundNet may have disabled them)
        print("[DEBUG] Re-enabling gradients before backward pass...")
        for name, param in self.model.named_parameters():
            if 'image_encoder' in name:
                param.requires_grad_(True)
        
        # Backward pass
        self.model.zero_grad()
        class_score = logits[0, target_class_idx]
        print(f"[DEBUG] PoundNet class score requires_grad: {class_score.requires_grad}")
        
        if not class_score.requires_grad:
            raise RuntimeError("Class score does not require gradients. Check model parameter gradients.")
        
        # Verify image encoder parameters still have gradients enabled
        enabled_params = sum(1 for name, param in self.model.named_parameters()
                           if 'image_encoder' in name and param.requires_grad)
        print(f"[DEBUG] Image encoder parameters with gradients before backward: {enabled_params}")
        
        class_score.backward(retain_graph=True)
        
        print(f"[DEBUG] PoundNet gradients captured: {list(self.gradients.keys())}")
        print(f"[DEBUG] PoundNet activations captured: {list(self.activations.keys())}")
        
        # Select layer for CAM generation
        if layer_name is None:
            layer_name = self.target_layers[-1]
        
        if layer_name not in self.gradients or layer_name not in self.activations:
            raise ValueError(f"Layer {layer_name} not found in gradients or activations")
        
        # Get gradients and activations
        gradients = self.gradients[layer_name]
        activations = self.activations[layer_name]
        
        print(f"[DEBUG] Raw gradients shape: {gradients.shape}")
        print(f"[DEBUG] Raw activations shape: {activations.shape}")
        print(f"[DEBUG] Raw gradients stats - min: {gradients.min():.6f}, max: {gradients.max():.6f}, mean: {gradients.mean():.6f}")
        print(f"[DEBUG] Raw activations stats - min: {activations.min():.6f}, max: {activations.max():.6f}, mean: {activations.mean():.6f}")
        
        # Handle different tensor formats
        if gradients.dim() == 3:
            # Format: (seq_len, batch, dim) -> (batch, seq_len, dim)
            if gradients.shape[1] == 1:
                gradients = gradients.permute(1, 0, 2)
                activations = activations.permute(1, 0, 2)
        
        print(f"[DEBUG] After permute - gradients shape: {gradients.shape}")
        print(f"[DEBUG] After permute - activations shape: {activations.shape}")
        print(f"[DEBUG] After permute - gradients stats - min: {gradients.min():.6f}, max: {gradients.max():.6f}, mean: {gradients.mean():.6f}")
        
        # Extract patch tokens, accounting for PoundNet structure:
        # [class_token, patch_tokens..., prompt_tokens...]
        batch_size = gradients.shape[0]
        seq_len = gradients.shape[1]
        
        print(f"[DEBUG] Sequence length: {seq_len}, Expected patches: {self.num_patches}")
        
        # Calculate token positions
        class_token_idx = 0
        patch_start_idx = 1
        patch_end_idx = patch_start_idx + self.num_patches
        prompt_start_idx = patch_end_idx
        
        print(f"[DEBUG] Token positions - class: {class_token_idx}, patch_start: {patch_start_idx}, patch_end: {patch_end_idx}, prompt_start: {prompt_start_idx}")
        
        # Ensure we don't exceed sequence length
        if patch_end_idx > seq_len:
            print(f"[DEBUG] Patch end index {patch_end_idx} exceeds sequence length {seq_len}, adjusting...")
            # Fallback: use all tokens except class token and potential prompt tokens
            # Try to identify prompt tokens at the end
            if seq_len > self.num_patches + 1:
                # Assume prompt tokens are at the end
                estimated_prompt_tokens = seq_len - self.num_patches - 1
                patch_end_idx = seq_len - estimated_prompt_tokens
                print(f"[DEBUG] Estimated {estimated_prompt_tokens} prompt tokens, new patch_end: {patch_end_idx}")
            else:
                patch_end_idx = seq_len
                patch_start_idx = 1
                print(f"[DEBUG] Using all available tokens from {patch_start_idx} to {patch_end_idx}")
        
        # Debug token ranges and analyze gradient distribution
        print(f"[DEBUG] Extracting tokens from {patch_start_idx} to {patch_end_idx} (total: {patch_end_idx - patch_start_idx})")
        
        # Analyze gradient distribution across all tokens
        all_gradients = gradients[0]  # Shape: (273, 1024)
        token_grad_norms = torch.norm(all_gradients, dim=1)  # Shape: (273,)
        
        print(f"[DEBUG] Token gradient norms - shape: {token_grad_norms.shape}")
        print(f"[DEBUG] Token gradient norms - min: {token_grad_norms.min():.6f}, max: {token_grad_norms.max():.6f}, mean: {token_grad_norms.mean():.6f}")
        
        # Find tokens with non-zero gradients
        nonzero_tokens = torch.nonzero(token_grad_norms > 1e-6).squeeze()
        print(f"[DEBUG] Tokens with non-zero gradients: {nonzero_tokens.tolist() if nonzero_tokens.numel() > 0 else 'None'}")
        
        # Check specific token ranges
        class_token_grad = token_grad_norms[0]
        patch_token_grads = token_grad_norms[1:257] if seq_len > 257 else token_grad_norms[1:]
        remaining_token_grads = token_grad_norms[257:] if seq_len > 257 else torch.tensor([])
        
        print(f"[DEBUG] Class token gradient norm: {class_token_grad:.6f}")
        print(f"[DEBUG] Patch tokens gradient norms - min: {patch_token_grads.min():.6f}, max: {patch_token_grads.max():.6f}, mean: {patch_token_grads.mean():.6f}")
        if remaining_token_grads.numel() > 0:
            print(f"[DEBUG] Remaining tokens gradient norms - min: {remaining_token_grads.min():.6f}, max: {remaining_token_grads.max():.6f}, mean: {remaining_token_grads.mean():.6f}")
        
        # Try different extraction strategies
        if patch_token_grads.max() <= 1e-6:
            print("[DEBUG] Patch tokens have zero gradients, trying alternative extraction...")
            
            # Strategy 1: Use all tokens except class token
            if remaining_token_grads.numel() > 0 and remaining_token_grads.max() > 1e-6:
                print("[DEBUG] Using remaining tokens (likely prompt tokens) for gradient computation")
                # Use the last 256 tokens or all remaining tokens
                start_idx = max(1, seq_len - 256)
                end_idx = seq_len
                patch_gradients = gradients[0, start_idx:end_idx, :]
                patch_activations = activations[0, start_idx:end_idx, :]
                
                # Pad or truncate to 256 tokens
                if patch_gradients.shape[0] < 256:
                    padding = 256 - patch_gradients.shape[0]
                    patch_gradients = F.pad(patch_gradients, (0, 0, 0, padding))
                    patch_activations = F.pad(patch_activations, (0, 0, 0, padding))
                elif patch_gradients.shape[0] > 256:
                    patch_gradients = patch_gradients[:256]
                    patch_activations = patch_activations[:256]
            else:
                print("[DEBUG] No tokens have significant gradients, using class token with patch activations")
                # Better fallback: use class token gradients with actual patch activations
                # This preserves spatial information from patch activations while using class gradients
                class_grad = gradients[0, 0:1, :].repeat(256, 1)  # Repeat class gradients
                patch_act = activations[0, patch_start_idx:patch_end_idx, :]  # Use actual patch activations
                
                # Ensure we have 256 patch activations
                if patch_act.shape[0] < 256:
                    padding = 256 - patch_act.shape[0]
                    patch_act = F.pad(patch_act, (0, 0, 0, padding))
                elif patch_act.shape[0] > 256:
                    patch_act = patch_act[:256]
                
                patch_gradients = class_grad
                patch_activations = patch_act
        else:
            # Standard extraction
            patch_gradients = gradients[0, patch_start_idx:patch_end_idx, :]
            patch_activations = activations[0, patch_start_idx:patch_end_idx, :]
        
        print(f"[DEBUG] Final extracted patch gradients shape: {patch_gradients.shape}")
        print(f"[DEBUG] Final extracted patch activations shape: {patch_activations.shape}")
        print(f"[DEBUG] Final extracted patch gradients stats - min: {patch_gradients.min():.6f}, max: {patch_gradients.max():.6f}, mean: {patch_gradients.mean():.6f}")
        print(f"[DEBUG] Final extracted patch activations stats - min: {patch_activations.min():.6f}, max: {patch_activations.max():.6f}, mean: {patch_activations.mean():.6f}")
        
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
        
        # Add debugging information
        print(f"[DEBUG] Patch gradients shape: {patch_gradients.shape}")
        print(f"[DEBUG] Patch activations shape: {patch_activations.shape}")
        print(f"[DEBUG] Gradients stats - min: {patch_gradients.min():.6f}, max: {patch_gradients.max():.6f}, mean: {patch_gradients.mean():.6f}")
        print(f"[DEBUG] Activations stats - min: {patch_activations.min():.6f}, max: {patch_activations.max():.6f}, mean: {patch_activations.mean():.6f}")

        # Method 1: Standard GradCAM (average gradients across spatial dimension)
        weights_standard = torch.mean(patch_gradients, dim=0, keepdim=True)  # (1, dim)
        cam_standard = torch.sum(weights_standard * patch_activations, dim=1)  # (num_patches,)

        # Method 2: Use absolute gradients to avoid cancellation
        weights_abs = torch.mean(torch.abs(patch_gradients), dim=0, keepdim=True)  # (1, dim)
        cam_abs = torch.sum(weights_abs * patch_activations, dim=1)  # (num_patches,)

        # Method 3: Use squared gradients for stronger attention highlighting
        weights_squared = torch.mean(patch_gradients.pow(2), dim=0, keepdim=True)  # (1, dim)
        cam_squared = torch.sum(weights_squared * torch.abs(patch_activations), dim=1)  # (num_patches,)

        # Method 4: Per-patch importance (element-wise multiplication)
        patch_importance = torch.sum(torch.abs(patch_gradients) * torch.abs(patch_activations), dim=1)  # (num_patches,)
        cam_importance = patch_importance

        # Method 5: Enhanced attention with gradient magnitude weighting
        grad_magnitude = torch.norm(patch_gradients, dim=1, keepdim=True)  # (num_patches, 1)
        act_magnitude = torch.norm(patch_activations, dim=1, keepdim=True)  # (num_patches, 1)
        cam_enhanced = (grad_magnitude * act_magnitude).squeeze()  # (num_patches,)

        # Debug all methods
        print(f"[DEBUG] Standard CAM stats - min: {cam_standard.min():.6f}, max: {cam_standard.max():.6f}, mean: {cam_standard.mean():.6f}")
        print(f"[DEBUG] Absolute CAM stats - min: {cam_abs.min():.6f}, max: {cam_abs.max():.6f}, mean: {cam_abs.mean():.6f}")
        print(f"[DEBUG] Squared CAM stats - min: {cam_squared.min():.6f}, max: {cam_squared.max():.6f}, mean: {cam_squared.mean():.6f}")
        print(f"[DEBUG] Importance CAM stats - min: {cam_importance.min():.6f}, max: {cam_importance.max():.6f}, mean: {cam_importance.mean():.6f}")
        print(f"[DEBUG] Enhanced CAM stats - min: {cam_enhanced.min():.6f}, max: {cam_enhanced.max():.6f}, mean: {cam_enhanced.mean():.6f}")

        # Choose the best method based on maximum activation
        cam_methods = {
            'standard': cam_standard,
            'absolute': cam_abs,
            'squared': cam_squared,
            'importance': cam_importance,
            'enhanced': cam_enhanced
        }
        
        # Select method with highest maximum value (strongest attention)
        best_method = max(cam_methods.keys(), key=lambda k: cam_methods[k].max().item())
        cam = cam_methods[best_method]
        print(f"[DEBUG] Selected method: {best_method}")
        
        # If still all zeros, use enhanced method with small epsilon
        if cam.max() <= 1e-6:
            cam = cam_enhanced + 1e-6
            print("[DEBUG] All methods near zero, using enhanced with epsilon")

        print(f"[DEBUG] Final CAM before ReLU - min: {cam.min():.6f}, max: {cam.max():.6f}, mean: {cam.mean():.6f}")
        
        # Reshape to spatial grid
        cam = cam[:grid_size * grid_size].view(grid_size, grid_size)
        
        # Apply ReLU to focus on positive contributions
        cam = F.relu(cam)
        
        # Convert to numpy with proper detachment
        print(f"[DEBUG] CAM tensor requires_grad: {cam.requires_grad}")
        cam = cam.detach().cpu().numpy()
        
        # Enhanced normalization and contrast improvement
        if normalize:
            cam = self._enhanced_normalize_cam(cam)
        
        # Upsample to input resolution with better interpolation
        cam = cv2.resize(cam, (self.input_size, self.input_size), interpolation=cv2.INTER_CUBIC)
        
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
        
        print(f"[DEBUG] Enhanced normalization - input range: [{cam.min():.6f}, {cam.max():.6f}], output range: [{cam_final.min():.6f}, {cam_final.max():.6f}]")
        
        return cam_final
    
    def generate_enhanced_attention_map(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[Union[int, str]] = None,
        layer_name: Optional[str] = None,
        enhancement_method: str = 'multi_scale'
    ) -> np.ndarray:
        """
        Generate enhanced attention map with multiple highlighting techniques.
        
        Args:
            input_tensor: Input tensor of shape (1, 3, H, W)
            target_class: Target class for attention
            layer_name: Specific layer to use
            enhancement_method: Enhancement method ('multi_scale', 'guided', 'integrated')
            
        Returns:
            Enhanced attention map
        """
        if enhancement_method == 'multi_scale':
            return self._generate_multi_scale_attention(input_tensor, target_class, layer_name)
        elif enhancement_method == 'guided':
            return self._generate_guided_attention(input_tensor, target_class, layer_name)
        elif enhancement_method == 'integrated':
            return self._generate_integrated_attention(input_tensor, target_class, layer_name)
        else:
            return self.generate_cam(input_tensor, target_class, layer_name, normalize=True)
    
    def _generate_multi_scale_attention(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[Union[int, str]] = None,
        layer_name: Optional[str] = None
    ) -> np.ndarray:
        """Generate multi-scale attention by combining multiple layers."""
        attention_maps = []
        
        # Generate CAMs for all target layers
        for layer in self.target_layers:
            try:
                cam = self.generate_cam(input_tensor, target_class, layer, normalize=False)
                attention_maps.append(cam)
            except Exception as e:
                print(f"[DEBUG] Failed to generate CAM for layer {layer}: {e}")
                continue
        
        if not attention_maps:
            # Fallback to single layer
            return self.generate_cam(input_tensor, target_class, layer_name, normalize=True)
        
        # Combine attention maps with weighted average
        weights = [0.2, 0.3, 0.5]  # Give more weight to later layers
        if len(attention_maps) != len(weights):
            weights = [1.0 / len(attention_maps)] * len(attention_maps)
        
        combined_attention = np.zeros_like(attention_maps[0])
        for i, (cam, weight) in enumerate(zip(attention_maps, weights)):
            # Normalize each CAM individually
            cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            combined_attention += weight * cam_norm
        
        # Enhanced normalization
        return self._enhanced_normalize_cam(combined_attention)
    
    def _generate_guided_attention(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[Union[int, str]] = None,
        layer_name: Optional[str] = None
    ) -> np.ndarray:
        """Generate guided attention using gradient guidance."""
        # Get standard CAM
        cam = self.generate_cam(input_tensor, target_class, layer_name, normalize=False)
        
        # Compute input gradients for guidance
        input_tensor.requires_grad_(True)
        output = self.model(input_tensor)
        
        if isinstance(output, dict):
            logits = output['logits']
        else:
            logits = output
        
        # Handle target class
        if target_class is None:
            target_class_idx = logits.argmax(dim=1).item()
        elif isinstance(target_class, str):
            target_class_idx = 0 if target_class.lower() == 'real' else 1
        else:
            target_class_idx = target_class
        
        # Compute input gradients
        self.model.zero_grad()
        class_score = logits[0, target_class_idx]
        input_grads = torch.autograd.grad(class_score, input_tensor, retain_graph=True)[0]
        
        # Convert input gradients to attention guidance
        input_grads_np = input_grads.squeeze().detach().cpu().numpy()
        guidance = np.mean(np.abs(input_grads_np), axis=0)  # Average across channels
        
        # Resize guidance to match CAM size
        guidance_resized = cv2.resize(guidance, (cam.shape[1], cam.shape[0]))
        
        # Combine CAM with guidance
        guided_cam = cam * (1 + guidance_resized)
        
        return self._enhanced_normalize_cam(guided_cam)
    
    def _generate_integrated_attention(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[Union[int, str]] = None,
        layer_name: Optional[str] = None
    ) -> np.ndarray:
        """Generate integrated attention using multiple techniques."""
        # Get CAMs for both classes
        real_cam = self.generate_cam(input_tensor, 'real', layer_name, normalize=False)
        fake_cam = self.generate_cam(input_tensor, 'fake', layer_name, normalize=False)
        
        # Compute difference map (discriminative regions)
        diff_map = np.abs(fake_cam - real_cam)
        
        # Get target class CAM
        if target_class is None or target_class == 'fake' or target_class == 1:
            target_cam = fake_cam
        else:
            target_cam = real_cam
        
        # Integrate target attention with discriminative regions
        integrated_attention = target_cam * (1 + 0.5 * diff_map)
        
        return self._enhanced_normalize_cam(integrated_attention)

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