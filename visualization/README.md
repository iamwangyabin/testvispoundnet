# PoundNet GradCAM Visualization

This module provides comprehensive GradCAM (Gradient-weighted Class Activation Mapping) visualization capabilities specifically designed for the PoundNet deepfake detection model.

## Overview

PoundNet uses a CLIP Vision Transformer (ViT-L/14) with learnable prompts for deepfake detection. This GradCAM implementation is specially adapted to handle:

- **Vision Transformer Architecture**: Patch-based input representation and transformer blocks
- **Learnable Prompts**: Visual prompt tokens that enhance the model's detection capabilities  
- **Real vs Fake Classification**: Binary classification with class-specific attention visualization
- **Multi-layer Analysis**: Attention pattern analysis across different transformer layers

## Features

### Core Capabilities
- ✅ **ViT-adapted GradCAM**: Handles patch tokens, class tokens, and prompt tokens
- ✅ **Spatial Reconstruction**: Converts 14×14 patch gradients to 224×224 heatmaps
- ✅ **Multi-layer Analysis**: Visualize attention patterns across transformer layers
- ✅ **Comparative Analysis**: Side-by-side Real vs Fake class visualizations
- ✅ **Batch Processing**: Efficient processing of large image datasets
- ✅ **Multiple Output Formats**: PNG, JPG, NumPy arrays, and JSON metadata

### Visualization Types
1. **Individual Class CAMs**: Separate visualizations for Real and Fake classes
2. **Comparative Analysis**: Side-by-side comparison of both classes
3. **Multi-layer Analysis**: Attention evolution across transformer layers
4. **Statistical Summaries**: Quantitative analysis of attention patterns

## Installation

### Requirements
Add the following to your `requirements.txt`:

```txt
matplotlib>=3.5.0
opencv-python>=4.5.0
pandas>=1.3.0
tqdm>=4.62.0
Pillow>=8.3.0
```

### Setup
The GradCAM module is already integrated into the PoundNet project structure:

```
visualization/
├── gradcam/
│   ├── __init__.py              # Main module exports
│   ├── core.py                  # Core ViT GradCAM implementation
│   ├── poundnet_gradcam.py      # PoundNet-specific wrapper
│   ├── utils.py                 # Visualization utilities
│   └── examples/
│       ├── generate_gradcam.py  # Single/batch image analysis
│       └── batch_process.py     # Advanced batch processing
└── README.md                    # This documentation
```

## Quick Start

### Basic Usage

```python
import torch
from visualization.gradcam import PoundNetGradCAM
from visualization.gradcam.utils import preprocess_image_for_gradcam, visualize_gradcam

# Load your PoundNet model (see examples for full loading code)
model = load_poundnet_model('cfgs/poundnet.yaml')

# Initialize GradCAM
gradcam = PoundNetGradCAM(model=model)

# Preprocess image
image_tensor, original_image = preprocess_image_for_gradcam('path/to/image.jpg')

# Generate prediction with CAM
result = gradcam.generate_prediction_with_cam(image_tensor)

print(f"Prediction: {result['predicted_class_name']}")
print(f"Confidence: {result['confidence']:.4f}")

# Visualize results
fig = visualize_gradcam(
    original_image, 
    result['cam'], 
    title=f"GradCAM - {result['predicted_class_name']}"
)
```

### Command Line Usage

#### Single Image Analysis
```bash
python visualization/gradcam/examples/generate_gradcam.py \
    --config cfgs/poundnet.yaml \
    --image path/to/image.jpg \
    --output_dir results/ \
    --individual --comparative --multilayer
```

#### Batch Processing (Regular Images)
```bash
python visualization/gradcam/examples/batch_process.py \
    --config cfgs/poundnet.yaml \
    --input_dir dataset/images/ \
    --output_dir results/batch/ \
    --max_images 100 \
    --save_cams
```

#### Batch Processing (Arrow Datasets - Compatible with test.py)
```bash
python visualization/gradcam/examples/batch_process_arrow.py \
    --config cfgs/poundnet.yaml \
    --output_dir results/arrow_batch/ \
    --max_images_per_subset 50 \
    --save_cams
```

## API Reference

### PoundNetGradCAM Class

The main class for PoundNet-specific GradCAM analysis.

```python
class PoundNetGradCAM:
    def __init__(
        self,
        model: nn.Module,
        target_layers: List[str] = None,
        patch_size: int = 14,
        input_size: int = 224,
        use_cuda: bool = True
    )
```

#### Key Methods

##### `generate_cam()`
Generate Class Activation Map for a specific class.

```python
def generate_cam(
    self,
    input_tensor: torch.Tensor,
    target_class: Union[int, str] = None,  # 0/'Real', 1/'Fake', or None
    layer_name: str = None,
    normalize: bool = True
) -> np.ndarray
```

##### `generate_prediction_with_cam()`
Generate prediction along with corresponding CAM.

```python
def generate_prediction_with_cam(
    self,
    input_tensor: torch.Tensor,
    layer_name: str = None,
    normalize: bool = True
) -> Dict[str, Any]
```

**Returns:**
```python
{
    'cam': np.ndarray,                    # GradCAM heatmap
    'predicted_class_name': str,          # 'Real' or 'Fake'
    'confidence': float,                  # Prediction confidence
    'real_prob': float,                   # Probability of Real class
    'fake_prob': float,                   # Probability of Fake class
    'logits': np.ndarray,                 # Raw model logits
    'probabilities': np.ndarray           # Softmax probabilities
}
```

##### `generate_comparative_cam()`
Generate CAMs for both Real and Fake classes.

```python
def generate_comparative_cam(
    self,
    input_tensor: torch.Tensor,
    layer_name: str = None
) -> Dict[str, np.ndarray]
```

##### `analyze_attention_patterns()`
Analyze attention patterns across multiple layers.

```python
def analyze_attention_patterns(
    self,
    input_tensor: torch.Tensor
) -> Dict[str, Dict[str, Any]]
```

### Visualization Functions

#### `visualize_gradcam()`
Create comprehensive GradCAM visualization.

```python
def visualize_gradcam(
    image: np.ndarray,
    cam: np.ndarray,
    title: str = "GradCAM Visualization",
    save_path: str = None,
    show_original: bool = True,
    show_heatmap: bool = True,
    show_overlay: bool = True,
    colormap: str = 'jet',
    alpha: float = 0.4
) -> plt.Figure
```

#### `visualize_comparative_gradcam()`
Create comparative Real vs Fake visualization.

```python
def visualize_comparative_gradcam(
    image: np.ndarray,
    real_cam: np.ndarray,
    fake_cam: np.ndarray,
    prediction_result: Dict[str, Any],
    title: str = "Comparative GradCAM Analysis",
    save_path: str = None
) -> plt.Figure
```

## Advanced Usage

### Custom Target Layers

Specify which transformer layers to analyze:

```python
# Target specific layers
target_layers = [
    'image_encoder.transformer.resblocks.6.ln_2',   # Early layer
    'image_encoder.transformer.resblocks.12.ln_2',  # Middle layer  
    'image_encoder.transformer.resblocks.23.ln_2'   # Late layer
]

gradcam = PoundNetGradCAM(model=model, target_layers=target_layers)
```

### Multi-layer Analysis

```python
# Analyze attention patterns across layers
layer_results = gradcam.analyze_attention_patterns(image_tensor)

# Visualize multi-layer analysis
fig = visualize_multi_layer_analysis(
    original_image,
    layer_results,
    title="Multi-Layer Attention Analysis"
)

# Create attention summary plot
summary_fig = create_attention_summary_plot(layer_results)
```

### Batch Processing with Statistics

```python
from visualization.gradcam.examples.batch_process import BatchGradCAMProcessor

processor = BatchGradCAMProcessor(
    config_path='cfgs/poundnet.yaml',
    target_layers=target_layers
)

results = processor.process_batch(
    image_paths=image_list,
    output_dir='results/',
    save_cams=True,
    save_summary=True
)

# Access statistics
stats = results['statistics']
print(f"Class distribution: {stats['class_distribution']}")
print(f"Average confidence: {stats['confidence_statistics']['mean']:.4f}")
```

## Understanding the Results

### Heatmap Interpretation

- **Red/Hot regions**: Areas that strongly contribute to the classification decision
- **Blue/Cold regions**: Areas with minimal contribution
- **Intensity**: Higher intensity indicates stronger contribution

### Class-Specific Analysis

- **Real Class CAM**: Shows regions that make the image appear authentic
- **Fake Class CAM**: Shows regions that indicate potential manipulation
- **Comparative Analysis**: Highlights differences in attention between classes

### Multi-Layer Insights

- **Early Layers**: Focus on low-level features (edges, textures)
- **Middle Layers**: Capture mid-level patterns and structures  
- **Late Layers**: Represent high-level semantic features for final decision

## Performance Considerations

### Memory Usage
- Each CAM requires ~200KB for a 224×224 image
- Batch processing automatically manages memory for large datasets
- Use `save_cams=False` for memory-efficient batch processing

### Processing Speed
- Single image: ~0.5-2 seconds (depending on GPU)
- Batch processing: ~100-500 images/minute
- Multi-layer analysis adds ~2-3x processing time

### GPU Recommendations
- Minimum: 4GB VRAM for single image processing
- Recommended: 8GB+ VRAM for efficient batch processing
- CPU processing is supported but significantly slower

## Troubleshooting

### Common Issues

#### "Layer not found in gradients"
```python
# Check available layers
for name, module in model.named_modules():
    if 'transformer' in name and 'ln_2' in name:
        print(name)
```

#### "CUDA out of memory"
```python
# Reduce batch size or disable CUDA
gradcam = PoundNetGradCAM(model=model, use_cuda=False)
```

#### "Unexpected tensor dimensions"
Ensure your model is in evaluation mode:
```python
model.eval()
```

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Examples and Tutorials

### Example 1: Basic Real vs Fake Analysis
```python
# Load model and image
model = load_poundnet_model('cfgs/poundnet.yaml')
gradcam = PoundNetGradCAM(model=model)
image_tensor, original_image = preprocess_image_for_gradcam('test_image.jpg')

# Generate comparative analysis
comparative_cams = gradcam.generate_comparative_cam(image_tensor)
prediction_result = gradcam.generate_prediction_with_cam(image_tensor)

# Visualize results
fig = visualize_comparative_gradcam(
    original_image,
    comparative_cams['real'],
    comparative_cams['fake'], 
    prediction_result,
    save_path='comparative_analysis.png'
)
```

### Example 2: Dataset Analysis
```python
# Process entire dataset
from pathlib import Path

image_paths = list(Path('dataset/').glob('*.jpg'))
processor = BatchGradCAMProcessor('cfgs/poundnet.yaml')

results = processor.process_batch(
    image_paths=image_paths,
    output_dir='analysis_results/',
    save_summary=True
)

# Analyze results
stats = results['statistics']
print(f"Detected {stats['class_distribution']['Fake']} fake images")
print(f"Average confidence: {stats['confidence_statistics']['mean']:.3f}")
```

## Citation

If you use this GradCAM implementation in your research, please cite:

```bibtex
@misc{wang2024pennywisepoundfoolishdeepfakedetection,
      title={Penny-Wise and Pound-Foolish in Deepfake Detection}, 
      author={Yabin Wang and Zhiwu Huang and Su Zhou and Adam Prugel-Bennett and Xiaopeng Hong},
      year={2024},
      eprint={2408.08412},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.08412}, 
}
```

## Contributing

Contributions are welcome! Please see the main project repository for contribution guidelines.

## License

This project is licensed under the same terms as the main PoundNet project.