# Arrow Dataset GradCAM Visualization

This enhanced version of `generate_gradcam.py` now supports direct processing of Arrow datasets with full GradCAM visualizations.

## New Features

- **Arrow Dataset Support**: Process Arrow datasets directly from PoundNet config files
- **Dataset Filtering**: Filter specific datasets and subsets
- **Full Visualizations**: Generate individual, comparative, and multi-layer GradCAM visualizations
- **Batch Processing**: Process multiple datasets and subsets in one command

## Usage Examples

### 1. Process Complete Arrow Dataset

```bash
python visualization/gradcam/examples/generate_gradcam.py \
    --config cfgs/poundnet.yaml \
    --arrow_dataset \
    --output_dir ./arrow_gradcam_output \
    --individual --comparative \
    --max_images_per_subset 10
```

### 2. Process Specific Datasets

```bash
python visualization/gradcam/examples/generate_gradcam.py \
    --config cfgs/poundnet.yaml \
    --arrow_dataset \
    --output_dir ./arrow_gradcam_output \
    --dataset_filter DiffusionForensics ForenSynths \
    --individual --comparative --multilayer
```

### 3. Process Specific Subsets

```bash
python visualization/gradcam/examples/generate_gradcam.py \
    --config cfgs/poundnet.yaml \
    --arrow_dataset \
    --output_dir ./arrow_gradcam_output \
    --subset_filter progan stylegan \
    --individual --comparative \
    --max_images_per_subset 5
```

### 4. Full Analysis with Custom Layers

```bash
python visualization/gradcam/examples/generate_gradcam.py \
    --config cfgs/poundnet.yaml \
    --arrow_dataset \
    --output_dir ./arrow_gradcam_output \
    --target_layers image_encoder.transformer.resblocks.6.ln_2 image_encoder.transformer.resblocks.12.ln_2 image_encoder.transformer.resblocks.23.ln_2 \
    --individual --comparative --multilayer \
    --max_images_per_subset 20
```

## Command Line Arguments

### Arrow Dataset Specific Arguments

- `--arrow_dataset`: Enable Arrow dataset processing mode
- `--max_images_per_subset INT`: Maximum images to process per dataset subset
- `--dataset_filter NAMES`: Filter specific datasets (e.g., DiffusionForensics ForenSynths)
- `--subset_filter NAMES`: Filter specific subsets (e.g., progan stylegan)

### Visualization Options

- `--individual`: Save individual class visualizations (Real and Fake separately)
- `--comparative`: Save comparative analysis (Real vs Fake side-by-side)
- `--multilayer`: Save multi-layer analysis across different transformer layers

### General Options

- `--config PATH`: Path to PoundNet configuration file
- `--checkpoint PATH`: Path to model checkpoint (optional, uses config default)
- `--output_dir PATH`: Output directory for results
- `--target_layers LAYERS`: Specific layers to analyze

## Output Structure

When processing Arrow datasets, the output directory will be organized as:

```
output_dir/
├── dataset1_name/
│   ├── subset1/
│   │   ├── image_000000/
│   │   │   ├── image_000000_prediction_gradcam.png
│   │   │   ├── image_000000_prediction_gradcam.npy
│   │   │   ├── image_000000_prediction_metadata.json
│   │   │   ├── image_000000_real_gradcam.png
│   │   │   ├── image_000000_fake_gradcam.png
│   │   │   ├── image_000000_comparative.png
│   │   │   └── image_000000_multilayer.png (if --multilayer)
│   │   ├── image_000001/
│   │   └── ...
│   └── subset2/
├── dataset2_name/
└── arrow_gradcam_summary.json
```

## Generated Visualizations

### Individual Class Visualizations
- **Real Class GradCAM**: Shows attention patterns when predicting "Real"
- **Fake Class GradCAM**: Shows attention patterns when predicting "Fake"

### Comparative Analysis
- **Side-by-side comparison**: Real vs Fake attention patterns
- **Prediction information**: Probabilities and confidence scores

### Multi-layer Analysis
- **Layer progression**: How attention patterns evolve through transformer layers
- **Statistical analysis**: Mean activation and attention statistics

## Performance Tips

1. **Use filtering**: Process specific datasets/subsets to reduce computation time
2. **Limit images**: Use `--max_images_per_subset` for quick testing
3. **Skip multilayer**: Omit `--multilayer` for faster processing
4. **GPU usage**: Ensure CUDA is available for faster processing

## Comparison with batch_process_arrow.py

| Feature | batch_process_arrow.py | generate_gradcam.py (enhanced) |
|---------|------------------------|--------------------------------|
| Arrow dataset support | ✅ | ✅ |
| Statistical analysis | ✅ | ✅ |
| CAM arrays (.npy) | ✅ | ✅ |
| **Visualization images** | ❌ | ✅ |
| **Individual class CAMs** | ❌ | ✅ |
| **Comparative analysis** | ❌ | ✅ |
| **Multi-layer analysis** | ❌ | ✅ |
| **Ready-to-view images** | ❌ | ✅ |

The enhanced `generate_gradcam.py` provides everything that `batch_process_arrow.py` offers, plus comprehensive visualization capabilities.