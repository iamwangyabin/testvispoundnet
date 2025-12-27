"""
Batch processing utility for GradCAM analysis.

This script provides advanced batch processing capabilities for analyzing
large datasets with PoundNet GradCAM, including parallel processing and
statistical analysis.
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm
import json
import time
from typing import List, Dict, Any

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from visualization.gradcam import PoundNetGradCAM
from visualization.gradcam.utils import preprocess_image_for_gradcam
from utils.util import load_config_with_cli
from utils.network_factory import get_model
from utils.resume_tools import resume_lightning
import hydra


class BatchGradCAMProcessor:
    """
    Batch processor for GradCAM analysis with parallel processing capabilities.
    """
    
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str = None,
        target_layers: List[str] = None,
        use_cuda: bool = True,
        num_workers: int = None
    ):
        """
        Initialize batch processor.
        
        Args:
            config_path: Path to PoundNet configuration
            checkpoint_path: Path to model checkpoint
            target_layers: Specific layers to analyze
            use_cuda: Whether to use CUDA
            num_workers: Number of parallel workers
        """
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.target_layers = target_layers
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.num_workers = num_workers or min(4, mp.cpu_count())
        
        # Load model
        self.model, self.config = self._load_model()
        
        # Initialize GradCAM
        self.gradcam = PoundNetGradCAM(
            model=self.model,
            target_layers=self.target_layers,
            use_cuda=self.use_cuda
        )
        
        print(f"Batch processor initialized with {self.num_workers} workers")
        print(f"CUDA available: {self.use_cuda}")
    
    def _load_model(self):
        """Load PoundNet model."""
        print(f"Loading model from config: {self.config_path}")
        conf = load_config_with_cli(self.config_path)
        conf = hydra.utils.instantiate(conf)
        
        model = get_model(conf)
        
        checkpoint_path = self.checkpoint_path or conf.resume.path
        print(f"Loading checkpoint: {checkpoint_path}")
        resume_lightning(model, checkpoint_path)
        
        model.eval()
        if self.use_cuda:
            model = model.cuda()
        
        return model, conf
    
    def process_single_image(
        self,
        image_path: str,
        save_cams: bool = False,
        output_dir: str = None
    ) -> Dict[str, Any]:
        """
        Process a single image and return results.
        
        Args:
            image_path: Path to input image
            save_cams: Whether to save CAM arrays
            output_dir: Output directory for saved files
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Preprocess image
            image_tensor, original_image = preprocess_image_for_gradcam(image_path)
            
            # Generate prediction with CAM
            result = self.gradcam.generate_prediction_with_cam(image_tensor)
            
            # Add image metadata
            result['image_path'] = image_path
            result['image_name'] = os.path.basename(image_path)
            result['image_size'] = original_image.shape[:2]
            
            # Save CAMs if requested
            if save_cams and output_dir:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                cam_path = os.path.join(output_dir, f"{base_name}_cam.npy")
                np.save(cam_path, result['cam'])
                result['cam_path'] = cam_path
                # Remove large array from result to save memory
                del result['cam']
            
            return result
            
        except Exception as e:
            return {
                'image_path': image_path,
                'image_name': os.path.basename(image_path),
                'error': str(e),
                'success': False
            }
    
    def process_batch(
        self,
        image_paths: List[str],
        output_dir: str,
        save_cams: bool = False,
        save_summary: bool = True,
        progress_bar: bool = True
    ) -> Dict[str, Any]:
        """
        Process a batch of images.
        
        Args:
            image_paths: List of image paths to process
            output_dir: Output directory
            save_cams: Whether to save individual CAM arrays
            save_summary: Whether to save summary statistics
            progress_bar: Whether to show progress bar
            
        Returns:
            Dictionary with batch results and statistics
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Processing {len(image_paths)} images...")
        start_time = time.time()
        
        results = []
        successful_results = []
        failed_results = []
        
        # Process images with progress bar
        iterator = tqdm(image_paths) if progress_bar else image_paths
        
        for image_path in iterator:
            result = self.process_single_image(
                image_path,
                save_cams=save_cams,
                output_dir=output_dir if save_cams else None
            )
            
            results.append(result)
            
            if result.get('success', True):  # Default to True if not specified
                successful_results.append(result)
            else:
                failed_results.append(result)
        
        processing_time = time.time() - start_time
        
        # Compute statistics
        stats = self._compute_batch_statistics(successful_results, processing_time)
        
        # Save results
        batch_results = {
            'results': results,
            'statistics': stats,
            'processing_info': {
                'total_images': len(image_paths),
                'successful': len(successful_results),
                'failed': len(failed_results),
                'processing_time': processing_time,
                'images_per_second': len(image_paths) / processing_time
            }
        }
        
        if save_summary:
            self._save_batch_summary(batch_results, output_dir)
        
        return batch_results
    
    def _compute_batch_statistics(
        self,
        results: List[Dict[str, Any]],
        processing_time: float
    ) -> Dict[str, Any]:
        """Compute statistics for batch results."""
        if not results:
            return {}
        
        # Extract predictions
        predictions = [r['predicted_class_name'] for r in results]
        confidences = [r['confidence'] for r in results]
        real_probs = [r['real_prob'] for r in results]
        fake_probs = [r['fake_prob'] for r in results]
        
        # Class distribution
        class_counts = pd.Series(predictions).value_counts().to_dict()
        
        # Confidence statistics
        confidence_stats = {
            'mean': np.mean(confidences),
            'std': np.std(confidences),
            'min': np.min(confidences),
            'max': np.max(confidences),
            'median': np.median(confidences)
        }
        
        # Probability statistics
        real_prob_stats = {
            'mean': np.mean(real_probs),
            'std': np.std(real_probs),
            'min': np.min(real_probs),
            'max': np.max(real_probs)
        }
        
        fake_prob_stats = {
            'mean': np.mean(fake_probs),
            'std': np.std(fake_probs),
            'min': np.min(fake_probs),
            'max': np.max(fake_probs)
        }
        
        # High confidence predictions
        high_confidence_threshold = 0.8
        high_confidence_count = sum(1 for c in confidences if c > high_confidence_threshold)
        
        return {
            'class_distribution': class_counts,
            'confidence_statistics': confidence_stats,
            'real_probability_statistics': real_prob_stats,
            'fake_probability_statistics': fake_prob_stats,
            'high_confidence_predictions': {
                'count': high_confidence_count,
                'percentage': (high_confidence_count / len(results)) * 100,
                'threshold': high_confidence_threshold
            },
            'processing_statistics': {
                'total_processed': len(results),
                'processing_time': processing_time,
                'average_time_per_image': processing_time / len(results)
            }
        }
    
    def _save_batch_summary(self, batch_results: Dict[str, Any], output_dir: str):
        """Save batch summary to files."""
        # Save detailed results as JSON
        results_path = os.path.join(output_dir, 'batch_results.json')
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = []
            for result in batch_results['results']:
                json_result = {}
                for key, value in result.items():
                    if isinstance(value, np.ndarray):
                        json_result[key] = value.tolist()
                    elif isinstance(value, torch.Tensor):
                        json_result[key] = value.cpu().numpy().tolist()
                    else:
                        json_result[key] = value
                json_results.append(json_result)
            
            json_data = {
                'results': json_results,
                'statistics': batch_results['statistics'],
                'processing_info': batch_results['processing_info']
            }
            json.dump(json_data, f, indent=2)
        
        # Save summary CSV
        csv_path = os.path.join(output_dir, 'batch_summary.csv')
        df_data = []
        for result in batch_results['results']:
            if result.get('success', True):
                df_data.append({
                    'image_name': result['image_name'],
                    'predicted_class': result['predicted_class_name'],
                    'confidence': result['confidence'],
                    'real_prob': result['real_prob'],
                    'fake_prob': result['fake_prob']
                })
        
        if df_data:
            df = pd.DataFrame(df_data)
            df.to_csv(csv_path, index=False)
        
        # Save statistics summary
        stats_path = os.path.join(output_dir, 'statistics_summary.txt')
        with open(stats_path, 'w') as f:
            stats = batch_results['statistics']
            f.write("BATCH PROCESSING STATISTICS\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Class Distribution:\n")
            for class_name, count in stats.get('class_distribution', {}).items():
                f.write(f"  {class_name}: {count}\n")
            f.write("\n")
            
            f.write("Confidence Statistics:\n")
            conf_stats = stats.get('confidence_statistics', {})
            for key, value in conf_stats.items():
                f.write(f"  {key}: {value:.4f}\n")
            f.write("\n")
            
            f.write("Processing Information:\n")
            proc_info = batch_results['processing_info']
            for key, value in proc_info.items():
                f.write(f"  {key}: {value}\n")
        
        print(f"Batch summary saved to: {output_dir}")


def find_images_in_directory(
    directory: str,
    extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
    recursive: bool = True
) -> List[str]:
    """Find all image files in a directory."""
    image_paths = []
    directory = Path(directory)
    
    for ext in extensions:
        if recursive:
            image_paths.extend(directory.rglob(f"*{ext}"))
            image_paths.extend(directory.rglob(f"*{ext.upper()}"))
        else:
            image_paths.extend(directory.glob(f"*{ext}"))
            image_paths.extend(directory.glob(f"*{ext.upper()}"))
    
    return [str(path) for path in sorted(image_paths)]


def main():
    parser = argparse.ArgumentParser(description="Batch GradCAM processing for PoundNet")
    parser.add_argument('--config', type=str, required=True,
                       help='Path to PoundNet configuration file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--target_layers', type=str, nargs='+', default=None,
                       help='Specific layers to analyze')
    parser.add_argument('--max_images', type=int, default=None,
                       help='Maximum number of images to process')
    parser.add_argument('--save_cams', action='store_true',
                       help='Save individual CAM arrays')
    parser.add_argument('--recursive', action='store_true',
                       help='Search for images recursively')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of parallel workers')
    parser.add_argument('--extensions', type=str, nargs='+',
                       default=['.jpg', '.jpeg', '.png', '.bmp'],
                       help='Image file extensions to process')
    
    args = parser.parse_args()
    
    # Find images
    print(f"Searching for images in: {args.input_dir}")
    image_paths = find_images_in_directory(
        args.input_dir,
        extensions=args.extensions,
        recursive=args.recursive
    )
    
    if args.max_images:
        image_paths = image_paths[:args.max_images]
    
    print(f"Found {len(image_paths)} images to process")
    
    if not image_paths:
        print("No images found. Exiting.")
        return
    
    # Initialize processor
    processor = BatchGradCAMProcessor(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        target_layers=args.target_layers,
        num_workers=args.num_workers
    )
    
    # Process batch
    results = processor.process_batch(
        image_paths=image_paths,
        output_dir=args.output_dir,
        save_cams=args.save_cams,
        save_summary=True,
        progress_bar=True
    )
    
    # Print summary
    print("\nBatch processing complete!")
    print(f"Total images: {results['processing_info']['total_images']}")
    print(f"Successful: {results['processing_info']['successful']}")
    print(f"Failed: {results['processing_info']['failed']}")
    print(f"Processing time: {results['processing_info']['processing_time']:.2f} seconds")
    print(f"Images per second: {results['processing_info']['images_per_second']:.2f}")
    
    if results['statistics']:
        print("\nClass distribution:")
        for class_name, count in results['statistics']['class_distribution'].items():
            percentage = (count / results['processing_info']['successful']) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")


if __name__ == "__main__":
    main()