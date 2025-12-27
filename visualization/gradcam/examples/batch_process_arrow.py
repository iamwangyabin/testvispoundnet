"""
Arrow dataset batch processing utility for GradCAM analysis.

This script provides batch processing capabilities for Arrow format datasets
that are compatible with PoundNet's test.py, supporting the same dataset
configurations used in the original testing pipeline.
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import time
from typing import List, Dict, Any
import io
from PIL import Image

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from visualization.gradcam import PoundNetGradCAM
from utils.util import load_config_with_cli
from utils.network_factory import get_model
from utils.resume_tools import resume_lightning
import utils  # Add this import for eval() to find utils module
import hydra
import data


class ArrowDatasetGradCAMProcessor:
    """
    GradCAM processor for Arrow format datasets compatible with PoundNet test.py.
    """
    
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str = None,
        target_layers: List[str] = None,
        use_cuda: bool = True
    ):
        """
        Initialize Arrow dataset processor.
        
        Args:
            config_path: Path to PoundNet configuration
            checkpoint_path: Path to model checkpoint
            target_layers: Specific layers to analyze
            use_cuda: Whether to use CUDA
        """
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.target_layers = target_layers
        self.use_cuda = use_cuda and torch.cuda.is_available()
        
        # Load model and config
        self.model, self.config = self._load_model()
        
        # Initialize GradCAM
        self.gradcam = PoundNetGradCAM(
            model=self.model,
            target_layers=self.target_layers,
            use_cuda=self.use_cuda
        )
        
        print(f"Arrow dataset processor initialized")
        print(f"CUDA available: {self.use_cuda}")
    
    def _load_model(self):
        """Load PoundNet model from config."""
        print(f"Loading model from config: {self.config_path}")
        conf = load_config_with_cli(self.config_path)
        conf = hydra.utils.instantiate(conf)
        
        model = get_model(conf)
        
        checkpoint_path = self.checkpoint_path or conf.resume.path
        print(f"Loading checkpoint: {checkpoint_path}")
        
        # Use the same resume function as test.py
        eval(conf.resume.target)(model, checkpoint_path)
        
        model.eval()
        if self.use_cuda:
            model = model.cuda()
        
        return model, conf
    
    def process_arrow_dataset(
        self,
        dataset_config: Dict[str, Any],
        output_dir: str,
        max_images_per_subset: int = None,
        save_cams: bool = False,
        save_summary: bool = True
    ) -> Dict[str, Any]:
        """
        Process Arrow dataset using the same configuration as test.py.
        
        Args:
            dataset_config: Dataset configuration from config file
            output_dir: Output directory for results
            max_images_per_subset: Maximum images per subset
            save_cams: Whether to save CAM arrays
            save_summary: Whether to save summary statistics
            
        Returns:
            Dictionary with processing results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        all_results = []
        dataset_summaries = {}
        
        print(f"Processing Arrow datasets...")
        print(f"Found {len(dataset_config['source'])} dataset sources")
        
        # Process each dataset source (same as test.py)
        for source_idx, sub_data in enumerate(dataset_config['source']):
            print(f"\nProcessing dataset {source_idx + 1}/{len(dataset_config['source'])}")
            print(f"Target: {sub_data['target']}")
            print(f"Data root: {sub_data['data_root']}")
            print(f"Benchmark: {sub_data['benchmark_name']}")
            
            # Process each subset in the source
            for subset_idx, sub_set in enumerate(sub_data['sub_sets']):
                print(f"\n  Processing subset: {sub_set}")
                
                try:
                    # Create dataset using the same method as test.py
                    dataset = eval(sub_data['target'])(
                        sub_data['data_root'], 
                        dataset_config['trsf'],
                        subset=sub_set, 
                        split=sub_data['split']
                    )
                    
                    # Create data loader
                    data_loader = torch.utils.data.DataLoader(
                        dataset, 
                        batch_size=1,  # Process one image at a time for GradCAM
                        num_workers=0,  # Avoid multiprocessing issues with GradCAM
                        shuffle=False
                    )
                    
                    # Process subset
                    subset_results = self._process_subset(
                        data_loader=data_loader,
                        subset_name=f"{sub_data['benchmark_name']}_{sub_set}",
                        output_dir=os.path.join(output_dir, sub_data['benchmark_name'], sub_set),
                        max_images=max_images_per_subset,
                        save_cams=save_cams
                    )
                    
                    # Add metadata
                    subset_results['benchmark_name'] = sub_data['benchmark_name']
                    subset_results['subset_name'] = sub_set
                    subset_results['data_root'] = sub_data['data_root']
                    
                    all_results.extend(subset_results['results'])
                    dataset_summaries[f"{sub_data['benchmark_name']}_{sub_set}"] = subset_results['summary']
                    
                    print(f"    Processed {len(subset_results['results'])} images")
                    print(f"    Success rate: {subset_results['summary']['success_rate']:.2%}")
                    
                except Exception as e:
                    print(f"    Error processing subset {sub_set}: {e}")
                    continue
        
        # Compute overall statistics
        overall_stats = self._compute_overall_statistics(all_results, dataset_summaries)
        
        # Save results
        if save_summary:
            self._save_arrow_dataset_summary(
                all_results, 
                dataset_summaries, 
                overall_stats, 
                output_dir
            )
        
        return {
            'results': all_results,
            'dataset_summaries': dataset_summaries,
            'overall_statistics': overall_stats
        }
    
    def _process_subset(
        self,
        data_loader,
        subset_name: str,
        output_dir: str,
        max_images: int = None,
        save_cams: bool = False
    ) -> Dict[str, Any]:
        """Process a single dataset subset."""
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        successful_count = 0
        failed_count = 0
        
        print(f"    Processing {len(data_loader)} images from {subset_name}")
        
        # Process images with progress bar
        for idx, (img_tensor, label) in enumerate(tqdm(data_loader, desc=f"  {subset_name}")):
            if max_images and idx >= max_images:
                break
            
            try:
                # Move to device
                if self.use_cuda:
                    img_tensor = img_tensor.cuda()
                
                # Generate GradCAM analysis
                result = self.gradcam.generate_prediction_with_cam(img_tensor)
                
                # Add metadata
                result.update({
                    'image_index': idx,
                    'true_label': label.item(),
                    'subset_name': subset_name,
                    'predicted_correct': (result['predicted_class_idx'] == label.item())
                })
                
                # Save CAM if requested
                if save_cams:
                    cam_path = os.path.join(output_dir, f"image_{idx:06d}_cam.npy")
                    np.save(cam_path, result['cam'])
                    result['cam_path'] = cam_path
                    # Remove large array to save memory
                    del result['cam']
                
                results.append(result)
                successful_count += 1
                
            except Exception as e:
                print(f"      Error processing image {idx}: {e}")
                failed_count += 1
                results.append({
                    'image_index': idx,
                    'true_label': label.item(),
                    'subset_name': subset_name,
                    'error': str(e),
                    'success': False
                })
        
        # Compute subset summary
        summary = self._compute_subset_summary(results, successful_count, failed_count)
        
        return {
            'results': results,
            'summary': summary
        }
    
    def _compute_subset_summary(
        self, 
        results: List[Dict], 
        successful_count: int, 
        failed_count: int
    ) -> Dict[str, Any]:
        """Compute summary statistics for a subset."""
        total_count = successful_count + failed_count
        
        if successful_count == 0:
            return {
                'total_images': total_count,
                'successful': successful_count,
                'failed': failed_count,
                'success_rate': 0.0
            }
        
        # Extract successful results
        successful_results = [r for r in results if r.get('success', True)]
        
        # Compute accuracy
        correct_predictions = sum(1 for r in successful_results if r.get('predicted_correct', False))
        accuracy = correct_predictions / successful_count if successful_count > 0 else 0.0
        
        # Compute confidence statistics
        confidences = [r['confidence'] for r in successful_results]
        real_probs = [r['real_prob'] for r in successful_results]
        fake_probs = [r['fake_prob'] for r in successful_results]
        
        # Class distribution
        predictions = [r['predicted_class_name'] for r in successful_results]
        class_counts = pd.Series(predictions).value_counts().to_dict()
        
        return {
            'total_images': total_count,
            'successful': successful_count,
            'failed': failed_count,
            'success_rate': successful_count / total_count,
            'accuracy': accuracy,
            'class_distribution': class_counts,
            'confidence_stats': {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences)
            },
            'real_prob_stats': {
                'mean': np.mean(real_probs),
                'std': np.std(real_probs)
            },
            'fake_prob_stats': {
                'mean': np.mean(fake_probs),
                'std': np.std(fake_probs)
            }
        }
    
    def _compute_overall_statistics(
        self, 
        all_results: List[Dict], 
        dataset_summaries: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """Compute overall statistics across all datasets."""
        successful_results = [r for r in all_results if r.get('success', True)]
        
        if not successful_results:
            return {}
        
        # Overall accuracy
        correct_predictions = sum(1 for r in successful_results if r.get('predicted_correct', False))
        overall_accuracy = correct_predictions / len(successful_results)
        
        # Overall class distribution
        predictions = [r['predicted_class_name'] for r in successful_results]
        overall_class_counts = pd.Series(predictions).value_counts().to_dict()
        
        # Confidence statistics
        confidences = [r['confidence'] for r in successful_results]
        
        # Per-dataset accuracy
        dataset_accuracies = {}
        for dataset_name, summary in dataset_summaries.items():
            dataset_accuracies[dataset_name] = summary.get('accuracy', 0.0)
        
        return {
            'total_processed': len(all_results),
            'total_successful': len(successful_results),
            'overall_accuracy': overall_accuracy,
            'overall_class_distribution': overall_class_counts,
            'overall_confidence_stats': {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences)
            },
            'dataset_accuracies': dataset_accuracies,
            'num_datasets': len(dataset_summaries)
        }
    
    def _save_arrow_dataset_summary(
        self,
        all_results: List[Dict],
        dataset_summaries: Dict[str, Dict],
        overall_stats: Dict[str, Any],
        output_dir: str
    ):
        """Save comprehensive summary of Arrow dataset processing."""
        
        # Save detailed results as JSON
        results_path = os.path.join(output_dir, 'arrow_dataset_results.json')
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = []
            for result in all_results:
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
                'dataset_summaries': dataset_summaries,
                'overall_statistics': overall_stats
            }
            json.dump(json_data, f, indent=2)
        
        # Save CSV summary
        csv_path = os.path.join(output_dir, 'arrow_dataset_summary.csv')
        successful_results = [r for r in all_results if r.get('success', True)]
        if successful_results:
            df_data = []
            for result in successful_results:
                df_data.append({
                    'subset_name': result['subset_name'],
                    'image_index': result['image_index'],
                    'true_label': result['true_label'],
                    'predicted_class': result['predicted_class_name'],
                    'confidence': result['confidence'],
                    'real_prob': result['real_prob'],
                    'fake_prob': result['fake_prob'],
                    'correct': result['predicted_correct']
                })
            
            df = pd.DataFrame(df_data)
            df.to_csv(csv_path, index=False)
        
        # Save statistics summary
        stats_path = os.path.join(output_dir, 'arrow_dataset_statistics.txt')
        with open(stats_path, 'w') as f:
            f.write("ARROW DATASET GRADCAM ANALYSIS SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("Overall Statistics:\n")
            f.write(f"  Total processed: {overall_stats.get('total_processed', 0)}\n")
            f.write(f"  Successful: {overall_stats.get('total_successful', 0)}\n")
            f.write(f"  Overall accuracy: {overall_stats.get('overall_accuracy', 0):.4f}\n")
            f.write(f"  Number of datasets: {overall_stats.get('num_datasets', 0)}\n\n")
            
            f.write("Class Distribution:\n")
            for class_name, count in overall_stats.get('overall_class_distribution', {}).items():
                percentage = (count / overall_stats.get('total_successful', 1)) * 100
                f.write(f"  {class_name}: {count} ({percentage:.1f}%)\n")
            f.write("\n")
            
            f.write("Per-Dataset Accuracy:\n")
            for dataset_name, accuracy in overall_stats.get('dataset_accuracies', {}).items():
                f.write(f"  {dataset_name}: {accuracy:.4f}\n")
            f.write("\n")
            
            f.write("Dataset Summaries:\n")
            for dataset_name, summary in dataset_summaries.items():
                f.write(f"\n  {dataset_name}:\n")
                f.write(f"    Total images: {summary.get('total_images', 0)}\n")
                f.write(f"    Success rate: {summary.get('success_rate', 0):.2%}\n")
                f.write(f"    Accuracy: {summary.get('accuracy', 0):.4f}\n")
                f.write(f"    Mean confidence: {summary.get('confidence_stats', {}).get('mean', 0):.4f}\n")
        
        print(f"Arrow dataset summary saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Arrow dataset GradCAM processing for PoundNet")
    parser.add_argument('--config', type=str, required=True,
                       help='Path to PoundNet configuration file (same as test.py)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--target_layers', type=str, nargs='+', default=None,
                       help='Specific layers to analyze')
    parser.add_argument('--max_images_per_subset', type=int, default=None,
                       help='Maximum images per dataset subset')
    parser.add_argument('--save_cams', action='store_true',
                       help='Save individual CAM arrays')
    parser.add_argument('--dataset_filter', type=str, nargs='+', default=None,
                       help='Filter specific datasets (e.g., DiffusionForensics ForenSynths)')
    parser.add_argument('--subset_filter', type=str, nargs='+', default=None,
                       help='Filter specific subsets (e.g., progan stylegan)')
    
    args = parser.parse_args()
    
    # Load configuration (same as test.py)
    print("Loading PoundNet configuration...")
    conf = load_config_with_cli(args.config)
    conf = hydra.utils.instantiate(conf)
    
    # Filter datasets if requested
    if args.dataset_filter or args.subset_filter:
        filtered_sources = []
        for source in conf.datasets.source:
            # Filter by dataset name
            if args.dataset_filter and source['benchmark_name'] not in args.dataset_filter:
                continue
            
            # Filter by subset names
            if args.subset_filter:
                filtered_subsets = [s for s in source['sub_sets'] if s in args.subset_filter]
                if not filtered_subsets:
                    continue
                source = dict(source)  # Make a copy
                source['sub_sets'] = filtered_subsets
            
            filtered_sources.append(source)
        
        conf.datasets.source = filtered_sources
        print(f"Filtered to {len(filtered_sources)} dataset sources")
    
    # Initialize processor
    processor = ArrowDatasetGradCAMProcessor(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        target_layers=args.target_layers
    )
    
    # Process datasets
    results = processor.process_arrow_dataset(
        dataset_config=conf.datasets,
        output_dir=args.output_dir,
        max_images_per_subset=args.max_images_per_subset,
        save_cams=args.save_cams,
        save_summary=True
    )
    
    # Print summary
    print("\nArrow dataset processing complete!")
    stats = results['overall_statistics']
    print(f"Total processed: {stats.get('total_processed', 0)}")
    print(f"Overall accuracy: {stats.get('overall_accuracy', 0):.4f}")
    print(f"Datasets analyzed: {stats.get('num_datasets', 0)}")
    
    if stats.get('overall_class_distribution'):
        print("\nClass distribution:")
        for class_name, count in stats['overall_class_distribution'].items():
            percentage = (count / stats.get('total_successful', 1)) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")


if __name__ == "__main__":
    main()