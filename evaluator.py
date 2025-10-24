"""
Evaluation Module for Dance Movement Analysis
Comprehensive evaluation of aesthetic and biomechanical metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix
from scipy.stats import pearsonr, spearmanr
from scipy import stats
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class DanceEvaluator:
    """
    Comprehensive evaluator for dance movement analysis
    Implements all metrics from Section 4.4 and 5 of the paper
    """
    
    def __init__(self, model: nn.Module, config: Dict, device: torch.device):
        """
        Initialize evaluator
        
        Args:
            model: Trained dance aesthetic model
            config: Configuration dictionary
            device: Device to use
        """
        self.model = model
        self.config = config
        self.device = device
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize metrics storage
        self.reset_metrics()
        
        # Visualization settings
        self.visualization_config = config.get('visualization', {})
        
    def reset_metrics(self):
        """Reset all stored metrics"""
        self.predictions = []
        self.ground_truth = []
        self.styles = []
        self.biomech_scores = []
        self.joint_stress_values = []
        self.cam_visualizations = []
        
    def evaluate_comprehensive(self, data_loader) -> Dict:
        """
        Perform comprehensive evaluation on dataset
        
        Args:
            data_loader: Data loader for evaluation
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        self.reset_metrics()
        
        # Disable gradient computation
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc='Evaluating')):
                # Move data to device
                frames = batch['frames'].to(self.device)
                aesthetic_scores = batch['aesthetic_scores'].to(self.device)
                sequence_lengths = batch.get('sequence_lengths', None)
                
                # Forward pass
                outputs = self.model(
                    frames,
                    sequence_lengths,
                    return_features=True
                )
                
                # Store predictions
                self._store_batch_results(batch, outputs)
                
                # Generate CAMs for subset
                if batch_idx < 5 and outputs.get('cams') is not None:
                    self._store_cam_visualizations(frames, outputs['cams'])
                    
        # Calculate all metrics
        metrics = self._calculate_all_metrics()
        
        # Generate analysis
        analysis = self._generate_analysis()
        
        return {
            'metrics': metrics,
            'analysis': analysis,
            'visualizations': self.cam_visualizations
        }
    
    def _store_batch_results(self, batch: Dict, outputs: Dict):
        """Store batch results for metric calculation"""
        # Aesthetic scores
        preds = outputs['aesthetic_score'].squeeze().cpu().numpy()
        targets = batch['aesthetic_scores'].cpu().numpy()
        
        self.predictions.extend(preds)
        self.ground_truth.extend(targets)
        
        # Dance styles
        if 'styles' in batch:
            self.styles.extend(batch['styles'])
            
        # Biomechanical scores
        if 'biomechanical_score' in outputs:
            biomech = outputs['biomechanical_score'].squeeze().cpu().numpy()
            self.biomech_scores.extend(biomech)
            
        # Joint stress
        if 'joint_angles' in outputs:
            stress = self._calculate_joint_stress(outputs['joint_angles'])
            self.joint_stress_values.extend(stress.cpu().numpy())
            
    def _calculate_joint_stress(self, joint_angles: torch.Tensor) -> torch.Tensor:
        """
        Calculate joint stress from angles
        
        Args:
            joint_angles: Joint angle tensor
            
        Returns:
            Joint stress values
        """
        # Define safe ranges for each joint type
        safe_ranges = {
            'head': (-0.5, 0.5),
            'shoulder': (-2.0, 2.0),
            'elbow': (0, 2.5),
            'wrist': (-1.0, 1.0),
            'hip': (-1.5, 1.5),
            'knee': (0, 2.3),
            'ankle': (-0.5, 0.5)
        }
        
        batch_size = joint_angles.shape[0]
        stress_values = torch.zeros(batch_size).to(joint_angles.device)
        
        # Calculate stress for each joint
        for i in range(joint_angles.shape[1]):
            # Determine joint type based on index
            if i < 3:
                min_val, max_val = safe_ranges['head']
            elif i < 7:
                min_val, max_val = safe_ranges['shoulder']
            elif i < 11:
                min_val, max_val = safe_ranges['elbow']
            elif i < 15:
                min_val, max_val = safe_ranges['wrist']
            elif i < 19:
                min_val, max_val = safe_ranges['hip']
            elif i < 23:
                min_val, max_val = safe_ranges['knee']
            else:
                min_val, max_val = safe_ranges['ankle']
                
            # Calculate deviation from safe range
            angles = joint_angles[:, i, :]
            below_min = torch.relu(min_val - angles)
            above_max = torch.relu(angles - max_val)
            
            stress = below_min + above_max
            stress_values += stress.mean(dim=1)
            
        # Normalize
        stress_values = stress_values / joint_angles.shape[1]
        
        return stress_values
    
    def _calculate_all_metrics(self) -> Dict:
        """Calculate all evaluation metrics"""
        predictions = np.array(self.predictions)
        ground_truth = np.array(self.ground_truth)
        
        # Basic metrics
        metrics = {
            'mse': mean_squared_error(ground_truth, predictions),
            'rmse': np.sqrt(mean_squared_error(ground_truth, predictions)),
            'mae': mean_absolute_error(ground_truth, predictions),
            'pearson_r': pearsonr(ground_truth, predictions)[0],
            'pearson_p': pearsonr(ground_truth, predictions)[1],
            'spearman_r': spearmanr(ground_truth, predictions)[0],
            'spearman_p': spearmanr(ground_truth, predictions)[1]
        }
        
        # R-squared
        ss_tot = np.sum((ground_truth - np.mean(ground_truth)) ** 2)
        ss_res = np.sum((ground_truth - predictions) ** 2)
        metrics['r_squared'] = 1 - (ss_res / ss_tot)
        
        # Mean and std of errors
        errors = predictions - ground_truth
        metrics['mean_error'] = np.mean(errors)
        metrics['std_error'] = np.std(errors)
        
        # Percentile errors
        metrics['median_absolute_error'] = np.median(np.abs(errors))
        metrics['percentile_25_error'] = np.percentile(np.abs(errors), 25)
        metrics['percentile_75_error'] = np.percentile(np.abs(errors), 75)
        metrics['percentile_95_error'] = np.percentile(np.abs(errors), 95)
        
        # Per-style metrics if available
        if self.styles:
            unique_styles = list(set(self.styles))
            for style in unique_styles:
                style_mask = [s == style for s in self.styles]
                style_preds = predictions[style_mask]
                style_truth = ground_truth[style_mask]
                
                if len(style_preds) > 1:
                    metrics[f'{style}_mse'] = mean_squared_error(style_truth, style_preds)
                    metrics[f'{style}_correlation'] = pearsonr(style_truth, style_preds)[0]
                    metrics[f'{style}_mae'] = mean_absolute_error(style_truth, style_preds)
                    
        # Biomechanical metrics if available
        if self.biomech_scores:
            biomech_scores = np.array(self.biomech_scores)
            metrics['mean_biomech_score'] = np.mean(biomech_scores)
            metrics['std_biomech_score'] = np.std(biomech_scores)
            
            # Correlation between aesthetic and biomechanical
            if len(biomech_scores) == len(predictions):
                metrics['aesthetic_biomech_correlation'] = pearsonr(predictions, biomech_scores)[0]
                
        # Joint stress metrics
        if self.joint_stress_values:
            joint_stress = np.array(self.joint_stress_values)
            metrics['mean_joint_stress'] = np.mean(joint_stress)
            metrics['std_joint_stress'] = np.std(joint_stress)
            metrics['max_joint_stress'] = np.max(joint_stress)
            
            # Percentage within safe limits
            safe_threshold = self.config['evaluation']['biomechanical']['joint_stress_threshold']
            metrics['percent_safe_poses'] = np.mean(joint_stress < safe_threshold) * 100
            
        # Inter-rater agreement simulation (approximation)
        if self.config['evaluation']['calculate_inter_rater']:
            metrics['fleiss_kappa'] = self._calculate_fleiss_kappa(predictions, ground_truth)
            
        return metrics
    
    def _calculate_fleiss_kappa(self, predictions: np.ndarray, 
                                ground_truth: np.ndarray) -> float:
        """
        Calculate Fleiss' Kappa for inter-rater agreement
        Using Cohen's kappa as approximation for continuous scores
        """
        from sklearn.metrics import cohen_kappa_score
        
        # Discretize scores into categories
        n_categories = 10
        pred_categories = np.digitize(predictions, bins=np.linspace(0, 10, n_categories+1))
        truth_categories = np.digitize(ground_truth, bins=np.linspace(0, 10, n_categories+1))
        
        # Calculate weighted kappa
        kappa = cohen_kappa_score(truth_categories, pred_categories, weights='quadratic')
        
        return kappa
    
    def _generate_analysis(self) -> Dict:
        """Generate detailed analysis of results"""
        predictions = np.array(self.predictions)
        ground_truth = np.array(self.ground_truth)
        
        analysis = {}
        
        # Error distribution analysis
        errors = predictions - ground_truth
        analysis['error_distribution'] = {
            'skewness': stats.skew(errors),
            'kurtosis': stats.kurtosis(errors),
            'normality_test': stats.normaltest(errors),
            'histogram': np.histogram(errors, bins=20)
        }
        
        # Performance by score range
        score_ranges = [(0, 3), (3, 6), (6, 8), (8, 10)]
        analysis['performance_by_range'] = {}
        
        for low, high in score_ranges:
            mask = (ground_truth >= low) & (ground_truth < high)
            if np.any(mask):
                range_preds = predictions[mask]
                range_truth = ground_truth[mask]
                
                analysis['performance_by_range'][f'{low}-{high}'] = {
                    'count': int(np.sum(mask)),
                    'mse': float(mean_squared_error(range_truth, range_preds)),
                    'mae': float(mean_absolute_error(range_truth, range_preds))
                }
                
        # Outlier analysis
        z_scores = np.abs(stats.zscore(errors))
        outliers = z_scores > 3
        analysis['outliers'] = {
            'count': int(np.sum(outliers)),
            'percentage': float(np.mean(outliers) * 100),
            'indices': np.where(outliers)[0].tolist()
        }
        
        # Confidence intervals
        confidence_level = 0.95
        analysis['confidence_intervals'] = {
            'mean_prediction': self._calculate_confidence_interval(predictions, confidence_level),
            'mean_error': self._calculate_confidence_interval(errors, confidence_level)
        }
        
        return analysis
    
    def _calculate_confidence_interval(self, data: np.ndarray, 
                                      confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for data"""
        mean = np.mean(data)
        se = stats.sem(data)
        interval = se * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
        
        return (float(mean - interval), float(mean + interval))
    
    def _store_cam_visualizations(self, frames: torch.Tensor, cams: torch.Tensor):
        """Store CAM visualizations for analysis"""
        # Get last frame from sequence
        last_frames = frames[:, -1]  # (batch_size, C, H, W)
        
        for i in range(min(5, frames.shape[0])):
            # Denormalize frame
            frame = last_frames[i].cpu().numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            frame = frame * std + mean
            frame = np.clip(frame, 0, 1)
            
            # Get CAM
            if cams.dim() > 2:
                cam = cams[i, 0].cpu().numpy() if cams.shape[1] > 0 else cams[i].cpu().numpy()
            else:
                cam = cams[i].cpu().numpy()
                
            # Resize CAM to match frame
            cam_resized = cv2.resize(cam, (frame.shape[1], frame.shape[0]))
            
            # Normalize CAM
            cam_normalized = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)
            
            # Store visualization data
            self.cam_visualizations.append({
                'frame': frame,
                'cam': cam_normalized,
                'overlay': self._create_cam_overlay(frame, cam_normalized)
            })
            
    def _create_cam_overlay(self, frame: np.ndarray, cam: np.ndarray, 
                            alpha: float = 0.5) -> np.ndarray:
        """Create CAM overlay on frame"""
        # Apply colormap to CAM
        cam_colored = plt.cm.jet(cam)[:, :, :3]
        
        # Create overlay
        overlay = frame * (1 - alpha) + cam_colored * alpha
        
        return overlay
    
    def generate_visualizations(self, data_loader, output_dir: Path):
        """
        Generate comprehensive visualizations
        
        Args:
            data_loader: Data loader for visualization
            output_dir: Directory to save visualizations
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate error distribution plot
        self._plot_error_distribution(output_dir)
        
        # Generate scatter plot
        self._plot_predictions_vs_truth(output_dir)
        
        # Generate per-style performance
        if self.styles:
            self._plot_per_style_performance(output_dir)
            
        # Generate CAM visualizations
        if self.cam_visualizations:
            self._save_cam_visualizations(output_dir)
            
        # Generate biomechanical analysis
        if self.biomech_scores:
            self._plot_biomechanical_analysis(output_dir)
            
    def _plot_error_distribution(self, output_dir: Path):
        """Plot error distribution"""
        predictions = np.array(self.predictions)
        ground_truth = np.array(self.ground_truth)
        errors = predictions - ground_truth
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram
        axes[0].hist(errors, bins=30, edgecolor='black', alpha=0.7)
        axes[0].axvline(x=0, color='red', linestyle='--', label='Zero Error')
        axes[0].set_xlabel('Prediction Error')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Error Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Q-Q plot
        stats.probplot(errors, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'error_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    def _plot_predictions_vs_truth(self, output_dir: Path):
        """Plot predictions vs ground truth"""
        predictions = np.array(self.predictions)
        ground_truth = np.array(self.ground_truth)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Scatter plot
        ax.scatter(ground_truth, predictions, alpha=0.5, s=20)
        
        # Perfect prediction line
        min_val = min(ground_truth.min(), predictions.min())
        max_val = max(ground_truth.max(), predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        # Fit line
        z = np.polyfit(ground_truth, predictions, 1)
        p = np.poly1d(z)
        ax.plot(ground_truth, p(ground_truth), 'b-', alpha=0.7, label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')
        
        # Annotations
        correlation = pearsonr(ground_truth, predictions)[0]
        ax.text(0.05, 0.95, f'r = {correlation:.3f}\nMSE = {mean_squared_error(ground_truth, predictions):.3f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Ground Truth Aesthetic Score')
        ax.set_ylabel('Predicted Aesthetic Score')
        ax.set_title('Predictions vs Ground Truth')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'predictions_vs_truth.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    def _plot_per_style_performance(self, output_dir: Path):
        """Plot performance per dance style"""
        predictions = np.array(self.predictions)
        ground_truth = np.array(self.ground_truth)
        
        # Calculate metrics per style
        unique_styles = list(set(self.styles))
        style_metrics = []
        
        for style in unique_styles:
            style_mask = [s == style for s in self.styles]
            style_preds = predictions[style_mask]
            style_truth = ground_truth[style_mask]
            
            if len(style_preds) > 1:
                style_metrics.append({
                    'Style': style,
                    'MSE': mean_squared_error(style_truth, style_preds),
                    'MAE': mean_absolute_error(style_truth, style_preds),
                    'Correlation': pearsonr(style_truth, style_preds)[0],
                    'Count': len(style_preds)
                })
                
        df = pd.DataFrame(style_metrics)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # MSE by style
        axes[0, 0].bar(df['Style'], df['MSE'])
        axes[0, 0].set_xlabel('Dance Style')
        axes[0, 0].set_ylabel('MSE')
        axes[0, 0].set_title('MSE by Dance Style')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # MAE by style
        axes[0, 1].bar(df['Style'], df['MAE'])
        axes[0, 1].set_xlabel('Dance Style')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].set_title('MAE by Dance Style')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Correlation by style
        axes[1, 0].bar(df['Style'], df['Correlation'])
        axes[1, 0].set_xlabel('Dance Style')
        axes[1, 0].set_ylabel('Correlation')
        axes[1, 0].set_title('Correlation by Dance Style')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].axhline(y=0.92, color='r', linestyle='--', label='Paper Result (0.92)')
        axes[1, 0].legend()
        
        # Sample count by style
        axes[1, 1].bar(df['Style'], df['Count'])
        axes[1, 1].set_xlabel('Dance Style')
        axes[1, 1].set_ylabel('Sample Count')
        axes[1, 1].set_title('Sample Distribution')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'per_style_performance.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    def _save_cam_visualizations(self, output_dir: Path):
        """Save CAM visualizations"""
        cam_dir = output_dir / 'cam_visualizations'
        cam_dir.mkdir(exist_ok=True)
        
        for i, viz in enumerate(self.cam_visualizations):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original frame
            axes[0].imshow(viz['frame'])
            axes[0].set_title('Original Frame')
            axes[0].axis('off')
            
            # CAM heatmap
            im = axes[1].imshow(viz['cam'], cmap='hot')
            axes[1].set_title('CAM Heatmap')
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], fraction=0.046)
            
            # Overlay
            axes[2].imshow(viz['overlay'])
            axes[2].set_title('CAM Overlay')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(cam_dir / f'cam_{i}.png', dpi=150, bbox_inches='tight')
            plt.close()
            
    def _plot_biomechanical_analysis(self, output_dir: Path):
        """Plot biomechanical analysis"""
        biomech_scores = np.array(self.biomech_scores)
        predictions = np.array(self.predictions)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Biomechanical score distribution
        axes[0].hist(biomech_scores, bins=30, edgecolor='black', alpha=0.7)
        axes[0].axvline(x=np.mean(biomech_scores), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(biomech_scores):.2f}')
        axes[0].set_xlabel('Biomechanical Efficiency Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Biomechanical Score Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Aesthetic vs Biomechanical correlation
        axes[1].scatter(predictions, biomech_scores, alpha=0.5)
        
        # Fit line
        z = np.polyfit(predictions, biomech_scores, 1)
        p = np.poly1d(z)
        axes[1].plot(predictions, p(predictions), 'r-', alpha=0.7)
        
        correlation = pearsonr(predictions, biomech_scores)[0]
        axes[1].set_xlabel('Aesthetic Score')
        axes[1].set_ylabel('Biomechanical Score')
        axes[1].set_title(f'Aesthetic vs Biomechanical (r={correlation:.3f})')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'biomechanical_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    def generate_report(self, metrics: Dict, analysis: Dict, output_path: Path):
        """
        Generate comprehensive evaluation report
        
        Args:
            metrics: Evaluation metrics
            analysis: Analysis results
            output_path: Path to save report
        """
        report = []
        report.append("="*60)
        report.append("DANCE MOVEMENT ANALYSIS - EVALUATION REPORT")
        report.append("="*60)
        report.append("")
        
        # Model Performance vs Paper Results
        report.append("MODEL PERFORMANCE (vs Paper Results)")
        report.append("-"*40)
        report.append(f"MSE: {metrics['mse']:.4f} (Paper: 0.25)")
        report.append(f"Correlation: {metrics['pearson_r']:.4f} (Paper: 0.92)")
        report.append(f"RMSE: {metrics['rmse']:.4f}")
        report.append(f"MAE: {metrics['mae']:.4f}")
        report.append("")
        
        # Statistical Significance
        report.append("STATISTICAL SIGNIFICANCE")
        report.append("-"*40)
        report.append(f"Pearson p-value: {metrics.get('pearson_p', 'N/A')}")
        report.append(f"Spearman p-value: {metrics.get('spearman_p', 'N/A')}")
        report.append("")
        
        # Biomechanical Metrics
        if 'mean_biomech_score' in metrics:
            report.append("BIOMECHANICAL ANALYSIS")
            report.append("-"*40)
            report.append(f"Mean Efficiency: {metrics['mean_biomech_score']:.3f}")
            report.append(f"Std Efficiency: {metrics['std_biomech_score']:.3f}")
            report.append(f"Safe Poses: {metrics.get('percent_safe_poses', 'N/A'):.1f}%")
            report.append(f"Aesthetic-Biomech Correlation: {metrics.get('aesthetic_biomech_correlation', 'N/A'):.3f}")
            report.append("")
            
        # Per-Style Performance
        style_metrics = [k for k in metrics.keys() if '_mse' in k and k != 'mse']
        if style_metrics:
            report.append("PER-STYLE PERFORMANCE")
            report.append("-"*40)
            for style_metric in style_metrics:
                style = style_metric.replace('_mse', '')
                report.append(f"{style}:")
                report.append(f"  MSE: {metrics[style_metric]:.4f}")
                report.append(f"  Correlation: {metrics.get(f'{style}_correlation', 'N/A'):.4f}")
                report.append(f"  MAE: {metrics.get(f'{style}_mae', 'N/A'):.4f}")
            report.append("")
            
        # Error Analysis
        if 'error_distribution' in analysis:
            report.append("ERROR ANALYSIS")
            report.append("-"*40)
            report.append(f"Skewness: {analysis['error_distribution']['skewness']:.3f}")
            report.append(f"Kurtosis: {analysis['error_distribution']['kurtosis']:.3f}")
            report.append(f"Outliers: {analysis['outliers']['count']} ({analysis['outliers']['percentage']:.1f}%)")
            report.append("")
            
        # Confidence Intervals
        if 'confidence_intervals' in analysis:
            report.append("95% CONFIDENCE INTERVALS")
            report.append("-"*40)
            ci_pred = analysis['confidence_intervals']['mean_prediction']
            report.append(f"Mean Prediction: [{ci_pred[0]:.3f}, {ci_pred[1]:.3f}]")
            ci_err = analysis['confidence_intervals']['mean_error']
            report.append(f"Mean Error: [{ci_err[0]:.3f}, {ci_err[1]:.3f}]")
            report.append("")
            
        # Inter-rater Agreement
        if 'fleiss_kappa' in metrics:
            report.append("INTER-RATER AGREEMENT")
            report.append("-"*40)
            report.append(f"Fleiss' Kappa: {metrics['fleiss_kappa']:.3f}")
            report.append("")
            
        report.append("="*60)
        report.append(f"Report generated at: {pd.Timestamp.now()}")
        report.append("="*60)
        
        # Save report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report))
            
        print(f"Report saved to {output_path}")
