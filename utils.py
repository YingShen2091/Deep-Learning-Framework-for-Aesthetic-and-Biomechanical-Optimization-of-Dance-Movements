"""
Utility modules for Dance Movement Analysis System
Includes logging, seed management, and other helper functions
"""

# ========== logger.py ==========

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(name: str, log_dir: str = './logs', 
                 level: str = 'INFO', console: bool = True) -> logging.Logger:
    """
    Setup logger with file and console handlers
    
    Args:
        name: Logger name
        log_dir: Directory to save log files
        level: Logging level
        console: Whether to output to console
        
    Returns:
        Configured logger
    """
    # Create log directory
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    log_file = log_dir / f'{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, level.upper()))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger


# ========== seed.py ==========

import random
import numpy as np
import torch


def set_all_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make cudnn deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"All random seeds set to {seed}")


# ========== metrics_utils.py ==========

import torch
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr


def calculate_aesthetic_metrics(predictions: np.ndarray, 
                               ground_truth: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive aesthetic evaluation metrics
    
    Args:
        predictions: Model predictions
        ground_truth: Ground truth labels
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'mse': mean_squared_error(ground_truth, predictions),
        'rmse': np.sqrt(mean_squared_error(ground_truth, predictions)),
        'mae': mean_absolute_error(ground_truth, predictions),
        'pearson_r': pearsonr(ground_truth, predictions)[0],
        'spearman_r': spearmanr(ground_truth, predictions)[0]
    }
    
    # R-squared
    ss_tot = np.sum((ground_truth - np.mean(ground_truth)) ** 2)
    ss_res = np.sum((ground_truth - predictions) ** 2)
    metrics['r_squared'] = 1 - (ss_res / ss_tot)
    
    return metrics


def calculate_biomechanical_metrics(joint_angles: torch.Tensor,
                                   safe_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    """
    Calculate biomechanical efficiency metrics
    
    Args:
        joint_angles: Joint angle tensor
        safe_ranges: Safe ranges for each joint type
        
    Returns:
        Dictionary of biomechanical metrics
    """
    batch_size = joint_angles.shape[0]
    stress_values = torch.zeros(batch_size)
    
    # Calculate stress for each joint
    for i in range(joint_angles.shape[1]):
        # Determine joint type and get safe range
        if i < 3:
            min_val, max_val = safe_ranges.get('head', (-0.5, 0.5))
        elif i < 7:
            min_val, max_val = safe_ranges.get('shoulder', (-2.0, 2.0))
        elif i < 11:
            min_val, max_val = safe_ranges.get('elbow', (0, 2.5))
        elif i < 15:
            min_val, max_val = safe_ranges.get('wrist', (-1.0, 1.0))
        elif i < 19:
            min_val, max_val = safe_ranges.get('hip', (-1.5, 1.5))
        elif i < 23:
            min_val, max_val = safe_ranges.get('knee', (0, 2.3))
        else:
            min_val, max_val = safe_ranges.get('ankle', (-0.5, 0.5))
            
        # Calculate deviation
        angles = joint_angles[:, i, :]
        below_min = torch.relu(min_val - angles)
        above_max = torch.relu(angles - max_val)
        
        stress = below_min + above_max
        stress_values += stress.mean(dim=1)
        
    # Normalize stress
    stress_values = stress_values / joint_angles.shape[1]
    
    metrics = {
        'mean_stress': float(stress_values.mean()),
        'max_stress': float(stress_values.max()),
        'percent_safe': float((stress_values < 0.5).float().mean() * 100)
    }
    
    return metrics


# ========== video_utils.py ==========

import cv2
import ffmpeg


def extract_video_info(video_path: str) -> Dict:
    """
    Extract video information
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary containing video information
    """
    try:
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams'] 
                            if stream['codec_type'] == 'video'), None)
        
        if video_stream:
            info = {
                'width': int(video_stream['width']),
                'height': int(video_stream['height']),
                'fps': eval(video_stream['r_frame_rate']),
                'duration': float(video_stream.get('duration', 0)),
                'nb_frames': int(video_stream.get('nb_frames', 0)),
                'codec': video_stream['codec_name']
            }
        else:
            # Fallback to OpenCV
            cap = cv2.VideoCapture(video_path)
            info = {
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'nb_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'duration': 0,
                'codec': 'unknown'
            }
            cap.release()
            
    except Exception as e:
        print(f"Error extracting video info: {e}")
        info = {}
        
    return info


def create_video_from_frames(frames: List[np.ndarray], output_path: str,
                            fps: float = 30.0, codec: str = 'mp4v'):
    """
    Create video from frame list
    
    Args:
        frames: List of frames
        output_path: Output video path
        fps: Frames per second
        codec: Video codec
    """
    if not frames:
        print("No frames to create video")
        return
        
    height, width = frames[0].shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        if frame.shape[2] == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame
            
        out.write(frame_bgr)
        
    out.release()
    print(f"Video saved to {output_path}")


# ========== data_utils.py ==========

def normalize_scores(scores: np.ndarray, min_val: float = 0, 
                    max_val: float = 10) -> np.ndarray:
    """
    Normalize scores to [0, 1] range
    
    Args:
        scores: Raw scores
        min_val: Minimum score value
        max_val: Maximum score value
        
    Returns:
        Normalized scores
    """
    return (scores - min_val) / (max_val - min_val)


def denormalize_scores(scores: np.ndarray, min_val: float = 0,
                      max_val: float = 10) -> np.ndarray:
    """
    Denormalize scores from [0, 1] to original range
    
    Args:
        scores: Normalized scores
        min_val: Minimum score value
        max_val: Maximum score value
        
    Returns:
        Denormalized scores
    """
    return scores * (max_val - min_val) + min_val


def split_sequences(sequence: np.ndarray, window_size: int,
                   stride: int = 1) -> List[np.ndarray]:
    """
    Split long sequence into overlapping windows
    
    Args:
        sequence: Input sequence
        window_size: Size of each window
        stride: Stride between windows
        
    Returns:
        List of windowed sequences
    """
    windows = []
    for i in range(0, len(sequence) - window_size + 1, stride):
        windows.append(sequence[i:i + window_size])
    return windows


# ========== visualization_utils.py ==========

import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_curves(history: Dict, save_path: Optional[str] = None):
    """
    Plot training and validation curves
    
    Args:
        history: Training history dictionary
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MSE curves
    axes[0, 1].plot(history['train_mse'], label='Train')
    axes[0, 1].plot(history['val_mse'], label='Validation')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE')
    axes[0, 1].set_title('MSE Curves')
    axes[0, 1].axhline(y=0.25, color='r', linestyle='--', label='Paper Result')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Correlation curves
    axes[1, 0].plot(history['train_correlation'], label='Train')
    axes[1, 0].plot(history['val_correlation'], label='Validation')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Correlation')
    axes[1, 0].set_title('Correlation Curves')
    axes[1, 0].axhline(y=0.92, color='r', linestyle='--', label='Paper Result')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate
    if 'learning_rate' in history:
        axes[1, 1].plot(history['learning_rate'])
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
        
    plt.close()


def visualize_pose_sequence(poses: np.ndarray, save_path: Optional[str] = None):
    """
    Visualize pose sequence as stick figures
    
    Args:
        poses: Pose sequence (frames, keypoints, coordinates)
        save_path: Path to save animation
    """
    # Skeleton connections for visualization
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Head
        (5, 6), (6, 7), (7, 8),  # Left arm
        (9, 10), (10, 11), (11, 12),  # Right arm
        (13, 14), (14, 15), (15, 16),  # Left leg
        (17, 18), (18, 19), (19, 20),  # Right leg
        (0, 5), (0, 9), (5, 9),  # Shoulders
        (13, 17), (13, 17)  # Hips
    ]
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Animation function would be implemented here
    # For simplicity, showing first frame
    if len(poses) > 0:
        pose = poses[0]
        
        # Draw connections
        for connection in connections:
            if connection[0] < len(pose) and connection[1] < len(pose):
                point1 = pose[connection[0]]
                point2 = pose[connection[1]]
                ax.plot([point1[0], point2[0]],
                       [point1[1], point2[1]],
                       [point1[2], point2[2]] if pose.shape[1] > 2 else [0, 0],
                       'b-')
                       
        # Draw keypoints
        ax.scatter(pose[:, 0], pose[:, 1],
                  pose[:, 2] if pose.shape[1] > 2 else np.zeros(len(pose)),
                  c='red', s=50)
                  
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Pose Visualization')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
        
    plt.close()


# ========== model_utils.py ==========

def count_parameters(model: torch.nn.Module) -> int:
    """
    Count total and trainable parameters in model
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def get_model_size(model: torch.nn.Module) -> float:
    """
    Get model size in MB
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        
    size_mb = (param_size + buffer_size) / 1024 / 1024
    
    return size_mb
