"""
Data Preprocessing and Loading Module
Handles video frames, motion capture data, pose estimation, and data augmentation
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
import pandas as pd
import json
import os
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import mediapipe as mp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import h5py


class DanceDataset(Dataset):
    """
    Dataset class for dance movement data
    Handles video frames, motion capture, and pose estimation data
    """
    
    def __init__(self,
                 data_root: str,
                 split: str = 'train',
                 sequence_length: int = 150,  # 5 seconds at 30 fps
                 frame_size: Tuple[int, int] = (224, 224),
                 use_motion_capture: bool = True,
                 use_pose_estimation: bool = True,
                 augment: bool = True,
                 dance_styles: List[str] = None,
                 cache_preprocessed: bool = True):
        """
        Initialize dataset
        
        Args:
            data_root: Root directory containing data
            split: 'train', 'val', or 'test'
            sequence_length: Number of frames per sequence
            frame_size: Size to resize frames to
            use_motion_capture: Whether to use motion capture data
            use_pose_estimation: Whether to use pose estimation
            augment: Whether to apply data augmentation
            dance_styles: List of dance styles to include (None for all)
            cache_preprocessed: Whether to cache preprocessed data
        """
        super(DanceDataset, self).__init__()
        
        self.data_root = Path(data_root)
        self.split = split
        self.sequence_length = sequence_length
        self.frame_size = frame_size
        self.use_motion_capture = use_motion_capture
        self.use_pose_estimation = use_pose_estimation
        self.augment = augment and (split == 'train')
        self.cache_preprocessed = cache_preprocessed
        
        # Dance styles
        if dance_styles is None:
            self.dance_styles = ['ballet', 'hip_hop', 'indian_classical', 'contemporary']
        else:
            self.dance_styles = dance_styles
            
        # Paths
        self.video_path = self.data_root / 'videos' / split
        self.mocap_path = self.data_root / 'mocap' / split
        self.annotations_path = self.data_root / 'annotations' / f'{split}.csv'
        self.pose_path = self.data_root / 'pose_data' / split
        self.cache_path = self.data_root / 'processed' / split
        
        # Create cache directory if needed
        if self.cache_preprocessed:
            self.cache_path.mkdir(parents=True, exist_ok=True)
            
        # Load annotations
        self.annotations = pd.read_csv(self.annotations_path)
        
        # Filter by dance styles if specified
        if self.dance_styles:
            self.annotations = self.annotations[
                self.annotations['style'].isin(self.dance_styles)
            ]
            
        # Initialize pose estimator
        if self.use_pose_estimation:
            self.pose_estimator = PoseEstimator()
            
        # Define data augmentation pipeline
        self.augmentation_pipeline = self._get_augmentation_pipeline()
        
        # Define normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet means
            std=[0.229, 0.224, 0.225]     # ImageNet stds
        )
        
        # Cache for preprocessed data
        self.cache = {}
        
    def _get_augmentation_pipeline(self) -> A.Compose:
        """
        Create augmentation pipeline using albumentations
        """
        if self.augment:
            return A.Compose([
                A.RandomRotate90(p=0.5),
                A.Flip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.ElasticTransform(
                    alpha=120,
                    sigma=120 * 0.05,
                    alpha_affine=120 * 0.03,
                    p=0.3
                ),
                A.OpticalDistortion(p=0.3),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(self.frame_size[0] * 0.1),
                    max_width=int(self.frame_size[1] * 0.1),
                    p=0.3
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset
        
        Args:
            idx: Index of sample
            
        Returns:
            Dictionary containing video frames, poses, mocap data, and labels
        """
        # Check cache first
        if self.cache_preprocessed and idx in self.cache:
            return self.cache[idx]
            
        # Get annotation row
        row = self.annotations.iloc[idx]
        
        # Load video frames
        video_frames = self._load_video_frames(row['video_file'])
        
        # Load motion capture data if available
        mocap_data = None
        if self.use_motion_capture and pd.notna(row.get('mocap_file')):
            mocap_data = self._load_motion_capture(row['mocap_file'])
            
        # Load or extract pose data
        pose_data = None
        if self.use_pose_estimation:
            if pd.notna(row.get('pose_file')):
                pose_data = self._load_pose_data(row['pose_file'])
            else:
                pose_data = self._extract_poses(video_frames)
                
        # Apply augmentations
        if self.augment:
            video_frames = self._augment_frames(video_frames)
            
        # Prepare sample dictionary
        sample = {
            'frames': video_frames,
            'aesthetic_score': torch.tensor(row['aesthetic_score'], dtype=torch.float32),
            'style': row['style'],
            'sequence_length': min(len(video_frames), self.sequence_length)
        }
        
        # Add optional data
        if mocap_data is not None:
            sample['mocap'] = mocap_data
            
        if pose_data is not None:
            sample['poses'] = pose_data
            
        # Add biomechanical score if available
        if 'biomech_score' in row:
            sample['biomechanical_score'] = torch.tensor(
                row['biomech_score'], dtype=torch.float32
            )
            
        # Add fluidity, expressiveness, precision scores if available
        for metric in ['fluidity', 'expressiveness', 'precision']:
            if metric in row:
                sample[metric] = torch.tensor(row[metric], dtype=torch.float32)
                
        # Cache if enabled
        if self.cache_preprocessed:
            self.cache[idx] = sample
            
        return sample
    
    def _load_video_frames(self, video_file: str) -> torch.Tensor:
        """
        Load and preprocess video frames
        
        Args:
            video_file: Path to video file
            
        Returns:
            Tensor of shape (seq_len, 3, H, W)
        """
        video_path = self.video_path / video_file
        
        # Check for cached processed frames
        cache_file = self.cache_path / f"{video_file}.h5"
        if cache_file.exists():
            with h5py.File(cache_file, 'r') as f:
                frames = torch.tensor(f['frames'][:])
                return frames
                
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        
        frames = []
        frame_count = 0
        
        while cap.isOpened() and frame_count < self.sequence_length:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize frame
            frame = cv2.resize(frame, self.frame_size)
            
            frames.append(frame)
            frame_count += 1
            
        cap.release()
        
        # Pad sequence if necessary
        while len(frames) < self.sequence_length:
            if len(frames) > 0:
                frames.append(frames[-1])  # Repeat last frame
            else:
                frames.append(np.zeros((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8))
                
        # Convert to tensor
        frames = np.stack(frames)
        
        # Apply normalization
        normalized_frames = []
        for frame in frames:
            augmented = self.augmentation_pipeline(image=frame)
            normalized_frames.append(augmented['image'])
            
        frames_tensor = torch.stack(normalized_frames)
        
        # Cache processed frames
        if self.cache_preprocessed:
            with h5py.File(cache_file, 'w') as f:
                f.create_dataset('frames', data=frames_tensor.numpy())
                
        return frames_tensor
    
    def _load_motion_capture(self, mocap_file: str) -> torch.Tensor:
        """
        Load motion capture data
        
        Args:
            mocap_file: Path to motion capture file
            
        Returns:
            Tensor of shape (seq_len, num_joints, 3)
        """
        mocap_path = self.mocap_path / mocap_file
        
        # Support different motion capture formats
        if mocap_file.endswith('.c3d'):
            mocap_data = self._load_c3d(mocap_path)
        elif mocap_file.endswith('.bvh'):
            mocap_data = self._load_bvh(mocap_path)
        elif mocap_file.endswith('.fbx'):
            mocap_data = self._load_fbx(mocap_path)
        else:
            # Default to numpy file
            mocap_data = np.load(mocap_path)
            
        # Ensure correct shape and length
        if len(mocap_data) > self.sequence_length:
            mocap_data = mocap_data[:self.sequence_length]
        elif len(mocap_data) < self.sequence_length:
            # Pad with last frame
            padding_length = self.sequence_length - len(mocap_data)
            padding = np.repeat(mocap_data[-1:], padding_length, axis=0)
            mocap_data = np.concatenate([mocap_data, padding], axis=0)
            
        return torch.tensor(mocap_data, dtype=torch.float32)
    
    def _load_c3d(self, file_path: Path) -> np.ndarray:
        """Load C3D motion capture file"""
        # Implementation would use c3d library
        # For now, return placeholder
        return np.random.randn(self.sequence_length, 25, 3)
    
    def _load_bvh(self, file_path: Path) -> np.ndarray:
        """Load BVH motion capture file"""
        # Implementation would use pymo or similar
        # For now, return placeholder
        return np.random.randn(self.sequence_length, 25, 3)
    
    def _load_fbx(self, file_path: Path) -> np.ndarray:
        """Load FBX motion capture file"""
        # Implementation would use FBX SDK
        # For now, return placeholder
        return np.random.randn(self.sequence_length, 25, 3)
    
    def _load_pose_data(self, pose_file: str) -> torch.Tensor:
        """
        Load pre-extracted pose data
        
        Args:
            pose_file: Path to pose data file
            
        Returns:
            Tensor of shape (seq_len, num_keypoints, 3)
        """
        pose_path = self.pose_path / pose_file
        
        if pose_file.endswith('.json'):
            with open(pose_path, 'r') as f:
                pose_data = json.load(f)
                
            # Convert to numpy array
            poses = []
            for frame_data in pose_data['frames']:
                keypoints = np.array(frame_data['keypoints'])
                poses.append(keypoints)
                
            pose_array = np.stack(poses)
        else:
            # Load numpy file
            pose_array = np.load(pose_path)
            
        # Ensure correct length
        if len(pose_array) > self.sequence_length:
            pose_array = pose_array[:self.sequence_length]
        elif len(pose_array) < self.sequence_length:
            padding_length = self.sequence_length - len(pose_array)
            padding = np.repeat(pose_array[-1:], padding_length, axis=0)
            pose_array = np.concatenate([pose_array, padding], axis=0)
            
        return torch.tensor(pose_array, dtype=torch.float32)
    
    def _extract_poses(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Extract poses from frames using pose estimator
        
        Args:
            frames: Video frames tensor
            
        Returns:
            Pose keypoints tensor
        """
        poses = []
        
        for frame in frames:
            # Convert tensor to numpy
            frame_np = frame.permute(1, 2, 0).numpy()
            
            # Denormalize
            frame_np = (frame_np * np.array([0.229, 0.224, 0.225]) + 
                       np.array([0.485, 0.456, 0.406]))
            frame_np = (frame_np * 255).astype(np.uint8)
            
            # Extract pose
            keypoints = self.pose_estimator.extract_pose(frame_np)
            poses.append(keypoints)
            
        return torch.tensor(np.stack(poses), dtype=torch.float32)
    
    def _augment_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal and spatial augmentations to frames
        
        Args:
            frames: Input frames tensor
            
        Returns:
            Augmented frames tensor
        """
        # Random temporal crop
        if self.augment and len(frames) > self.sequence_length:
            start_idx = np.random.randint(0, len(frames) - self.sequence_length)
            frames = frames[start_idx:start_idx + self.sequence_length]
            
        # Random speed augmentation (temporal)
        if self.augment and np.random.random() < 0.5:
            speed_factor = np.random.uniform(0.8, 1.2)
            indices = np.linspace(0, len(frames) - 1, 
                                int(len(frames) / speed_factor)).astype(int)
            frames = frames[indices]
            
        return frames


class PoseEstimator:
    """
    Pose estimation using MediaPipe or OpenPose
    """
    
    def __init__(self, use_mediapipe: bool = True):
        """
        Initialize pose estimator
        
        Args:
            use_mediapipe: If True, use MediaPipe; otherwise use OpenPose
        """
        self.use_mediapipe = use_mediapipe
        
        if use_mediapipe:
            # Initialize MediaPipe
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.num_keypoints = 33
        else:
            # Initialize OpenPose
            # This would require OpenPose installation
            self.num_keypoints = 25
            
    def extract_pose(self, image: np.ndarray) -> np.ndarray:
        """
        Extract pose keypoints from image
        
        Args:
            image: Input image (H, W, 3)
            
        Returns:
            Keypoints array of shape (num_keypoints, 3)
        """
        if self.use_mediapipe:
            # Process with MediaPipe
            results = self.pose.process(image)
            
            if results.pose_landmarks:
                keypoints = []
                for landmark in results.pose_landmarks.landmark:
                    keypoints.append([landmark.x, landmark.y, landmark.visibility])
                return np.array(keypoints)
            else:
                # Return zeros if no pose detected
                return np.zeros((self.num_keypoints, 3))
        else:
            # OpenPose processing would go here
            # For now, return placeholder
            return np.random.randn(self.num_keypoints, 3)


class DanceDataCollator:
    """
    Custom collator for batching dance sequences
    """
    
    def __init__(self, pad_to_max: bool = True):
        self.pad_to_max = pad_to_max
        
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of samples
        
        Args:
            batch: List of sample dictionaries
            
        Returns:
            Batched dictionary
        """
        # Find max sequence length in batch
        max_seq_len = max(sample['sequence_length'] for sample in batch)
        
        # Initialize batched tensors
        batch_size = len(batch)
        first_sample = batch[0]
        
        # Get frame dimensions
        _, c, h, w = first_sample['frames'].shape
        
        # Initialize batched frames
        if self.pad_to_max:
            batched_frames = torch.zeros(batch_size, max_seq_len, c, h, w)
        else:
            batched_frames = []
            
        # Collect other data
        aesthetic_scores = []
        styles = []
        sequence_lengths = []
        
        # Optional data
        has_mocap = 'mocap' in first_sample
        has_poses = 'poses' in first_sample
        
        if has_mocap:
            _, num_joints, joint_dim = first_sample['mocap'].shape
            batched_mocap = torch.zeros(batch_size, max_seq_len, num_joints, joint_dim)
            
        if has_poses:
            _, num_keypoints, keypoint_dim = first_sample['poses'].shape
            batched_poses = torch.zeros(batch_size, max_seq_len, num_keypoints, keypoint_dim)
            
        # Process each sample
        for i, sample in enumerate(batch):
            seq_len = sample['sequence_length']
            sequence_lengths.append(seq_len)
            
            # Add frames
            if self.pad_to_max:
                batched_frames[i, :seq_len] = sample['frames'][:seq_len]
            else:
                batched_frames.append(sample['frames'][:seq_len])
                
            # Add scores and metadata
            aesthetic_scores.append(sample['aesthetic_score'])
            styles.append(sample['style'])
            
            # Add optional data
            if has_mocap:
                batched_mocap[i, :seq_len] = sample['mocap'][:seq_len]
                
            if has_poses:
                batched_poses[i, :seq_len] = sample['poses'][:seq_len]
                
        # Create output dictionary
        collated = {
            'frames': batched_frames if self.pad_to_max else torch.stack(batched_frames),
            'aesthetic_scores': torch.stack(aesthetic_scores),
            'styles': styles,
            'sequence_lengths': torch.tensor(sequence_lengths)
        }
        
        if has_mocap:
            collated['mocap'] = batched_mocap
            
        if has_poses:
            collated['poses'] = batched_poses
            
        return collated


def create_data_loaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training, validation, and testing
    
    Args:
        config: Configuration dictionary
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = DanceDataset(
        data_root=config['data_root'],
        split='train',
        sequence_length=config['sequence_length'],
        frame_size=tuple(config['frame_size']),
        use_motion_capture=config.get('use_motion_capture', True),
        use_pose_estimation=config.get('use_pose_estimation', True),
        augment=True,
        dance_styles=config.get('dance_styles'),
        cache_preprocessed=config.get('cache_preprocessed', True)
    )
    
    val_dataset = DanceDataset(
        data_root=config['data_root'],
        split='val',
        sequence_length=config['sequence_length'],
        frame_size=tuple(config['frame_size']),
        use_motion_capture=config.get('use_motion_capture', True),
        use_pose_estimation=config.get('use_pose_estimation', True),
        augment=False,
        dance_styles=config.get('dance_styles'),
        cache_preprocessed=config.get('cache_preprocessed', True)
    )
    
    test_dataset = DanceDataset(
        data_root=config['data_root'],
        split='test',
        sequence_length=config['sequence_length'],
        frame_size=tuple(config['frame_size']),
        use_motion_capture=config.get('use_motion_capture', True),
        use_pose_estimation=config.get('use_pose_estimation', True),
        augment=False,
        dance_styles=config.get('dance_styles'),
        cache_preprocessed=config.get('cache_preprocessed', True)
    )
    
    # Create collator
    collator = DanceDataCollator(pad_to_max=True)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        collate_fn=collator,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        collate_fn=collator,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        collate_fn=collator,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
