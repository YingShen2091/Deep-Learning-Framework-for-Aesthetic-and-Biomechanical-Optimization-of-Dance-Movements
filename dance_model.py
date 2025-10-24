"""
Dance Movement Aesthetic Evaluation and Optimization Model
Implementation of CNN-LSTM architecture with gated attention mechanism
Based on the paper: Deep Learning Framework for Aesthetic and Biomechanical Optimization of Dance Movements
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, List, Dict, Optional
import numpy as np


class SpatialFeatureExtractor(nn.Module):
    """
    CNN-based spatial feature extractor using ResNet-50 backbone
    Extracts multi-scale features from dance frames
    """
    
    def __init__(self, pretrained: bool = True, feature_dim: int = 512):
        super(SpatialFeatureExtractor, self).__init__()
        
        # Load pre-trained ResNet-50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Multi-scale feature extraction layers
        self.conv1_features = None
        self.conv2_features = None
        self.conv3_features = None
        
        # Feature projection layers for different scales
        self.proj_conv1 = nn.Linear(256 * 56 * 56, feature_dim)
        self.proj_conv2 = nn.Linear(512 * 28 * 28, feature_dim)
        self.proj_conv3 = nn.Linear(2048 * 7 * 7, feature_dim)
        
        # Final projection layer
        self.final_projection = nn.Linear(feature_dim * 3, feature_dim)
        
        # Batch normalization
        self.bn = nn.BatchNorm1d(feature_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through CNN
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
            
        Returns:
            Spatial features and multi-scale feature maps
        """
        batch_size = x.size(0)
        
        # Extract features at different scales
        features = {}
        
        # Pass through ResNet layers
        x = self.backbone[0](x)  # Conv1
        x = self.backbone[1](x)
        x = self.backbone[2](x)
        x = self.backbone[3](x)
        
        x = self.backbone[4](x)  # Layer1
        features['layer1'] = x.clone()
        f1_flat = x.view(batch_size, -1)
        f1_proj = self.proj_conv1(f1_flat)
        
        x = self.backbone[5](x)  # Layer2
        features['layer2'] = x.clone()
        f2_flat = x.view(batch_size, -1)
        f2_proj = self.proj_conv2(f2_flat)
        
        x = self.backbone[6](x)  # Layer3
        x = self.backbone[7](x)  # Layer4
        features['layer3'] = x.clone()
        f3_flat = x.view(batch_size, -1)
        f3_proj = self.proj_conv3(f3_flat)
        
        # Concatenate multi-scale features
        multi_scale_features = torch.cat([f1_proj, f2_proj, f3_proj], dim=-1)
        
        # Final projection
        spatial_features = self.final_projection(multi_scale_features)
        spatial_features = self.bn(spatial_features)
        spatial_features = F.relu(spatial_features)
        spatial_features = self.dropout(spatial_features)
        
        return spatial_features, features


class TemporalModelingLSTM(nn.Module):
    """
    Bidirectional LSTM for temporal dependency modeling in dance sequences
    """
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, 
                 num_layers: int = 2, dropout: float = 0.3):
        super(TemporalModelingLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
        # Temporal attention mechanism
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=8,
            dropout=dropout
        )
        
    def forward(self, x: torch.Tensor, sequence_lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through LSTM
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            sequence_lengths: Actual lengths of sequences for masking
            
        Returns:
            Temporal features and hidden states
        """
        batch_size, seq_len, _ = x.size()
        
        # Pack sequences if lengths provided
        if sequence_lengths is not None:
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, sequence_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, (h_n, c_n) = self.lstm(x_packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True
            )
        else:
            lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply layer normalization
        lstm_out = self.layer_norm(lstm_out)
        
        # Apply temporal attention
        attn_out, attn_weights = self.temporal_attention(
            lstm_out, lstm_out, lstm_out
        )
        
        # Combine LSTM output with attention
        temporal_features = lstm_out + attn_out
        
        # Get final hidden state (concatenate forward and backward)
        h_n_combined = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        
        return temporal_features, h_n_combined


class GatedFeatureFusion(nn.Module):
    """
    Gated attention mechanism for fusing spatial and temporal features
    As described in Equation 6-9 of the paper
    """
    
    def __init__(self, spatial_dim: int = 512, temporal_dim: int = 512):
        super(GatedFeatureFusion, self).__init__()
        
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        
        # Gate computation
        self.gate_layer = nn.Linear(spatial_dim + temporal_dim, spatial_dim + temporal_dim)
        
        # Feature transformation layers
        self.spatial_transform = nn.Linear(spatial_dim, spatial_dim)
        self.temporal_transform = nn.Linear(temporal_dim, temporal_dim)
        
        # Batch normalization
        self.bn_spatial = nn.BatchNorm1d(spatial_dim)
        self.bn_temporal = nn.BatchNorm1d(temporal_dim)
        
    def forward(self, spatial_features: torch.Tensor, temporal_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse spatial and temporal features using gated mechanism
        
        Args:
            spatial_features: Spatial features from CNN
            temporal_features: Temporal features from LSTM
            
        Returns:
            Fused features
        """
        # Concatenate features for gate computation
        concat_features = torch.cat([spatial_features, temporal_features], dim=-1)
        
        # Compute gate values (Equation 6)
        gate = torch.sigmoid(self.gate_layer(concat_features))
        
        # Split gate for spatial and temporal
        gate_spatial = gate[:, :self.spatial_dim]
        gate_temporal = gate[:, self.spatial_dim:]
        
        # Transform features (Equations 7-8)
        spatial_gated = gate_spatial * torch.tanh(self.spatial_transform(spatial_features))
        temporal_gated = (1 - gate_temporal) * torch.tanh(self.temporal_transform(temporal_features))
        
        # Apply batch normalization
        if spatial_features.dim() == 2:
            spatial_gated = self.bn_spatial(spatial_gated)
            temporal_gated = self.bn_temporal(temporal_gated)
        
        # Combine gated features (Equation 9)
        fused_features = torch.cat([spatial_gated, temporal_gated], dim=-1)
        
        return fused_features


class ClassActivationMapping(nn.Module):
    """
    Class Activation Maps for interpretability
    Highlights key body parts contributing to aesthetic evaluation
    """
    
    def __init__(self, feature_dim: int = 2048, num_classes: int = 1):
        super(ClassActivationMapping, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Final classification layer
        self.fc = nn.Linear(feature_dim, num_classes)
        
        # Store feature maps for CAM generation
        self.feature_maps = None
        self.fc_weights = None
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate CAMs and predictions
        
        Args:
            x: Feature maps from CNN backbone
            
        Returns:
            Predictions and CAM heatmaps
        """
        batch_size, channels, h, w = x.size()
        
        # Store feature maps for CAM generation
        self.feature_maps = x.clone()
        
        # Global average pooling
        gap_features = self.gap(x).view(batch_size, -1)
        
        # Get predictions
        predictions = self.fc(gap_features)
        
        # Generate CAMs
        self.fc_weights = self.fc.weight.data
        cams = self.generate_cams()
        
        return predictions, cams
    
    def generate_cams(self) -> torch.Tensor:
        """
        Generate Class Activation Maps
        
        Returns:
            CAM heatmaps
        """
        if self.feature_maps is None or self.fc_weights is None:
            return None
            
        batch_size, channels, h, w = self.feature_maps.size()
        
        # Initialize CAMs
        cams = torch.zeros(batch_size, self.num_classes, h, w).to(self.feature_maps.device)
        
        # Generate CAM for each class
        for cls in range(self.num_classes):
            # Get weights for this class
            weights = self.fc_weights[cls].unsqueeze(0).unsqueeze(2).unsqueeze(3)
            
            # Weighted combination of feature maps
            cam = torch.sum(weights * self.feature_maps, dim=1)
            
            # Apply ReLU (only positive contributions)
            cam = F.relu(cam)
            
            # Normalize CAM
            cam_min = cam.view(batch_size, -1).min(dim=1)[0].view(batch_size, 1, 1)
            cam_max = cam.view(batch_size, -1).max(dim=1)[0].view(batch_size, 1, 1)
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
            
            cams[:, cls] = cam
            
        return cams


class DanceAestheticModel(nn.Module):
    """
    Complete dance aesthetic evaluation model
    Combines CNN, LSTM, CAM, and feature fusion
    """
    
    def __init__(self, 
                 spatial_feature_dim: int = 512,
                 temporal_hidden_dim: int = 256,
                 num_lstm_layers: int = 2,
                 dropout: float = 0.3,
                 num_aesthetic_classes: int = 10,
                 enable_cam: bool = True):
        super(DanceAestheticModel, self).__init__()
        
        # Spatial feature extractor (CNN)
        self.spatial_extractor = SpatialFeatureExtractor(
            pretrained=True,
            feature_dim=spatial_feature_dim
        )
        
        # Temporal modeling (LSTM)
        self.temporal_model = TemporalModelingLSTM(
            input_dim=spatial_feature_dim,
            hidden_dim=temporal_hidden_dim,
            num_layers=num_lstm_layers,
            dropout=dropout
        )
        
        # Feature fusion
        self.feature_fusion = GatedFeatureFusion(
            spatial_dim=spatial_feature_dim,
            temporal_dim=temporal_hidden_dim * 2  # Bidirectional
        )
        
        # CAM module
        self.enable_cam = enable_cam
        if enable_cam:
            self.cam_module = ClassActivationMapping(
                feature_dim=2048,
                num_classes=1  # Aesthetic score
            )
        
        # Final aesthetic score prediction
        fusion_dim = spatial_feature_dim + temporal_hidden_dim * 2
        self.aesthetic_head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)  # Single aesthetic score
        )
        
        # Biomechanical efficiency head
        self.biomech_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)  # Biomechanical score
        )
        
        # Joint angle prediction for optimization
        self.joint_angle_head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 25 * 3)  # 25 joints * 3 angles (x, y, z)
        )
        
    def forward(self, 
                video_frames: torch.Tensor,
                sequence_lengths: Optional[torch.Tensor] = None,
                return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through complete model
        
        Args:
            video_frames: Input video frames (batch_size, seq_len, 3, H, W)
            sequence_lengths: Actual lengths of sequences
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary containing predictions and features
        """
        batch_size, seq_len, channels, height, width = video_frames.size()
        
        # Extract spatial features for each frame
        spatial_features_seq = []
        all_feature_maps = []
        
        for t in range(seq_len):
            frame = video_frames[:, t]
            spatial_feat, feat_maps = self.spatial_extractor(frame)
            spatial_features_seq.append(spatial_feat)
            all_feature_maps.append(feat_maps)
        
        # Stack spatial features
        spatial_features = torch.stack(spatial_features_seq, dim=1)
        
        # Temporal modeling
        temporal_features, final_hidden = self.temporal_model(
            spatial_features, sequence_lengths
        )
        
        # Get temporal feature at last timestep
        if sequence_lengths is not None:
            # Get features at actual last timestep for each sequence
            last_temporal = []
            for i, length in enumerate(sequence_lengths):
                last_temporal.append(temporal_features[i, length-1])
            last_temporal_features = torch.stack(last_temporal)
        else:
            last_temporal_features = temporal_features[:, -1]
        
        # Get last spatial features
        if sequence_lengths is not None:
            last_spatial = []
            for i, length in enumerate(sequence_lengths):
                last_spatial.append(spatial_features[i, length-1])
            last_spatial_features = torch.stack(last_spatial)
        else:
            last_spatial_features = spatial_features[:, -1]
        
        # Feature fusion
        fused_features = self.feature_fusion(
            last_spatial_features,
            last_temporal_features
        )
        
        # Predictions
        aesthetic_score = self.aesthetic_head(fused_features)
        biomech_score = self.biomech_head(fused_features)
        joint_angles = self.joint_angle_head(fused_features)
        
        # Reshape joint angles
        joint_angles = joint_angles.view(batch_size, 25, 3)
        
        # Generate CAMs if enabled
        cams = None
        if self.enable_cam and all_feature_maps:
            # Use last frame's feature maps for CAM
            last_feat_maps = all_feature_maps[-1]['layer3']
            _, cams = self.cam_module(last_feat_maps)
        
        # Prepare output dictionary
        outputs = {
            'aesthetic_score': aesthetic_score,
            'biomechanical_score': biomech_score,
            'joint_angles': joint_angles,
            'cams': cams
        }
        
        if return_features:
            outputs['spatial_features'] = spatial_features
            outputs['temporal_features'] = temporal_features
            outputs['fused_features'] = fused_features
            
        return outputs


class MultiHeadAttentionComparison(nn.Module):
    """
    Implementation of different attention mechanisms for comparison
    As shown in Figure 4 of the paper
    """
    
    def __init__(self, feature_dim: int = 512, num_heads: int = 8):
        super(MultiHeadAttentionComparison, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        # Different attention mechanisms
        
        # 1. Convolution-integrated multi-head attention
        self.conv_attention = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.ReLU(),
            nn.MultiheadAttention(feature_dim, num_heads)
        )
        
        # 2. Context-mapped transformer attention
        self.context_attention = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4
        )
        
        # 3. BERT-style multi-head self-attention
        self.bert_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.1
        )
        
        # 4. Transformer-based residual learning attention
        self.residual_attention = ResidualAttentionBlock(
            feature_dim=feature_dim,
            num_heads=num_heads
        )
        
        # 5. MFMHLA (Multi-Frame Multi-Head Learning Attention) - Paper's proposed method
        self.mfmhla = MultiFrameMultiHeadAttention(
            feature_dim=feature_dim,
            num_heads=num_heads
        )


class ResidualAttentionBlock(nn.Module):
    """
    Residual attention block with transformer-based multi-head attention
    """
    
    def __init__(self, feature_dim: int = 512, num_heads: int = 8):
        super(ResidualAttentionBlock, self).__init__()
        
        self.attention = nn.MultiheadAttention(feature_dim, num_heads)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(),
            nn.Linear(feature_dim * 4, feature_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x


class MultiFrameMultiHeadAttention(nn.Module):
    """
    Multi-Frame Multi-Head Learning Attention (MFMHLA)
    Proposed method that integrates spatial and sequential information
    """
    
    def __init__(self, feature_dim: int = 512, num_heads: int = 8, num_frames: int = 5):
        super(MultiFrameMultiHeadAttention, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.num_frames = num_frames
        
        # Multi-scale attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(feature_dim, num_heads)
            for _ in range(3)  # 3 scales
        ])
        
        # Frame-wise attention
        self.frame_attention = nn.MultiheadAttention(
            feature_dim * num_frames,
            num_heads
        )
        
        # Resolution-specific projections
        self.resolution_projections = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim)
            for _ in range(3)
        ])
        
        # Final fusion layer
        self.fusion = nn.Linear(feature_dim * 3, feature_dim)
        
    def forward(self, x: torch.Tensor, multi_scale_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through MFMHLA
        
        Args:
            x: Input features
            multi_scale_features: Features at different resolutions
            
        Returns:
            Attention-weighted features
        """
        attended_features = []
        
        # Apply attention at each scale
        for i, (attn_layer, feat, proj) in enumerate(
            zip(self.attention_layers, multi_scale_features, self.resolution_projections)
        ):
            # Project features
            feat_proj = proj(feat)
            
            # Apply attention
            attn_out, _ = attn_layer(feat_proj, feat_proj, feat_proj)
            attended_features.append(attn_out)
        
        # Concatenate multi-scale attended features
        multi_scale_attended = torch.cat(attended_features, dim=-1)
        
        # Final fusion
        output = self.fusion(multi_scale_attended)
        
        return output
