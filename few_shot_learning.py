"""
Few-Shot Learning Module for Dance Movement Analysis
Implements meta-learning and transfer learning for new dance styles
Based on the paper's few-shot learning approach
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import copy
from torch.utils.data import DataLoader, Dataset


class FewShotConfig:
    """Configuration for few-shot learning"""
    
    # Meta-learning parameters
    num_inner_steps: int = 5  # Gradient steps in inner loop
    inner_lr: float = 0.01  # Learning rate for inner loop
    outer_lr: float = 0.001  # Learning rate for meta-update
    
    # Few-shot task setup
    n_way: int = 5  # Number of classes per task
    k_shot: int = 5  # Number of examples per class in support set
    q_query: int = 15  # Number of examples per class in query set
    
    # Training parameters
    meta_batch_size: int = 4  # Number of tasks per meta-update
    num_meta_epochs: int = 100
    
    # Model parameters
    feature_dim: int = 2048
    hidden_dim: int = 512
    
    # Prototypical networks parameters
    distance_metric: str = 'euclidean'  # or 'cosine'
    temperature: float = 1.0  # Temperature for distance scaling


class MAML(nn.Module):
    """
    Model-Agnostic Meta-Learning (MAML) for dance movement analysis
    Enables rapid adaptation to new dance styles
    """
    
    def __init__(self, base_model: nn.Module, config: FewShotConfig):
        super(MAML, self).__init__()
        
        self.base_model = base_model
        self.config = config
        
        # Meta-learnable parameters
        self.meta_parameters = OrderedDict()
        for name, param in base_model.named_parameters():
            if param.requires_grad:
                self.meta_parameters[name] = param.clone()
        
        # Meta optimizer
        self.meta_optimizer = optim.Adam(
            self.meta_parameters.values(),
            lr=config.outer_lr
        )
        
    def forward(self, x: torch.Tensor, params: Optional[OrderedDict] = None) -> torch.Tensor:
        """
        Forward pass with optional custom parameters
        """
        if params is None:
            return self.base_model(x)
        else:
            # Use functional forward pass with custom parameters
            return self._functional_forward(x, params)
    
    def _functional_forward(self, x: torch.Tensor, params: OrderedDict) -> torch.Tensor:
        """
        Functional forward pass using provided parameters
        Required for MAML gradient computation
        """
        # This is a simplified version - in practice, need to handle all layers
        # Create a copy of the model with new parameters
        model_copy = copy.deepcopy(self.base_model)
        
        # Replace parameters
        for name, param in params.items():
            # Navigate to the parameter and replace it
            parts = name.split('.')
            module = model_copy
            for part in parts[:-1]:
                module = getattr(module, part)
            setattr(module, parts[-1], param)
        
        return model_copy(x)
    
    def inner_loop(self, support_data: torch.Tensor, support_labels: torch.Tensor,
                   num_steps: int = None) -> OrderedDict:
        """
        Inner loop adaptation on support set
        """
        if num_steps is None:
            num_steps = self.config.num_inner_steps
        
        # Clone parameters for task-specific adaptation
        task_params = OrderedDict()
        for name, param in self.meta_parameters.items():
            task_params[name] = param.clone()
        
        # Inner loop optimization
        for step in range(num_steps):
            # Forward pass
            predictions = self._functional_forward(support_data, task_params)
            
            # Compute loss
            loss = F.mse_loss(predictions.squeeze(), support_labels)
            
            # Compute gradients
            grads = torch.autograd.grad(
                loss,
                task_params.values(),
                create_graph=True
            )
            
            # Update parameters
            task_params = OrderedDict()
            for (name, param), grad in zip(self.meta_parameters.items(), grads):
                task_params[name] = param - self.config.inner_lr * grad
        
        return task_params
    
    def meta_update(self, tasks: List[Tuple[torch.Tensor, torch.Tensor,
                                           torch.Tensor, torch.Tensor]]):
        """
        Meta-update using multiple tasks
        
        Args:
            tasks: List of (support_data, support_labels, query_data, query_labels)
        """
        meta_loss = 0
        
        for support_data, support_labels, query_data, query_labels in tasks:
            # Inner loop adaptation
            adapted_params = self.inner_loop(support_data, support_labels)
            
            # Evaluate on query set
            query_predictions = self._functional_forward(query_data, adapted_params)
            task_loss = F.mse_loss(query_predictions.squeeze(), query_labels)
            
            meta_loss += task_loss
        
        # Meta-update
        meta_loss /= len(tasks)
        
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        # Update meta parameters
        for name, param in self.base_model.named_parameters():
            if name in self.meta_parameters:
                self.meta_parameters[name] = param.clone()
        
        return meta_loss.item()


class PrototypicalNetwork(nn.Module):
    """
    Prototypical Networks for few-shot dance style classification
    Creates prototypes from support set and classifies query examples
    """
    
    def __init__(self, feature_extractor: nn.Module, config: FewShotConfig):
        super(PrototypicalNetwork, self).__init__()
        
        self.feature_extractor = feature_extractor
        self.config = config
        
        # Additional projection layer for prototype space
        self.projection = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 128)  # Prototype dimension
        )
    
    def compute_prototypes(self, support_features: torch.Tensor,
                          support_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute class prototypes from support set
        
        Args:
            support_features: Features of shape (n_way * k_shot, feature_dim)
            support_labels: Labels of shape (n_way * k_shot,)
        
        Returns:
            Prototypes of shape (n_way, prototype_dim)
        """
        n_way = self.config.n_way
        k_shot = self.config.k_shot
        
        # Reshape features
        support_features = support_features.reshape(n_way, k_shot, -1)
        
        # Compute mean prototype for each class
        prototypes = support_features.mean(dim=1)  # (n_way, feature_dim)
        
        # Project to prototype space
        prototypes = self.projection(prototypes)
        
        return prototypes
    
    def compute_distances(self, query_features: torch.Tensor,
                         prototypes: torch.Tensor) -> torch.Tensor:
        """
        Compute distances between query features and prototypes
        
        Args:
            query_features: Features of shape (n_query, feature_dim)
            prototypes: Prototypes of shape (n_way, prototype_dim)
        
        Returns:
            Distances of shape (n_query, n_way)
        """
        # Project query features
        query_features = self.projection(query_features)
        
        if self.config.distance_metric == 'euclidean':
            # Euclidean distance
            distances = torch.cdist(query_features, prototypes)
        elif self.config.distance_metric == 'cosine':
            # Cosine similarity (converted to distance)
            query_norm = F.normalize(query_features, dim=-1)
            proto_norm = F.normalize(prototypes, dim=-1)
            similarities = torch.mm(query_norm, proto_norm.t())
            distances = 1 - similarities
        else:
            raise ValueError(f"Unknown distance metric: {self.config.distance_metric}")
        
        return distances
    
    def forward(self, support_data: torch.Tensor, support_labels: torch.Tensor,
                query_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for prototypical networks
        
        Args:
            support_data: Support set data
            support_labels: Support set labels
            query_data: Query set data
        
        Returns:
            Predicted probabilities for query set
        """
        # Extract features
        support_features = self.feature_extractor(support_data)
        query_features = self.feature_extractor(query_data)
        
        # Compute prototypes
        prototypes = self.compute_prototypes(support_features, support_labels)
        
        # Compute distances
        distances = self.compute_distances(query_features, prototypes)
        
        # Convert to probabilities (negative distance as logits)
        logits = -distances / self.config.temperature
        probabilities = F.softmax(logits, dim=-1)
        
        return probabilities


class RelationNetwork(nn.Module):
    """
    Relation Network for few-shot learning
    Learns a similarity metric between support and query examples
    """
    
    def __init__(self, feature_extractor: nn.Module, 
                 feature_dim: int = 2048,
                 relation_dim: int = 8):
        super(RelationNetwork, self).__init__()
        
        self.feature_extractor = feature_extractor
        
        # Feature embedding network
        self.feature_embedding = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # Relation module (learns similarity)
        self.relation_module = nn.Sequential(
            nn.Linear(256 * 2, 512),  # Concatenated features
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, relation_dim),
            nn.ReLU(),
            nn.Linear(relation_dim, 1),
            nn.Sigmoid()  # Output similarity score [0, 1]
        )
    
    def forward(self, support_data: torch.Tensor, support_labels: torch.Tensor,
                query_data: torch.Tensor) -> torch.Tensor:
        """
        Compute relation scores between support and query examples
        """
        # Extract and embed features
        support_features = self.feature_embedding(
            self.feature_extractor(support_data)
        )
        query_features = self.feature_embedding(
            self.feature_extractor(query_data)
        )
        
        n_way = len(torch.unique(support_labels))
        k_shot = len(support_labels) // n_way
        n_query = len(query_data)
        
        # Compute support prototypes (mean of each class)
        support_features = support_features.reshape(n_way, k_shot, -1)
        support_prototypes = support_features.mean(dim=1)  # (n_way, feature_dim)
        
        # Expand for pairwise comparison
        support_prototypes = support_prototypes.unsqueeze(0).expand(n_query, -1, -1)
        query_features = query_features.unsqueeze(1).expand(-1, n_way, -1)
        
        # Concatenate pairs
        relation_pairs = torch.cat([support_prototypes, query_features], dim=-1)
        
        # Compute relation scores
        relation_scores = self.relation_module(relation_pairs).squeeze(-1)
        
        return relation_scores


class TransferLearning(nn.Module):
    """
    Transfer learning module for adapting pre-trained models to new dance styles
    """
    
    def __init__(self, pretrained_model: nn.Module, num_classes: int = 10):
        super(TransferLearning, self).__init__()
        
        self.pretrained_model = pretrained_model
        
        # Freeze early layers (keep learned features)
        for param in list(pretrained_model.parameters())[:-10]:
            param.requires_grad = False
        
        # Add task-specific layers
        self.task_specific = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Style-specific adaptation modules
        self.style_adapters = nn.ModuleDict()
        
    def add_style_adapter(self, style_name: str, adapter_dim: int = 64):
        """
        Add a style-specific adapter module
        """
        adapter = nn.Sequential(
            nn.Linear(2048, adapter_dim),
            nn.ReLU(),
            nn.Linear(adapter_dim, 2048),
            nn.Sigmoid()
        )
        self.style_adapters[style_name] = adapter
    
    def forward(self, x: torch.Tensor, style: Optional[str] = None) -> torch.Tensor:
        """
        Forward pass with optional style-specific adaptation
        """
        # Get base features
        features = self.pretrained_model(x)
        
        # Apply style-specific adaptation if available
        if style is not None and style in self.style_adapters:
            adapter = self.style_adapters[style]
            adaptation = adapter(features)
            features = features * adaptation  # Element-wise modulation
        
        # Task-specific classification
        output = self.task_specific(features)
        
        return output


class MetaLearningTrainer:
    """
    Trainer for few-shot learning methods
    """
    
    def __init__(self, model: nn.Module, config: FewShotConfig,
                 device: torch.device = torch.device('cuda')):
        self.model = model
        self.config = config
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Optimizer for non-MAML methods
        if not isinstance(model, MAML):
            self.optimizer = optim.Adam(model.parameters(), lr=config.outer_lr)
    
    def train_episode(self, support_data: torch.Tensor,
                     support_labels: torch.Tensor,
                     query_data: torch.Tensor,
                     query_labels: torch.Tensor) -> float:
        """
        Train on a single few-shot episode
        """
        self.model.train()
        
        if isinstance(self.model, MAML):
            # MAML training
            tasks = [(support_data, support_labels, query_data, query_labels)]
            loss = self.model.meta_update(tasks)
        else:
            # Standard few-shot training
            predictions = self.model(support_data, support_labels, query_data)
            
            # Compute loss
            if predictions.dim() == 2:  # Classification
                loss = F.cross_entropy(predictions, query_labels)
            else:  # Regression
                loss = F.mse_loss(predictions.squeeze(), query_labels)
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            loss = loss.item()
        
        return loss
    
    def evaluate(self, support_data: torch.Tensor,
                support_labels: torch.Tensor,
                query_data: torch.Tensor,
                query_labels: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate on a few-shot task
        """
        self.model.eval()
        
        with torch.no_grad():
            if isinstance(self.model, MAML):
                # MAML evaluation
                adapted_params = self.model.inner_loop(support_data, support_labels)
                predictions = self.model._functional_forward(query_data, adapted_params)
            else:
                # Standard evaluation
                predictions = self.model(support_data, support_labels, query_data)
            
            # Compute metrics
            if predictions.dim() == 2:  # Classification
                loss = F.cross_entropy(predictions, query_labels).item()
                accuracy = (predictions.argmax(dim=1) == query_labels).float().mean().item()
                metrics = {'loss': loss, 'accuracy': accuracy}
            else:  # Regression
                loss = F.mse_loss(predictions.squeeze(), query_labels).item()
                correlation = torch.corrcoef(
                    torch.stack([predictions.squeeze(), query_labels])
                )[0, 1].item()
                metrics = {'loss': loss, 'correlation': correlation}
        
        return metrics


class DanceStyleDataset(Dataset):
    """
    Dataset for few-shot learning on dance styles
    """
    
    def __init__(self, data_path: str, styles: List[str],
                 examples_per_style: int = 100):
        self.data_path = data_path
        self.styles = styles
        self.examples_per_style = examples_per_style
        
        # Load data (placeholder - implement actual loading)
        self.data = {}
        self.labels = {}
        
        for idx, style in enumerate(styles):
            # Load dance sequences for this style
            style_data = self._load_style_data(style)
            self.data[style] = style_data
            self.labels[style] = idx
    
    def _load_style_data(self, style: str) -> torch.Tensor:
        """
        Load dance data for a specific style
        """
        # Placeholder - implement actual data loading
        # Should return tensor of shape (num_examples, sequence_length, features)
        return torch.randn(self.examples_per_style, 30, 2048)
    
    def sample_episode(self, n_way: int, k_shot: int, 
                       q_query: int) -> Tuple[torch.Tensor, torch.Tensor,
                                            torch.Tensor, torch.Tensor]:
        """
        Sample a few-shot episode
        """
        # Sample n_way styles
        selected_styles = np.random.choice(self.styles, n_way, replace=False)
        
        support_data = []
        support_labels = []
        query_data = []
        query_labels = []
        
        for idx, style in enumerate(selected_styles):
            style_data = self.data[style]
            
            # Sample k_shot + q_query examples
            indices = np.random.choice(
                len(style_data),
                k_shot + q_query,
                replace=False
            )
            
            # Split into support and query
            support_indices = indices[:k_shot]
            query_indices = indices[k_shot:]
            
            support_data.append(style_data[support_indices])
            support_labels.extend([idx] * k_shot)
            
            query_data.append(style_data[query_indices])
            query_labels.extend([idx] * q_query)
        
        # Convert to tensors
        support_data = torch.cat(support_data, dim=0)
        support_labels = torch.tensor(support_labels)
        query_data = torch.cat(query_data, dim=0)
        query_labels = torch.tensor(query_labels)
        
        return support_data, support_labels, query_data, query_labels
    
    def __len__(self):
        return len(self.styles) * self.examples_per_style
    
    def __getitem__(self, idx):
        # For standard dataset access
        style_idx = idx // self.examples_per_style
        example_idx = idx % self.examples_per_style
        
        style = self.styles[style_idx]
        data = self.data[style][example_idx]
        label = self.labels[style]
        
        return data, label


class AdaptiveLearningRate:
    """
    Adaptive learning rate for meta-learning
    """
    
    def __init__(self, initial_lr: float = 0.01,
                 min_lr: float = 0.0001,
                 patience: int = 5):
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.patience = patience
        
        self.current_lr = initial_lr
        self.best_loss = float('inf')
        self.patience_counter = 0
    
    def update(self, loss: float) -> float:
        """
        Update learning rate based on loss
        """
        if loss < self.best_loss:
            self.best_loss = loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
            if self.patience_counter >= self.patience:
                # Reduce learning rate
                self.current_lr = max(self.current_lr * 0.5, self.min_lr)
                self.patience_counter = 0
        
        return self.current_lr
