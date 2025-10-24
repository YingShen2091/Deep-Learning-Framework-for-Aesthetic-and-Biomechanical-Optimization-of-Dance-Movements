"""
Training Module for Dance Movement Analysis
Implements complete training pipeline with aesthetic evaluation, biomechanical optimization,
CAM visualization, and reinforcement learning integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
import json
import yaml
import time
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import wandb
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Import custom modules (these would be imported from the created files)
# from models.dance_model import DanceAestheticModel
# from models.reinforcement_learning import PPOAgent, DanceEnvironment, RolloutBuffer
# from data.data_loader import create_data_loaders
# from utils.metrics import AestheticMetrics
# from utils.visualization import CAMVisualizer


class DanceTrainer:
    """
    Complete training pipeline for dance movement analysis
    """
    
    def __init__(self, config: Dict):
        """
        Initialize trainer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set random seeds for reproducibility
        self._set_seeds(config.get('seed', 42))
        
        # Initialize model
        self.model = self._build_model()
        
        # Initialize optimizers
        self.optimizer, self.scheduler = self._build_optimizers()
        
        # Initialize loss functions
        self.criterion_aesthetic = nn.MSELoss()
        self.criterion_biomech = nn.MSELoss()
        self.criterion_joint = nn.MSELoss()
        
        # Loss weights
        self.loss_weights = {
            'aesthetic': config.get('loss_weight_aesthetic', 1.0),
            'biomechanical': config.get('loss_weight_biomech', 0.5),
            'joint': config.get('loss_weight_joint', 0.3)
        }
        
        # Initialize RL components if enabled
        self.use_rl = config.get('use_reinforcement_learning', True)
        if self.use_rl:
            self.rl_agent, self.rl_env = self._build_rl_components()
            self.rl_update_frequency = config.get('rl_update_frequency', 100)
            
        # Initialize metrics tracking
        self.metrics = AestheticMetrics()
        
        # Initialize visualization
        self.visualizer = CAMVisualizer()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Mixed precision training
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Logging
        self._setup_logging()
        
        # Checkpointing
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        
    def _build_model(self) -> nn.Module:
        """Build and initialize the model"""
        from models.dance_model import DanceAestheticModel
        
        model = DanceAestheticModel(
            spatial_feature_dim=self.config.get('spatial_feature_dim', 512),
            temporal_hidden_dim=self.config.get('temporal_hidden_dim', 256),
            num_lstm_layers=self.config.get('num_lstm_layers', 2),
            dropout=self.config.get('dropout', 0.3),
            num_aesthetic_classes=self.config.get('num_aesthetic_classes', 10),
            enable_cam=self.config.get('enable_cam', True)
        )
        
        # Load pretrained weights if available
        if self.config.get('pretrained_checkpoint'):
            checkpoint = torch.load(self.config['pretrained_checkpoint'])
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded pretrained model from {self.config['pretrained_checkpoint']}")
            
        model = model.to(self.device)
        
        # Multi-GPU training
        if torch.cuda.device_count() > 1 and self.config.get('use_multi_gpu', True):
            model = nn.DataParallel(model)
            
        return model
    
    def _build_optimizers(self) -> Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]:
        """Build optimizer and learning rate scheduler"""
        # Optimizer selection
        optimizer_type = self.config.get('optimizer', 'adam')
        lr = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        if optimizer_type == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999)
            )
        elif optimizer_type == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
            
        # Learning rate scheduler
        scheduler_type = self.config.get('scheduler', 'step')
        
        if scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config.get('lr_step_size', 20),
                gamma=self.config.get('lr_gamma', 0.1)
            )
        elif scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.get('max_epochs', 50),
                eta_min=1e-6
            )
        elif scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.1,
                patience=5,
                verbose=True
            )
        else:
            scheduler = None
            
        return optimizer, scheduler
    
    def _build_rl_components(self):
        """Build reinforcement learning components"""
        from models.reinforcement_learning import PPOAgent, DanceEnvironment
        
        # Create environment
        env = DanceEnvironment(
            aesthetic_model=self.model,
            max_steps=self.config.get('rl_max_steps', 100),
            w1=self.config.get('rl_w1', 0.6),
            w2=self.config.get('rl_w2', 0.3),
            w3=self.config.get('rl_w3', 0.1)
        )
        
        # Create PPO agent
        observation_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        agent = PPOAgent(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=self.config.get('rl_hidden_dim', 256),
            lr_actor=self.config.get('rl_lr_actor', 3e-4),
            lr_critic=self.config.get('rl_lr_critic', 3e-4),
            gamma=self.config.get('rl_gamma', 0.99),
            gae_lambda=self.config.get('rl_gae_lambda', 0.95),
            epsilon=self.config.get('rl_epsilon', 0.2),
            c1=self.config.get('rl_c1', 0.5),
            c2=self.config.get('rl_c2', 0.01),
            max_grad_norm=self.config.get('rl_max_grad_norm', 0.5)
        )
        
        agent = agent.to(self.device)
        
        return agent, env
    
    def _setup_logging(self):
        """Setup logging and experiment tracking"""
        # TensorBoard
        self.tb_writer = SummaryWriter(
            log_dir=Path(self.config['log_dir']) / 'tensorboard'
        )
        
        # Weights & Biases
        if self.config.get('use_wandb', False):
            wandb.init(
                project=self.config.get('wandb_project', 'dance-ai'),
                config=self.config,
                name=self.config.get('experiment_name', 'dance-training')
            )
            
        # MLflow
        if self.config.get('use_mlflow', False):
            mlflow.set_experiment(self.config.get('mlflow_experiment', 'dance-analysis'))
            mlflow.start_run(run_name=self.config.get('experiment_name'))
            mlflow.log_params(self.config)
            
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        epoch_losses = {
            'total': 0,
            'aesthetic': 0,
            'biomechanical': 0,
            'joint': 0
        }
        
        epoch_metrics = {
            'mse': 0,
            'mae': 0,
            'correlation': 0
        }
        
        predictions_all = []
        targets_all = []
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            frames = batch['frames'].to(self.device)
            aesthetic_scores = batch['aesthetic_scores'].to(self.device)
            sequence_lengths = batch['sequence_lengths'].to(self.device)
            
            # Optional data
            biomech_scores = batch.get('biomechanical_scores')
            if biomech_scores is not None:
                biomech_scores = biomech_scores.to(self.device)
                
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(frames, sequence_lengths, return_features=True)
                    
                    # Calculate losses
                    loss_aesthetic = self.criterion_aesthetic(
                        outputs['aesthetic_score'].squeeze(),
                        aesthetic_scores
                    )
                    
                    loss_biomech = 0
                    if biomech_scores is not None:
                        loss_biomech = self.criterion_biomech(
                            outputs['biomechanical_score'].squeeze(),
                            biomech_scores
                        )
                        
                    # Joint angle regularization
                    loss_joint = torch.mean(torch.abs(outputs['joint_angles']))
                    
                    # Total loss
                    total_loss = (
                        self.loss_weights['aesthetic'] * loss_aesthetic +
                        self.loss_weights['biomechanical'] * loss_biomech +
                        self.loss_weights['joint'] * loss_joint
                    )
                    
                # Backward pass with mixed precision
                self.scaler.scale(total_loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.get('max_grad_norm', 1.0)
                )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
            else:
                # Standard forward pass
                outputs = self.model(frames, sequence_lengths, return_features=True)
                
                # Calculate losses
                loss_aesthetic = self.criterion_aesthetic(
                    outputs['aesthetic_score'].squeeze(),
                    aesthetic_scores
                )
                
                loss_biomech = 0
                if biomech_scores is not None:
                    loss_biomech = self.criterion_biomech(
                        outputs['biomechanical_score'].squeeze(),
                        biomech_scores
                    )
                    
                # Joint angle regularization
                loss_joint = torch.mean(torch.abs(outputs['joint_angles']))
                
                # Total loss
                total_loss = (
                    self.loss_weights['aesthetic'] * loss_aesthetic +
                    self.loss_weights['biomechanical'] * loss_biomech +
                    self.loss_weights['joint'] * loss_joint
                )
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.get('max_grad_norm', 1.0)
                )
                
                # Optimizer step
                self.optimizer.step()
                
            # Update metrics
            epoch_losses['total'] += total_loss.item()
            epoch_losses['aesthetic'] += loss_aesthetic.item()
            if biomech_scores is not None:
                epoch_losses['biomechanical'] += loss_biomech.item()
            epoch_losses['joint'] += loss_joint.item()
            
            # Store predictions for metrics
            predictions = outputs['aesthetic_score'].squeeze().detach().cpu().numpy()
            targets = aesthetic_scores.detach().cpu().numpy()
            predictions_all.extend(predictions)
            targets_all.extend(targets)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss.item(),
                'aes': loss_aesthetic.item()
            })
            
            # Reinforcement learning update
            if self.use_rl and (self.global_step + 1) % self.rl_update_frequency == 0:
                self._rl_optimization_step(outputs['fused_features'].detach())
                
            # Logging
            if (batch_idx + 1) % self.config.get('log_frequency', 10) == 0:
                self._log_training_step(epoch_losses, batch_idx)
                
            # Visualization
            if (batch_idx + 1) % self.config.get('vis_frequency', 100) == 0:
                self._visualize_cams(frames[0], outputs['cams'][0] if outputs['cams'] is not None else None)
                
            self.global_step += 1
            
        # Calculate epoch metrics
        predictions_all = np.array(predictions_all)
        targets_all = np.array(targets_all)
        
        epoch_metrics['mse'] = mean_squared_error(targets_all, predictions_all)
        epoch_metrics['mae'] = mean_absolute_error(targets_all, predictions_all)
        epoch_metrics['correlation'], _ = pearsonr(targets_all, predictions_all)
        
        # Average losses
        num_batches = len(train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            
        return {**epoch_losses, **epoch_metrics}
    
    def validate(self, val_loader) -> Dict[str, float]:
        """
        Validate model
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        val_losses = {
            'total': 0,
            'aesthetic': 0,
            'biomechanical': 0,
            'joint': 0
        }
        
        predictions_all = []
        targets_all = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                # Move batch to device
                frames = batch['frames'].to(self.device)
                aesthetic_scores = batch['aesthetic_scores'].to(self.device)
                sequence_lengths = batch['sequence_lengths'].to(self.device)
                
                # Optional data
                biomech_scores = batch.get('biomechanical_scores')
                if biomech_scores is not None:
                    biomech_scores = biomech_scores.to(self.device)
                    
                # Forward pass
                outputs = self.model(frames, sequence_lengths)
                
                # Calculate losses
                loss_aesthetic = self.criterion_aesthetic(
                    outputs['aesthetic_score'].squeeze(),
                    aesthetic_scores
                )
                
                loss_biomech = 0
                if biomech_scores is not None:
                    loss_biomech = self.criterion_biomech(
                        outputs['biomechanical_score'].squeeze(),
                        biomech_scores
                    )
                    
                loss_joint = torch.mean(torch.abs(outputs['joint_angles']))
                
                total_loss = (
                    self.loss_weights['aesthetic'] * loss_aesthetic +
                    self.loss_weights['biomechanical'] * loss_biomech +
                    self.loss_weights['joint'] * loss_joint
                )
                
                # Update losses
                val_losses['total'] += total_loss.item()
                val_losses['aesthetic'] += loss_aesthetic.item()
                if biomech_scores is not None:
                    val_losses['biomechanical'] += loss_biomech.item()
                val_losses['joint'] += loss_joint.item()
                
                # Store predictions
                predictions = outputs['aesthetic_score'].squeeze().cpu().numpy()
                targets = aesthetic_scores.cpu().numpy()
                predictions_all.extend(predictions)
                targets_all.extend(targets)
                
        # Calculate metrics
        predictions_all = np.array(predictions_all)
        targets_all = np.array(targets_all)
        
        val_metrics = {
            'mse': mean_squared_error(targets_all, predictions_all),
            'mae': mean_absolute_error(targets_all, predictions_all),
            'correlation': pearsonr(targets_all, predictions_all)[0],
            'spearman': spearmanr(targets_all, predictions_all)[0]
        }
        
        # Average losses
        num_batches = len(val_loader)
        for key in val_losses:
            val_losses[key] /= num_batches
            
        # Calculate inter-rater agreement if available
        if self.config.get('calculate_inter_rater', False):
            val_metrics['fleiss_kappa'] = self._calculate_fleiss_kappa(predictions_all, targets_all)
            
        return {**val_losses, **val_metrics}
    
    def _rl_optimization_step(self, features: torch.Tensor):
        """
        Perform reinforcement learning optimization step
        
        Args:
            features: Current features from the model
        """
        if not self.use_rl:
            return
            
        from models.reinforcement_learning import RolloutBuffer
        
        # Create rollout buffer
        rollout_buffer = RolloutBuffer(
            buffer_size=self.config.get('rl_buffer_size', 2048),
            observation_dim=self.rl_env.observation_space.shape[0],
            action_dim=self.rl_env.action_space.shape[0]
        )
        
        # Collect rollouts
        for _ in range(self.config.get('rl_n_rollouts', 10)):
            obs = self.rl_env.reset()
            
            for step in range(self.config.get('rl_max_steps', 100)):
                # Get action from agent
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
                action, log_prob, value = self.rl_agent.get_action(obs_tensor)
                
                # Step environment
                next_obs, reward, done, info = self.rl_env.step(action.cpu().numpy())
                
                # Add to buffer
                rollout_buffer.add(
                    observation=obs_tensor,
                    action=action,
                    reward=reward,
                    value=value.item(),
                    log_prob=log_prob.item(),
                    done=done
                )
                
                obs = next_obs
                
                if done:
                    break
                    
        # Update RL agent
        rl_stats = self.rl_agent.update(
            rollout_buffer.get(),
            n_epochs=self.config.get('rl_n_epochs', 10),
            batch_size=self.config.get('rl_batch_size', 64)
        )
        
        # Log RL statistics
        for key, value in rl_stats.items():
            self.tb_writer.add_scalar(f'RL/{key}', value, self.global_step)
            
    def _log_training_step(self, losses: Dict, batch_idx: int):
        """Log training metrics"""
        # TensorBoard logging
        for key, value in losses.items():
            self.tb_writer.add_scalar(f'Train/Loss_{key}', value, self.global_step)
            
        # Current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.tb_writer.add_scalar('Train/LearningRate', current_lr, self.global_step)
        
        # Weights & Biases logging
        if self.config.get('use_wandb', False):
            wandb.log({
                **{f'train_loss_{k}': v for k, v in losses.items()},
                'learning_rate': current_lr,
                'global_step': self.global_step
            })
            
    def _visualize_cams(self, frames: torch.Tensor, cams: Optional[torch.Tensor]):
        """Visualize Class Activation Maps"""
        if cams is None:
            return
            
        # Get last frame
        last_frame = frames[-1].cpu().numpy().transpose(1, 2, 0)
        
        # Denormalize frame
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        last_frame = last_frame * std + mean
        last_frame = np.clip(last_frame, 0, 1)
        
        # Get CAM
        cam = cams[0].cpu().numpy() if cams.dim() > 1 else cams.cpu().numpy()
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original frame
        axes[0].imshow(last_frame)
        axes[0].set_title('Original Frame')
        axes[0].axis('off')
        
        # CAM heatmap
        axes[1].imshow(cam, cmap='hot')
        axes[1].set_title('CAM Heatmap')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(last_frame)
        axes[2].imshow(cam, alpha=0.5, cmap='hot')
        axes[2].set_title('CAM Overlay')
        axes[2].axis('off')
        
        # Save figure
        cam_path = Path(self.config['log_dir']) / 'cams' / f'cam_step_{self.global_step}.png'
        cam_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(cam_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Log to TensorBoard
        self.tb_writer.add_figure('CAM/Visualization', fig, self.global_step)
        
    def save_checkpoint(self, metrics: Dict, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{self.current_epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            
        # Save RL agent if used
        if self.use_rl:
            rl_checkpoint = {
                'agent_state_dict': self.rl_agent.state_dict(),
                'agent_optimizer_actor': self.rl_agent.actor_optimizer.state_dict(),
                'agent_optimizer_critic': self.rl_agent.critic_optimizer.state_dict()
            }
            rl_path = self.checkpoint_dir / f'rl_agent_epoch_{self.current_epoch}.pt'
            torch.save(rl_checkpoint, rl_path)
            
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
        
    def train(self, train_loader, val_loader, test_loader=None):
        """
        Main training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Optional test data loader
        """
        max_epochs = self.config.get('max_epochs', 50)
        early_stopping_patience = self.config.get('early_stopping_patience', 5)
        
        print(f"Starting training for {max_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.current_epoch, max_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate(val_loader)
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['total'])
                else:
                    self.scheduler.step()
                    
            # Logging
            print(f"\nEpoch {epoch}/{max_epochs}")
            print(f"Train - Loss: {train_metrics['total']:.4f}, "
                  f"MSE: {train_metrics['mse']:.4f}, "
                  f"Correlation: {train_metrics['correlation']:.4f}")
            print(f"Val - Loss: {val_metrics['total']:.4f}, "
                  f"MSE: {val_metrics['mse']:.4f}, "
                  f"Correlation: {val_metrics['correlation']:.4f}")
            
            # TensorBoard logging
            for key, value in train_metrics.items():
                self.tb_writer.add_scalar(f'Train/Epoch_{key}', value, epoch)
            for key, value in val_metrics.items():
                self.tb_writer.add_scalar(f'Val/Epoch_{key}', value, epoch)
                
            # Weights & Biases logging
            if self.config.get('use_wandb', False):
                wandb.log({
                    **{f'train_{k}': v for k, v in train_metrics.items()},
                    **{f'val_{k}': v for k, v in val_metrics.items()},
                    'epoch': epoch
                })
                
            # Check for improvement
            if val_metrics['total'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total']
                self.patience_counter = 0
                self.save_checkpoint(val_metrics, is_best=True)
                print(f"New best model saved! Val loss: {self.best_val_loss:.4f}")
            else:
                self.patience_counter += 1
                self.save_checkpoint(val_metrics, is_best=False)
                
            # Early stopping
            if self.patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break
                
        # Final test evaluation
        if test_loader:
            print("\nEvaluating on test set...")
            test_metrics = self.validate(test_loader)
            print(f"Test - MSE: {test_metrics['mse']:.4f}, "
                  f"Correlation: {test_metrics['correlation']:.4f}")
            
            # Save test results
            with open(self.checkpoint_dir / 'test_results.json', 'w') as f:
                json.dump(test_metrics, f, indent=4)
                
        # Cleanup
        self.tb_writer.close()
        if self.config.get('use_wandb', False):
            wandb.finish()
        if self.config.get('use_mlflow', False):
            mlflow.end_run()
            
        print("\nTraining completed!")
        
    def _calculate_fleiss_kappa(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Calculate Fleiss' Kappa for inter-rater agreement
        
        Args:
            predictions: Model predictions
            targets: Ground truth labels
            
        Returns:
            Fleiss' Kappa score
        """
        from sklearn.metrics import cohen_kappa_score
        
        # Discretize continuous scores into categories
        n_categories = 10
        pred_categories = np.digitize(predictions, bins=np.linspace(0, 10, n_categories+1))
        target_categories = np.digitize(targets, bins=np.linspace(0, 10, n_categories+1))
        
        # Calculate Cohen's kappa as approximation
        kappa = cohen_kappa_score(target_categories, pred_categories, weights='quadratic')
        
        return kappa


class AestheticMetrics:
    """
    Metrics calculation for aesthetic evaluation
    """
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset all metrics"""
        self.predictions = []
        self.targets = []
        self.styles = []
        
    def update(self, predictions: np.ndarray, targets: np.ndarray, styles: List[str] = None):
        """Update metrics with new predictions"""
        self.predictions.extend(predictions)
        self.targets.extend(targets)
        if styles:
            self.styles.extend(styles)
            
    def compute(self) -> Dict[str, float]:
        """Compute all metrics"""
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        metrics = {
            'mse': mean_squared_error(targets, predictions),
            'rmse': np.sqrt(mean_squared_error(targets, predictions)),
            'mae': mean_absolute_error(targets, predictions),
            'pearson_r': pearsonr(targets, predictions)[0],
            'spearman_r': spearmanr(targets, predictions)[0]
        }
        
        # Per-style metrics if available
        if self.styles:
            unique_styles = list(set(self.styles))
            for style in unique_styles:
                style_mask = [s == style for s in self.styles]
                style_preds = predictions[style_mask]
                style_targets = targets[style_mask]
                
                if len(style_preds) > 1:
                    metrics[f'{style}_mse'] = mean_squared_error(style_targets, style_preds)
                    metrics[f'{style}_correlation'] = pearsonr(style_targets, style_preds)[0]
                    
        return metrics


class CAMVisualizer:
    """
    Visualization tools for Class Activation Maps
    """
    
    def __init__(self, colormap: str = 'jet'):
        self.colormap = colormap
        
    def visualize_cam(self, image: np.ndarray, cam: np.ndarray, 
                      alpha: float = 0.5) -> np.ndarray:
        """
        Overlay CAM on image
        
        Args:
            image: Original image
            cam: Class activation map
            alpha: Transparency factor
            
        Returns:
            Overlayed image
        """
        # Resize CAM to match image size
        cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
        
        # Normalize CAM
        cam_normalized = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min())
        
        # Apply colormap
        cam_colored = plt.cm.get_cmap(self.colormap)(cam_normalized)[:, :, :3]
        
        # Overlay on image
        overlay = image * (1 - alpha) + cam_colored * alpha
        
        return overlay
    
    def create_comparison_grid(self, images: List[np.ndarray], 
                              cams: List[np.ndarray]) -> np.ndarray:
        """
        Create grid of images with CAM overlays
        
        Args:
            images: List of images
            cams: List of CAMs
            
        Returns:
            Grid image
        """
        overlays = []
        for img, cam in zip(images, cams):
            overlay = self.visualize_cam(img, cam)
            overlays.append(overlay)
            
        # Create grid
        n = len(overlays)
        grid_size = int(np.ceil(np.sqrt(n)))
        
        grid = np.zeros((grid_size * images[0].shape[0], 
                        grid_size * images[0].shape[1], 3))
        
        for idx, overlay in enumerate(overlays):
            i = idx // grid_size
            j = idx % grid_size
            
            start_i = i * images[0].shape[0]
            end_i = (i + 1) * images[0].shape[0]
            start_j = j * images[0].shape[1]
            end_j = (j + 1) * images[0].shape[1]
            
            grid[start_i:end_i, start_j:end_j] = overlay
            
        return grid
