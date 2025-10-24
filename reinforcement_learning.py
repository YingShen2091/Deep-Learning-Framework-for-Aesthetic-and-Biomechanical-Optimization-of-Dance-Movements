"""
Reinforcement Learning Module for Dance Movement Optimization
Implementation of Proximal Policy Optimization (PPO) with GAE
Based on Algorithm 1 and 2 from the paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict, List, Optional
from collections import deque
import gym
from gym import spaces
from dataclasses import dataclass
import copy


@dataclass
class RLConfig:
    """Configuration for RL training"""
    # PPO hyperparameters
    clip_epsilon: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Optimization
    learning_rate: float = 3e-4
    num_epochs: int = 10
    mini_batch_size: int = 64
    
    # Environment
    num_joints: int = 25  # Number of body joints
    max_joint_adjustment: float = 0.1  # Maximum angle adjustment in radians
    
    # Reward weights (w1, w2, w3 from Algorithm 1)
    aesthetic_weight: float = 0.6
    biomechanical_weight: float = 0.3
    smoothness_weight: float = 0.1
    
    # Training
    num_steps: int = 2048
    num_processes: int = 4
    
    # Joint limits (in radians)
    joint_limits: Dict[str, Tuple[float, float]] = None
    
    def __post_init__(self):
        if self.joint_limits is None:
            # Default joint limits based on biomechanical constraints
            self.joint_limits = {
                'neck': (-0.5, 0.5),
                'shoulder': (-3.14, 3.14),
                'elbow': (0, 2.7),
                'wrist': (-1.5, 1.5),
                'hip': (-2.0, 0.5),
                'knee': (-2.5, 0),
                'ankle': (-0.5, 0.5),
                'spine': (-0.8, 0.8)
            }


class DanceEnvironment(gym.Env):
    """
    Custom Gym environment for dance movement optimization
    Implements MDP formulation from Algorithm 1
    """
    
    def __init__(self, model: nn.Module, config: RLConfig,
                 reference_sequences: Optional[torch.Tensor] = None):
        super(DanceEnvironment, self).__init__()
        
        self.model = model
        self.config = config
        self.reference_sequences = reference_sequences
        self.device = next(model.parameters()).device
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-config.max_joint_adjustment,
            high=config.max_joint_adjustment,
            shape=(config.num_joints * 3,),  # 3D adjustments for each joint
            dtype=np.float32
        )
        
        # State space: concatenated CNN-LSTM features
        state_dim = 2048 + 512  # Spatial + Temporal features
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )
        
        # Initialize state
        self.current_state = None
        self.current_pose = None
        self.timestep = 0
        self.max_timesteps = 30
        
        # History for temporal consistency
        self.pose_history = deque(maxlen=5)
        self.action_history = deque(maxlen=5)
        
    def reset(self, initial_pose: Optional[np.ndarray] = None) -> np.ndarray:
        """Reset environment to initial state"""
        if initial_pose is None:
            # Initialize with neutral pose
            initial_pose = np.zeros((self.config.num_joints, 3))
        
        self.current_pose = initial_pose
        self.timestep = 0
        self.pose_history.clear()
        self.action_history.clear()
        
        # Get initial state from model features
        self.current_state = self._get_state_features(self.current_pose)
        
        return self.current_state
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return new state, reward, done flag, and info
        Implements transition dynamics from Algorithm 1
        """
        # Apply joint angle adjustments (∆θt)
        action = action.reshape(self.config.num_joints, 3)
        
        # Apply biomechanical constraints
        adjusted_pose = self._apply_joint_constraints(
            self.current_pose + action
        )
        
        # Store history for temporal consistency
        self.pose_history.append(adjusted_pose)
        self.action_history.append(action)
        
        # Get new state features
        new_state = self._get_state_features(adjusted_pose)
        
        # Calculate reward (Equation from Algorithm 1: Rt = w1·A(st) + w2·B(st) - w3·||∆θt||²)
        reward = self._calculate_reward(
            adjusted_pose,
            action,
            new_state
        )
        
        # Update environment
        self.current_pose = adjusted_pose
        self.current_state = new_state
        self.timestep += 1
        
        # Check if episode is done
        done = self.timestep >= self.max_timesteps
        
        # Additional info for logging
        info = {
            'aesthetic_score': self._get_aesthetic_score(new_state),
            'biomechanical_score': self._get_biomechanical_score(adjusted_pose),
            'pose': adjusted_pose,
            'timestep': self.timestep
        }
        
        return new_state, reward, done, info
    
    def _get_state_features(self, pose: np.ndarray) -> np.ndarray:
        """
        Extract state features from pose using CNN-LSTM model
        State st = [St; Ht] as defined in Algorithm 1
        """
        # Convert pose to appropriate input format
        # This would involve rendering the pose or using motion capture data
        # For now, we'll create a placeholder
        
        with torch.no_grad():
            # Create dummy video frame from pose (in practice, use pose rendering)
            dummy_frame = torch.randn(1, 1, 3, 224, 224).to(self.device)
            
            # Get model features
            outputs = self.model(dummy_frame, return_features=True)
            
            spatial_features = outputs['spatial_features'].squeeze()
            temporal_features = outputs['temporal_features'].squeeze()
            
            # Concatenate features
            state = torch.cat([
                spatial_features.mean(0),  # Average over sequence
                temporal_features.mean(0)
            ])
            
        return state.cpu().numpy()
    
    def _apply_joint_constraints(self, pose: np.ndarray) -> np.ndarray:
        """
        Apply biomechanical joint limits
        Ensures |∆θt| ≤ θmax as specified in Algorithm 1
        """
        constrained_pose = pose.copy()
        
        # Apply joint-specific constraints
        for joint_idx in range(self.config.num_joints):
            # Get joint type (simplified mapping)
            joint_type = self._get_joint_type(joint_idx)
            
            if joint_type in self.config.joint_limits:
                min_angle, max_angle = self.config.joint_limits[joint_type]
                
                # Constrain each dimension
                for dim in range(3):
                    constrained_pose[joint_idx, dim] = np.clip(
                        constrained_pose[joint_idx, dim],
                        min_angle,
                        max_angle
                    )
        
        return constrained_pose
    
    def _get_joint_type(self, joint_idx: int) -> str:
        """Map joint index to joint type"""
        # Simplified mapping - in practice, use proper skeleton definition
        joint_mapping = {
            0: 'neck', 1: 'neck',
            2: 'shoulder', 3: 'shoulder', 4: 'shoulder', 5: 'shoulder',
            6: 'elbow', 7: 'elbow', 8: 'elbow', 9: 'elbow',
            10: 'wrist', 11: 'wrist', 12: 'wrist', 13: 'wrist',
            14: 'hip', 15: 'hip', 16: 'hip', 17: 'hip',
            18: 'knee', 19: 'knee', 20: 'knee', 21: 'knee',
            22: 'ankle', 23: 'ankle', 24: 'spine'
        }
        return joint_mapping.get(joint_idx, 'spine')
    
    def _calculate_reward(self, pose: np.ndarray, action: np.ndarray,
                         state: np.ndarray) -> float:
        """
        Calculate reward function from Algorithm 1
        Rt = w1 · A(st) + w2 · B(st) - w3 · ||∆θt||²
        """
        # Aesthetic score A(st) - normalized to [0, 1]
        aesthetic_score = self._get_aesthetic_score(state)
        
        # Biomechanical efficiency B(st) - normalized to [0, 1]
        biomechanical_score = self._get_biomechanical_score(pose)
        
        # Action magnitude penalty ||∆θt||²
        action_penalty = np.sum(action ** 2)
        
        # Calculate weighted reward
        reward = (
            self.config.aesthetic_weight * aesthetic_score +
            self.config.biomechanical_weight * biomechanical_score -
            self.config.smoothness_weight * action_penalty
        )
        
        # Add temporal consistency bonus
        if len(self.pose_history) > 1:
            temporal_consistency = self._calculate_temporal_consistency()
            reward += 0.1 * temporal_consistency
        
        return reward
    
    def _get_aesthetic_score(self, state: np.ndarray) -> float:
        """
        Get normalized aesthetic score from model
        A(st) from Algorithm 1
        """
        with torch.no_grad():
            # Use the model's aesthetic predictor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Reshape for model input
            dummy_input = state_tensor.unsqueeze(1)  # Add sequence dimension
            
            # Get aesthetic score (placeholder - in practice, use proper model inference)
            score = torch.sigmoid(torch.randn(1)).item()
        
        return score
    
    def _get_biomechanical_score(self, pose: np.ndarray) -> float:
        """
        Calculate biomechanical efficiency score
        B(st) = 1 - (1/T) * Σ||θi - θsafe||² from Algorithm 1
        """
        # Define safe joint angles (injury-minimizing positions)
        safe_angles = {
            'neck': np.array([0, 0, 0]),
            'shoulder': np.array([0.2, 0, 0]),
            'elbow': np.array([1.5, 0, 0]),
            'wrist': np.array([0, 0, 0]),
            'hip': np.array([-0.3, 0, 0]),
            'knee': np.array([-0.2, 0, 0]),
            'ankle': np.array([0, 0, 0]),
            'spine': np.array([0, 0, 0])
        }
        
        total_deviation = 0
        for joint_idx in range(self.config.num_joints):
            joint_type = self._get_joint_type(joint_idx)
            if joint_type in safe_angles:
                safe_pose = safe_angles[joint_type]
                deviation = np.sum((pose[joint_idx] - safe_pose) ** 2)
                total_deviation += deviation
        
        # Normalize to [0, 1]
        max_deviation = self.config.num_joints * 3 * (np.pi ** 2)  # Maximum possible deviation
        biomechanical_score = 1 - (total_deviation / max_deviation)
        
        return np.clip(biomechanical_score, 0, 1)
    
    def _calculate_temporal_consistency(self) -> float:
        """Calculate smoothness of movement over time"""
        if len(self.pose_history) < 2:
            return 0
        
        # Calculate difference between consecutive poses
        pose_diff = np.diff(np.array(self.pose_history), axis=0)
        
        # Penalize large changes
        consistency_score = 1 - np.mean(np.abs(pose_diff))
        
        return np.clip(consistency_score, 0, 1)


class PPOAgent(nn.Module):
    """
    PPO Agent for dance movement optimization
    Implements Algorithm 2 from the paper
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 hidden_dim: int = 256, config: RLConfig = None):
        super(PPOAgent, self).__init__()
        
        self.config = config or RLConfig()
        
        # Actor network (policy πφ)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Critic network (value function Vψ)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Action distribution
        self.action_log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for both actor and critic"""
        action_mean = self.actor(state)
        value = self.critic(state)
        return action_mean, value
    
    def get_action(self, state: torch.Tensor, 
                   deterministic: bool = False
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy
        
        Returns:
            action, log_prob, value
        """
        action_mean, value = self.forward(state)
        
        if deterministic:
            action = action_mean
            log_prob = torch.zeros_like(action).sum(-1)
        else:
            # Create normal distribution
            action_std = self.action_log_std.exp()
            dist = torch.distributions.Normal(action_mean, action_std)
            
            # Sample action
            action = dist.sample()
            
            # Calculate log probability
            log_prob = dist.log_prob(action).sum(-1)
        
        return action, log_prob, value
    
    def evaluate_actions(self, states: torch.Tensor, 
                        actions: torch.Tensor
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update
        
        Returns:
            log_probs, values, entropy
        """
        action_mean, values = self.forward(states)
        
        action_std = self.action_log_std.exp()
        dist = torch.distributions.Normal(action_mean, action_std)
        
        log_probs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1).mean()
        
        return log_probs, values, entropy


class PPOTrainer:
    """
    PPO training implementation following Algorithm 2
    """
    
    def __init__(self, agent: PPOAgent, env: DanceEnvironment, 
                 config: RLConfig):
        self.agent = agent
        self.env = env
        self.config = config
        self.device = next(agent.parameters()).device
        
        # Optimizer
        self.optimizer = optim.Adam(
            agent.parameters(),
            lr=config.learning_rate,
            eps=1e-5
        )
        
        # Storage for rollout data
        self.rollout_storage = RolloutStorage(
            num_steps=config.num_steps,
            num_processes=config.num_processes,
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            device=self.device
        )
        
    def train(self, num_updates: int = 1000):
        """
        Main training loop implementing Algorithm 2
        """
        num_steps = self.config.num_steps
        num_processes = self.config.num_processes
        
        # Initialize environments (parallel)
        states = torch.zeros(num_processes, *self.env.observation_space.shape)
        
        for update in range(num_updates):
            # Collect trajectories
            with torch.no_grad():
                for step in range(num_steps):
                    # Get actions from policy
                    actions, log_probs, values = self.agent.get_action(
                        states.to(self.device)
                    )
                    
                    # Execute actions in environment
                    next_states = []
                    rewards = []
                    dones = []
                    infos = []
                    
                    for proc_idx in range(num_processes):
                        state, reward, done, info = self.env.step(
                            actions[proc_idx].cpu().numpy()
                        )
                        next_states.append(state)
                        rewards.append(reward)
                        dones.append(done)
                        infos.append(info)
                    
                    # Store in rollout buffer
                    self.rollout_storage.insert(
                        states=states,
                        actions=actions,
                        log_probs=log_probs,
                        values=values,
                        rewards=torch.tensor(rewards),
                        dones=torch.tensor(dones)
                    )
                    
                    states = torch.tensor(next_states)
            
            # Compute returns and advantages using GAE
            self._compute_gae()
            
            # PPO update
            self._ppo_update()
            
            # Clear rollout storage
            self.rollout_storage.clear()
            
            # Logging
            if update % 10 == 0:
                print(f"Update {update}: Training in progress...")
    
    def _compute_gae(self):
        """
        Compute Generalized Advantage Estimation
        Implements GAE calculation from Algorithm 2
        """
        gamma = self.config.gamma
        gae_lambda = self.config.gae_lambda
        
        advantages = torch.zeros_like(self.rollout_storage.rewards)
        last_advantage = 0
        
        # Calculate advantages backwards
        for t in reversed(range(len(self.rollout_storage.rewards))):
            if t == len(self.rollout_storage.rewards) - 1:
                next_value = 0
            else:
                next_value = self.rollout_storage.values[t + 1]
            
            # δt = Rt + γV(st+1) - V(st)
            delta = (self.rollout_storage.rewards[t] + 
                    gamma * next_value * (1 - self.rollout_storage.dones[t]) - 
                    self.rollout_storage.values[t])
            
            # At = Σ(γλ)^l * δt+l
            advantages[t] = delta + gamma * gae_lambda * last_advantage * (1 - self.rollout_storage.dones[t])
            last_advantage = advantages[t]
        
        self.rollout_storage.advantages = advantages
        self.rollout_storage.returns = advantages + self.rollout_storage.values
    
    def _ppo_update(self):
        """
        PPO policy and value function update
        Implements the clipped objective from Algorithm 2
        """
        # Get all data from storage
        states = self.rollout_storage.states
        actions = self.rollout_storage.actions
        old_log_probs = self.rollout_storage.log_probs
        returns = self.rollout_storage.returns
        advantages = self.rollout_storage.advantages
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO epochs
        for epoch in range(self.config.num_epochs):
            # Mini-batch updates
            batch_size = self.config.mini_batch_size
            num_samples = states.size(0)
            
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                
                batch_states = states[start_idx:end_idx]
                batch_actions = actions[start_idx:end_idx]
                batch_old_log_probs = old_log_probs[start_idx:end_idx]
                batch_returns = returns[start_idx:end_idx]
                batch_advantages = advantages[start_idx:end_idx]
                
                # Get current policy evaluation
                log_probs, values, entropy = self.agent.evaluate_actions(
                    batch_states,
                    batch_actions
                )
                
                # Calculate ratio rt(φ) = πφ(at|st) / πφold(at|st)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 
                                  1 - self.config.clip_epsilon,
                                  1 + self.config.clip_epsilon) * batch_advantages
                
                # Policy loss (negative because we maximize)
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value function loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Total loss
                loss = (policy_loss + 
                       self.config.value_loss_coef * value_loss - 
                       self.config.entropy_coef * entropy)
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.agent.parameters(),
                    self.config.max_grad_norm
                )
                self.optimizer.step()


class RolloutStorage:
    """Storage for trajectory data during PPO training"""
    
    def __init__(self, num_steps: int, num_processes: int,
                 state_shape: tuple, action_shape: tuple,
                 device: torch.device):
        self.num_steps = num_steps
        self.num_processes = num_processes
        self.device = device
        
        # Initialize storage tensors
        self.states = torch.zeros(num_steps, num_processes, *state_shape)
        self.actions = torch.zeros(num_steps, num_processes, *action_shape)
        self.log_probs = torch.zeros(num_steps, num_processes)
        self.values = torch.zeros(num_steps, num_processes)
        self.rewards = torch.zeros(num_steps, num_processes)
        self.dones = torch.zeros(num_steps, num_processes)
        
        self.advantages = None
        self.returns = None
        
        self.step = 0
    
    def insert(self, states, actions, log_probs, values, rewards, dones):
        """Insert new data into storage"""
        self.states[self.step] = states
        self.actions[self.step] = actions
        self.log_probs[self.step] = log_probs.squeeze()
        self.values[self.step] = values.squeeze()
        self.rewards[self.step] = rewards
        self.dones[self.step] = dones.float()
        
        self.step = (self.step + 1) % self.num_steps
    
    def clear(self):
        """Clear storage after update"""
        self.step = 0


class SemiEncoderGenerator(nn.Module):
    """
    Semi-encoder generator for creating optimized dance sequences
    Generates new movements based on learned policy
    """
    
    def __init__(self, latent_dim: int = 128, pose_dim: int = 75):
        super(SemiEncoderGenerator, self).__init__()
        
        # Encoder for existing poses
        self.encoder = nn.Sequential(
            nn.Linear(pose_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.Tanh()
        )
        
        # Generator for new poses
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, pose_dim),
            nn.Tanh()
        )
        
        # Temporal consistency network
        self.temporal_net = nn.LSTM(
            input_size=latent_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )
    
    def encode(self, pose: torch.Tensor) -> torch.Tensor:
        """Encode pose to latent space"""
        return self.encoder(pose)
    
    def generate(self, latent: torch.Tensor) -> torch.Tensor:
        """Generate new pose from latent code"""
        return self.generator(latent)
    
    def generate_sequence(self, initial_pose: torch.Tensor,
                         sequence_length: int = 30) -> torch.Tensor:
        """Generate optimized dance sequence"""
        batch_size = initial_pose.size(0)
        
        # Encode initial pose
        latent = self.encode(initial_pose)
        
        # Generate sequence
        sequence = []
        hidden = None
        
        for t in range(sequence_length):
            # Process through temporal network
            latent = latent.unsqueeze(1)  # Add sequence dimension
            latent, hidden = self.temporal_net(latent, hidden)
            latent = latent.squeeze(1)
            
            # Generate pose
            pose = self.generate(latent)
            sequence.append(pose)
            
            # Use generated pose for next step
            latent = self.encode(pose)
        
        return torch.stack(sequence, dim=1)
