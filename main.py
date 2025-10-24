"""
Main execution script for Dance Movement Analysis System
Entry point for training, evaluation, inference, and optimization
"""

import argparse
import yaml
import torch
import os
import sys
from pathlib import Path
import logging
from datetime import datetime
import json
import numpy as np
from typing import Dict, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import custom modules
from models.dance_model import DanceAestheticModel
from models.reinforcement_learning import PPOAgent, DanceEnvironment
from models.few_shot_learning import FewShotLearner, FewShotDanceDataset
from data.data_loader import create_data_loaders
from training.trainer import DanceTrainer
from evaluation.evaluator import DanceEvaluator
from inference.inference_engine import InferenceEngine
from utils.logger import setup_logger
from utils.seed import set_all_seeds


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_mode(args, config):
    """
    Main training pipeline
    """
    logger = setup_logger('train', config['logging']['log_dir'])
    logger.info("Starting training mode...")
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Set seeds for reproducibility
    set_all_seeds(config['seed'])
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(config)
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = DanceTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    logger.info("Starting training...")
    trainer.train(train_loader, val_loader, test_loader)
    
    # Final evaluation
    logger.info("Running final evaluation on test set...")
    test_metrics = trainer.validate(test_loader)
    
    # Save results
    results_path = Path(config['training']['checkpoint_dir']) / 'final_results.json'
    with open(results_path, 'w') as f:
        json.dump(test_metrics, f, indent=4)
    
    logger.info(f"Training completed! Results saved to {results_path}")
    logger.info(f"Test Metrics: {json.dumps(test_metrics, indent=2)}")
    
    # Print key metrics from paper
    print("\n" + "="*50)
    print("FINAL RESULTS (Paper Metrics)")
    print("="*50)
    print(f"Aesthetic Score MSE: {test_metrics.get('mse', 'N/A'):.4f} (Paper: 0.25)")
    print(f"Correlation Coefficient: {test_metrics.get('correlation', 'N/A'):.4f} (Paper: 0.92)")
    print(f"Biomechanical Efficiency: {test_metrics.get('biomechanical', 'N/A'):.4f}")
    print("="*50)


def evaluate_mode(args, config):
    """
    Evaluation mode for trained models
    """
    logger = setup_logger('evaluate', config['logging']['log_dir'])
    logger.info("Starting evaluation mode...")
    
    # Set seeds
    set_all_seeds(config['seed'])
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model = DanceAestheticModel(
        spatial_feature_dim=config['model']['spatial_feature_dim'],
        temporal_hidden_dim=config['model']['temporal_hidden_dim'],
        num_lstm_layers=config['model']['num_lstm_layers'],
        dropout=config['model']['dropout'],
        num_aesthetic_classes=config['model']['num_aesthetic_classes'],
        enable_cam=config['model']['enable_cam']
    )
    
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Create evaluator
    evaluator = DanceEvaluator(model, config, device)
    
    # Load test data
    _, _, test_loader = create_data_loaders(config)
    
    # Run comprehensive evaluation
    logger.info("Running comprehensive evaluation...")
    results = evaluator.evaluate_comprehensive(test_loader)
    
    # Save results
    output_dir = Path(args.output_dir) if args.output_dir else Path('./evaluation_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    metrics_path = output_dir / f'metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(metrics_path, 'w') as f:
        json.dump(results['metrics'], f, indent=4)
    
    # Generate visualizations
    if config['visualization']['save_visualizations']:
        logger.info("Generating visualizations...")
        evaluator.generate_visualizations(test_loader, output_dir)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for key, value in results['metrics'].items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print("="*50)
    
    logger.info(f"Evaluation completed! Results saved to {output_dir}")


def optimize_mode(args, config):
    """
    Reinforcement learning optimization mode
    """
    logger = setup_logger('optimize', config['logging']['log_dir'])
    logger.info("Starting optimization mode...")
    
    # Set seeds
    set_all_seeds(config['seed'])
    
    # Load base model
    logger.info(f"Loading base model from {args.checkpoint}")
    base_model = DanceAestheticModel(
        spatial_feature_dim=config['model']['spatial_feature_dim'],
        temporal_hidden_dim=config['model']['temporal_hidden_dim'],
        num_lstm_layers=config['model']['num_lstm_layers'],
        dropout=config['model']['dropout'],
        num_aesthetic_classes=config['model']['num_aesthetic_classes'],
        enable_cam=config['model']['enable_cam']
    )
    
    checkpoint = torch.load(args.checkpoint)
    base_model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_model = base_model.to(device)
    
    # Create RL environment
    logger.info("Creating RL environment...")
    env = DanceEnvironment(
        aesthetic_model=base_model,
        max_steps=config['reinforcement_learning']['rl_max_steps'],
        w1=config['reinforcement_learning']['rl_w1'],
        w2=config['reinforcement_learning']['rl_w2'],
        w3=config['reinforcement_learning']['rl_w3']
    )
    
    # Create PPO agent
    logger.info("Initializing PPO agent...")
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = PPOAgent(
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_dim=config['reinforcement_learning']['rl_hidden_dim'],
        lr_actor=config['reinforcement_learning']['rl_lr_actor'],
        lr_critic=config['reinforcement_learning']['rl_lr_critic'],
        gamma=config['reinforcement_learning']['rl_gamma'],
        gae_lambda=config['reinforcement_learning']['rl_gae_lambda'],
        epsilon=config['reinforcement_learning']['rl_epsilon']
    ).to(device)
    
    # Training loop
    logger.info("Starting RL optimization...")
    n_episodes = args.n_episodes if args.n_episodes else 1000
    
    for episode in range(n_episodes):
        # Reset environment
        obs = env.reset()
        episode_reward = 0
        episode_aesthetic = []
        episode_biomech = []
        
        for step in range(env.max_steps):
            # Get action from agent
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            action, log_prob, value = agent.get_action(obs_tensor)
            
            # Step environment
            next_obs, reward, done, info = env.step(action.squeeze().cpu().numpy())
            
            # Track metrics
            episode_reward += reward
            episode_aesthetic.append(info['aesthetic_score'])
            episode_biomech.append(info['biomechanical_score'])
            
            obs = next_obs
            
            if done:
                break
        
        # Log episode results
        if (episode + 1) % 10 == 0:
            avg_aesthetic = np.mean(episode_aesthetic)
            avg_biomech = np.mean(episode_biomech)
            logger.info(f"Episode {episode+1}/{n_episodes} - "
                       f"Reward: {episode_reward:.2f}, "
                       f"Aesthetic: {avg_aesthetic:.2f}, "
                       f"Biomech: {avg_biomech:.2f}")
            
            # Save checkpoint
            if (episode + 1) % 100 == 0:
                checkpoint_path = Path(args.output_dir) / f'rl_agent_episode_{episode+1}.pt'
                torch.save({
                    'agent_state_dict': agent.state_dict(),
                    'episode': episode + 1,
                    'metrics': {
                        'reward': episode_reward,
                        'aesthetic': avg_aesthetic,
                        'biomech': avg_biomech
                    }
                }, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    logger.info("Optimization completed!")
    
    # Print improvement statistics (as reported in paper)
    print("\n" + "="*50)
    print("OPTIMIZATION RESULTS")
    print("="*50)
    print(f"Expected Aesthetic Improvement: 15% (Paper result)")
    print(f"Expected Biomechanical Improvement: 10% (Paper result)")
    print("="*50)


def few_shot_mode(args, config):
    """
    Few-shot learning mode for adapting to new dance styles
    """
    logger = setup_logger('few_shot', config['logging']['log_dir'])
    logger.info("Starting few-shot learning mode...")
    
    # Set seeds
    set_all_seeds(config['seed'])
    
    # Load base model
    logger.info(f"Loading base model from {args.checkpoint}")
    base_model = DanceAestheticModel(
        spatial_feature_dim=config['model']['spatial_feature_dim'],
        temporal_hidden_dim=config['model']['temporal_hidden_dim'],
        num_lstm_layers=config['model']['num_lstm_layers'],
        dropout=config['model']['dropout'],
        num_aesthetic_classes=config['model']['num_aesthetic_classes'],
        enable_cam=False  # Disable CAM for few-shot
    )
    
    checkpoint = torch.load(args.checkpoint)
    base_model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_model = base_model.to(device)
    
    # Create few-shot dataset
    logger.info("Creating few-shot dataset...")
    dataset = FewShotDanceDataset(
        data_root=config['data_root'],
        dance_styles=args.new_styles.split(',') if args.new_styles else ['new_style'],
        n_way=config['few_shot_learning']['n_way'],
        k_shot=config['few_shot_learning']['k_shot'],
        q_queries=config['few_shot_learning']['q_queries'],
        sequence_length=config['sequence_length'],
        frame_size=tuple(config['frame_size'])
    )
    
    # Create few-shot learner
    logger.info(f"Initializing few-shot learner with method: {config['few_shot_learning']['method']}")
    learner = FewShotLearner(
        base_model=base_model,
        method=config['few_shot_learning']['method'],
        device=device
    )
    
    # Meta-training
    if args.train:
        logger.info("Starting meta-training...")
        n_epochs = config['few_shot_learning']['n_meta_epochs']
        
        for epoch in range(n_epochs):
            # Train epoch
            train_metrics = learner.train_epoch(
                dataset,
                n_tasks=config['few_shot_learning']['n_tasks_per_epoch'],
                meta_lr=config['few_shot_learning']['meta_lr']
            )
            
            # Log metrics
            logger.info(f"Epoch {epoch+1}/{n_epochs} - "
                       f"Loss: {train_metrics['loss']:.4f}, "
                       f"Accuracy: {train_metrics['accuracy']:.4f}")
            
            # Evaluate
            if (epoch + 1) % 10 == 0:
                eval_metrics = learner.evaluate(dataset, n_tasks=50)
                logger.info(f"Evaluation - "
                           f"Accuracy: {eval_metrics['accuracy']:.4f} "
                           f"± {eval_metrics['accuracy_ci_95']:.4f}")
    
    # Adaptation to new style
    if args.adapt:
        logger.info("Adapting to new dance style...")
        
        # Sample support set for adaptation
        support_set, support_labels, _, _ = dataset.sample_task()
        support_set = support_set.to(device)
        support_labels = support_labels.to(device)
        
        # Adapt model
        adapted_model = learner.adapt_to_new_style(
            support_set,
            support_labels.float(),  # Convert to scores
            n_adaptation_steps=config['few_shot_learning']['n_adaptation_steps']
        )
        
        # Save adapted model
        output_path = Path(args.output_dir) / 'adapted_model.pt'
        torch.save(adapted_model.state_dict(), output_path)
        logger.info(f"Adapted model saved to {output_path}")
    
    # Print few-shot results (as reported in paper Table 4)
    print("\n" + "="*50)
    print("FEW-SHOT LEARNING RESULTS")
    print("="*50)
    print("Expected performance with different shot numbers (Paper results):")
    print("1-shot: Correlation 0.60, MSE 0.35")
    print("5-shot: Correlation 0.85, MSE 0.20")
    print("10-shot: Correlation 0.92, MSE 0.15")
    print("="*50)


def inference_mode(args, config):
    """
    Inference mode for processing new dance videos
    """
    logger = setup_logger('inference', config['logging']['log_dir'])
    logger.info("Starting inference mode...")
    
    # Initialize inference engine
    logger.info("Initializing inference engine...")
    engine = InferenceEngine(
        checkpoint_path=args.checkpoint,
        config=config,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Process input
    if args.video:
        logger.info(f"Processing video: {args.video}")
        results = engine.process_video(
            video_path=args.video,
            output_dir=args.output_dir,
            generate_cam=args.generate_cam,
            optimize=args.optimize
        )
        
        # Print results
        print("\n" + "="*50)
        print("INFERENCE RESULTS")
        print("="*50)
        print(f"Aesthetic Score: {results['aesthetic_score']:.2f}/10")
        print(f"Biomechanical Efficiency: {results['biomechanical_score']:.2f}")
        print(f"Fluidity: {results.get('fluidity', 'N/A')}")
        print(f"Expressiveness: {results.get('expressiveness', 'N/A')}")
        print(f"Precision: {results.get('precision', 'N/A')}")
        
        if args.optimize:
            print("\nOptimization Results:")
            print(f"Aesthetic Improvement: {results['optimization']['aesthetic_improvement']:.1%}")
            print(f"Biomechanical Improvement: {results['optimization']['biomech_improvement']:.1%}")
            print(f"Optimized video saved to: {results['optimization']['output_path']}")
        
        if args.generate_cam:
            print(f"\nCAM visualizations saved to: {results['cam_path']}")
        
        print("="*50)
        
    elif args.batch_dir:
        logger.info(f"Processing batch directory: {args.batch_dir}")
        results = engine.process_batch(
            batch_dir=args.batch_dir,
            output_dir=args.output_dir
        )
        
        # Save batch results
        results_path = Path(args.output_dir) / 'batch_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Batch processing completed! Results saved to {results_path}")
        
    else:
        logger.error("Please provide either --video or --batch_dir for inference")
        sys.exit(1)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Dance Movement Analysis System')
    
    # Mode selection
    parser.add_argument('mode', choices=['train', 'evaluate', 'optimize', 'few_shot', 'inference'],
                       help='Operation mode')
    
    # Common arguments
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory')
    
    # Training arguments
    parser.add_argument('--resume', type=str,
                       help='Resume training from checkpoint')
    
    # Optimization arguments
    parser.add_argument('--n_episodes', type=int,
                       help='Number of RL episodes')
    
    # Few-shot arguments
    parser.add_argument('--new_styles', type=str,
                       help='Comma-separated list of new dance styles')
    parser.add_argument('--train', action='store_true',
                       help='Run meta-training')
    parser.add_argument('--adapt', action='store_true',
                       help='Adapt to new style')
    
    # Inference arguments
    parser.add_argument('--video', type=str,
                       help='Path to input video')
    parser.add_argument('--batch_dir', type=str,
                       help='Directory containing videos for batch processing')
    parser.add_argument('--generate_cam', action='store_true',
                       help='Generate CAM visualizations')
    parser.add_argument('--optimize', action='store_true',
                       help='Apply RL optimization')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Execute selected mode
    if args.mode == 'train':
        train_mode(args, config)
    elif args.mode == 'evaluate':
        evaluate_mode(args, config)
    elif args.mode == 'optimize':
        optimize_mode(args, config)
    elif args.mode == 'few_shot':
        few_shot_mode(args, config)
    elif args.mode == 'inference':
        inference_mode(args, config)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == '__main__':
    main()
