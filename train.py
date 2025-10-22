#!/usr/bin/env python3
"""
Main training script for TinyZero A*PO.
"""
import os
import sys
import argparse
import logging
import time
import json
from typing import Dict, Any, Optional

import torch
import torch.distributed as dist

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_local_config, get_training_config
from data.generation import create_dataset, load_problems
from model.model import load_base_model, setup_model_for_training
from model.fsdp_utils import setup_distributed, setup_fsdp, save_checkpoint, load_checkpoint, cleanup_distributed
from rollout.value_model import ValueModel, train_value_model
from rollout.generator import RolloutGenerator
from rollout.cache import save_rollouts, load_rollouts, cache_exists, get_rollout_cache_path
from training.astar_po import AStarPO
from training.optimizer import (
    create_optimizer_and_scheduler,
    create_value_optimizer_and_scheduler,
    get_lr
)
from training.reward import compute_rewards_batch, compute_advantages, normalize_advantages
from eval.evaluate import evaluate, run_distributed_evaluation

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_training(config, use_distributed: bool = False):
    """
    Setup training environment.
    
    Args:
        config: Configuration object
        use_distributed: Whether to use distributed training
        
    Returns:
        Tuple of (model, value_model, tokenizer, train_problems, eval_problems)
    """
    # Setup distributed training if requested
    if use_distributed:
        distributed_info = setup_distributed()
        config.rank = distributed_info["rank"]
        config.world_size = distributed_info["world_size"]
        config.local_rank = distributed_info["local_rank"]
        device = distributed_info["device"]
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config.rank = 0
        config.world_size = 1
        config.local_rank = 0
    
    logger.info(f"Training on device: {device}")
    
    # Load model and tokenizer
    model, tokenizer = load_base_model(
        config.model.model_name,
        device=device,
        torch_dtype=torch.bfloat16
    )
    
    # Setup model for training
    model = setup_model_for_training(model)
    
    # Create value model
    value_model = ValueModel(
        base_model=model,
        hidden_size=config.value_model.hidden_size,
        num_layers=config.value_model.num_layers,
        dropout=config.value_model.dropout
    )
    
    # Setup FSDP if distributed
    if use_distributed and config.world_size > 1:
        model = setup_fsdp(model, config)
        value_model = setup_fsdp(value_model, config)
    
    # Load or create datasets
    train_data_path = os.path.join(config.data.data_dir, "train_problems.json")
    eval_data_path = os.path.join(config.data.data_dir, "eval_problems.json")
    
    if os.path.exists(train_data_path) and os.path.exists(eval_data_path):
        logger.info("Loading existing datasets")
        train_problems = load_problems(train_data_path)
        eval_problems = load_problems(eval_data_path)
    else:
        logger.info("Creating new datasets")
        train_problems, eval_problems = create_dataset(config)
    
    logger.info(f"Loaded {len(train_problems)} train problems, {len(eval_problems)} eval problems")
    
    return model, value_model, tokenizer, train_problems, eval_problems


def train_iteration(
    iteration: int,
    model: torch.nn.Module,
    value_model: torch.nn.Module,
    tokenizer: Any,
    train_problems: list,
    astar_po: AStarPO,
    policy_optimizer: torch.optim.Optimizer,
    value_optimizer: torch.optim.Optimizer,
    policy_scheduler: torch.optim.lr_scheduler._LRScheduler,
    value_scheduler: torch.optim.lr_scheduler._LRScheduler,
    config
) -> Dict[str, float]:
    """
    Single training iteration.
    
    Args:
        iteration: Current iteration number
        model: Policy model
        value_model: Value model
        tokenizer: Tokenizer
        train_problems: Training problems
        astar_po: A*PO trainer
        policy_optimizer: Policy optimizer
        value_optimizer: Value optimizer
        policy_scheduler: Policy scheduler
        value_scheduler: Value scheduler
        config: Configuration
        
    Returns:
        Dictionary with training metrics
    """
    logger.info(f"Starting iteration {iteration}")
    
    # Create rollout generator
    generator = RolloutGenerator(model, value_model, tokenizer, config)
    
    # Check if rollouts are cached
    cache_path = get_rollout_cache_path(iteration)
    
    if cache_exists(iteration):
        logger.info(f"Loading cached rollouts from {cache_path}")
        rollouts = load_rollouts(cache_path)
    else:
        # Generate rollouts
        logger.info("Generating rollouts...")
        rollouts = generator.generate_rollouts(
            train_problems,
            num_rollouts_per_problem=config.astar.num_rollouts_per_problem
        )
        
        # Save rollouts to cache
        logger.info(f"Saving rollouts to cache: {cache_path}")
        save_rollouts(rollouts, cache_path)

    max_rollouts = getattr(config.training, "max_rollouts_per_iteration", None)
    if max_rollouts is not None and len(rollouts) > max_rollouts:
        logger.info(f"Limiting rollouts to first {max_rollouts} entries (from {len(rollouts)})")
        rollouts = rollouts[:max_rollouts]
    
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    # Extract ground truth answers
    if rollouts and "ground_truth_answer" in rollouts[0]:
        ground_truths = [r.get("ground_truth_answer") for r in rollouts]
    else:
        ground_truths = []
        for problem in train_problems:
            if hasattr(problem, 'answer'):
                answer = problem.answer
            else:
                answer = problem["answer"]
            
            for _ in range(config.astar.num_rollouts_per_problem):
                ground_truths.append(answer)
        ground_truths = ground_truths[:len(rollouts)]

    # Compute rewards
    logger.info("Computing rewards...")
    rewards = compute_rewards_batch(rollouts, ground_truths)
    rewards = rewards.to(device=device, dtype=dtype)
    
    # Compute advantages
    logger.info("Computing advantages...")
    advantages = astar_po.compute_advantages(rollouts, rewards)
    advantages = advantages.to(device=device, dtype=dtype)
    
    # Train policy
    logger.info("Training policy...")
    policy_metrics = astar_po.train_step(
        rollouts, rewards, advantages, policy_optimizer, value_optimizer
    )
    
    # Train value model (less frequently)
    value_metrics = {}
    if iteration % config.value_model.update_frequency == 0:
        logger.info("Training value model...")
        value_metrics = train_value_model(
            value_model, rollouts, rewards, value_optimizer, config
        )
    
    # Update schedulers
    policy_scheduler.step()
    value_scheduler.step()
    
    # Get learning rates
    policy_lr = get_lr(policy_optimizer)
    value_lr = get_lr(value_optimizer)
    
    # Compile metrics
    metrics = {
        "iteration": iteration,
        "policy_lr": policy_lr,
        "value_lr": value_lr,
        **policy_metrics,
        **value_metrics
    }
    
    logger.info(f"Iteration {iteration} completed: policy_loss={policy_metrics['policy_loss']:.4f}")
    
    return metrics


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train TinyZero A*PO")
    parser.add_argument("--config", type=str, default="local", choices=["local", "training"],
                       help="Configuration to use")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--distributed", action="store_true",
                       help="Use distributed training")
    parser.add_argument("--eval-only", action="store_true",
                       help="Only run evaluation")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to checkpoint for evaluation")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config == "local":
        config = get_local_config()
    else:
        config = get_training_config()
    
    logger.info(f"Using configuration: {args.config}")
    logger.info(f"Model: {config.model.model_name}")
    logger.info(f"Training iterations: {config.training.num_iterations}")
    
    try:
        # Setup training
        model, value_model, tokenizer, train_problems, eval_problems = setup_training(
            config, use_distributed=args.distributed
        )
        
        if args.eval_only:
            # Evaluation only
            if args.checkpoint:
                logger.info(f"Loading checkpoint from {args.checkpoint}")
                load_checkpoint(model, None, args.checkpoint)
                if value_model is not None:
                    load_checkpoint(value_model, None, args.checkpoint)
            
            # Run evaluation
            logger.info("Running evaluation...")
            eval_results = run_distributed_evaluation(
                model, value_model, tokenizer, eval_problems, config
            )
            
            logger.info("Evaluation results:")
            for key, value in eval_results.items():
                logger.info(f"  {key}: {value:.4f}")
            
            return
        
        # Create optimizers and schedulers
        policy_optimizer, policy_scheduler = create_optimizer_and_scheduler(
            model, config, total_steps=config.training.num_iterations
        )
        
        value_optimizer, value_scheduler = create_value_optimizer_and_scheduler(
            value_model, config, total_steps=config.training.num_iterations
        )
        
        # Create A*PO trainer
        astar_po = AStarPO(model, value_model, config)
        
        # Resume from checkpoint if specified
        start_iteration = 0
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            checkpoint = load_checkpoint(model, policy_optimizer, args.resume, policy_scheduler)
            if value_model is not None:
                load_checkpoint(value_model, value_optimizer, args.resume, value_scheduler)
            start_iteration = checkpoint.get("iteration", 0) + 1
        
        # Training loop
        logger.info("Starting training loop")
        
        for iteration in range(start_iteration, config.training.num_iterations):
            try:
                # Training iteration
                metrics = train_iteration(
                    iteration, model, value_model, tokenizer, train_problems,
                    astar_po, policy_optimizer, value_optimizer,
                    policy_scheduler, value_scheduler, config
                )
                
                # Log metrics
                logger.info(f"Iteration {iteration} metrics:")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        logger.info(f"  {key}: {value:.4f}")
                    else:
                        logger.info(f"  {key}: {value}")
                
                # Evaluation
                if (getattr(config.logging, "eval_frequency", 0) and
                        # iteration > 0 and
                        iteration % config.logging.eval_frequency == 0):
                    logger.info("Running evaluation...")
                    eval_results = run_distributed_evaluation(
                        model, value_model, tokenizer, eval_problems, config
                    )
                    
                    logger.info("Evaluation results:")
                    for key, value in eval_results.items():
                        logger.info(f"  {key}: {value:.4f}")
                
                # Save checkpoint
                if (getattr(config.logging, "save_frequency", 0) and
                        iteration > 0 and
                        iteration % config.logging.save_frequency == 0):
                    checkpoint_path = os.path.join(
                        config.logging.checkpoint_dir,
                        f"checkpoint_iter_{iteration}.pt"
                    )
                    save_checkpoint(
                        model, policy_optimizer, iteration, checkpoint_path,
                        policy_scheduler, {"value_model": value_model}
                    )
                    logger.info(f"Checkpoint saved: {checkpoint_path}")
                
            except KeyboardInterrupt:
                logger.info("Training interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in iteration {iteration}: {e}")
                raise
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        # Cleanup distributed training
        if args.distributed:
            cleanup_distributed()


if __name__ == "__main__":
    main()
