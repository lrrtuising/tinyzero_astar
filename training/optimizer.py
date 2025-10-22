"""
Optimizer configuration and utilities for TinyZero A*PO.
"""
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    ConstantLR
)
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


def create_optimizer(
    model: torch.nn.Module,
    config,
    parameter_groups: Optional[List[Dict]] = None
) -> torch.optim.Optimizer:
    """
    Create optimizer for policy model.
    
    Args:
        model: Model to optimize
        config: Configuration object
        parameter_groups: Optional parameter groups with different learning rates
        
    Returns:
        Optimizer
    """
    if parameter_groups is None:
        # Use all parameters with same learning rate
        parameters = model.parameters()
        lr = config.training.learning_rate
    else:
        # Use parameter groups
        parameters = parameter_groups
        lr = config.training.learning_rate  # Base learning rate
    
    optimizer = optim.AdamW(
        parameters,
        lr=lr,
        betas=(config.training.beta1, config.training.beta2),
        eps=config.training.epsilon,
        weight_decay=config.training.weight_decay
    )
    
    logger.info(f"Created optimizer with lr={lr}, weight_decay={config.training.weight_decay}")
    return optimizer


def create_value_optimizer(
    value_model: torch.nn.Module,
    config
) -> torch.optim.Optimizer:
    """
    Create optimizer for value model.
    
    Args:
        value_model: Value model to optimize
        config: Configuration object
        
    Returns:
        Value model optimizer
    """
    optimizer = optim.AdamW(
        value_model.parameters(),
        lr=config.training.value_model_lr,
        betas=(config.training.beta1, config.training.beta2),
        eps=config.training.epsilon,
        weight_decay=config.training.weight_decay
    )
    
    logger.info(f"Created value optimizer with lr={config.training.value_model_lr}")
    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config,
    total_steps: Optional[int] = None
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer
        config: Configuration object
        total_steps: Total training steps (for cosine annealing)
        
    Returns:
        Learning rate scheduler
    """
    if total_steps is None:
        total_steps = config.training.num_iterations
    
    # Warmup + cosine annealing
    if config.training.warmup_steps > 0:
        # Linear warmup
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=config.training.warmup_steps
        )
        
        # Cosine annealing after warmup
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - config.training.warmup_steps,
            eta_min=config.training.learning_rate * 0.1
        )
        
        # Sequential scheduler
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[config.training.warmup_steps]
        )
    else:
        # Cosine annealing only
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=config.training.learning_rate * 0.1
        )
    
    logger.info(f"Created scheduler with warmup_steps={config.training.warmup_steps}, total_steps={total_steps}")
    return scheduler


def create_parameter_groups(
    model: torch.nn.Module,
    config,
    lr_multipliers: Optional[Dict[str, float]] = None
) -> List[Dict]:
    """
    Create parameter groups with different learning rates.
    
    Args:
        model: Model
        config: Configuration object
        lr_multipliers: Learning rate multipliers for different parameter types
        
    Returns:
        List of parameter group dictionaries
    """
    if lr_multipliers is None:
        lr_multipliers = {
            "embeddings": 0.1,
            "layers": 1.0,
            "lm_head": 1.0
        }
    
    parameter_groups = []
    base_lr = config.training.learning_rate
    
    # Group parameters by type
    embeddings = []
    layers = []
    lm_head = []
    other = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if "embed" in name.lower():
            embeddings.append(param)
        elif "layer" in name.lower() or "transformer" in name.lower():
            layers.append(param)
        elif "lm_head" in name.lower() or "head" in name.lower():
            lm_head.append(param)
        else:
            other.append(param)
    
    # Create parameter groups
    if embeddings:
        parameter_groups.append({
            "params": embeddings,
            "lr": base_lr * lr_multipliers.get("embeddings", 1.0),
            "name": "embeddings"
        })
    
    if layers:
        parameter_groups.append({
            "params": layers,
            "lr": base_lr * lr_multipliers.get("layers", 1.0),
            "name": "layers"
        })
    
    if lm_head:
        parameter_groups.append({
            "params": lm_head,
            "lr": base_lr * lr_multipliers.get("lm_head", 1.0),
            "name": "lm_head"
        })
    
    if other:
        parameter_groups.append({
            "params": other,
            "lr": base_lr,
            "name": "other"
        })
    
    logger.info(f"Created {len(parameter_groups)} parameter groups")
    for group in parameter_groups:
        logger.info(f"  {group['name']}: {len(group['params'])} params, lr={group['lr']}")
    
    return parameter_groups


def clip_gradients(
    model: torch.nn.Module,
    max_norm: float,
    norm_type: float = 2.0
) -> float:
    """
    Clip gradients to prevent exploding gradients.
    
    Args:
        model: Model
        max_norm: Maximum gradient norm
        norm_type: Type of norm to use
        
    Returns:
        Total gradient norm before clipping
    """
    total_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=max_norm,
        norm_type=norm_type
    )
    
    return total_norm.item()


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    Get current learning rate from optimizer.
    
    Args:
        optimizer: Optimizer
        
    Returns:
        Current learning rate
    """
    return optimizer.param_groups[0]['lr']


def get_all_lrs(optimizer: torch.optim.Optimizer) -> Dict[str, float]:
    """
    Get learning rates for all parameter groups.
    
    Args:
        optimizer: Optimizer
        
    Returns:
        Dictionary with learning rates for each group
    """
    lrs = {}
    for i, group in enumerate(optimizer.param_groups):
        name = group.get('name', f'group_{i}')
        lrs[name] = group['lr']
    
    return lrs


def create_optimizer_and_scheduler(
    model: torch.nn.Module,
    config,
    parameter_groups: Optional[List[Dict]] = None,
    total_steps: Optional[int] = None
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """
    Create both optimizer and scheduler.
    
    Args:
        model: Model to optimize
        config: Configuration object
        parameter_groups: Optional parameter groups
        total_steps: Total training steps
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    optimizer = create_optimizer(model, config, parameter_groups)
    scheduler = create_scheduler(optimizer, config, total_steps)
    
    return optimizer, scheduler


def create_value_optimizer_and_scheduler(
    value_model: torch.nn.Module,
    config,
    total_steps: Optional[int] = None
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """
    Create optimizer and scheduler for value model.
    
    Args:
        value_model: Value model
        config: Configuration object
        total_steps: Total training steps
        
    Returns:
        Tuple of (value_optimizer, value_scheduler)
    """
    optimizer = create_value_optimizer(value_model, config)
    scheduler = create_scheduler(optimizer, config, total_steps)
    
    return optimizer, scheduler


def save_optimizer_state(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    path: str
) -> None:
    """
    Save optimizer and scheduler state.
    
    Args:
        optimizer: Optimizer
        scheduler: Scheduler
        path: Path to save state
    """
    state = {
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }
    torch.save(state, path)
    logger.info(f"Optimizer state saved to {path}")


def load_optimizer_state(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    path: str
) -> None:
    """
    Load optimizer and scheduler state.
    
    Args:
        optimizer: Optimizer
        scheduler: Scheduler
        path: Path to load state from
    """
    state = torch.load(path)
    optimizer.load_state_dict(state['optimizer_state_dict'])
    scheduler.load_state_dict(state['scheduler_state_dict'])
    logger.info(f"Optimizer state loaded from {path}")


def get_optimizer_info(optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
    """
    Get information about optimizer.
    
    Args:
        optimizer: Optimizer
        
    Returns:
        Dictionary with optimizer information
    """
    info = {
        'num_groups': len(optimizer.param_groups),
        'learning_rates': get_all_lrs(optimizer),
        'optimizer_type': type(optimizer).__name__
    }
    
    # Add parameter group details
    for i, group in enumerate(optimizer.param_groups):
        group_info = {
            'lr': group['lr'],
            'weight_decay': group.get('weight_decay', 0),
            'num_params': len(group['params'])
        }
        info[f'group_{i}'] = group_info
    
    return info


if __name__ == "__main__":
    # Test optimizer utilities
    print("Testing optimizer utilities...")
    
    # Create a simple test model
    import torch.nn as nn
    
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
        
        def forward(self, x):
            return self.linear(x)
    
    model = TestModel()
    
    # Test parameter groups
    from config import get_local_config
    config = get_local_config()
    
    parameter_groups = create_parameter_groups(model, config)
    print(f"Created {len(parameter_groups)} parameter groups")
    
    # Test optimizer creation
    optimizer = create_optimizer(model, config, parameter_groups)
    print(f"Created optimizer: {type(optimizer).__name__}")
    
    # Test scheduler
    scheduler = create_scheduler(optimizer, config, total_steps=100)
    print(f"Created scheduler: {type(scheduler).__name__}")
    
    # Test learning rate
    lr = get_lr(optimizer)
    print(f"Current learning rate: {lr}")
    
    print("Optimizer utilities tests completed")
