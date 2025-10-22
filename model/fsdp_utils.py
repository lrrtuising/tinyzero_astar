"""
PyTorch FSDP utilities for distributed training.
"""
import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
    ShardedStateDictConfig,
    CPUOffload,
    BackwardPrefetch,
    ShardingStrategy,
    MixedPrecision
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
    ModuleWrapPolicy
)
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional, Dict, Any, Union
import logging

logger = logging.getLogger(__name__)


def setup_distributed() -> Dict[str, int]:
    """
    Initialize distributed training.
    
    Returns:
        Dictionary with rank, world_size, local_rank
    """
    # Initialize process group
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    logger.info(f"Distributed training initialized. Rank: {rank}, World size: {world_size}, Local rank: {local_rank}")
    
    return {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "device": device
    }


def setup_fsdp(
    model: torch.nn.Module,
    config,
    auto_wrap_policy: Optional[ModuleWrapPolicy] = None
) -> FSDP:
    """
    Wrap model with FSDP for distributed training.
    
    Args:
        model: Model to wrap
        config: Configuration object with FSDP settings
        auto_wrap_policy: Auto wrap policy for transformer layers
        
    Returns:
        FSDP-wrapped model
    """
    # Set up mixed precision
    if config.fsdp.mixed_precision == "bf16":
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16
        )
    elif config.fsdp.mixed_precision == "fp16":
        mixed_precision = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16
        )
    else:
        mixed_precision = None
    
    # Set up CPU offload
    cpu_offload = CPUOffload(offload_params=config.fsdp.cpu_offload)
    
    # Set up sharding strategy
    if config.fsdp.sharding_strategy == "FULL_SHARD":
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif config.fsdp.sharding_strategy == "SHARD_GRAD_OP":
        sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
    elif config.fsdp.sharding_strategy == "NO_SHARD":
        sharding_strategy = ShardingStrategy.NO_SHARD
    else:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    
    # Set up backward prefetch
    if config.fsdp.backward_prefetch == "BACKWARD_PRE":
        backward_prefetch = BackwardPrefetch.BACKWARD_PRE
    elif config.fsdp.backward_prefetch == "BACKWARD_POST":
        backward_prefetch = BackwardPrefetch.BACKWARD_POST
    else:
        backward_prefetch = BackwardPrefetch.BACKWARD_PRE
    
    # Default auto wrap policy for transformer models
    if auto_wrap_policy is None:
        # Try to detect transformer layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            # Qwen2.5 style
            transformer_layer_name = "layers"
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            # GPT style
            transformer_layer_name = "h"
        else:
            # Fallback to module name matching
            transformer_layer_name = "layers"
        
        auto_wrap_policy = transformer_auto_wrap_policy(
            transformer_layer_cls={
                getattr(model, transformer_layer_name, None).__class__ if hasattr(model, transformer_layer_name) else None
            }
        )
    
    # Wrap model with FSDP
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        cpu_offload=cpu_offload,
        sharding_strategy=sharding_strategy,
        backward_prefetch=backward_prefetch,
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        forward_prefetch=True
    )
    
    logger.info(f"Model wrapped with FSDP. Sharding strategy: {sharding_strategy}")
    
    return fsdp_model


def save_checkpoint(
    model: Union[FSDP, torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    iteration: int,
    path: str,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    additional_state: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save FSDP-compatible checkpoint.
    
    Args:
        model: FSDP model or regular model
        optimizer: Optimizer
        iteration: Current iteration
        path: Path to save checkpoint
        scheduler: Learning rate scheduler (optional)
        additional_state: Additional state to save (optional)
    """
    if isinstance(model, FSDP):
        # Save FSDP state dict
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            model_state_dict = model.state_dict()
    else:
        # Regular model state dict
        model_state_dict = model.state_dict()
    
    # Prepare checkpoint
    checkpoint = {
        "iteration": iteration,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
    }
    
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    if additional_state is not None:
        checkpoint.update(additional_state)
    
    # Save checkpoint (only rank 0 saves)
    if dist.get_rank() == 0:
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    # Synchronize all processes
    dist.barrier()


def load_checkpoint(
    model: Union[FSDP, torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    path: str,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Load FSDP-compatible checkpoint.
    
    Args:
        model: FSDP model or regular model
        optimizer: Optimizer
        path: Path to checkpoint
        scheduler: Learning rate scheduler (optional)
        device: Device to load on
        
    Returns:
        Dictionary with loaded state
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    checkpoint = torch.load(path, map_location=device)
    
    if isinstance(model, FSDP):
        # Load FSDP state dict
        load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, load_policy):
            model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Regular model state dict
        model.load_state_dict(checkpoint["model_state_dict"])
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # Load scheduler state if available
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    logger.info(f"Checkpoint loaded from {path}")
    
    return checkpoint


def cleanup_distributed() -> None:
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed training cleaned up")


def get_fsdp_info(model: FSDP) -> Dict[str, Any]:
    """
    Get information about FSDP model.
    
    Args:
        model: FSDP model
        
    Returns:
        Dictionary with FSDP information
    """
    return {
        "sharding_strategy": model.sharding_strategy,
        "mixed_precision": model.mixed_precision,
        "cpu_offload": model.cpu_offload,
        "device_id": model.device_id,
        "sync_module_states": model.sync_module_states,
        "forward_prefetch": model.forward_prefetch
    }


def broadcast_tensor(tensor: torch.Tensor, src_rank: int = 0) -> torch.Tensor:
    """
    Broadcast tensor from source rank to all ranks.
    
    Args:
        tensor: Tensor to broadcast
        src_rank: Source rank
        
    Returns:
        Broadcasted tensor
    """
    dist.broadcast(tensor, src=src_rank)
    return tensor


def gather_tensors(tensor: torch.Tensor, dst_rank: int = 0) -> Optional[torch.Tensor]:
    """
    Gather tensors from all ranks to destination rank.
    
    Args:
        tensor: Tensor to gather
        dst_rank: Destination rank
        
    Returns:
        Gathered tensor (only on dst_rank), None otherwise
    """
    if dist.get_rank() == dst_rank:
        gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.gather(tensor, gathered, dst=dst_rank)
        return torch.cat(gathered, dim=0)
    else:
        dist.gather(tensor, dst=dst_rank)
        return None


def all_reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    All-reduce tensor across all ranks.
    
    Args:
        tensor: Tensor to all-reduce
        
    Returns:
        All-reduced tensor
    """
    dist.all_reduce(tensor)
    tensor.div_(dist.get_world_size())
    return tensor


def is_main_process() -> bool:
    """Check if current process is main process (rank 0)."""
    return dist.get_rank() == 0


def get_world_size() -> int:
    """Get world size."""
    return dist.get_world_size() if dist.is_initialized() else 1


def get_rank() -> int:
    """Get current rank."""
    return dist.get_rank() if dist.is_initialized() else 0


if __name__ == "__main__":
    # Test FSDP utilities (requires distributed setup)
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("FSDP utilities loaded successfully")
    print("Note: These utilities require distributed training setup to test fully")
