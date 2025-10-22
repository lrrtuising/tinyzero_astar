"""
Configuration for TinyZero A*PO implementation.
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Model configuration."""
    model_name: str = "Qwen/Qwen2.5-3B"
    max_seq_length: int = 512
    vocab_size: int = 152064  # Qwen2.5 vocab size
    hidden_size: int = 3072
    num_attention_heads: int = 24
    num_hidden_layers: int = 28


@dataclass
class TrainingConfig:
    """Training configuration."""
    num_iterations: int = 100
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    value_model_lr: float = 3e-5
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    max_rollouts_per_iteration: Optional[int] = None


@dataclass
class AStarConfig:
    """A* search configuration."""
    num_rollouts_per_problem: int = 4
    beam_width: int = 8
    max_search_depth: int = 200
    exploration_weight: float = 0.1
    max_generation_length: int = 300
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50


@dataclass
class ValueModelConfig:
    """Value model configuration."""
    hidden_size: int = 512
    num_layers: int = 2
    dropout: float = 0.1
    update_frequency: int = 5  # Update every N policy updates
    value_loss_weight: float = 0.5


@dataclass
class FSDPConfig:
    """FSDP configuration."""
    sharding_strategy: str = "FULL_SHARD"
    mixed_precision: str = "bf16"  # or "fp16"
    cpu_offload: bool = False
    auto_wrap_policy: str = "transformer_layer"
    backward_prefetch: str = "BACKWARD_PRE"


@dataclass
class DataConfig:
    """Data configuration."""
    num_train_problems: int = 1000
    num_eval_problems: int = 200
    min_digits: int = 2
    max_digits: int = 3
    seed: int = 42
    data_dir: str = "data"


@dataclass
class LoggingConfig:
    """Logging configuration."""
    use_wandb: bool = False
    wandb_project: str = "tinyzero-astar"
    wandb_entity: Optional[str] = None
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    eval_frequency: int = 10
    save_frequency: int = 20
    log_level: str = "INFO"


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    astar: AStarConfig = AStarConfig()
    value_model: ValueModelConfig = ValueModelConfig()
    fsdp: FSDPConfig = FSDPConfig()
    data: DataConfig = DataConfig()
    logging: LoggingConfig = LoggingConfig()
    
    # Distributed training
    local_rank: int = 0
    world_size: int = 1
    rank: int = 0
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Set distributed training parameters from environment
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.rank = int(os.environ.get("RANK", 0))
        
        # Create directories
        os.makedirs(self.logging.log_dir, exist_ok=True)
        os.makedirs(self.logging.checkpoint_dir, exist_ok=True)
        os.makedirs(self.data.data_dir, exist_ok=True)


def get_local_config() -> Config:
    """Get configuration for local testing (smaller scale)."""
    config = Config()
    
    # Smaller scale for local testing
    config.model.model_name = "Qwen/Qwen2.5-0.5B"
    config.model.hidden_size = 1024
    config.model.num_attention_heads = 16
    config.model.num_hidden_layers = 24
    config.training.num_iterations = 1
    config.training.batch_size = 1
    config.training.gradient_accumulation_steps = 1
    config.training.max_rollouts_per_iteration = 4
    config.astar.num_rollouts_per_problem = 2
    config.astar.beam_width = 4
    config.logging.eval_frequency = 1000
    config.logging.save_frequency = 1
    config.data.num_train_problems = 100
    config.data.num_eval_problems = 20
    config.data.min_digits = 2
    config.data.max_digits = 2
    
    return config


def get_training_config() -> Config:
    """Get configuration for full training."""
    return Config()


if __name__ == "__main__":
    # Test configuration
    config = get_local_config()
    print("Local config created successfully")
    print(f"Model: {config.model.model_name}")
    print(f"Training iterations: {config.training.num_iterations}")
    print(f"Data problems: {config.data.num_train_problems}")
