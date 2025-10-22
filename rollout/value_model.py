"""
Value model for A* heuristic estimation in TinyZero A*PO.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class ValueModel(nn.Module):
    """
    Value model that estimates the expected future reward for partial generations.
    This serves as the heuristic function h(n) in A* search.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        share_embeddings: bool = True
    ):
        """
        Initialize value model.
        
        Args:
            base_model: Base language model
            hidden_size: Hidden size for value head
            num_layers: Number of layers in value head
            dropout: Dropout rate
            share_embeddings: Whether to share embeddings with base model
        """
        super().__init__()
        
        self.base_model = base_model
        self.share_embeddings = share_embeddings
        
        # Get model dimensions
        if hasattr(base_model, 'config'):
            self.hidden_size = base_model.config.hidden_size
        elif hasattr(base_model, 'model') and hasattr(base_model.model, 'embed_tokens'):
            # Try to get hidden size from embedding layer
            self.hidden_size = base_model.model.embed_tokens.embedding_dim
        else:
            # Fallback
            self.hidden_size = 3072  # Qwen2.5-3B default
        
        # Value head architecture
        value_layers = []
        current_size = self.hidden_size
        
        for i in range(num_layers):
            next_size = hidden_size if i < num_layers - 1 else 1
            value_layers.append(nn.Linear(current_size, next_size))
            
            if i < num_layers - 1:
                value_layers.append(nn.ReLU())
                value_layers.append(nn.Dropout(dropout))
            
            current_size = next_size
        
        self.value_head = nn.Sequential(*value_layers)
        
        # Move to same device and dtype as base model
        device = next(base_model.parameters()).device
        dtype = next(base_model.parameters()).dtype
        self.to(device=device, dtype=dtype)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"Value model initialized with {sum(p.numel() for p in self.parameters()):,} parameters")
    
    def _init_weights(self):
        """Initialize weights of the value head."""
        for module in self.value_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of value model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_hidden_states: Whether to return hidden states
            
        Returns:
            Value estimates [batch_size, seq_len] or (values, hidden_states)
        """
        # Get hidden states from base model
        with torch.no_grad() if not self.training else torch.enable_grad():
            if hasattr(self.base_model, 'model'):
                # Qwen2.5 style
                outputs = self.base_model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                hidden_states = outputs.hidden_states[-1]  # Last layer
            else:
                # Fallback
                outputs = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                hidden_states = outputs.hidden_states[-1]
        
        # Compute value estimates for each position
        values = self.value_head(hidden_states).squeeze(-1)  # [batch_size, seq_len]
        
        if return_hidden_states:
            return values, hidden_states
        else:
            return values
    
    def get_value_estimates(
        self,
        states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get value estimates for batch of states.
        Used as heuristic h(n) in A* search.
        
        Args:
            states: Token sequences [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Value estimates [batch_size, seq_len]
        """
        return self.forward(states, attention_mask)
    
    def get_final_values(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get final value estimates (at the last valid position).
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Final value estimates [batch_size]
        """
        values = self.forward(input_ids, attention_mask)
        
        if attention_mask is not None:
            # Get values at last valid position
            seq_lengths = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
            final_values = values.gather(1, seq_lengths.unsqueeze(1)).squeeze(1)
        else:
            # Use last position
            final_values = values[:, -1]
        
        return final_values


def train_value_model(
    value_model: ValueModel,
    rollouts: List[Dict],
    rewards: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    config,
    gamma: float = 0.99
) -> Dict[str, float]:
    """
    Train value model using rollout data.
    
    Args:
        value_model: Value model to train
        rollouts: List of rollout dictionaries
        rewards: Rewards for each rollout [num_rollouts]
        optimizer: Optimizer for value model
        config: Configuration object
        gamma: Discount factor for returns
        
    Returns:
        Dictionary with training metrics
    """
    value_model.train()
    
    # Prepare training data
    input_ids_list = []
    attention_masks_list = []
    returns_list = []
    
    for i, rollout in enumerate(rollouts):
        tokens = rollout["tokens"]
        log_probs = rollout["log_probs"]
        reward = rewards[i].item()
        
        # Compute returns (Monte Carlo for now)
        returns = [reward] * len(tokens)
        
        # Convert to tensors
        input_ids = torch.tensor(tokens, dtype=torch.long, device=value_model.base_model.device)
        attention_mask = torch.ones_like(input_ids)
        
        input_ids_list.append(input_ids)
        attention_masks_list.append(attention_mask)
        returns_list.extend(returns)
    
    # Pad sequences
    max_len = max(len(seq) for seq in input_ids_list)
    
    padded_input_ids = []
    padded_attention_masks = []
    padded_returns = []
    
    for i, (input_ids, attention_mask) in enumerate(zip(input_ids_list, attention_masks_list)):
        seq_len = len(input_ids)
        pad_len = max_len - seq_len
        
        if pad_len > 0:
            pad_token_id = 0  # Assuming 0 is pad token
            input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_token_id, device=input_ids.device)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, device=attention_mask.device)])
        
        padded_input_ids.append(input_ids)
        padded_attention_masks.append(attention_mask)
        
        # Pad returns
        rollout_returns = returns_list[i*seq_len:(i+1)*seq_len]
        rollout_returns.extend([0.0] * pad_len)  # 0 for padded positions
        padded_returns.extend(rollout_returns)
    
    # Stack into batches
    input_ids_batch = torch.stack(padded_input_ids)
    attention_masks_batch = torch.stack(padded_attention_masks)
    target_dtype = next(value_model.parameters()).dtype
    returns_batch = torch.tensor(
        padded_returns,
        device=value_model.base_model.device,
        dtype=target_dtype
    ).view(input_ids_batch.shape)
    
    # Forward pass
    predicted_values = value_model.get_value_estimates(input_ids_batch, attention_masks_batch)
    
    # Compute loss (MSE between predicted and actual returns)
    # Only compute loss on non-padded positions
    valid_mask = attention_masks_batch.bool()
    loss = F.mse_loss(
        predicted_values[valid_mask],
        returns_batch[valid_mask]
    )
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(value_model.parameters(), config.training.max_grad_norm)
    
    optimizer.step()
    
    # Compute metrics
    with torch.no_grad():
        mse = F.mse_loss(predicted_values[valid_mask], returns_batch[valid_mask]).item()
        mae = F.l1_loss(predicted_values[valid_mask], returns_batch[valid_mask]).item()
        
        # Value prediction accuracy (within some tolerance)
        tolerance = 0.1
        correct = torch.abs(predicted_values[valid_mask] - returns_batch[valid_mask]) < tolerance
        accuracy = correct.float().mean().item()
    
    return {
        "value_loss": loss.item(),
        "value_mse": mse,
        "value_mae": mae,
        "value_accuracy": accuracy
    }


def compute_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 0.99,
    lambda_gae: float = 0.95
) -> torch.Tensor:
    """
    Compute advantage estimates using GAE (Generalized Advantage Estimation).
    
    Args:
        rewards: Rewards [num_rollouts]
        values: Value estimates [num_rollouts]
        gamma: Discount factor
        lambda_gae: GAE parameter
        
    Returns:
        Advantage estimates [num_rollouts]
    """
    # For now, use simple advantage = reward - value
    # TODO: Implement proper GAE
    advantages = rewards - values
    return advantages


def normalize_advantages(advantages: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Normalize advantages for stable training.
    
    Args:
        advantages: Advantage estimates
        epsilon: Small value to avoid division by zero
        
    Returns:
        Normalized advantages
    """
    if advantages.std() < epsilon:
        return advantages
    
    return (advantages - advantages.mean()) / (advantages.std() + epsilon)


if __name__ == "__main__":
    # Test value model
    import torch
    from transformers import AutoModelForCausalLM
    
    # Create a simple test
    print("Testing value model...")
    
    # This would require a real model to test fully
    print("Value model implementation completed")
    print("Note: Full testing requires a loaded language model")
