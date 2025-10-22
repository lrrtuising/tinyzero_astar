"""
A* Policy Optimization (A*PO) algorithm implementation.
"""
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Any
import logging

from training.reward import compute_advantages, normalize_advantages

logger = logging.getLogger(__name__)


class AStarPO:
    """
    A* Policy Optimization trainer.
    Combines A* search for rollout generation with policy gradient optimization.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        value_model: torch.nn.Module,
        config
    ):
        """
        Initialize A*PO trainer.
        
        Args:
            model: Policy model
            value_model: Value model
            config: Configuration object
        """
        self.model = model
        self.value_model = value_model
        self.config = config
        
        # Training parameters
        self.kl_penalty_weight = getattr(config.training, 'kl_penalty_weight', 0.1)
        self.entropy_bonus_weight = getattr(config.training, 'entropy_bonus_weight', 0.01)
        self.ppo_clip_epsilon = getattr(config.training, 'ppo_clip_epsilon', 0.2)
        self.use_ppo = getattr(config.training, 'use_ppo', True)
        
        # Store old policy for PPO
        self.old_policy_log_probs = None
        
        logger.info(f"A*PO trainer initialized with PPO={self.use_ppo}, KL penalty={self.kl_penalty_weight}")
    
    def train_step(
        self,
        rollouts: List[Dict[str, Any]],
        rewards: torch.Tensor,
        advantages: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        value_optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict[str, float]:
        """
        Single training step for A*PO.
        
        Args:
            rollouts: List of rollout dictionaries
            rewards: Rewards for each rollout
            advantages: Advantage estimates
            optimizer: Policy optimizer
            value_optimizer: Value model optimizer (optional)
            
        Returns:
            Dictionary with loss components
        """
        self.model.train()
        if self.value_model is not None:
            self.value_model.train()
        
        # Prepare training data
        policy_loss = self._compute_policy_loss(rollouts, advantages)
        
        # Compute value loss if value model is provided
        value_loss = 0.0
        if self.value_model is not None and value_optimizer is not None:
            value_loss = self._compute_value_loss(rollouts, rewards)
        
        # Total loss
        total_loss = policy_loss + value_loss
        
        # Backward pass
        optimizer.zero_grad()
        if value_optimizer is not None:
            value_optimizer.zero_grad()
        
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
        if self.value_model is not None:
            torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), self.config.training.max_grad_norm)
        
        # Optimizer step
        optimizer.step()
        if value_optimizer is not None:
            value_optimizer.step()
        
        # Update old policy for PPO
        if self.use_ppo:
            self._update_old_policy(rollouts)
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item() if isinstance(value_loss, torch.Tensor) else value_loss,
            "total_loss": total_loss.item(),
            "kl_divergence": self._compute_kl_divergence(rollouts),
            "entropy": self._compute_entropy(rollouts)
        }
    
    def _compute_policy_loss(
        self,
        rollouts: List[Dict[str, Any]],
        advantages: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute policy gradient loss.
        
        Args:
            rollouts: List of rollout dictionaries
            advantages: Advantage estimates
            
        Returns:
            Policy loss
        """
        if self.use_ppo:
            return self._compute_ppo_loss(rollouts, advantages)
        else:
            return self._compute_basic_policy_loss(rollouts, advantages)
    
    def _compute_basic_policy_loss(
        self,
        rollouts: List[Dict[str, Any]],
        advantages: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute basic policy gradient loss.
        
        Args:
            rollouts: List of rollout dictionaries
            advantages: Advantage estimates
            
        Returns:
            Policy loss
        """
        losses = []
        
        for i, rollout in enumerate(rollouts):
            tokens = rollout["tokens"]
            log_probs = rollout["log_probs"]
            advantage = advantages[i]
            
            # Convert to tensors
            if isinstance(tokens, list):
                tokens = torch.tensor(tokens, device=self.model.device)
            if isinstance(log_probs, list):
                log_probs = torch.tensor(log_probs, device=self.model.device)
            
            # Compute current log probabilities
            current_log_probs = self._get_current_log_probs(tokens)
            
            # Policy gradient loss: -log_prob * advantage
            policy_loss = -(current_log_probs * advantage).mean()
            losses.append(policy_loss)
        
        return torch.stack(losses).mean()
    
    def _compute_ppo_loss(
        self,
        rollouts: List[Dict[str, Any]],
        advantages: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute PPO-style clipped policy loss.
        
        Args:
            rollouts: List of rollout dictionaries
            advantages: Advantage estimates
            
        Returns:
            PPO loss
        """
        losses = []
        
        for i, rollout in enumerate(rollouts):
            tokens = rollout["tokens"]
            old_log_probs = rollout["log_probs"]
            advantage = advantages[i]
            
            # Convert to tensors
            if isinstance(tokens, list):
                tokens = torch.tensor(tokens, device=self.model.device)
            if isinstance(old_log_probs, list):
                old_log_probs = torch.tensor(old_log_probs, device=self.model.device)
            
            # Compute current log probabilities
            current_log_probs = self._get_current_log_probs(tokens)
            
            # Ensure same length by truncating to minimum length
            min_len = min(len(current_log_probs), len(old_log_probs))
            current_log_probs = current_log_probs[:min_len]
            old_log_probs = old_log_probs[:min_len]
            
            # Compute ratio
            ratio = torch.exp(current_log_probs - old_log_probs)
            
            # PPO clipped objective
            clipped_ratio = torch.clamp(ratio, 1 - self.ppo_clip_epsilon, 1 + self.ppo_clip_epsilon)
            
            # Policy loss
            policy_loss = -torch.min(ratio * advantage, clipped_ratio * advantage).mean()
            losses.append(policy_loss)
        
        return torch.stack(losses).mean()
    
    def _compute_value_loss(
        self,
        rollouts: List[Dict[str, Any]],
        rewards: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute value function loss.
        
        Args:
            rollouts: List of rollout dictionaries
            rewards: Rewards for each rollout
            
        Returns:
            Value loss
        """
        if self.value_model is None:
            return torch.tensor(0.0, device=self.model.device)
        
        losses = []
        
        for i, rollout in enumerate(rollouts):
            tokens = rollout["tokens"]
            reward = rewards[i]
            
            # Convert to tensor
            if isinstance(tokens, list):
                tokens = torch.tensor(tokens, device=self.model.device).unsqueeze(0)
            
            # Get value prediction
            attention_mask = torch.ones_like(tokens)
            predicted_value = self.value_model.get_final_values(tokens, attention_mask)
            
            # MSE loss
            value_loss = F.mse_loss(predicted_value, reward.unsqueeze(0))
            losses.append(value_loss)
        
        return torch.stack(losses).mean()
    
    def _get_current_log_probs(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Get current log probabilities for tokens.
        
        Args:
            tokens: Token sequence [seq_len]
            
        Returns:
            Log probabilities [seq_len]
        """
        if len(tokens.shape) == 1:
            tokens = tokens.unsqueeze(0)
        
        # Forward pass
        outputs = self.model(tokens)
        logits = outputs.logits
        
        # Get log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Extract log probabilities for actual tokens
        seq_len = tokens.size(1)
        log_probs_selected = log_probs[0, :seq_len-1, :].gather(1, tokens[0, 1:].unsqueeze(1)).squeeze(1)
        
        return log_probs_selected
    
    def _compute_kl_divergence(self, rollouts: List[Dict[str, Any]]) -> float:
        """
        Compute KL divergence between old and new policy.
        
        Args:
            rollouts: List of rollout dictionaries
            
        Returns:
            Average KL divergence
        """
        if not self.use_ppo or self.old_policy_log_probs is None:
            return 0.0
        
        kl_divs = []
        
        for rollout in rollouts:
            tokens = rollout["tokens"]
            old_log_probs = rollout["log_probs"]
            
            # Convert to tensors
            if isinstance(tokens, list):
                tokens = torch.tensor(tokens, device=self.model.device)
            if isinstance(old_log_probs, list):
                old_log_probs = torch.tensor(old_log_probs, device=self.model.device)
            
            # Get current log probabilities
            current_log_probs = self._get_current_log_probs(tokens)

            min_len = min(len(current_log_probs), len(old_log_probs))
            current_log_probs = current_log_probs[:min_len]
            old_log_probs = old_log_probs[:min_len]

            # KL divergence: KL(old || new) = E[log(old) - log(new)]
            kl_div = (old_log_probs - current_log_probs).mean().item()
            kl_divs.append(kl_div)
        
        return sum(kl_divs) / len(kl_divs) if kl_divs else 0.0
    
    def _compute_entropy(self, rollouts: List[Dict[str, Any]]) -> float:
        """
        Compute entropy of the policy.
        
        Args:
            rollouts: List of rollout dictionaries
            
        Returns:
            Average entropy
        """
        entropies = []
        
        for rollout in rollouts:
            tokens = rollout["tokens"]
            
            # Convert to tensor
            if isinstance(tokens, list):
                tokens = torch.tensor(tokens, device=self.model.device).unsqueeze(0)
            
            # Forward pass
            outputs = self.model(tokens)
            logits = outputs.logits
            
            # Compute entropy
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1).mean().item()
            entropies.append(entropy)
        
        return sum(entropies) / len(entropies) if entropies else 0.0
    
    def _update_old_policy(self, rollouts: List[Dict[str, Any]]) -> None:
        """
        Update old policy log probabilities for PPO.
        
        Args:
            rollouts: List of rollout dictionaries
        """
        self.old_policy_log_probs = []
        
        for rollout in rollouts:
            tokens = rollout["tokens"]
            
            # Convert to tensor
            if isinstance(tokens, list):
                tokens = torch.tensor(tokens, device=self.model.device)
            
            # Get current log probabilities
            with torch.no_grad():
                current_log_probs = self._get_current_log_probs(tokens)
                self.old_policy_log_probs.append(current_log_probs.cpu())
    
    def compute_advantages(
        self,
        rollouts: List[Dict[str, Any]],
        rewards: torch.Tensor,
        values: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute advantage estimates.
        
        Args:
            rollouts: List of rollout dictionaries
            rewards: Rewards for each rollout
            values: Value estimates (optional)
            
        Returns:
            Advantage estimates
        """
        if values is None:
            # Use rewards as baseline
            advantages = rewards - rewards.mean()
        else:
            # Use value estimates as baseline
            advantages = rewards - values
        
        # Normalize advantages
        advantages = normalize_advantages(advantages)
        
        return advantages


if __name__ == "__main__":
    # Test A*PO implementation
    print("A*PO implementation completed")
    print("Note: Full testing requires loaded models and rollouts")
