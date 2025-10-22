"""
A* search algorithm for guided rollout generation in TinyZero A*PO.
"""
import torch
import torch.nn.functional as F
import heapq
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class SearchNode:
    """
    Represents a node in the A* search tree.
    """
    token_sequence: List[int]
    log_prob: float  # g-cost: cumulative log probability
    value_estimate: float  # h-cost: value model estimate
    parent: Optional['SearchNode'] = None
    depth: int = 0
    is_terminal: bool = False
    
    def __post_init__(self):
        """Compute f-score after initialization."""
        self.f_score = self.log_prob + self.value_estimate
    
    def __lt__(self, other: 'SearchNode') -> bool:
        """Comparison for priority queue (lower f-score = higher priority)."""
        return self.f_score < other.f_score
    
    def __eq__(self, other: 'SearchNode') -> bool:
        """Equality comparison."""
        return self.token_sequence == other.token_sequence
    
    def __hash__(self) -> int:
        """Hash for set operations."""
        return hash(tuple(self.token_sequence))
    
    def get_path(self) -> List[int]:
        """Get the full token sequence from root to this node."""
        return self.token_sequence.copy()
    
    def get_log_prob_sequence(self) -> List[float]:
        """Get the sequence of log probabilities leading to this node."""
        if self.parent is None:
            return []
        
        parent_log_probs = self.parent.get_log_prob_sequence()
        parent_log_probs.append(self.log_prob - (self.parent.log_prob if self.parent else 0))
        return parent_log_probs


class AStarSearcher:
    """
    A* search for guided rollout generation.
    Uses value model as heuristic function h(n).
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        value_model: torch.nn.Module,
        tokenizer: Any,
        config
    ):
        """
        Initialize A* searcher.
        
        Args:
            model: Language model for policy
            value_model: Value model for heuristic
            tokenizer: Tokenizer
            config: Configuration object
        """
        self.model = model
        self.value_model = value_model
        self.tokenizer = tokenizer
        self.config = config
        
        # Search parameters
        self.beam_width = config.astar.beam_width
        self.max_depth = config.astar.max_search_depth
        self.max_generation_length = config.astar.max_generation_length
        self.temperature = config.astar.temperature
        self.top_p = config.astar.top_p
        self.top_k = config.astar.top_k
        self.exploration_weight = config.astar.exploration_weight
        
        # Special tokens
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        
        logger.info(f"A* searcher initialized with beam_width={self.beam_width}, max_depth={self.max_depth}")
    
    def search(
        self,
        prompt: str,
        target_answer: Optional[int] = None,
        num_rollouts: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Run A* search to generate multiple rollouts.
        
        Args:
            prompt: Input prompt
            target_answer: Target answer (for reward computation)
            num_rollouts: Number of rollouts to generate
            
        Returns:
            List of rollout dictionaries
        """
        # Tokenize prompt
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        
        # Initialize search
        initial_node = SearchNode(
            token_sequence=prompt_tokens,
            log_prob=0.0,
            value_estimate=0.0,
            parent=None,
            depth=0
        )
        
        # A* search
        rollouts = self._astar_search(initial_node, num_rollouts)
        
        # Format rollouts
        formatted_rollouts = []
        for rollout in rollouts:
            formatted_rollout = {
                "tokens": rollout["tokens"],
                "log_probs": rollout["log_probs"],
                "value_estimates": rollout["value_estimates"],
                "generated_text": self.tokenizer.decode(rollout["tokens"], skip_special_tokens=True),
                "f_score": rollout["f_score"],
                "depth": rollout["depth"]
            }
            formatted_rollouts.append(formatted_rollout)
        
        return formatted_rollouts
    
    def _astar_search(
        self,
        initial_node: SearchNode,
        num_rollouts: int
    ) -> List[Dict[str, Any]]:
        """
        Perform A* search.
        
        Args:
            initial_node: Starting node
            num_rollouts: Number of rollouts to generate
            
        Returns:
            List of rollout dictionaries
        """
        # Priority queue (open set)
        open_set = [initial_node]
        heapq.heapify(open_set)
        
        # Closed set (explored nodes)
        closed_set: Set[Tuple] = set()
        
        # Completed rollouts
        completed_rollouts = []
        
        # Search statistics
        nodes_expanded = 0
        max_open_size = 0
        
        while open_set and len(completed_rollouts) < num_rollouts:
            # Get node with lowest f-score
            current_node = heapq.heappop(open_set)
            nodes_expanded += 1
            
            # Check if terminal
            if self._is_terminal(current_node):
                # Create rollout from this path
                rollout = self._create_rollout_from_path(current_node)
                completed_rollouts.append(rollout)
                continue
            
            # Add to closed set
            closed_set.add(tuple(current_node.token_sequence))
            
            # Expand node
            successors = self._expand_node(current_node)
            
            for successor in successors:
                # Skip if already explored
                if tuple(successor.token_sequence) in closed_set:
                    continue
                
                # Add to open set
                heapq.heappush(open_set, successor)
            
            # Update statistics
            max_open_size = max(max_open_size, len(open_set))
            
            # Limit open set size to prevent memory issues
            if len(open_set) > self.beam_width * 10:
                # Keep only the best nodes
                best_nodes = heapq.nsmallest(self.beam_width, open_set)
                open_set = best_nodes
                heapq.heapify(open_set)
        
        logger.info(f"A* search completed: {len(completed_rollouts)} rollouts, {nodes_expanded} nodes expanded, max_open={max_open_size}")
        
        return completed_rollouts
    
    def _expand_node(self, node: SearchNode) -> List[SearchNode]:
        """
        Expand a node by generating successor nodes.
        
        Args:
            node: Node to expand
            
        Returns:
            List of successor nodes
        """
        if node.depth >= self.max_depth:
            return []
        
        # Get next token candidates from policy
        next_tokens, log_probs = self._get_next_token_candidates(node)
        
        successors = []
        for token, log_prob in zip(next_tokens, log_probs):
            # Create new token sequence
            new_sequence = node.token_sequence + [token]
            
            # Compute cumulative log probability (g-cost)
            new_log_prob = node.log_prob + log_prob
            
            # Get value estimate (h-cost)
            value_estimate = self._get_value_estimate(new_sequence)
            
            # Create successor node
            successor = SearchNode(
                token_sequence=new_sequence,
                log_prob=new_log_prob,
                value_estimate=value_estimate,
                parent=node,
                depth=node.depth + 1
            )
            
            successors.append(successor)
        
        return successors
    
    def _get_next_token_candidates(
        self,
        node: SearchNode
    ) -> Tuple[List[int], List[float]]:
        """
        Get next token candidates from policy model.
        
        Args:
            node: Current node
            
        Returns:
            Tuple of (token_ids, log_probs)
        """
        # Prepare input
        device = next(self.model.parameters()).device
        input_ids = torch.tensor([node.token_sequence], device=device)
        
        with torch.no_grad():
            # Get logits
            outputs = self.model(input_ids)
            logits = outputs.logits[0, -1, :]  # Last position
            
            # Apply temperature
            logits = logits / self.temperature
            
            # Apply top-k filtering
            if self.top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, min(self.top_k, logits.size(0)))
                filtered_logits = torch.full_like(logits, float('-inf'))
                filtered_logits[top_k_indices] = top_k_logits
                logits = filtered_logits
            
            # Apply top-p filtering
            if self.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > self.top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = float('-inf')
            
            # Sample tokens
            probs = F.softmax(logits, dim=-1)
            token_probs = torch.multinomial(probs, num_samples=min(self.beam_width, len(probs.nonzero())))
            
            # Get log probabilities
            log_probs = F.log_softmax(logits, dim=-1)[token_probs]
            
            return token_probs.tolist(), log_probs.tolist()
    
    def _get_value_estimate(self, token_sequence: List[int]) -> float:
        """
        Get value estimate for a token sequence.
        
        Args:
            token_sequence: Sequence of token IDs
            
        Returns:
            Value estimate
        """
        if len(token_sequence) == 0:
            return 0.0
        
        # Prepare input
        device = next(self.value_model.base_model.parameters()).device
        input_ids = torch.tensor([token_sequence], device=device)
        attention_mask = torch.ones_like(input_ids)
        
        with torch.no_grad():
            # Get value estimate
            values = self.value_model.get_value_estimates(input_ids, attention_mask)
            final_value = values[0, -1].item()  # Last position
        
        return final_value
    
    def _is_terminal(self, node: SearchNode) -> bool:
        """
        Check if a node is terminal.
        
        Args:
            node: Node to check
            
        Returns:
            True if terminal
        """
        # Check length limit
        if len(node.token_sequence) >= self.max_generation_length:
            return True
        
        # Check for EOS token
        if node.token_sequence and node.token_sequence[-1] == self.eos_token_id:
            return True
        
        # Check depth limit
        if node.depth >= self.max_depth:
            return True
        
        return False
    
    def _create_rollout_from_path(self, terminal_node: SearchNode) -> Dict[str, Any]:
        """
        Create rollout dictionary from a terminal node path.
        
        Args:
            terminal_node: Terminal node
            
        Returns:
            Rollout dictionary
        """
        # Get full path
        tokens = terminal_node.get_path()
        log_probs = terminal_node.get_log_prob_sequence()
        
        # Get value estimates for each step
        value_estimates = []
        current_sequence = []
        
        for i, token in enumerate(tokens):
            current_sequence.append(token)
            if i > 0:  # Skip initial prompt
                value_estimate = self._get_value_estimate(current_sequence)
                value_estimates.append(value_estimate)
        
        return {
            "tokens": tokens,
            "log_probs": log_probs,
            "value_estimates": value_estimates,
            "f_score": terminal_node.f_score,
            "depth": terminal_node.depth
        }


def batch_astar_search(
    searcher: AStarSearcher,
    prompts: List[str],
    num_rollouts_per_prompt: int = 4
) -> List[List[Dict[str, Any]]]:
    """
    Run A* search for multiple prompts in batch.
    
    Args:
        searcher: A* searcher
        prompts: List of prompts
        num_rollouts_per_prompt: Number of rollouts per prompt
        
    Returns:
        List of rollout lists (one per prompt)
    """
    all_rollouts = []
    
    for prompt in prompts:
        rollouts = searcher.search(prompt, num_rollouts=num_rollouts_per_prompt)
        all_rollouts.append(rollouts)
    
    return all_rollouts


if __name__ == "__main__":
    # Test A* search implementation
    print("A* search implementation completed")
    print("Note: Full testing requires loaded models and tokenizer")
