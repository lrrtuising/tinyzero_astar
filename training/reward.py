"""
Reward computation for TinyZero A*PO training.
"""
import torch
import re
from typing import List, Dict, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


def compute_reward(
    rollout: Dict[str, Any],
    ground_truth_answer: int,
    accuracy_weight: float = 1.0,
    format_weight: float = 0.1,
    length_penalty: float = 0.01
) -> float:
    """
    Compute reward for a single rollout.
    
    Args:
        rollout: Rollout dictionary
        ground_truth_answer: Correct answer
        accuracy_weight: Weight for accuracy reward
        format_weight: Weight for format reward
        length_penalty: Penalty for long generations
        
    Returns:
        Scalar reward value
    """
    # Extract answer from rollout
    extracted_answer = rollout.get("extracted_answer")
    generated_text = rollout.get("generated_text", "")
    
    # Accuracy reward
    if extracted_answer is not None and extracted_answer == ground_truth_answer:
        accuracy_reward = 1.0
    else:
        accuracy_reward = -1.0  # Penalty for wrong answer
    
    # Format reward (bonus for properly formatted output)
    format_reward = _compute_format_reward(generated_text)
    
    # Length penalty
    length_penalty_value = _compute_length_penalty(generated_text, length_penalty)
    
    # Total reward
    total_reward = (
        accuracy_weight * accuracy_reward +
        format_weight * format_reward -
        length_penalty_value
    )
    
    return total_reward


def compute_rewards_batch(
    rollouts: List[Dict[str, Any]],
    ground_truths: List[int]
) -> torch.Tensor:
    """
    Compute rewards for batch of rollouts.
    
    Args:
        rollouts: List of rollout dictionaries
        ground_truths: List of ground truth answers
        
    Returns:
        Tensor of rewards [num_rollouts]
    """
    rewards = []
    
    for rollout, ground_truth in zip(rollouts, ground_truths):
        reward = compute_reward(rollout, ground_truth)
        rewards.append(reward)
    
    return torch.tensor(rewards, dtype=torch.float32)


def _compute_format_reward(generated_text: str) -> float:
    """
    Compute format reward for generated text.
    
    Args:
        generated_text: Generated text
        
    Returns:
        Format reward (0.0 to 1.0)
    """
    # Check for step-by-step reasoning indicators
    step_indicators = [
        "step", "first", "second", "third", "next", "then", "finally",
        "let's", "we", "multiply", "add", "carry", "result"
    ]
    
    # Check for mathematical notation
    math_indicators = [
        "=", "+", "-", "*", "×", "÷", "/", "(", ")"
    ]
    
    # Count indicators
    step_count = sum(1 for indicator in step_indicators if indicator.lower() in generated_text.lower())
    math_count = sum(1 for indicator in math_indicators if indicator in generated_text)
    
    # Normalize rewards
    step_reward = min(step_count / 3.0, 1.0)  # Cap at 1.0
    math_reward = min(math_count / 5.0, 1.0)  # Cap at 1.0
    
    # Combined format reward
    format_reward = (step_reward + math_reward) / 2.0
    
    return format_reward


def _compute_length_penalty(generated_text: str, penalty_rate: float) -> float:
    """
    Compute length penalty for generated text.
    
    Args:
        generated_text: Generated text
        penalty_rate: Penalty rate per character
        
    Returns:
        Length penalty value
    """
    length = len(generated_text)
    return penalty_rate * length


def parse_answer(text: str) -> Optional[int]:
    """
    Extract numerical answer from text using multiple patterns.
    
    Args:
        text: Text to parse
        
    Returns:
        Parsed integer or None
    """
    # Remove extra whitespace
    text = text.strip()
    
    # Try multiple patterns
    patterns = [
        r'= (\d+)',  # "= 123"
        r'answer[:\s]*(\d+)',  # "answer: 123"
        r'result[:\s]*(\d+)',  # "result: 123"
        r'(\d+)$',  # Number at end
        r'(\d+)\s*\.?\s*$',  # Number at end with optional period
        r'(\d+)\s*$',  # Number at end with whitespace
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        if matches:
            try:
                # Take the last match (most likely final answer)
                answer = int(matches[-1])
                return answer
            except ValueError:
                continue
    
    # Fallback: find all numbers and take the largest
    numbers = re.findall(r'\d+', text)
    if numbers:
        try:
            return max(int(num) for num in numbers)
        except ValueError:
            pass
    
    return None


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
    # TODO: Implement proper GAE for sequential rollouts
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


def compute_rollout_accuracy(
    rollouts: List[Dict[str, Any]],
    ground_truths: List[int]
) -> float:
    """
    Compute accuracy for a batch of rollouts.
    
    Args:
        rollouts: List of rollout dictionaries
        ground_truths: List of ground truth answers
        
    Returns:
        Accuracy (0.0 to 1.0)
    """
    if not rollouts or not ground_truths:
        return 0.0
    
    correct = 0
    total = len(rollouts)
    
    for rollout, ground_truth in zip(rollouts, ground_truths):
        extracted_answer = rollout.get("extracted_answer")
        if extracted_answer is not None and extracted_answer == ground_truth:
            correct += 1
    
    return correct / total


def compute_format_correctness(rollouts: List[Dict[str, Any]]) -> float:
    """
    Compute format correctness for a batch of rollouts.
    
    Args:
        rollouts: List of rollout dictionaries
        
    Returns:
        Format correctness (0.0 to 1.0)
    """
    if not rollouts:
        return 0.0
    
    parseable = 0
    total = len(rollouts)
    
    for rollout in rollouts:
        extracted_answer = rollout.get("extracted_answer")
        if extracted_answer is not None:
            parseable += 1
    
    return parseable / total


def get_reward_statistics(
    rollouts: List[Dict[str, Any]],
    ground_truths: List[int]
) -> Dict[str, float]:
    """
    Get comprehensive reward statistics.
    
    Args:
        rollouts: List of rollout dictionaries
        ground_truths: List of ground truth answers
        
    Returns:
        Dictionary with reward statistics
    """
    if not rollouts:
        return {}
    
    # Compute rewards
    rewards = compute_rewards_batch(rollouts, ground_truths)
    
    # Basic statistics
    mean_reward = rewards.mean().item()
    std_reward = rewards.std().item()
    min_reward = rewards.min().item()
    max_reward = rewards.max().item()
    
    # Accuracy
    accuracy = compute_rollout_accuracy(rollouts, ground_truths)
    
    # Format correctness
    format_correctness = compute_format_correctness(rollouts)
    
    # Answer extraction rate
    extracted_answers = [r.get("extracted_answer") for r in rollouts]
    extraction_rate = sum(1 for ans in extracted_answers if ans is not None) / len(rollouts)
    
    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "min_reward": min_reward,
        "max_reward": max_reward,
        "accuracy": accuracy,
        "format_correctness": format_correctness,
        "extraction_rate": extraction_rate,
        "total_rollouts": len(rollouts)
    }


if __name__ == "__main__":
    # Test reward computation
    print("Testing reward computation...")
    
    # Test parse_answer function
    test_texts = [
        "The answer is 123",
        "= 456",
        "result: 789",
        "123",
        "The final result is 999.",
        "No clear answer"
    ]
    
    for text in test_texts:
        answer = parse_answer(text)
        print(f"'{text}' -> {answer}")
    
    # Test reward computation
    test_rollout = {
        "extracted_answer": 123,
        "generated_text": "Let's solve this step by step. First, we multiply 12 * 34 = 408"
    }
    
    reward = compute_reward(test_rollout, 123)
    print(f"Reward for correct answer: {reward}")
    
    reward_wrong = compute_reward(test_rollout, 456)
    print(f"Reward for wrong answer: {reward_wrong}")
    
    print("Reward computation tests completed")
