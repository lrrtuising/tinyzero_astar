"""
Rollout caching utilities for TinyZero A*PO.
"""
import json
import pickle
import os
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def save_rollouts(rollouts: List[Dict[str, Any]], filepath: str) -> None:
    """
    Save rollouts to disk.
    
    Args:
        rollouts: List of rollout dictionaries
        filepath: Path to save file
    """
    # Convert tensors to lists for JSON serialization
    serializable_rollouts = []
    
    for rollout in rollouts:
        serializable_rollout = {}
        for key, value in rollout.items():
            if key in ["tokens", "log_probs", "value_estimates"]:
                # Convert tensor to list
                if hasattr(value, 'tolist'):
                    serializable_rollout[key] = value.tolist()
                else:
                    serializable_rollout[key] = value
            else:
                serializable_rollout[key] = value
        
        serializable_rollouts.append(serializable_rollout)
    
    # Save as JSON
    with open(filepath, 'w') as f:
        json.dump(serializable_rollouts, f, indent=2)
    
    logger.info(f"Saved {len(rollouts)} rollouts to {filepath}")


def load_rollouts(filepath: str) -> List[Dict[str, Any]]:
    """
    Load rollouts from disk.
    
    Args:
        filepath: Path to load file from
        
    Returns:
        List of rollout dictionaries
    """
    with open(filepath, 'r') as f:
        rollouts = json.load(f)
    
    logger.info(f"Loaded {len(rollouts)} rollouts from {filepath}")
    return rollouts


def save_rollouts_pickle(rollouts: List[Dict[str, Any]], filepath: str) -> None:
    """
    Save rollouts using pickle (preserves tensors).
    
    Args:
        rollouts: List of rollout dictionaries
        filepath: Path to save file
    """
    with open(filepath, 'wb') as f:
        pickle.dump(rollouts, f)
    
    logger.info(f"Saved {len(rollouts)} rollouts to {filepath} (pickle)")


def load_rollouts_pickle(filepath: str) -> List[Dict[str, Any]]:
    """
    Load rollouts using pickle.
    
    Args:
        filepath: Path to load file from
        
    Returns:
        List of rollout dictionaries
    """
    with open(filepath, 'rb') as f:
        rollouts = pickle.load(f)
    
    logger.info(f"Loaded {len(rollouts)} rollouts from {filepath} (pickle)")
    return rollouts


def get_rollout_cache_path(iteration: int, cache_dir: str = "rollout_cache") -> str:
    """
    Get cache file path for specific iteration.
    
    Args:
        iteration: Training iteration
        cache_dir: Cache directory
        
    Returns:
        Cache file path
    """
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"rollouts_iter_{iteration}.json")


def cache_exists(iteration: int, cache_dir: str = "rollout_cache") -> bool:
    """
    Check if rollout cache exists for iteration.
    
    Args:
        iteration: Training iteration
        cache_dir: Cache directory
        
    Returns:
        True if cache exists
    """
    cache_path = get_rollout_cache_path(iteration, cache_dir)
    return os.path.exists(cache_path)


def get_rollout_statistics(rollouts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get statistics about rollouts.
    
    Args:
        rollouts: List of rollout dictionaries
        
    Returns:
        Dictionary with statistics
    """
    if not rollouts:
        return {}
    
    # Basic statistics
    total_rollouts = len(rollouts)
    avg_length = sum(len(r.get("tokens", [])) for r in rollouts) / total_rollouts
    avg_f_score = sum(r.get("f_score", 0) for r in rollouts) / total_rollouts
    avg_depth = sum(r.get("depth", 0) for r in rollouts) / total_rollouts
    
    # Answer extraction statistics
    extracted_answers = [r.get("extracted_answer") for r in rollouts if r.get("extracted_answer") is not None]
    extraction_rate = len(extracted_answers) / total_rollouts
    
    # Accuracy statistics
    correct_answers = 0
    if rollouts and "ground_truth_answer" in rollouts[0]:
        for rollout in rollouts:
            if (rollout.get("extracted_answer") is not None and 
                rollout.get("extracted_answer") == rollout.get("ground_truth_answer")):
                correct_answers += 1
    
    accuracy = correct_answers / total_rollouts if total_rollouts > 0 else 0
    
    return {
        "total_rollouts": total_rollouts,
        "avg_length": avg_length,
        "avg_f_score": avg_f_score,
        "avg_depth": avg_depth,
        "extraction_rate": extraction_rate,
        "accuracy": accuracy,
        "correct_answers": correct_answers
    }


if __name__ == "__main__":
    # Test rollout caching
    test_rollouts = [
        {
            "tokens": [1, 2, 3],
            "log_probs": [0.1, 0.2, 0.3],
            "value_estimates": [0.5, 0.6, 0.7],
            "generated_text": "Test generation",
            "f_score": 1.5,
            "depth": 3,
            "extracted_answer": 123,
            "ground_truth_answer": 123
        }
    ]
    
    # Test saving and loading
    save_rollouts(test_rollouts, "test_rollouts.json")
    loaded_rollouts = load_rollouts("test_rollouts.json")
    
    print(f"Original: {len(test_rollouts)} rollouts")
    print(f"Loaded: {len(loaded_rollouts)} rollouts")
    
    # Test statistics
    stats = get_rollout_statistics(test_rollouts)
    print(f"Statistics: {stats}")
    
    # Cleanup
    os.remove("test_rollouts.json")
    print("Test completed successfully")
