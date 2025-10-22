"""
Evaluation module for TinyZero A*PO.
"""
import torch
import time
from typing import List, Dict, Optional, Tuple, Any
import logging

from rollout.generator import RolloutGenerator
from training.reward import (
    compute_rollout_accuracy,
    compute_format_correctness,
    get_reward_statistics
)

logger = logging.getLogger(__name__)


def _get_problem_field(problem: Any, field: str, default: Any = None) -> Any:
    """
    Safely get a field from either a dict or Problem dataclass.
    """
    if isinstance(problem, dict):
        return problem.get(field, default)
    return getattr(problem, field, default)


def evaluate(
    model: torch.nn.Module,
    value_model: torch.nn.Module,
    tokenizer: Any,
    eval_dataset: List[Dict],
    config,
    num_rollouts_per_problem: int = 4
) -> Dict[str, float]:
    """
    Run evaluation on validation set.
    
    Args:
        model: Policy model
        value_model: Value model
        tokenizer: Tokenizer
        eval_dataset: Evaluation dataset
        config: Configuration object
        num_rollouts_per_problem: Number of rollouts per problem
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"Starting evaluation on {len(eval_dataset)} problems")
    
    # Create rollout generator
    generator = RolloutGenerator(model, value_model, tokenizer, config)
    
    # Generate rollouts for evaluation
    start_time = time.time()
    rollouts = generator.generate_rollouts(
        eval_dataset,
        num_rollouts_per_problem=num_rollouts_per_problem
    )
    generation_time = time.time() - start_time
    
    # Extract ground truth answers
    ground_truths = [_get_problem_field(problem, "answer") for problem in eval_dataset]
    
    # Compute metrics
    accuracy = compute_rollout_accuracy(rollouts, ground_truths)
    format_correctness = compute_format_correctness(rollouts)
    reward_stats = get_reward_statistics(rollouts, ground_truths)
    
    # Compute rollout statistics
    rollout_stats = generator.get_rollout_statistics(rollouts)
    
    # Log sample generations
    log_samples(eval_dataset[:5], rollouts[:5*num_rollouts_per_problem], n=5)
    
    # Compile results
    results = {
        "accuracy": accuracy,
        "format_correctness": format_correctness,
        "generation_time": generation_time,
        "problems_per_second": len(eval_dataset) / generation_time,
        **reward_stats,
        **rollout_stats
    }
    
    logger.info(f"Evaluation completed: accuracy={accuracy:.3f}, format_correctness={format_correctness:.3f}")
    return results


def compute_accuracy(
    predictions: List[Optional[int]],
    ground_truths: List[int]
) -> float:
    """
    Calculate accuracy.
    
    Args:
        predictions: List of predicted answers (can be None)
        ground_truths: List of ground truth answers
        
    Returns:
        Accuracy (0.0 to 1.0)
    """
    if not predictions or not ground_truths:
        return 0.0
    
    correct = 0
    total = len(predictions)
    
    for pred, truth in zip(predictions, ground_truths):
        if pred is not None and pred == truth:
            correct += 1
    
    return correct / total


def log_samples(
    problems: List[Dict],
    rollouts: List[Dict],
    n: int = 5
) -> None:
    """
    Log sample generations for debugging.
    
    Args:
        problems: List of problem dictionaries
        rollouts: List of rollout dictionaries
        n: Number of samples to log
    """
    logger.info("Sample generations:")
    
    for i in range(min(n, len(problems))):
        problem = problems[i]
        problem_rollouts = [r for r in rollouts if r.get("problem_id") == i]
        
        if not problem_rollouts:
            continue
        
        prompt_text = _get_problem_field(problem, "problem", "<unknown>")
        answer = _get_problem_field(problem, "answer", "<unknown>")
        
        logger.info(f"\nProblem {i+1}: {prompt_text}")
        logger.info(f"Ground truth: {answer}")
        
        for j, rollout in enumerate(problem_rollouts[:2]):  # Show first 2 rollouts
            generated_text = rollout.get("generated_text", "")
            extracted_answer = rollout.get("extracted_answer")
            f_score = rollout.get("f_score", 0.0)
            
            logger.info(f"  Rollout {j+1}: {generated_text[:100]}...")
            logger.info(f"    Extracted answer: {extracted_answer}")
            logger.info(f"    F-score: {f_score:.3f}")


def compute_format_correctness_eval(generations: List[str]) -> float:
    """
    Check if outputs follow expected format.
    
    Args:
        generations: List of generated texts
        
    Returns:
        Percentage of parseable outputs
    """
    if not generations:
        return 0.0
    
    parseable = 0
    total = len(generations)
    
    for generation in generations:
        # Try to extract a number from the generation
        import re
        numbers = re.findall(r'\d+', generation)
        if numbers:
            parseable += 1
    
    return parseable / total


def evaluate_single_problem(
    model: torch.nn.Module,
    value_model: torch.nn.Module,
    tokenizer: Any,
    problem: Dict,
    config,
    num_rollouts: int = 4
) -> Dict[str, Any]:
    """
    Evaluate on a single problem.
    
    Args:
        model: Policy model
        value_model: Value model
        tokenizer: Tokenizer
        problem: Problem dictionary
        config: Configuration object
        num_rollouts: Number of rollouts to generate
        
    Returns:
        Evaluation results for single problem
    """
    generator = RolloutGenerator(model, value_model, tokenizer, config)
    
    # Generate rollouts
    rollouts = generator.generate_single_rollout(problem, num_rollouts)
    
    # Compute metrics
    ground_truth = _get_problem_field(problem, "answer")
    predictions = [r.get("extracted_answer") for r in rollouts]
    
    accuracy = compute_accuracy(predictions, [ground_truth] * len(predictions))
    format_correctness = compute_format_correctness_eval([r.get("generated_text", "") for r in rollouts])
    
    # Get rollout statistics
    rollout_stats = generator.get_rollout_statistics(rollouts)
    
    return {
        "accuracy": accuracy,
        "format_correctness": format_correctness,
        "ground_truth": ground_truth,
        "predictions": predictions,
        "rollouts": rollouts,
        **rollout_stats
    }


def evaluate_difficulty_levels(
    model: torch.nn.Module,
    value_model: torch.nn.Module,
    tokenizer: Any,
    eval_dataset: List[Dict],
    config
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate on different difficulty levels.
    
    Args:
        model: Policy model
        value_model: Value model
        tokenizer: Tokenizer
        eval_dataset: Evaluation dataset
        config: Configuration object
        
    Returns:
        Dictionary with results for each difficulty level
    """
    # Group problems by difficulty
    difficulty_groups = {}
    for problem in eval_dataset:
        difficulty = _get_problem_field(problem, "difficulty", "unknown")
        if difficulty not in difficulty_groups:
            difficulty_groups[difficulty] = []
        difficulty_groups[difficulty].append(problem)
    
    results = {}
    
    for difficulty, problems in difficulty_groups.items():
        logger.info(f"Evaluating {len(problems)} {difficulty} problems")
        
        # Evaluate this difficulty level
        difficulty_results = evaluate(
            model, value_model, tokenizer, problems, config
        )
        
        results[difficulty] = difficulty_results
    
    return results


def run_distributed_evaluation(
    model: torch.nn.Module,
    value_model: torch.nn.Module,
    tokenizer: Any,
    eval_dataset: List[Dict],
    config
) -> Dict[str, float]:
    """
    Run evaluation in distributed setting.
    
    Args:
        model: Policy model
        value_model: Value model
        tokenizer: Tokenizer
        eval_dataset: Evaluation dataset
        config: Configuration object
        
    Returns:
        Aggregated evaluation results
    """
    try:
        import torch.distributed as dist
        
        if not dist.is_initialized():
            # Not in distributed mode, run regular evaluation
            return evaluate(model, value_model, tokenizer, eval_dataset, config)
        
        # Split dataset across ranks
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        # Distribute problems across ranks
        problems_per_rank = len(eval_dataset) // world_size
        start_idx = rank * problems_per_rank
        end_idx = start_idx + problems_per_rank if rank < world_size - 1 else len(eval_dataset)
        
        local_problems = eval_dataset[start_idx:end_idx]
        
        # Run local evaluation
        local_results = evaluate(model, value_model, tokenizer, local_problems, config)
        
        # Gather results from all ranks
        if rank == 0:
            all_results = [local_results]
            for _ in range(world_size - 1):
                result = dist.recv()
                all_results.append(result)
        else:
            dist.send(local_results, dst=0)
            return {}
        
        # Aggregate results
        aggregated_results = {}
        for key in local_results.keys():
            if isinstance(local_results[key], (int, float)):
                # Average numerical metrics
                values = [result[key] for result in all_results]
                aggregated_results[key] = sum(values) / len(values)
            else:
                # Take from rank 0
                aggregated_results[key] = all_results[0][key]
        
        return aggregated_results
        
    except ImportError:
        # torch.distributed not available
        return evaluate(model, value_model, tokenizer, eval_dataset, config)


def benchmark_generation_speed(
    model: torch.nn.Module,
    value_model: torch.nn.Module,
    tokenizer: Any,
    test_problems: List[Dict],
    config,
    num_rollouts: int = 4
) -> Dict[str, float]:
    """
    Benchmark generation speed.
    
    Args:
        model: Policy model
        value_model: Value model
        tokenizer: Tokenizer
        test_problems: Test problems
        config: Configuration object
        num_rollouts: Number of rollouts per problem
        
    Returns:
        Benchmark results
    """
    generator = RolloutGenerator(model, value_model, tokenizer, config)
    
    # Warmup
    if test_problems:
        generator.generate_single_rollout(test_problems[0], 1)
    
    # Benchmark
    start_time = time.time()
    
    for problem in test_problems:
        generator.generate_single_rollout(problem, num_rollouts)
    
    total_time = time.time() - start_time
    
    # Compute metrics
    total_rollouts = len(test_problems) * num_rollouts
    rollouts_per_second = total_rollouts / total_time
    problems_per_second = len(test_problems) / total_time
    
    return {
        "total_time": total_time,
        "total_rollouts": total_rollouts,
        "rollouts_per_second": rollouts_per_second,
        "problems_per_second": problems_per_second,
        "avg_time_per_rollout": total_time / total_rollouts
    }


if __name__ == "__main__":
    # Test evaluation functions
    print("Testing evaluation functions...")
    
    # Test accuracy computation
    predictions = [123, 456, None, 789]
    ground_truths = [123, 456, 999, 789]
    accuracy = compute_accuracy(predictions, ground_truths)
    print(f"Accuracy: {accuracy:.3f}")
    
    # Test format correctness
    generations = [
        "The answer is 123",
        "No clear answer",
        "Result: 456",
        "Invalid format"
    ]
    format_correctness = compute_format_correctness_eval(generations)
    print(f"Format correctness: {format_correctness:.3f}")
    
    print("Evaluation tests completed")
