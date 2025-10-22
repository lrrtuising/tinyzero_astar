"""
Data generation for multiplication problems.
"""
import random
import json
import os
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class Problem:
    """Represents a multiplication problem."""
    problem: str
    operand1: int
    operand2: int
    answer: int
    difficulty: str  # "easy", "medium", "hard"


def generate_multiplication_problems(
    n: int, 
    min_digits: int = 2, 
    max_digits: int = 3,
    seed: int = 42
) -> List[Problem]:
    """
    Generate n random multiplication problems.
    
    Args:
        n: Number of problems to generate
        min_digits: Minimum number of digits for operands
        max_digits: Maximum number of digits for operands
        seed: Random seed for reproducibility
        
    Returns:
        List of Problem objects
    """
    random.seed(seed)
    problems = []
    
    for _ in range(n):
        # Generate operand lengths
        digits1 = random.randint(min_digits, max_digits)
        digits2 = random.randint(min_digits, max_digits)
        
        # Generate operands
        operand1 = random.randint(10**(digits1-1), 10**digits1 - 1)
        operand2 = random.randint(10**(digits2-1), 10**digits2 - 1)
        
        # Calculate answer
        answer = operand1 * operand2
        
        # Create problem string
        problem = f"{operand1} * {operand2} = "
        
        # Determine difficulty
        total_digits = digits1 + digits2
        if total_digits <= 4:
            difficulty = "easy"
        elif total_digits <= 6:
            difficulty = "medium"
        else:
            difficulty = "hard"
        
        problems.append(Problem(
            problem=problem,
            operand1=operand1,
            operand2=operand2,
            answer=answer,
            difficulty=difficulty
        ))
    
    return problems


def format_prompt(problem: str) -> str:
    """
    Format problem into model input prompt.
    
    Args:
        problem: Problem string (e.g., "234 * 567 = ")
        
    Returns:
        Formatted prompt for the model
    """
    return f"Solve step by step:\n{problem}"


def create_dataset(config) -> Tuple[List[Problem], List[Problem]]:
    """
    Create train and eval datasets.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (train_problems, eval_problems)
    """
    # Generate training problems
    train_problems = generate_multiplication_problems(
        n=config.data.num_train_problems,
        min_digits=config.data.min_digits,
        max_digits=config.data.max_digits,
        seed=config.data.seed
    )
    
    # Generate evaluation problems
    eval_problems = generate_multiplication_problems(
        n=config.data.num_eval_problems,
        min_digits=config.data.min_digits,
        max_digits=config.data.max_digits,
        seed=config.data.seed + 1  # Different seed for eval
    )
    
    # Save datasets to disk for reproducibility
    save_problems(train_problems, os.path.join(config.data.data_dir, "train_problems.json"))
    save_problems(eval_problems, os.path.join(config.data.data_dir, "eval_problems.json"))
    
    return train_problems, eval_problems


def save_problems(problems: List[Problem], filepath: str) -> None:
    """
    Save problems to JSON file.
    
    Args:
        problems: List of Problem objects
        filepath: Path to save file
    """
    data = []
    for problem in problems:
        data.append({
            "problem": problem.problem,
            "operand1": problem.operand1,
            "operand2": problem.operand2,
            "answer": problem.answer,
            "difficulty": problem.difficulty
        })
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_problems(filepath: str) -> List[Problem]:
    """
    Load problems from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        List of Problem objects
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    problems = []
    for item in data:
        problems.append(Problem(
            problem=item["problem"],
            operand1=item["operand1"],
            operand2=item["operand2"],
            answer=item["answer"],
            difficulty=item["difficulty"]
        ))
    
    return problems


def get_problem_statistics(problems: List[Problem]) -> Dict:
    """
    Get statistics about a set of problems.
    
    Args:
        problems: List of Problem objects
        
    Returns:
        Dictionary with statistics
    """
    total = len(problems)
    difficulty_counts = {}
    digit_counts = {}
    
    for problem in problems:
        # Count by difficulty
        difficulty_counts[problem.difficulty] = difficulty_counts.get(problem.difficulty, 0) + 1
        
        # Count by total digits
        total_digits = len(str(problem.operand1)) + len(str(problem.operand2))
        digit_counts[total_digits] = digit_counts.get(total_digits, 0) + 1
    
    return {
        "total_problems": total,
        "difficulty_distribution": difficulty_counts,
        "digit_distribution": digit_counts,
        "min_answer": min(p.answer for p in problems),
        "max_answer": max(p.answer for p in problems),
        "avg_answer": sum(p.answer for p in problems) / total
    }


def create_difficulty_specific_dataset(
    n_easy: int = 300,
    n_medium: int = 300, 
    n_hard: int = 400,
    seed: int = 42
) -> List[Problem]:
    """
    Create dataset with specific difficulty distribution.
    
    Args:
        n_easy: Number of easy problems (2x2 digits)
        n_medium: Number of medium problems (3x2 digits)
        n_hard: Number of hard problems (3x3 digits)
        seed: Random seed
        
    Returns:
        List of Problem objects
    """
    problems = []
    
    # Easy problems (2x2 digits)
    easy_problems = generate_multiplication_problems(n_easy, 2, 2, seed)
    problems.extend(easy_problems)
    
    # Medium problems (3x2 digits)
    medium_problems = generate_multiplication_problems(n_medium, 3, 2, seed + 1)
    problems.extend(medium_problems)
    
    # Hard problems (3x3 digits)
    hard_problems = generate_multiplication_problems(n_hard, 3, 3, seed + 2)
    problems.extend(hard_problems)
    
    # Shuffle the combined dataset
    random.seed(seed + 3)
    random.shuffle(problems)
    
    return problems


if __name__ == "__main__":
    # Test data generation
    from config import get_local_config
    
    config = get_local_config()
    
    # Generate test problems
    problems = generate_multiplication_problems(10, 2, 3)
    
    print("Generated problems:")
    for i, problem in enumerate(problems[:5]):
        print(f"{i+1}. {problem.problem} (answer: {problem.answer}, difficulty: {problem.difficulty})")
    
    # Test dataset creation
    train_problems, eval_problems = create_dataset(config)
    
    print(f"\nTrain dataset: {len(train_problems)} problems")
    print(f"Eval dataset: {len(eval_problems)} problems")
    
    # Print statistics
    train_stats = get_problem_statistics(train_problems)
    print(f"\nTrain statistics: {train_stats}")
    
    eval_stats = get_problem_statistics(eval_problems)
    print(f"Eval statistics: {eval_stats}")
