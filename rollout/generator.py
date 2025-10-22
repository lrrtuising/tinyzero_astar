"""
Rollout generator for TinyZero A*PO using A* search.
"""
import torch
import re
from typing import List, Dict, Optional, Tuple, Any
import logging

from .astar_search import AStarSearcher
from model.model import get_generation_config

logger = logging.getLogger(__name__)


class RolloutGenerator:
    """
    Generates rollouts using A* search for entire dataset.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        value_model: torch.nn.Module,
        tokenizer: Any,
        config
    ):
        """
        Initialize rollout generator.
        
        Args:
            model: Language model for policy
            value_model: Value model for A* heuristic
            tokenizer: Tokenizer
            config: Configuration object
        """
        self.model = model
        self.value_model = value_model
        self.tokenizer = tokenizer
        self.config = config
        
        # Initialize A* searcher
        self.astar_searcher = AStarSearcher(
            model=model,
            value_model=value_model,
            tokenizer=tokenizer,
            config=config
        )
        
        logger.info("Rollout generator initialized")
    
    def generate_rollouts(
        self,
        problems: List[Dict],
        num_rollouts_per_problem: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Generate rollouts for batch of problems.
        
        Args:
            problems: List of problem dictionaries
            num_rollouts_per_problem: Number of rollouts per problem
            
        Returns:
            List of rollout dictionaries
        """
        all_rollouts = []
        
        for i, problem in enumerate(problems):
            logger.info(f"Generating rollouts for problem {i+1}/{len(problems)}")
            
            # Handle both Problem objects and dictionaries
            if hasattr(problem, 'problem'):
                # Problem object
                problem_text = problem.problem
                answer = problem.answer
            else:
                # Dictionary
                problem_text = problem["problem"]
                answer = problem["answer"]
            
            # Format prompt
            prompt = self._format_prompt(problem_text)
            
            # Generate rollouts using A* search
            rollouts = self.astar_searcher.search(
                prompt=prompt,
                target_answer=answer,
                num_rollouts=num_rollouts_per_problem
            )
            
            # Add problem metadata to each rollout
            for rollout in rollouts:
                rollout["problem_id"] = i
                rollout["prompt"] = prompt
                rollout["ground_truth_answer"] = answer
                rollout["extracted_answer"] = self._extract_answer(rollout["generated_text"])
            
            all_rollouts.extend(rollouts)
        
        logger.info(f"Generated {len(all_rollouts)} rollouts for {len(problems)} problems")
        return all_rollouts
    
    def _format_prompt(self, problem: str) -> str:
        """
        Format problem into model input prompt.
        
        Args:
            problem: Problem string (e.g., "234 * 567 = ")
            
        Returns:
            Formatted prompt
        """
        return f"Solve step by step:\n{problem}"
    
    def _extract_answer(self, generated_text: str) -> Optional[int]:
        """
        Extract final answer from generated text.
        
        Args:
            generated_text: Generated text from model
            
        Returns:
            Extracted answer as integer, or None if not found
        """
        # Remove the prompt part if present
        if "Solve step by step:" in generated_text:
            generated_text = generated_text.split("Solve step by step:")[-1].strip()
        
        # Try multiple patterns to extract answer
        patterns = [
            r'= (\d+)',  # "= 123"
            r'answer[:\s]*(\d+)',  # "answer: 123" or "answer 123"
            r'(\d+)$',  # Number at end of line
            r'(\d+)\s*$',  # Number at end with optional whitespace
            r'(\d+)\s*\.?\s*$',  # Number at end with optional period
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, generated_text, re.IGNORECASE | re.MULTILINE)
            if matches:
                try:
                    # Take the last match (most likely to be the final answer)
                    answer = int(matches[-1])
                    return answer
                except ValueError:
                    continue
        
        # If no pattern matches, try to find any number in the text
        numbers = re.findall(r'\d+', generated_text)
        if numbers:
            try:
                # Take the largest number (likely to be the answer)
                return max(int(num) for num in numbers)
            except ValueError:
                pass
        
        return None
    
    def batch_generate(
        self,
        prompts: List[str],
        generation_config: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Efficient batched generation (fallback to regular generation).
        
        Args:
            prompts: List of prompts
            generation_config: Generation configuration
            
        Returns:
            List of generation dictionaries
        """
        if generation_config is None:
            generation_config = get_generation_config()
        
        # For now, use A* search for each prompt individually
        # TODO: Implement true batching for efficiency
        all_generations = []
        
        for prompt in prompts:
            rollouts = self.astar_searcher.search(prompt, num_rollouts=1)
            if rollouts:
                generation = rollouts[0]
                generation["prompt"] = prompt
                all_generations.append(generation)
        
        return all_generations
    
    def generate_single_rollout(
        self,
        problem: Dict,
        num_rollouts: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Generate rollouts for a single problem.
        
        Args:
            problem: Problem dictionary or Problem object
            num_rollouts: Number of rollouts to generate
            
        Returns:
            List of rollout dictionaries
        """
        # Handle both Problem objects and dictionaries
        if hasattr(problem, 'problem'):
            # Problem object
            problem_text = problem.problem
            answer = problem.answer
        else:
            # Dictionary
            problem_text = problem["problem"]
            answer = problem["answer"]
        
        prompt = self._format_prompt(problem_text)
        
        rollouts = self.astar_searcher.search(
            prompt=prompt,
            target_answer=answer,
            num_rollouts=num_rollouts
        )
        
        # Add problem metadata
        for rollout in rollouts:
            rollout["problem_id"] = 0
            rollout["prompt"] = prompt
            rollout["ground_truth_answer"] = answer
            rollout["extracted_answer"] = self._extract_answer(rollout["generated_text"])
        
        return rollouts
    
    def get_rollout_statistics(self, rollouts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about generated rollouts.
        
        Args:
            rollouts: List of rollout dictionaries
            
        Returns:
            Dictionary with statistics
        """
        if not rollouts:
            return {}
        
        # Basic statistics
        total_rollouts = len(rollouts)
        avg_length = sum(len(r["tokens"]) for r in rollouts) / total_rollouts
        avg_f_score = sum(r["f_score"] for r in rollouts) / total_rollouts
        avg_depth = sum(r["depth"] for r in rollouts) / total_rollouts
        
        # Answer extraction statistics
        extracted_answers = [r["extracted_answer"] for r in rollouts if r["extracted_answer"] is not None]
        extraction_rate = len(extracted_answers) / total_rollouts
        
        # Accuracy statistics (if ground truth available)
        correct_answers = 0
        if rollouts and "ground_truth_answer" in rollouts[0]:
            for rollout in rollouts:
                if (rollout["extracted_answer"] is not None and 
                    rollout["extracted_answer"] == rollout["ground_truth_answer"]):
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


def create_rollout_generator(
    model: torch.nn.Module,
    value_model: torch.nn.Module,
    tokenizer: Any,
    config
) -> RolloutGenerator:
    """
    Create rollout generator.
    
    Args:
        model: Language model
        value_model: Value model
        tokenizer: Tokenizer
        config: Configuration
        
    Returns:
        RolloutGenerator instance
    """
    return RolloutGenerator(model, value_model, tokenizer, config)


if __name__ == "__main__":
    # Test rollout generator
    print("Rollout generator implementation completed")
    print("Note: Full testing requires loaded models and tokenizer")
