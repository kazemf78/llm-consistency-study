# ============================================================================
# File: src/llm_consistency/evaluators/registry.py
# ============================================================================
"""
Registry for dataset evaluators.
"""

from typing import Dict, Type, Any
from llm_consistency.evaluators.base import Evaluator


# Global registry (populated by imports)
EVALUATOR_REGISTRY: Dict[str, Type[Evaluator]] = {}


def register_evaluator(dataset: str, evaluator_class: Type[Evaluator]):
    """
    Register an evaluator for a dataset.
    
    Args:
        dataset: Dataset name
        evaluator_class: Evaluator class
    """
    EVALUATOR_REGISTRY[dataset] = evaluator_class


def get_evaluator(dataset: str, **kwargs) -> Evaluator:
    """
    Factory function to get evaluator for a dataset.
    
    Args:
        dataset: Dataset name
        **kwargs: Arguments to pass to evaluator constructor
        
    Returns:
        Evaluator instance
        
    Example:
        evaluator = get_evaluator("SimpleQA", judge_model="gpt-4o")
    """
    if dataset not in EVALUATOR_REGISTRY:
        raise ValueError(
            f"No evaluator registered for dataset '{dataset}'. "
            f"Available: {list(EVALUATOR_REGISTRY.keys())}"
        )
    
    evaluator_class = EVALUATOR_REGISTRY[dataset]
    return evaluator_class(**kwargs)



