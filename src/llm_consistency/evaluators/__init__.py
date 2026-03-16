
# ============================================================================
# File: src/llm_consistency/evaluators/__init__.py
# ============================================================================
"""
Evaluators for different datasets.

Usage:
    from llm_consistency.evaluators import SimpleQAEvaluator, get_evaluator
    
    # Direct instantiation
    evaluator = SimpleQAEvaluator(judge_model="gpt-4o")
    
    # Factory pattern
    evaluator = get_evaluator("SimpleQA", judge_model="gpt-4o")
"""

from llm_consistency.evaluators.base import Evaluator, EnsembleEvaluator
from llm_consistency.evaluators.simple_qa import SimpleQAEvaluator
from llm_consistency.evaluators.truthful_qa import TruthfulQAEvaluator
# from llm_consistency.evaluators.gsm8k import GSM8KEvaluator
# from llm_consistency.evaluators.math500 import MATH500Evaluator
from llm_consistency.evaluators.registry import (
    get_evaluator,
    register_evaluator,
    EVALUATOR_REGISTRY,
)

__all__ = [
    # Base classes
    "Evaluator",
    "EnsembleEvaluator",
    
    # Dataset-specific evaluators
    "SimpleQAEvaluator",
    "TruthfulQAEvaluator",
    # "GSM8KEvaluator",
    # "MATH500Evaluator",
    
    # Factory
    "get_evaluator",
    "register_evaluator",
    "EVALUATOR_REGISTRY",
]