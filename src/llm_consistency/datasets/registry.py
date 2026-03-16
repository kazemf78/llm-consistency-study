# src/llm_consistency/datasets/registry.py

DATASET_REGISTRY = {
    "SimpleQA": {
        "path": "simple_qa_test_set.csv",
        "question_col": "problem",
        "min_length": 10,
        "task_type": "factual_qa",
    },
    "TruthfulQA": {
        "path": "TruthfulQA.csv",
        "question_col": "Question",
        "min_length": 8,
        "task_type": "factual_qa",
    },
    "gsm8k_test": {
        "path": "openai_gsm8k_main_test.csv",
        "question_col": "question",
        "min_length": 10,
        "task_type": "math",
    },
    "math_500_test": {
        "path": "HuggingFaceH4_MATH_500_test.csv",
        "question_col": "problem",
        "min_length": 10,
        "task_type": "math",
    },
    "aime_2024_train": {
        "path": "HuggingFaceH4_aime_2024_train.csv",
        "question_col": "problem",
        "min_length": 10,
        "task_type": "math",
    },
}

def get_dataset_spec(name: str) -> dict:
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset: {name}")
    return DATASET_REGISTRY[name]

# tddo: Add answer_col in future if needed!
