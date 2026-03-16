# LLM Consistency: Controlled Evaluation Framework

This repository implements a **modular, reproducible framework for studying LLM consistency and reliability** under meaning-preserving paraphrases.

It supports:

- Dataset expansion via controlled paraphrase generation  
- Zero-temperature answer generation across multiple models  
- LLM-based and ground-truth evaluation  
- Structured artifact logging for full reproducibility  
- Post-hoc consistency and reliability analysis via notebooks  

The framework is designed to make experiments:
- Reproducible  
- Inspectable  
- Model-agnostic  
- Easily extensible  

---

# Repository Structure

```
.
├── datasets/                  # Raw datasets (CSV)
├── run_artifacts/             # All experiment outputs (versioned)
├── src/llm_consistency/       # Core framework
│   ├── models/                # OpenAI, HF local, vLLM backends
│   ├── pipeline/              # Paraphrasing, answer generation, evaluation
│   ├── evaluators/            # Dataset-specific graders
│   ├── metrics/               # Semantic & consistency metrics
│   ├── prompts/               # Prompt templates
│   ├── datasets/              # Dataset registry
│   ├── io/                    # Artifact management
│   └── core/                  # Shared utilities
├── analysis_*.ipynb           # Post-hoc analysis notebooks
├── pyproject.toml             # Project configuration
└── uv.lock
```

---

# Installation

We use **uv** for environment and dependency management.

### 1. Install uv

```bash
pip install uv
```

### 2. Sync environment

```bash
uv sync
```

---

# Environment Variables

Set required API keys:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...   # if using custom endpoint
```

---

# Experimental Pipeline

Each experiment follows a controlled multi-stage pipeline:

## 1️⃣ Paraphrase Generation

- Meaning-preserving expansion
- Supports:
  - plain paraphrasing
  - toned paraphrasing
- Saved under:
  ```
  run_artifacts/<experiment_name>/paraphrases/
  ```

Config saved in:
```
config.paraphrases.json
```

---

## 2️⃣ Answer Generation

- Zero-temperature generation (temperature=0)
- Supports:
  - OpenAI API models
  - HuggingFace local models
  - vLLM backends
- Outputs per-model CSV files
- Saved under:
  ```
  run_artifacts/<experiment_name>/answers/
  ```

Config saved in:
```
config.answer_generation.json
```

---

## 3️⃣ Evaluation (Optional)

Some datasets support structured grading.

Evaluation outputs:
```
run_artifacts/<experiment_name>/evaluation/grades/
```

Config saved in:
```
config.evaluation.json
```

---

# Datasets Currently Included

Located under `datasets/`:

- SimpleQA
- TruthfulQA
- GSM8K
- MATH-500
- AIME 2024 (train)

Datasets are registered via:

```
src/llm_consistency/datasets/registry.py
```

---

# Running Experiments

Experiments are organized by dataset and experiment flag.

Example artifact directories:

```
run_artifacts/
├── SimpleQA_v00/
├── SimpleQA_test1500/
├── gsm8k_test_v00/
├── math_500_test_v00/
├── TruthfulQA_v00/
```

Each directory is **self-contained and reproducible**.

---

# Experiment Naming Convention

Experiment artifact directories follow:

```
<dataset>_<experiment_flag>/
```

The `experiment_flag` is intentionally flexible and encodes what makes the run distinguishable, such as:

- Version tags (`v00`)
- Dataset subset size (`test1500`)
- Evaluation variant
- Mitigation strategy
- Prompting configuration
- Model grouping

### Examples

```
SimpleQA_v00
SimpleQA_test1500
SimpleQA_test1500_mitigated
gsm8k_test_v00
math_500_test_v00
TruthfulQA_v00
```

Each directory is **self-contained and reproducible**, storing:

- Paraphrases
- Model answers
- Evaluation outputs (if applicable)
- Exact configuration files

This flexible naming scheme prioritizes clarity and traceability over rigid schema enforcement.

---

# Models

Supported model backends:

- `openai_api.py`
- `hf_local.py`
- `vllm_local.py`

Model construction handled by:

```
models/factory.py
```

---

# Prompts

Stored as plain text under:

```
src/llm_consistency/prompts/
```

Structure:

```
prompts/
├── answer/
├── paraphrase/
├── evaluation/
```

Prompts are version-controlled and referenced in run configs.

---

# Analysis

Post-hoc analysis notebooks:

- `analysis_SimpleQA_dataset.ipynb`
- `analysis_math_datasets.ipynb`

These compute:

- Original accuracy
- Conservative accuracy
- Maximum accuracy
- Reliability gap
- Latent capability gap
- Distribution shifts
- Semantic equivalence metrics

---

# Reproducibility Design

Each run stores:

- Paraphrases
- Model answers
- Grading results (if applicable)
- Exact configs
- Prompt references

Everything needed to fully reconstruct the experiment lives inside:

```
run_artifacts/<experiment_name>/
```

---

# Extending the Framework

### ➤ Add a new dataset

Register it in:
```
datasets/registry.py
```

### ➤ Add a new evaluator

Add it under:
```
evaluators/
```
and register in `evaluators/registry.py`.

### ➤ Add a new model backend

Add a model class under:
```
models/
```
and register in `factory.py`.

---

# Research Focus

This framework is built to study:

- LLM reliability beyond raw accuracy
- Consistency under meaning-preserving perturbations
- Latent vs displayed capability
- Conservative vs maximum accuracy
- Semantic robustness
