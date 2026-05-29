"""
benchmark_judge.py
------------------
Benchmarks LLM judge models against human annotations for semantic equivalence.

Usage:
    python benchmark_judge.py \
        --annotations path/to/annotations.csv \
        --judge-models gpt-4.1,gpt-4o \
        --prompt-names default,detailed \
        --output-dir path/to/output \
        --api-conc 16

    uv run benchmark_judge.py \
        --annotations annotation_splits_v00/all_datasets.csv \
        --judge-models gpt-5.4-mini,gpt-5,gpt-4.1,gpt-4.1-mini,gpt-4o \
        --prompt-names v0,default,detailed,with_examples \
        --output-dir annotation_splits_v00/tmp_benchmark_judge \
        --api-conc 64

    uv run benchmark_judge.py \
        --annotations annotation_splits_v00/all_datasets.csv \
        --judge-models gpt-5.4-mini,gpt-5,gpt-4.1,gpt-4.1-mini,gpt-4o \
        --prompt-names v0,v1,v2 \
        --output-dir annotation_splits_v00/tmp_benchmark_judge \
        --api-conc 64
Assumptions:
    - human_judge column: "same" = equivalent (positive), anything else = not equivalent (negative)
    - Configurable via --positive-label
"""

import time
import hashlib
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict
from dotenv import load_dotenv

from llm_consistency.models.factory import get_llm, is_api_model

load_dotenv()

# ---------------------------------------------------------------------------
# Prompt registry
# ---------------------------------------------------------------------------
# Add new prompts here — no other file needs changing.
# Each prompt must contain {orig} and {cand} placeholders.
# Must return a single token: YES or NO (case-insensitive prefix is checked).

PROMPTS: Dict[str, str] = {
    "v0": (
        """You are a precise evaluator of meaning.
Determine if the candidate paraphrase expresses the SAME intent and meaning as the original question, without adding/removing constraints or changing sentiment.
In other words, could they be used interchangeably in the same context? and if they are questions, do they ask the same thing? and imply the same thing? and carry the same assumptions? and can they be both answered the same way?

Original: {orig}
Candidate: {cand}

Respond with EXACTLY one token: YES or NO."""
    ),
    "default": (
        "Do the following two questions ask for exactly the same information?\n\n"
        "Original: {orig}\n"
        "Candidate: {cand}\n\n"
        "Answer YES if they are semantically equivalent, NO otherwise. "
        "Reply with only YES or NO."
    ),
    "detailed": (
        "You are evaluating whether two questions are semantically equivalent — "
        "that is, a correct answer to one would also be a correct answer to the other.\n\n"
        "Rules:\n"
        "- Minor wording changes, tone shifts, or rephrasing: YES\n"
        "- Different scope, added constraints, or changed subject: NO\n"
        "- If in doubt, answer NO\n\n"
        "Original: {orig}\n"
        "Candidate: {cand}\n\n"
        "Reply with only YES or NO."
    ),
    "with_examples": (
        "Decide if two questions are semantically equivalent.\n\n"
        "Examples:\n"
        "  Original: 'Who wrote Hamlet?' | Candidate: 'Who is the author of Hamlet?' → YES\n"
        "  Original: 'Who wrote Hamlet?' | Candidate: 'Who directed the film Hamlet?' → NO\n"
        "  Original: 'What year did WWII end?' | Candidate: 'When did the Second World War conclude?' → YES\n"
        "  Original: 'What is the capital of France?' | Candidate: 'What is the largest city in France?' → NO\n\n"
        "Now evaluate:\n"
        "Original: {orig}\n"
        "Candidate: {cand}\n\n"
        "Reply with only YES or NO."
    ),
    "v1": (
        """You are a strict semantic equivalence evaluator.

Your task is to determine whether two questions are EXACTLY equivalent in meaning and information content.

Two questions are equivalent if:

* They ask for the SAME information
* They impose the SAME constraints and assumptions
* They would ALWAYS lead to the SAME answer

Ignore differences in:

* Tone (e.g., polite, casual, inquisitive)
* Writing style or phrasing
* Order of information
* Minor rewording or synonyms
* Whether the question is phrased as a statement or a direct question

Allow:

* Reordering of details
* Compression or expansion of wording
* Referring to previously defined objects (e.g., "the given expression", "the diagram shown"), as long as the meaning is preserved

Mark as NOT equivalent ONLY if there is ANY semantic difference, including:

* Missing or altered numerical values, entities, or conditions
* Changed logical relationships (e.g., dependencies, timing, causality)
* Ambiguity that could lead to a different answer
* Loss of necessary information required to solve the problem

Be conservative: if there is ANY possible difference in the answer, output NO.

---

Original: {orig}
Candidate: {cand}

Respond with EXACTLY one token:
YES or NO
"""
    ),
    "v2": (
"""
You are a precise evaluator of question equivalence.

Your goal is to determine whether the Candidate question is EXACTLY equivalent to the Original question.

Two questions are equivalent if:

* They ask for the SAME information
* They impose the SAME constraints and assumptions
* They would ALWAYS produce the SAME answer

---

IGNORE differences in:

* Tone (polite, casual, inquisitive, etc.)
* Writing style or phrasing
* Order of information
* Minor rewording or synonyms
* Statement vs question form

---

ALLOW:

* Reordering of details
* Compression or expansion of wording
* Referring to objects implicitly (e.g., “the given expression”, “the diagram shown”), as long as the meaning is preserved

---

CRITICAL — mark as NO if ANY of the following occurs:

1. Missing or changed information:

* Any numbers, entities, or constraints are missing or altered

2. Logical differences:

* Different dependencies, conditions, or sequence of events

3. Ambiguity:

* The Candidate could lead to a different interpretation or answer

4. REASONING LEAKAGE (VERY IMPORTANT):

* The Candidate partially solves the problem
* Introduces derived values not explicitly stated in the Original
* Simplifies or transforms the problem using computation
* Removes intermediate reasoning steps by precomputing them

Even if the final answer would be the same, this is NOT equivalent.

---

Be strict:
If there is ANY possibility the two questions could behave differently, output NO.

---

### Examples:

Example 1 (VALID paraphrase)
Original: How many times did Phoebe impersonate Estelle in one episode?
Candidate: In a single episode, how often did Phoebe mimic Estelle?
Answer: YES

---

Example 2 (VALID paraphrase, reordered)
Original: Julie sold 14 glasses and the boys sold an equal number out of 32 total. How many more did Julie sell than Micah?
Candidate: Given 32 total glasses, with Julie selling 14 and the boys splitting the rest equally, how many more did Julie sell than Micah?
Answer: YES

---

Example 3 (INVALID — missing condition)
Original: A farmer buys 42 bales of hay weighing 75 pounds each... How many trips are needed?
Candidate: How many trips are needed to transport the farmer’s purchases with a 2250-pound truck?
Answer: NO

Reason: Missing key numerical details required to solve the problem.

---

Example 4 (INVALID — reasoning leakage)
Original: Rozanne uses 4 dozen eggs and 2 more eggs. Each glass needs 5 eggs. How many trays can she make?
Candidate: Rozanne uses 50 eggs, with each glass requiring 5 eggs. How many trays can she make?
Answer: NO

Reason: The candidate precomputes 4 dozen + 2 = 50, removing part of the reasoning.

---

Example 5 (INVALID — logical change)
Original: Walter tells his lawyer after his friend gives $200.
Candidate: The lawyer donates based on neighbors and online fund contributions.
Answer: NO

Reason: The dependency on the friend’s contribution is removed.

---

Example 6 (VALID — math reference)
Original: Find θ such that tanθ = [long expression]
Candidate: What is θ such that the given expression for tanθ holds?
Answer: YES

---

Now evaluate:

Original: {orig}
Candidate: {cand}

Respond with EXACTLY one token:
YES or NO
"""
    )
}


# ---------------------------------------------------------------------------
# Judge batch runner — uses your model classes directly
# ---------------------------------------------------------------------------

def _run_judge_batch(
    rows: List[Dict],
    model_name: str,
    prompt_template: str,
    api_conc: int = 16,
) -> List[Dict]:
    """
    Run judge over rows using OpenAIAPILLM or get_llm (local support included).
    Returns enriched row dicts with 'llm_raw' and 'llm_equiv' added.
    """
    prompts = [
        prompt_template.format(
            orig=row["original_question"],
            cand=row["paraphrased_question"],
        )
        for row in rows
    ]

    # max_output_tokens=10: judge only needs YES/NO
    llm = get_llm(model_name, max_tokens=1024, temperature=0.0)
    llm.prepare()

    if is_api_model(model_name):
        raw_responses = llm.batch(prompts, concurrency=api_conc)
    else:
        raw_responses = llm.batch(prompts)

    return [
        {
            **row,
            "llm_raw": raw,
            "llm_equiv": raw.strip().upper().startswith("Y"),
        }
        for row, raw in zip(rows, raw_responses)
    ]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(df: pd.DataFrame, positive_label: str) -> Dict:
    """
    Compute agreement metrics between human_judge and llm_equiv.

    positive_label: the human_judge value that means "equivalent" (e.g. "same").
    """
    human_pos = df["human_judge"].str.strip().str.lower() == positive_label.lower()
    llm_pos = df["llm_equiv"].astype(bool)

    tp = (human_pos & llm_pos).sum()
    tn = (~human_pos & ~llm_pos).sum()
    fp = (~human_pos & llm_pos).sum()
    fn = (human_pos & ~llm_pos).sum()

    total = len(df)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    # False positive rate: LLM says "same" but human says "different"
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    # False negative rate: LLM says "different" but human says "same"
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    return {
        "total": total,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_positive_rate": round(fpr, 4),  # LLM over-accepts
        "false_negative_rate": round(fnr, 4),  # LLM over-rejects
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark(
    annotations_path: str,
    judge_models: List[str],
    prompt_names: List[str],
    output_dir: str,
    positive_label: str = "same",
    api_conc: int = 16,
    datasets_filter: Optional[List[str]] = None,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ann = pd.read_csv(annotations_path)
    print(f"Loaded {len(ann)} annotated rows from {annotations_path}")

    # Validate required columns
    required = {"original_question", "paraphrased_question", "human_judge"}
    missing = required - set(ann.columns)
    if missing:
        raise ValueError(f"Annotations file missing columns: {missing}")

    if datasets_filter:
        if "dataset" in ann.columns:
            ann = ann[ann["dataset"].isin(datasets_filter)]
            print(f"Filtered to datasets {datasets_filter}: {len(ann)} rows")
        else:
            print("Warning: --datasets filter specified but no 'dataset' column found, ignoring.")

    # Add stable content hash for future cache compatibility
    ann["para_hash"] = ann.apply(
        lambda r: hashlib.sha256(
            f"{r['original_question']}|||{r['paraphrased_question']}".encode()
        ).hexdigest()[:16],
        axis=1,
    )

    rows = ann.to_dict("records")

    all_metrics = []

    for model in judge_models:
        for prompt_name in prompt_names:
            if prompt_name not in PROMPTS:
                print(f"⚠️  Unknown prompt '{prompt_name}', skipping. Available: {list(PROMPTS.keys())}")
                continue

            print(f"\n{'='*60}")
            print(f"Judge: {model} | Prompt: {prompt_name}")
            print(f"{'='*60}")
            st = time.time()

            enriched = _run_judge_batch(
                rows,
                model_name=model,
                prompt_template=PROMPTS[prompt_name],
                api_conc=api_conc,
            )

            elapsed = round(time.time() - st, 2)
            pred_df = pd.DataFrame(enriched)
            pred_df["judge_model"] = model
            pred_df["prompt_name"] = prompt_name

            metrics = _compute_metrics(pred_df, positive_label=positive_label)
            metrics["judge_model"] = model
            metrics["prompt_name"] = prompt_name
            metrics["elapsed_sec"] = elapsed

            all_metrics.append(metrics)

            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}  (of LLM 'same', how many are truly same)")
            print(f"  Recall:    {metrics['recall']:.4f}  (of truly same, how many LLM catches)")
            print(f"  F1:        {metrics['f1']:.4f}")
            print(f"  FPR:       {metrics['false_positive_rate']:.4f}  (LLM over-accepts)")
            print(f"  FNR:       {metrics['false_negative_rate']:.4f}  (LLM over-rejects)")
            print(f"  Time: {elapsed}s")

            # Save per-combination predictions
            safe_model = model.replace("/", "_")
            pred_path = out / f"predictions_{safe_model}_{prompt_name}.csv"
            pred_df.to_csv(pred_path, index=False)
            print(f"  Saved predictions → {pred_path}")

    # Save summary
    summary_df = pd.DataFrame(all_metrics).sort_values("f1", ascending=False)
    summary_path = out / "benchmark_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n{'='*60}")
    print(f"📊 Benchmark summary saved → {summary_path}")
    print(summary_df[["judge_model", "prompt_name", "accuracy", "precision", "recall", "f1",
                       "false_positive_rate", "false_negative_rate"]].to_string(index=False))

    return summary_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark LLM judges vs human annotations")
    parser.add_argument("--annotations", required=True,
                        help="Path to annotated CSV with human_judge column")
    parser.add_argument("--judge-models", type=csv_list, default="gpt-4.1-mini",
                        help="Comma-separated judge model names")
    parser.add_argument("--prompt-names", type=csv_list, default="default",
                        help=f"Comma-separated prompt names. Available: {list(PROMPTS.keys())}")
    parser.add_argument("--output-dir", default="benchmark_results",
                        help="Directory to save results")
    parser.add_argument("--positive-label", default="same",
                        help="human_judge value that means 'equivalent' (default: 'same')")
    parser.add_argument("--api-conc", type=int, default=16,
                        help="API concurrency")
    parser.add_argument("--datasets", type=csv_list, default=None,
                        help="Filter to specific datasets (comma-separated)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(
        annotations_path=args.annotations,
        judge_models=args.judge_models,
        prompt_names=args.prompt_names,
        output_dir=args.output_dir,
        positive_label=args.positive_label,
        api_conc=args.api_conc,
        datasets_filter=args.datasets,
    )

