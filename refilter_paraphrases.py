# refilter_paraphrases.py

import time
import hashlib
import argparse
import pandas as pd
from typing import List, Optional, Dict
from dotenv import load_dotenv
from tqdm import tqdm

from llm_consistency.core.paths import ProjectPaths
from llm_consistency.models.factory import get_llm, is_api_model
from llm_consistency.prompts import load_prompt

load_dotenv()

# ---------------------------------------------------------------------------
# Prompt registry
# ---------------------------------------------------------------------------
PROMPTS: Dict[str, str] = {
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
        "  Original: 'Who wrote Hamlet?' | Candidate: 'Who is the author of Hamlet?' -> YES\n"
        "  Original: 'Who wrote Hamlet?' | Candidate: 'Who directed the film Hamlet?' -> NO\n"
        "  Original: 'What year did WWII end?' | Candidate: 'When did the Second World War conclude?' -> YES\n"
        "  Original: 'What is the capital of France?' | Candidate: 'What is the largest city in France?' -> NO\n\n"
        "Now evaluate:\n"
        "Original: {orig}\n"
        "Candidate: {cand}\n\n"
        "Reply with only YES or NO."
    ),
}

# ---------------------------------------------------------------------------
# Hashing utilities
# ---------------------------------------------------------------------------

def _para_hash(original: str, paraphrased: str) -> str:
    key = f"{original.strip()}|||{paraphrased.strip()}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _orig_hash(original: str) -> str:
    return hashlib.sha256(original.strip().encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Judge batch runner WITH CHUNK SAVING
# ---------------------------------------------------------------------------

def _run_judge_batch(
    rows: List[Dict],
    model_name: str,
    prompt_template: str,
    api_conc: int = 16,
    chunk_size: int = 100,
    max_tokens: int = 32,
    save_every_chunk: bool = True,
    intermediate_path: Optional[str] = None,
) -> List[Dict]:
    llm = get_llm(model_name, max_tokens=max_tokens, temperature=0.0)
    llm.prepare()

    results = []
    yes_count = 0
    no_count = 0

    chunks = [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]

    with tqdm(total=len(rows), unit="pair", desc=f"Judging [{model_name}]") as pbar:
        for chunk_idx, chunk in enumerate(chunks):
            prompts = [
                prompt_template.format(
                    orig=row["original_question"],
                    cand=row["paraphrased_question"],
                )
                for row in chunk
            ]

            if is_api_model(model_name):
                raw_responses = llm.batch(prompts, concurrency=min(api_conc, len(prompts)))
            else:
                raw_responses = llm.batch(prompts)

            for row, raw in zip(chunk, raw_responses):
                is_equiv = raw.strip().upper().startswith("Y")

                results.append({
                    **row,
                    "llm_raw": raw,
                    "llm_equiv": is_equiv,
                    "judgment_source": "llm",
                })

                if is_equiv:
                    yes_count += 1
                else:
                    no_count += 1

            pbar.update(len(chunk))
            pbar.set_postfix(kept=yes_count, removed=no_count, refresh=True)

            # -------------------------
            # SAVE AFTER EACH CHUNK
            # -------------------------
            if save_every_chunk and intermediate_path is not None:
                pd.DataFrame(results).to_csv(intermediate_path, index=False)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_refilter(
    dataset: str,
    experiment_flag: str,
    judge_model: str,
    prompt_name: str,
    annotations_path: Optional[str] = None,
    reuse_annotations: bool = False,
    positive_label: str = "same",
    api_conc: int = 16,
    chunk_size: int = 100,
    max_tokens: int = 1024,
    output_tag: Optional[str] = None,
):
    if prompt_name not in PROMPTS:
        PROMPTS[prompt_name] = load_prompt(f"paraphrase/{prompt_name}")

    paths = ProjectPaths()
    rp = paths.run_paths(dataset, experiment_flag)

    paraphrases_path = rp.paraphrases_file()
    filtered_path = rp.paraphrases_filtered_file(judge_model, prompt_name, output_tag)
    full_path = rp.paraphrases_full_judged_file(judge_model, prompt_name, output_tag)
    removed_path = rp.paraphrases_removed_file(judge_model, prompt_name, output_tag)

    # 🔥 intermediate file
    intermediate_path = str(full_path).replace(".csv", "_partial.csv")

    print(f"Dataset: {dataset} / {experiment_flag}")
    print(f"Input: {paraphrases_path}")
    print(f"Intermediate: {intermediate_path}")

    para_df = pd.read_csv(paraphrases_path)

    para_df["para_hash"] = para_df.apply(
        lambda r: _para_hash(r["original_question"], r["paraphrased_question"]), axis=1
    )
    para_df["orig_hash"] = para_df["original_question"].apply(_orig_hash)

    annotation_lookup = {}
    if annotations_path and reuse_annotations:
        ann = pd.read_csv(annotations_path)
        for _, row in ann.iterrows():
            h = _para_hash(row["original_question"], row["paraphrased_question"])
            annotation_lookup[h] = str(row["human_judge"]).strip().lower()

    rows_to_judge, rows_from_annotation = [], []

    for _, row in para_df.iterrows():
        h = row["para_hash"]
        if h in annotation_lookup:
            rows_from_annotation.append({
                **row.to_dict(),
                "llm_raw": f"[human annotation]",
                "llm_equiv": annotation_lookup[h] == positive_label.lower(),
                "judgment_source": "human",
            })
        else:
            rows_to_judge.append(row.to_dict())

    judged_rows = []

    if rows_to_judge:
        judged_rows = _run_judge_batch(
            rows_to_judge,
            model_name=judge_model,
            prompt_template=PROMPTS[prompt_name],
            api_conc=api_conc,
            chunk_size=chunk_size,
            max_tokens=max_tokens,
            save_every_chunk=True,
            intermediate_path=intermediate_path,
        )

    all_judged = rows_from_annotation + judged_rows
    judged_df = pd.DataFrame(all_judged)

    kept_df = judged_df[judged_df["llm_equiv"] == True]
    removed_df = judged_df[judged_df["llm_equiv"] == False]

    kept_df.to_csv(filtered_path, index=False)
    judged_df.to_csv(full_path, index=False)
    removed_df.to_csv(removed_path, index=False)

    print("\nDone.")
    return filtered_path, full_path, removed_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--experiment-flag", required=True)
    parser.add_argument("--judge-model", default="gpt-4.1")
    parser.add_argument("--prompt-name", default="detailed")
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--reuse-annotations", action="store_true")
    parser.add_argument("--positive-label", default="same")
    parser.add_argument("--api-conc", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--output-tag", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_refilter(
        dataset=args.dataset,
        experiment_flag=args.experiment_flag,
        judge_model=args.judge_model,
        prompt_name=args.prompt_name,
        annotations_path=args.annotations,
        reuse_annotations=args.reuse_annotations,
        positive_label=args.positive_label,
        api_conc=args.api_conc,
        chunk_size=args.chunk_size,
        max_tokens=args.max_tokens,
        output_tag=args.output_tag,
    )