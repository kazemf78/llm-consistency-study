import os
import re
import glob
import math
import time
import random
import argparse
import pandas as pd
from typing import List, Tuple, Dict, Optional
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dotenv import load_dotenv

# from llm_consistency.models.hf_local import HFLocalLLM
from llm_consistency.models.openai_api import OpenAIAPILLM
from llm_consistency.metrics.semantic import nli_consistency_matrix_batched_fast

from pathlib import Path
from llm_consistency.core.paths import ProjectPaths
from llm_consistency.datasets.registry import get_dataset_spec
from llm_consistency.io.artifacts import save_pipeline_config
from llm_consistency.prompts import load_prompt

load_dotenv()

# ---------------------------- Helper Functions ----------------------------

_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _words(s: str) -> List[str]:
    return _WORD_RE.findall(s.lower())


def _lexical_overlap_ratio(a: str, b: str) -> float:
    """Returns ratio = (#common words) / min(len(a_words), len(b_words))."""
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return 0.0
    ca, cb = Counter(wa), Counter(wb)
    common = sum((ca & cb).values())
    return common / max(1, min(len(wa), len(wb)))


def _clean_line(s: str) -> str:
    """Normalize a single-line LLM output."""
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^(?:[-*•]\s+|\d+\.\s+)", "", s)
    return s




TONES: List[Tuple[str, str]] = [
    ("formal", "more professional phrasing"),
    ("casual", "relaxed, conversational feel"),
    # ("urgent", "subtle urgency without being pushy"),
    ("polite", "courteous tone (include 'please' if natural)"),
    ("confident", "decisive tone, no hedging"),
    ("neutral", "objective, even tone"),
    ("supportive", "reassuring, friendly tone"),
    ("inquisitive", "curious tone, optionally as a rhetorical question"),
]
# ---------------------------- Prompt Templates ----------------------------

PLAIN_TEMPLATE = load_prompt("paraphrase/plain")
TONED_TEMPLATE = load_prompt("paraphrase/toned")
JUDGE_EQUIV_TEMPLATE = load_prompt("paraphrase/judge_equivalence")


# ---------------------------- Paraphrase Generator ----------------------------

class ParaphraseGenerator:
    def __init__(self, resp_model: str = "gpt-4.1-mini", judge_model: str = "gpt-4.1-mini", api_key: Optional[str] = None):
        self.llm = OpenAIAPILLM(model=resp_model, api_key=api_key)
        self.judge = OpenAIAPILLM(model=judge_model, api_key=api_key)
        self.llm.prepare()
        self.judge.prepare()

    def _gen_plain_batch(self, q: str, n: int) -> List[Tuple[str, str]]:
        variations = [
            "avoid repeating key nouns verbatim if natural",
            "prefer different clause ordering",
            "vary syntax; try a different opener",
            "use synonyms for content words where safe",
            "compress slightly while keeping meaning",
            "expand slightly with harmless connective words",
            "switch from statement to question form (or vice versa) while preserving intent",
            "try a passive/active voice change if natural",
        ]
        prompts, tips = [], []
        for _ in range(n):
            tip = random.choice(variations)
            tips.append(tip)
            prompts.append(PLAIN_TEMPLATE.format(q=q) + f"\nDiversity tip: {tip}\n")

        outs = self.llm.batch(prompts, concurrency=10)
        cleaned = [(_clean_line(o), tip) for o, tip in zip(outs, tips) if o.strip()]
        return cleaned

    def _gen_toned_batch(self, q: str, n: int) -> List[Tuple[str, str]]:
        picks = random.sample(TONES, k=min(n, len(TONES))) if n <= len(TONES) else [random.choice(TONES) for _ in range(n)]
        prompts = [TONED_TEMPLATE.format(q=q, tone=t, desc=d) for (t, d) in picks]
        outs = self.llm.batch(prompts, concurrency=6)
        cleaned = [(_clean_line(o), f"{t} ({_d})") for o, (t, _d) in zip(outs, picks) if o and o.strip()]
        return cleaned

    def _judge_equivalence(self, orig: str, candidate: str) -> bool:
        prompt = JUDGE_EQUIV_TEMPLATE.format(orig=orig, cand=candidate)
        verdict = self.judge.single(prompt).strip().upper()
        return verdict.startswith("Y")  # YES

    def _passes_overlap(self, candidate: str, against: List[str], overlap_max: float) -> bool:
        return all(_lexical_overlap_ratio(candidate, ref) < overlap_max for ref in against)

    def generate(self, input_prompt: str, cfg: Dict[str, float], time_limit: Optional[int] = None) -> Dict[str, List[str]]:
        """Main paraphrase generation routine."""
        accepted_plain: List[Tuple[str, str]] = []  # (paraphrase, tip)
        accepted_toned: List[Tuple[str, str]] = []  # (paraphrase, tone)
        round_idx = 0
        paraphrase_attempts = 0
        st_time = time.time()

        while (
            len(accepted_plain) + len(accepted_toned) < cfg["target_total"]
            and paraphrase_attempts < cfg["max_attempts"]
        ):
            if time_limit and time.time() - st_time > time_limit:
                print("⏰ Time limit reached, stopping paraphrase generation.")
                break

            print(
                f"⏳ Round {round_idx + 1}: "
                f"{len(accepted_plain)} plain, {len(accepted_toned)} toned (attempts={paraphrase_attempts})"
            )

            plain_raw, toned_raw = [], []
            if len(accepted_plain) < cfg["target_plain"]:
                plain_raw = self._gen_plain_batch(input_prompt, cfg["round_plain"])
                paraphrase_attempts += cfg["round_plain"]
            if len(accepted_toned) < cfg["target_toned"]:
                toned_raw = self._gen_toned_batch(input_prompt, cfg["round_toned"])
                paraphrase_attempts += cfg["round_toned"]

            already = [input_prompt] + [p for p, _ in accepted_plain] + [p for p, _ in accepted_toned]

            # --- filter plain ---
            for cand, tip in plain_raw:
                if not self._passes_overlap(cand, already, cfg["overlap_max"]):
                    continue
                if not self._judge_equivalence(input_prompt, cand):
                    continue
                accepted_plain.append((cand, tip))
                already.append(cand)
                if len(accepted_plain) >= cfg["target_plain"]:
                    break

            # --- filter toned ---
            for cand, tone_label in toned_raw:
                if not self._passes_overlap(cand, already, cfg["overlap_max"]):
                    continue
                if not self._judge_equivalence(input_prompt, cand):
                    continue
                accepted_toned.append((cand, tone_label))
                already.append(cand)
                if len(accepted_toned) >= cfg["target_toned"]:
                    break

            # --- Local NLI filter ---
            try:
                all_candidates = [p for p, _ in accepted_plain] + [p for p, _ in accepted_toned]
                if all_candidates:
                    answers = [input_prompt] + all_candidates
                    nli_scores = nli_consistency_matrix_batched_fast(answers)
                    keep_plain, keep_toned = [], []
                    for i in range(1, len(answers)):
                        entail_score = nli_scores.iloc[0, i]
                        reverse_score = nli_scores.iloc[i, 0]
                        if (entail_score + reverse_score) / 2.0 >= 0.75:
                            text = answers[i]
                            keep_plain += [(p, tip) for (p, tip) in accepted_plain if p == text]
                            keep_toned += [(p, tone) for (p, tone) in accepted_toned if p == text]
                    accepted_plain, accepted_toned = keep_plain, keep_toned
                    print(f"🧠 NLI filter kept {len(keep_plain) + len(keep_toned)} total paraphrases (out of {len(all_candidates)}, {len(all_candidates) - (len(keep_plain) + len(keep_toned))}).")
            except Exception as e:
                print(f"⚠️ Local NLI filter failed: {e}")

            round_idx += 1
            print(len(accepted_plain), len(accepted_toned))
        final_plain = [p for p, _ in accepted_plain][:cfg["target_plain"]]
        final_toned = [p for p, _ in accepted_toned][:cfg["target_toned"]]
        print(f"### {len(final_plain)}, {len(final_toned)} ###")
        return {
            "all": final_plain + final_toned,
            "plain_with_tips": accepted_plain[:cfg["target_plain"]],
            "toned_with_tones": accepted_toned[:cfg["target_toned"]],
        }



# ---------------------------- Main Runner ----------------------------

def run_paraphrase_generation_pipeline(
    dataset: str = "SimpleQA",
    target_per_question: int = 10,
    num_rows: str = "5",
    experiment_flag: str = "v0",
    resume: bool = True,
    time_limit: int = 75,
    save_every: int = 5,
    executor_workers: int = 4,
):
    """
    Run the paraphrase generation pipeline for a dataset.
    """

    paths = ProjectPaths()
    rp = paths.run_paths(dataset, experiment_flag)
    rp.ensure_dirs()
    dataset_cfg = get_dataset_spec(dataset)
    dataset_path = paths.dataset_file(dataset_cfg["path"])
    df = pd.read_csv(dataset_path)

    if num_rows.strip().upper() == "ALL":
        num_rows_to_process = len(df)
    else:
        num_rows_to_process = int(num_rows)
    num_questions_to_paraphrase = len(df)

    if os.getenv("LEGACY_OUTPUT") == "1":
        paraphraser_dir = Path(f"paraphraser_data_{experiment_flag}_{dataset}")
        paraphraser_dir.mkdir(exist_ok=True)


    # Config dictionary
    target_plain = math.ceil(target_per_question * 0.7)
    target_toned = math.ceil(target_per_question * 0.4)
    cfg = {
        "target_total": target_per_question,
        "target_plain": target_plain,
        "target_toned": target_toned,
        "round_plain": target_plain * 2,
        "round_toned": target_toned * 2,
        "max_attempts": target_per_question * 12,
        "overlap_max": 0.8,
        "time_limit_per_question": time_limit,
    }

    RESPONSES_MODEL = os.getenv("PARA_MODEL", "gpt-4.1-mini")
    JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4.1-mini")

    run_config = {
        "dataset": dataset,
        "target_per_question": target_per_question,
        "num_rows": num_rows,
        "experiment_flag": experiment_flag,
        "response_model": RESPONSES_MODEL,
        "judge_model": JUDGE_MODEL,
        "paraphrase_cfg": cfg,
        "prompts": {
            "plain": "paraphrase/plain",
            "toned": "paraphrase/toned",
            "judge_equivalence": "paraphrase/judge_equivalence",
        }
    }
    save_pipeline_config(
        run_dir=rp.run_dir,
        pipeline_name="paraphrases",
        cfg=run_config,
    )

    # Dummy :D
    final_path = rp.paraphrases_file()
    if resume and final_path.exists():
        print(f"Last final file: {final_path}")
        # return

    last_file, last_index = rp.latest_paraphrase_checkpoint()

    timeout_count = 0
    new_rows, generated_questions, ids_processed = [], set(), set()
    if last_file:
        print(f"🔁 Resuming from {last_file}")
        generated_df = pd.read_csv(last_file)
        generated_questions = set(generated_df["original_question"].tolist())
        new_rows = generated_df.to_dict(orient="records")
        ids_processed = set(generated_df["idx"].tolist())

    executor = ThreadPoolExecutor(max_workers=executor_workers)
    gen = ParaphraseGenerator(resp_model=RESPONSES_MODEL, judge_model=JUDGE_MODEL)

    for idx, row in df.iloc[:num_rows_to_process].iterrows():
        question = row[dataset_cfg["question_col"]]
        if resume and (question in generated_questions or idx < last_index):
            continue
        if len(question.split()) < dataset_cfg["min_length"]:
            print(f"⚠️ Skipping short question (idx={idx})")
            continue

        print(f"\n🧩 [{idx}] Generating paraphrases for: {question[:80]}")
        st_time = time.time()
        future = executor.submit(gen.generate, question, cfg, time_limit)
        try:
            result = future.result(timeout=time_limit + 5)
        except FuturesTimeout:
            timeout_count += 1
            print(f"⚠️ Timeout on idx={idx}")
            result = {"all": [], "plain_with_tips": [], "toned_with_tones": []}

        rows_to_append = []
        for p, tip in result.get("plain_with_tips", []):
            rows_to_append.append({
                "idx": idx,
                "original_question": question,
                "paraphrased_question": p,
                "tip_or_tone": tip,
                "type": "plain",
            })

        for p, tone in result.get("toned_with_tones", []):
            rows_to_append.append({
                "idx": idx,
                "original_question": question,
                "paraphrased_question": p,
                "tip_or_tone": tone,
                "type": "toned",
            })
        new_rows += rows_to_append[:cfg["target_total"]]  # same as target_per_question
        ids_processed.add(idx)
        if len(ids_processed) % save_every == 0 and len(rows_to_append) > 0:
            save_path = rp.paraphrases_checkpoint_file(upto=idx+1)
            pd.DataFrame(new_rows).to_csv(save_path, index=False)
            print(f"💾 Progress saved at {save_path}")
        
        print(f"⏱️ Took {round(time.time() - st_time, 2)}s for idx={idx}")
        if len(ids_processed) >= num_questions_to_paraphrase:
            break
    print(f"TIMEOUT_COUNT: {timeout_count}")
    print(f"Saving the final result with {len(new_rows)} rows for dataset={dataset}, num_rows_processed={num_rows_to_process}, num_questions_paraphrased={len(ids_processed)}")

    save_path = rp.paraphrases_checkpoint_file(upto=num_rows_to_process)
    pd.DataFrame(new_rows).to_csv(save_path, index=False)
    final_path = rp.paraphrases_file()
    pd.DataFrame(new_rows).to_csv(final_path, index=False)
    print(f"✅ Final results saved: {final_path}")
    executor.shutdown(wait=False, cancel_futures=True)
    print(time.ctime())


# ---------------------------- CLI Entry ----------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Paraphrase generation pipeline")
    parser.add_argument("--dataset", type=str, default="SimpleQA")
    parser.add_argument("--target-per-question", type=int, default=10)
    parser.add_argument("--num-rows", type=str, default="5")
    parser.add_argument("--experiment-flag", type=str, default="v0")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Disable resume (default: resume is enabled)")
    parser.set_defaults(resume=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_paraphrase_generation_pipeline(
        dataset=args.dataset,
        target_per_question=args.target_per_question,
        num_rows=args.num_rows,
        experiment_flag=args.experiment_flag,
        resume=args.resume,
    )
