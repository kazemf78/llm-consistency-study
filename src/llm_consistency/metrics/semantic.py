from typing import List, Literal
import os, json, numpy as np, pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache

import time

def timed(fn):
    def wrapper(*a, **kw):
        t0 = time.time()
        out = fn(*a, **kw)
        print(f"{fn.__name__} took {time.time()-t0:.2f}s")
        return out
    return wrapper
# =======================
# helper
# =======================
def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

# =======================
# 1) semantic embeddings
# =======================
@lru_cache(maxsize=2)
def _load_sentence_model(model_name: str):
    """Cache local embedding model to avoid reload overhead."""
    from sentence_transformers import SentenceTransformer
    device = "cuda" if _has_cuda() else "cpu"
    print(f"[embedding_consistency_matrix] Loading {model_name} on {device}")
    return SentenceTransformer(model_name, device=device)

@timed
def embedding_consistency_matrix(
    answers: List[str],
    backend: Literal["local", "openai"] = "local",
    openai_model: str = "text-embedding-3-large",
    local_model: str = "sentence-transformers/all-mpnet-base-v2",
    batch_size: int = 64,
) -> pd.DataFrame:
    """
    Compute pairwise cosine similarity of embeddings for a list of answers.

    - backend="local": uses SentenceTransformers
    - backend="openai": uses OpenAI Embeddings API
    """
    if len(answers) == 0:
        raise ValueError("answers list is empty")

    if backend == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        X = []
        for i in range(0, len(answers), batch_size):
            batch = answers[i:i + batch_size]
            r = client.embeddings.create(model=openai_model, input=batch)
            X.extend([d.embedding for d in r.data])
        X = np.array(X, dtype=np.float32)

    else:  # Local embedding model
        model = _load_sentence_model(local_model)
        X = model.encode(
            answers,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False
        )

    # Compute cosine similarity
    M = cosine_similarity(X)
    idx = [f"{i+1:02d}" for i in range(len(answers))]
    return pd.DataFrame(M, index=idx, columns=idx)


# =======================
# 2) NLI (prob-based)
# =======================

# =======================
# 3) LLM-as-judge semantic (score)
# =======================

import json, os, hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, pandas as pd
from openai import OpenAI

SYSTEM_PROMPT_CONSISTENCY = (
    "You are a strict evaluator of factual consistency between two short answers to the SAME question. "
    "Focus on the CLAIMED numbers/quantities and named entities along with the claimed facts. "
    "If the numerical claims differ (e.g., 120 vs 172), this is a CONTRADICTION unless explicitly hedged to be the same range. "
    "Return a compact JSON object ONLY."
)

CONSISTENCY_USER_TMPL = """You will judge if these two answers are factually consistent.

Answer A:
{a}

Answer B:
{b}

Rules:
- Consider whether both can be simultaneously true.
- Pay special attention to numeric claims (e.g., counts, dates). If numbers conflict, mark as contradiction.
- If they say essentially the same fact (even paraphrased), mark as consistent/entails.
- If you cannot tell, mark as neutral/uncertain.

Output strictly as JSON with keys:
- "decision": one of ["entails","contradicts","neutral"]
- "consistency_score": float in [0,1] where higher = more consistent (e.g., entails≈1.0, neutral≈0.5, contradicts≈0.0)
- "rationale": short phrase (<= 20 words)
"""

# New, NLI-specific prompts (different from the semantic ones)
SYSTEM_PROMPT_NLI = (
    "You perform Natural Language Inference (NLI) between TWO answers to the SAME question. "
    "Decide if the second answer can be true given the first. "
    "Focus on factual content, especially numbers/dates/counts/entities. "
    "Return JSON only."
)

NLI_USER_TMPL = """NLI classification (B given A).

Answer A:
{a}

Answer B:
{b}

Label rules:
- "entails": B is supported/implied by A (or equivalent).
- "contradicts": A and B cannot both be true (e.g., conflicting numbers for the same quantity).
- "neutral": cannot determine; could both be true but not implied; or different quantities.

Numeric guidance:
- Exact number vs different exact number for same quantity → contradicts.
- About/approximately allows ±5% (or ±1 if number < 30). If ranges overlap, not contradiction.

Output JSON only:
{{"decision": "entails" | "contradicts" | "neutral"}}
"""

def _hash_pair_dir(a: str, b: str) -> str:
    h = hashlib.sha256()
    h.update(b"DIR|||"); h.update(a.encode("utf-8")); h.update(b"->"); h.update(b.encode("utf-8"))
    return h.hexdigest()

def _hash_pair(a, b):
    h = hashlib.sha256()
    h.update(a.encode("utf-8")); h.update(b" ||| "); h.update(b.encode("utf-8"))
    return h.hexdigest()

def _parse_json_block(txt: str) -> dict:
    txt = txt.strip()
    if txt.startswith("```"):
        # strip fences if present
        txt = txt.strip("`")
        # keep only the first JSON-looking chunk
        s = txt.find("{"); e = txt.rfind("}")
        txt = txt[s:e+1] if (s != -1 and e != -1) else "{}"
    else:
        s = txt.find("{"); e = txt.rfind("}")
        txt = txt[s:e+1] if (s != -1 and e != -1) else "{}"
    try:
        return json.loads(txt)
    except Exception:
        return {}

@timed
def llm_judge_pairwise_fast(
    answers,
    model="gpt-4o-mini",
    temperature=0.0,
    max_workers=8,
    cache: dict | None = None,
    return_labels=False,
):
    """
    One mutual judgment per unordered pair (upper triangle), parallel + optional cache.
    Returns: scores_df (and labels_df if return_labels=True)
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    n = len(answers)
    scores = np.zeros((n, n), dtype=np.float32)
    labels = np.full((n, n), "entails", dtype=object)
    for i in range(n):
        scores[i, i] = 1.0

    def _judge_pair(i, j):
        A, B = answers[i], answers[j]
        key = _hash_pair(A, B)
        if cache is not None and key in cache:
            return i, j, cache[key]

        try:
            resp = client.chat.completions.create(
                model=model, temperature=temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_CONSISTENCY},
                    {"role": "user", "content": CONSISTENCY_USER_TMPL.format(a=A, b=B)},
                ],
            )
            data = _parse_json_block(resp.choices[0].message.content)
            score = float(data.get("consistency_score", 0.0))
            lab = str(data.get("decision", "neutral")).lower()
            if lab not in {"entails", "contradicts", "neutral"}:
                lab = "neutral"
        except Exception:
            score, lab = 0.5, "neutral"

        if cache is not None:
            cache[key] = (score, lab)
        return i, j, (score, lab)

    futures = []
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i in range(n):
            for j in range(i + 1, n):
                futures.append(ex.submit(_judge_pair, i, j))
        for fut in as_completed(futures):
            i, j, (s, lab) = fut.result()
            scores[i, j] = scores[j, i] = s
            labels[i, j] = labels[j, i] = lab

    idx = [f"{k+1:02d}" for k in range(n)]
    df_scores = pd.DataFrame(scores, index=idx, columns=idx)
    if return_labels:
        df_labels = pd.DataFrame(labels, index=idx, columns=idx)
        return df_scores, df_labels
    return df_scores

@timed
def llm_judge_nli_bidirectional_fast(
    answers,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_workers: int = 8,
    cache: dict | None = None,
    return_numeric: bool = True,
    retries: int = 0,
    fail_policy: str = "skip",
    backoff_sec: float = 0.6,
    return_stats: bool = True,   # NEW
):
    import time
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    n = len(answers)

    # ---- stats ----
    total_pairs     = n * (n - 1) // 2
    failed_dirs     = 0     # number of AB/BA direction calls that failed (after retries)
    disposed_pairs  = 0     # number of (i,j) pairs disposed due to any missing direction

    labels = np.full((n, n), None, dtype=object)
    for i in range(n): labels[i, i] = "entails"

    def _call_once(A, B):
        resp = client.chat.completions.create(
            model=model, temperature=temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_NLI},
                {"role": "user", "content": NLI_USER_TMPL.format(a=A, b=B)},
            ],
        )
        data = _parse_json_block(resp.choices[0].message.content)
        # print(f"DEBUG _call_once response: {data}")
        d = str(data.get("decision", "")).lower()
        if d not in {"entails", "contradicts", "neutral"}:
            raise ValueError("Invalid or missing 'decision'")
        return d

    def _decide(i, j, A, B, dirflag: str):
        key = (dirflag, _hash_pair_dir(A, B))
        if cache is not None and key in cache:
            return (i, j, dirflag, cache[key])

        attempt = 0
        while True:
            try:
                d = _call_once(A, B)
                if cache is not None:
                    cache[key] = d
                return (i, j, dirflag, d)
            except Exception:
                attempt += 1
                print(f"Warning: LLM NLI call failed for pair ({i},{j}) dir={dirflag} attempt {attempt}")
                # print("Exception details:", json.dumps(str(Exception), indent=2))
                # import traceback
                # traceback.format_exc()
                if attempt > retries:
                    if fail_policy == "raise":
                        raise
                    return (i, j, dirflag, None)  # signal failed direction
                time.sleep(backoff_sec * (1.6 ** (attempt - 1)))

    # launch parallel AB/BA
    futs = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i in range(n):
            for j in range(i + 1, n):
                A, B = answers[i], answers[j]
                futs.append(ex.submit(_decide, i, j, A, B, "AB"))
                futs.append(ex.submit(_decide, i, j, B, A, "BA"))

        tmp = {}  # (i,j) -> {"AB": label_or_None, "BA": label_or_None}
        for f in as_completed(futs):
            i, j, dirflag, lab = f.result()
            if lab is None:
                failed_dirs += 1
            key = (i, j)
            if key not in tmp: tmp[key] = {}
            tmp[key][dirflag] = lab

    # numeric scores DF (keep means; diagonal = 1.0)
    scores = np.full((n, n), np.nan, dtype=np.float32)
    for i in range(n): scores[i, i] = 1.0

    # aggregate: contradicts wins; else need BOTH; numeric = mean; label by threshold with equality→"borderline"
    for (i, j), d in tmp.items():
        ab, ba = d.get("AB", None), d.get("BA", None)

        # hard contradict
        if ab == "contradicts" or ba == "contradicts":
            labels[i, j] = labels[j, i] = "contradicts"
            scores[i, j] = scores[j, i] = 0.0
            continue

        # require BOTH directions; if any missing → dispose
        if (ab is None) or (ba is None):
            labels[i, j] = labels[j, i] = None
            scores[i, j] = scores[j, i] = np.nan
            disposed_pairs += 1
            continue

        # map/mean (keep 0.75 as 0.75)
        map_num = {"entails": 1.0, "neutral": 0.5}
        v_ab, v_ba = map_num.get(ab), map_num.get(ba)
        if v_ab is None or v_ba is None:
            labels[i, j] = labels[j, i] = None
            scores[i, j] = scores[j, i] = np.nan
            disposed_pairs += 1
            continue

        mean_score = (v_ab + v_ba) / 2.0
        scores[i, j] = scores[j, i] = mean_score

        if mean_score > 0.75:
            labels[i, j] = labels[j, i] = "entails"
        elif mean_score < 0.75:
            labels[i, j] = labels[j, i] = "neutral"
        else:
            labels[i, j] = labels[j, i] = "borderline"  # equality: don't coerce to 0.5 or 1.0

    idx = [f"{k+1:02d}" for k in range(n)]
    df_labels = pd.DataFrame(labels, index=idx, columns=idx)

    if not return_numeric and not return_stats:
        return df_labels, None

    df_numeric = pd.DataFrame(scores, index=idx, columns=idx)

    if return_stats:
        stats = {
            "total_pairs": total_pairs,
            "failed_dirs": failed_dirs,      # count of AB/BA calls that returned None
            "disposed_pairs": disposed_pairs # pairs dropped due to missing direction
        }
        return df_labels, df_numeric, stats

    return df_labels, df_numeric

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, numpy as np, pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, numpy as np, pandas as pd

from functools import lru_cache
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from contextlib import nullcontext

@lru_cache(maxsize=4)
def _load_nli(model_name: str, use_fp16: bool):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and use_fp16:
        model = model.half()
    model = model.to(device).eval()
    return tok, model, device

@timed
def nli_consistency_matrix_batched_fast(
    answers,
    model_name: str = "facebook/bart-large-mnli",
    init_batch_size: int = 128,  # ✅ adaptive start
    max_length: int = 256,
    use_fp16: bool = True,
    aggregate: bool = True,
    mode: str = "prob",          # "prob" -> probabilities, "label" -> 0/0.5/1
):
    """
    Compute NLI entailment scores or discrete labels for all answer pairs.
    Adaptive batch size: automatically reduces on OOM errors.
    mode="prob"  -> continuous entailment probability in [0,1]
    mode="label" -> numeric class mapping: contradiction=0, neutral=0.5, entailment=1
    """
    tok, model, device = _load_nli(model_name, use_fp16)
    n = len(answers)
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    M = np.eye(n, dtype=np.float32)
    lid = {"contradiction": 0, "neutral": 1, "entailment": 2}

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if (device == "cuda" and use_fp16)
        else nullcontext()
    )

    bs = init_batch_size
    k = 0
    with torch.inference_mode(), autocast_ctx:
        while k < len(pairs):
            chunk = pairs[k : k + bs]
            A = [answers[i] for i, j in chunk]
            B = [answers[j] for i, j in chunk]

            try:
                inputs = tok(
                    A, B,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=max_length
                ).to(device)

                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()

                if mode == "prob":
                    vals = probs[:, lid["entailment"]]  # continuous
                elif mode == "label":
                    idxs = probs.argmax(axis=1)
                    map_num = {0: 0.0, 1: 0.5, 2: 1.0}
                    vals = np.array([map_num[i] for i in idxs], dtype=np.float32)
                else:
                    raise ValueError("mode must be 'prob' or 'label'")

                for (i, j), v in zip(chunk, vals):
                    M[i, j] = v

                k += bs  # ✅ move to next batch

            except RuntimeError as e:
                if "CUDA out of memory" in str(e) and bs > 1:
                    bs = max(1, bs // 2)
                    print(f"[nli_consistency_matrix_batched_fast] OOM → reducing batch size to {bs}")
                    torch.cuda.empty_cache()
                else:
                    raise

    if aggregate:
        M = 0.5 * (M + M.T)
        np.fill_diagonal(M, 1.0)

    idx = [f"{i+1:02d}" for i in range(n)]
    return pd.DataFrame(M, index=idx, columns=idx)


import asyncio
from openai import AsyncOpenAI

# NOT DEBUGGED!
async def llm_judge_nli_bidirectional_fast_async(
    answers,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    cache: dict | None = None,
    retries: int = 1,
    fail_policy: str = "skip",
    backoff_sec: float = 0.6,
    return_numeric: bool = True,
    return_stats: bool = True,
):
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    n = len(answers)

    total_pairs = n * (n - 1) // 2
    failed_dirs, disposed_pairs = 0, 0
    labels = np.full((n, n), None, dtype=object)
    for i in range(n): labels[i, i] = "entails"

    async def call_once(A, B):
        """Single async API call with minimal parsing and retry."""
        for attempt in range(retries + 1):
            try:
                r = await client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_NLI},
                        {"role": "user", "content": NLI_USER_TMPL.format(a=A, b=B)},
                    ],
                )
                data = _parse_json_block(r.choices[0].message.content)
                d = str(data.get("decision", "")).lower()
                if d not in {"entails", "contradicts", "neutral"}:
                    raise ValueError("bad label")
                return d
            except Exception:
                if attempt == retries:
                    return None
                await asyncio.sleep(backoff_sec * (1.6 ** attempt))

    async def decide(i, j, A, B, dirflag):
        key = (dirflag, _hash_pair_dir(A, B))
        if cache is not None and key in cache:
            return (i, j, dirflag, cache[key])

        d = await call_once(A, B)
        if cache is not None:
            cache[key] = d
        return (i, j, dirflag, d)

    # Launch all AB/BA direction calls concurrently
    tasks = []
    for i in range(n):
        for j in range(i + 1, n):
            A, B = answers[i], answers[j]
            tasks.append(decide(i, j, A, B, "AB"))
            tasks.append(decide(i, j, B, A, "BA"))

    tmp = {}
    for i, j, dirflag, lab in await asyncio.gather(*tasks):
        if lab is None:
            failed_dirs += 1
        key = (i, j)
        if key not in tmp: tmp[key] = {}
        tmp[key][dirflag] = lab

    # ---- aggregation identical to your original ----
    scores = np.full((n, n), np.nan, dtype=np.float32)
    for i in range(n): scores[i, i] = 1.0

    for (i, j), d in tmp.items():
        ab, ba = d.get("AB"), d.get("BA")
        if ab == "contradicts" or ba == "contradicts":
            labels[i, j] = labels[j, i] = "contradicts"
            scores[i, j] = scores[j, i] = 0.0
            continue
        if (ab is None) or (ba is None):
            labels[i, j] = labels[j, i] = None
            disposed_pairs += 1
            continue
        map_num = {"entails": 1.0, "neutral": 0.5}
        v_ab, v_ba = map_num.get(ab), map_num.get(ba)
        if v_ab is None or v_ba is None:
            labels[i, j] = labels[j, i] = None
            disposed_pairs += 1
            continue

        mean_score = (v_ab + v_ba) / 2.0
        scores[i, j] = scores[j, i] = mean_score
        labels[i, j] = labels[j, i] = (
            "entails" if mean_score > 0.75
            else "neutral" if mean_score < 0.75
            else "borderline"
        )

    idx = [f"{k+1:02d}" for k in range(n)]
    df_labels = pd.DataFrame(labels, index=idx, columns=idx)
    df_numeric = pd.DataFrame(scores, index=idx, columns=idx)

    if return_stats:
        stats = dict(total_pairs=total_pairs, failed_dirs=failed_dirs, disposed_pairs=disposed_pairs)
        return df_labels, df_numeric, stats
    return df_labels, df_numeric

import re
import spacy
from functools import lru_cache
from tqdm import tqdm

@lru_cache(maxsize=1)
def get_spacy_model():
    """Load the SpaCy NER model once (GPU if available)."""
    try:
        spacy.require_gpu()
        print("[SpaCy] Using GPU for en_core_web_trf")
    except Exception:
        print("[SpaCy] GPU not available, using CPU")
    return spacy.load("en_core_web_trf") # or "en_core_web_md" for a more light-weight option

def preprocess_answers(answers, batch_size: int = 32, show_progress: bool = True):
    """
    Enriches each text with named entities and numeric values.

    Args:
        answers (List[str]): List of answer strings.
        batch_size (int): Number of texts to process per batch.
        show_progress (bool): Whether to show a tqdm progress bar.

    Returns:
        List[str]: Enriched answer texts.
    """
    nlp = get_spacy_model()
    enriched = []
    iterator = nlp.pipe(answers, batch_size=batch_size)
    if show_progress:
        iterator = tqdm(iterator, total=len(answers), desc="NER enrichment")

    for doc in iterator:
        ents = [f"{e.label_}:{e.text}" for e in doc.ents]
        nums = re.findall(r"\d+\.?\d*", doc.text)
        ent_text = "; ".join(ents + [f"NUM:{n}" for n in nums])
        enriched.append(f"[Facts: {ent_text}] {doc.text}" if ent_text else doc.text)

    return enriched

# 🧩 Example usage on a DataFrame
# df["answer_enriched"] = preprocess_answers(df["answer"].tolist(), batch_size=32)
# df.to_csv("answers_with_facts.csv", index=False)
