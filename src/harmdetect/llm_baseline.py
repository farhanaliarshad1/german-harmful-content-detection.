"""Zero-/few-shot LLM baseline for fake news, hate speech and toxicity.

One call per post returns all three labels ("a,b,c"). Evaluation uses the merged
corpus, where each row is annotated for one task (-1 elsewhere); metrics for a
task are computed only on rows labelled for it.

Works with any OpenAI-compatible chat endpoint:
  * OpenAI:  export OPENAI_API_KEY=...            --model gpt-4o-mini
  * Ollama:  ollama serve                          --base_url http://localhost:11434/v1 --model llama3.1:8b

Answers are cached to JSONL so interrupted runs resume. Unparseable answers are
counted as *invalid* and reported, never silently mapped to class 0.

Usage:
    python -m harmdetect.llm_baseline --data data/merged_dataset.csv --model gpt-4o-mini --n 600 --few_shot
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from harmdetect.data import clean_text
from harmdetect.prompts import TASK_ORDER, build_messages

LABEL_COLUMNS = {"fake_news": "isFake", "hate_speech": "isHateSpeech", "toxicity": "isToxic"}
ANSWER_RE = re.compile(r"([01])\s*,\s*([01])\s*,\s*([01])")


def parse_labels(answer):
    """Parse '0,1,1' (tolerating spaces / surrounding text). None if there is no valid triple."""
    m = ANSWER_RE.search(answer or "")
    return tuple(int(g) for g in m.groups()) if m else None


async def classify_all(df: pd.DataFrame, args, cache_path: Path) -> dict:
    from openai import AsyncOpenAI  # optional dependency

    done: dict = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            row["pred"] = tuple(row["pred"]) if row["pred"] is not None else None
            done[row["id"]] = row
    todo = df[~df["id"].isin(list(done))]
    print(f"{len(done)} cached, {len(todo)} to classify with {args.model}")

    client = AsyncOpenAI(base_url=args.base_url, api_key=os.environ.get("OPENAI_API_KEY", "ollama"))
    sem = asyncio.Semaphore(args.concurrency)

    async def one(idx: int, text: str, fh) -> None:
        async with sem:
            answer = None
            for attempt in range(3):
                try:
                    resp = await client.chat.completions.create(
                        model=args.model,
                        messages=build_messages(text, few_shot=args.few_shot),
                        temperature=0.0,
                        max_tokens=10,
                    )
                    answer = resp.choices[0].message.content
                    break
                except Exception as exc:  # network / rate limit
                    answer = f"ERROR: {exc}"
                    await asyncio.sleep(2 ** attempt)
            row = {"id": idx, "answer": answer, "pred": parse_labels(answer)}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            done[idx] = row

    with cache_path.open("a", encoding="utf-8") as fh:
        await asyncio.gather(*(one(int(r.id), r.text, fh) for r in todo.itertuples()))
    return done


def evaluate(df: pd.DataFrame, results: dict) -> dict:
    """Per-task metrics on the rows annotated for that task; invalid answers reported separately."""
    valid = df["id"].map(lambda i: results[i]["pred"] is not None)
    report = {"n": len(df), "invalid_answers": int((~valid).sum())}
    for t, task in enumerate(TASK_ORDER):
        col = LABEL_COLUMNS[task]
        rows = df[(df[col] != -1) & valid]
        if rows.empty:
            continue
        y = rows[col].astype(int)
        yhat = rows["id"].map(lambda i: results[i]["pred"][t])
        p, r, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
        report[task] = {
            "n": len(rows),
            "accuracy": round(accuracy_score(y, yhat), 4),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
        }
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/merged_dataset.csv")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--base_url", default=None, help="OpenAI-compatible endpoint (e.g. Ollama)")
    p.add_argument("--n", type=int, default=600, help="sample size, stratified by task (0 = all)")
    p.add_argument("--few_shot", action="store_true")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--out_dir", default="outputs/llm")
    args = p.parse_args()

    df = pd.read_csv(args.data).dropna(subset=["text"])
    df["text"] = df["text"].map(clean_text)
    if args.n:
        task_of_row = df[list(LABEL_COLUMNS.values())].ne(-1).idxmax(axis=1)
        df = df.groupby(task_of_row, group_keys=False).apply(
            lambda g: g.sample(min(len(g), args.n // 3), random_state=42)
        )
    df = df.reset_index(drop=True)
    df["id"] = df.index

    tag = f"{args.model.replace(':', '_').replace('/', '_')}_{'fewshot' if args.few_shot else 'zeroshot'}"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = asyncio.run(classify_all(df, args, out / f"{tag}.jsonl"))
    report = {"model": args.model, "few_shot": args.few_shot, **evaluate(df, results)}
    (out / f"{tag}_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
