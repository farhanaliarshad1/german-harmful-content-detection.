"""Zero-/few-shot LLM baseline for fake-news detection.

Works with any OpenAI-compatible chat endpoint:
  * OpenAI:  export OPENAI_API_KEY=...            --model gpt-4o-mini
  * Ollama:  ollama serve                          --base_url http://localhost:11434/v1 --model llama3.1:8b

Results are cached to a JSONL file so interrupted runs resume where they stopped.
Unparseable answers are counted as *invalid* and reported separately instead of
silently being mapped to class 0.

Usage:
    python -m fakenews.llm_baseline --data data/fake_news.csv --model gpt-4o-mini --n 500 --few_shot
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

from fakenews.data import load_defakts
from fakenews.prompts import build_messages

ANSWER_RE = re.compile(r"\b([01])\b")


def parse_label(answer: str) -> int | None:
    m = ANSWER_RE.search(answer or "")
    return int(m.group(1)) if m else None


async def classify_all(texts, ids, args, cache_path: Path) -> dict:
    from openai import AsyncOpenAI  # optional dependency

    done = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[row["id"]] = row
    todo = [(i, t) for i, t in zip(ids, texts) if i not in done]
    print(f"{len(done)} cached, {len(todo)} to classify with {args.model}")

    client = AsyncOpenAI(base_url=args.base_url, api_key=os.environ.get("OPENAI_API_KEY", "ollama"))
    sem = asyncio.Semaphore(args.concurrency)

    async def one(idx, text, fh):
        async with sem:
            for attempt in range(3):
                try:
                    resp = await client.chat.completions.create(
                        model=args.model,
                        messages=build_messages(text, few_shot=args.few_shot),
                        temperature=0.0,
                        max_tokens=5,
                    )
                    answer = resp.choices[0].message.content
                    break
                except Exception as exc:  # network / rate limit
                    if attempt == 2:
                        answer = f"ERROR: {exc}"
                    await asyncio.sleep(2 ** attempt)
            row = {"id": idx, "answer": answer, "pred": parse_label(answer)}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            done[idx] = row

    with cache_path.open("a", encoding="utf-8") as fh:
        await asyncio.gather(*(one(i, t, fh) for i, t in todo))
    return done


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/fake_news.csv")
    p.add_argument("--label_col", default="binary_label")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--base_url", default=None, help="OpenAI-compatible endpoint (e.g. Ollama)")
    p.add_argument("--n", type=int, default=500, help="stratified sample size (0 = all)")
    p.add_argument("--few_shot", action="store_true")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--out_dir", default="outputs/llm")
    args = p.parse_args()

    df = load_defakts(args.data, label_col=args.label_col)
    if args.n:
        df = df.groupby("label", group_keys=False).apply(
            lambda g: g.sample(min(len(g), args.n // 2), random_state=42)
        )
    df = df.reset_index(drop=True)
    df["id"] = df.index

    tag = f"{args.model.replace(':', '_').replace('/', '_')}_{'fewshot' if args.few_shot else 'zeroshot'}"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = asyncio.run(classify_all(df["text"].tolist(), df["id"].tolist(), args, out / f"{tag}.jsonl"))

    df["pred"] = df["id"].map(lambda i: results[i]["pred"])
    valid = df.dropna(subset=["pred"])
    p_, r_, f1, _ = precision_recall_fscore_support(valid["label"], valid["pred"].astype(int), average="binary", zero_division=0)
    summary = {
        "model": args.model,
        "few_shot": args.few_shot,
        "n": len(df),
        "invalid_answers": int(df["pred"].isna().sum()),
        "accuracy": round(accuracy_score(valid["label"], valid["pred"].astype(int)), 4),
        "precision": round(p_, 4),
        "recall": round(r_, 4),
        "f1": round(f1, 4),
    }
    (out / f"{tag}_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
