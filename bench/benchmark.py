#!/usr/bin/env python3
"""Compare plain English, terse English, and Sempra prompting on a local LM Studio model.

Run from a machine that can reach the server (stdlib only, no installs):

    python3 bench/benchmark.py --base-url http://192.168.86.24:1234 \
        --model gemma-4-26b-a4b-it-uncensored-abliterix-mlx-int5-affine --runs 3

Token counts come from the server's own `usage` block, so they are exact for your model.
Results are also written to bench/results.json.
"""
import argparse
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODES = {
    "plain": "Reason step by step, then give the answer.",
    "terse": (ROOT / "prompts" / "terse_english.txt").read_text(),
    "sempra": (ROOT / "prompts" / "sempra_engine.txt").read_text(),
    "sempra_fewshot": (ROOT / "prompts" / "sempra_engine_fewshot.txt").read_text(),
}

ANSWER_SUFFIX = "\n\nEnd your reply with exactly one line: ANSWER: <final answer>"


def chat(base_url, model, system, user, max_tokens, temperature, timeout):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    start = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    return data, time.time() - start


def extract_answer(text):
    matches = re.findall(r"ANSWER:\s*(.+)", text, flags=re.IGNORECASE)
    return matches[-1].strip().strip("*`.") if matches else None


def numbers_match(got, want):
    try:
        return abs(float(got) - float(want)) <= 0.02 * max(1.0, abs(float(want)))
    except ValueError:
        return False


def is_correct(task, text):
    got = extract_answer(text)
    if got is None:
        return False
    for alt in task["answer"].split("|"):
        if alt.lower() in got.lower() or numbers_match(re.sub(r"[^\d.\-]", "", got), alt):
            return True
    return False


def spec_ok(task, text):
    low = text.lower()
    return all(s.lower() in low for s in task["must_include"])


def sempra_shaped(text):
    return bool(re.search(r"\w+[~!]\[", text)) and ">" in text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://192.168.86.24:1234")
    ap.add_argument("--model", default="gemma-4-26b-a4b-it-uncensored-abliterix-mlx-int5-affine")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--out", default=str(ROOT / "bench" / "results.json"))
    args = ap.parse_args()

    tasks = json.loads((ROOT / "bench" / "tasks.json").read_text())
    rows = []
    for mode in args.modes.split(","):
        for task in tasks:
            for run in range(args.runs):
                try:
                    data, secs = chat(args.base_url, args.model, MODES[mode],
                                      task["prompt"] + ANSWER_SUFFIX,
                                      args.max_tokens, args.temperature, args.timeout)
                except (urllib.error.URLError, TimeoutError, OSError) as err:
                    sys.exit(f"Cannot reach {args.base_url}: {err}")
                msg = data["choices"][0]["message"]
                text = (msg.get("reasoning_content") or "") + "\n" + (msg.get("content") or "")
                usage = data.get("usage", {})
                ok = is_correct(task, text) if task["type"] == "answer" else spec_ok(task, text)
                rows.append({
                    "mode": mode, "task": task["id"], "run": run, "ok": ok,
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "sempra_shaped": sempra_shaped(msg.get("content") or ""),
                    "seconds": round(secs, 2),
                    "output": msg.get("content"),
                    "reasoning": msg.get("reasoning_content"),
                })
                print(f"{mode:15} {task['id']:8} ok={ok!s:5} out={rows[-1]['completion_tokens']:5} tok", flush=True)

    Path(args.out).write_text(json.dumps(rows, indent=2))
    base = None
    print(f"\n{'mode':15} {'correct':>8} {'out tok':>8} {'vs plain':>9} {'sys+in tok':>11} {'sempra-shaped':>14}")
    for mode in args.modes.split(","):
        r = [x for x in rows if x["mode"] == mode]
        out = statistics.mean(x["completion_tokens"] for x in r)
        base = out if mode == "plain" or base is None else base
        print(f"{mode:15} {sum(x['ok'] for x in r) / len(r):8.0%} {out:8.0f} {out / base - 1:9.0%} "
              f"{statistics.mean(x['prompt_tokens'] for x in r):11.0f} "
              f"{sum(x['sempra_shaped'] for x in r) / len(r):14.0%}")
    print(f"\nFull outputs: {args.out}")


if __name__ == "__main__":
    main()
