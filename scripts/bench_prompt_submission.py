"""Measure what a sampling round actually spends its time on.

The driver hands vLLM the whole transcript as a *string* every round, so a ~7,000
token prompt is re-tokenised and re-hashed into prefix-cache blocks ~78 times per
respondent.  Backing that cost out of the completed 4090 run left a wide band
(6-17 ms per prompt per round); this measures it directly.

Two questions:

1. Does the per-round host cost scale with the *number* of prompts or with their
   total *length*?  Held at a constant token budget, a per-prompt cost grows with
   the group size and a per-token cost does not.
2. How much of it does passing ``prompt_token_ids`` instead of text remove?

Real transcripts from the completed run are used as prompts, cut at a
``Response: `` boundary, so the prompts are exactly the shape the sampler sees.

Usage: python scripts/bench_prompt_submission.py [--model Qwen/Qwen2.5-7B]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from silicon_sampling.pfander import paths
from silicon_sampling.sampling.engine import EngineConfig, VLLMEngine, configure_runtime

#: (group size, prompt tokens) at a roughly constant ~97k total, which is what the
#: 4090's 119,600-token KV cache holds with room for the draws in flight.
SHAPES = ((15, 6500), (30, 3250), (60, 1600))
DRAWS = 4
MAX_TOKENS = 7
REPEATS = 5


def load_prompts(count: int, tokenizer, target_tokens: int) -> list[str]:
    """Real transcripts, cut at a ``Response: `` boundary to ``target_tokens``."""
    files = sorted(paths.RAW.rglob("*.txt"))[: count * 3]
    if len(files) < count:
        raise SystemExit(
            f"need {count} transcripts under {paths.RAW}, found {len(files)}"
        )
    prompts = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) < target_tokens:
            continue
        # Cut to length, then back up to the last "Response: " so the prompt ends
        # where the sampler's prompts end.
        head = tokenizer.decode(ids[:target_tokens])
        marker = head.rfind("Response: ")
        if marker < 0:
            continue
        prompts.append(head[: marker + len("Response: ")])
        if len(prompts) == count:
            return prompts
    raise SystemExit(
        f"only {len(prompts)} of {count} transcripts reached {target_tokens} tokens"
    )


def timed(fn, repeats: int = REPEATS) -> float:
    """Median seconds over ``repeats`` calls, discarding the first."""
    samples = []
    for index in range(repeats + 1):
        started = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - started
        if index:
            samples.append(elapsed)
    return statistics.median(samples)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=EngineConfig.model)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    configure_runtime(paths.CACHE)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    config = EngineConfig(
        model=args.model,
        max_model_len=8192,
        gpu_memory_utilization=0.96,
        max_num_seqs=512,
    )
    results = []
    with VLLMEngine(config) as engine:
        print(f"[bench] KV cache: {engine.kv_cache_tokens()} tokens", flush=True)
        for group, length in SHAPES:
            prompts = load_prompts(group, tokenizer, length)
            token_ids = [
                tokenizer(prompt, add_special_tokens=False)["input_ids"]
                for prompt in prompts
            ]
            actual = statistics.mean(len(ids) for ids in token_ids)
            params = [
                engine.params(max_tokens=MAX_TOKENS, n=DRAWS, seed=1000 + index)
                for index in range(group)
            ]

            # Tokenisation alone, the part token ids would remove outright.
            t_tokenise = timed(
                lambda: [
                    tokenizer(prompt, add_special_tokens=False)["input_ids"]
                    for prompt in prompts
                ]
            )

            # Warm the prefix cache, then time both submission paths.
            engine._llm.generate(prompts, params, use_tqdm=False)
            t_text = timed(
                lambda: engine._llm.generate(prompts, params, use_tqdm=False)
            )
            token_prompts = [{"prompt_token_ids": ids} for ids in token_ids]
            engine._llm.generate(token_prompts, params, use_tqdm=False)
            t_ids = timed(
                lambda: engine._llm.generate(token_prompts, params, use_tqdm=False)
            )

            row = {
                "group": group,
                "prompt_tokens": round(actual),
                "total_prompt_tokens": round(actual * group),
                "round_text_ms": round(t_text * 1e3, 1),
                "round_ids_ms": round(t_ids * 1e3, 1),
                "tokenise_ms": round(t_tokenise * 1e3, 1),
                "saved_ms": round((t_text - t_ids) * 1e3, 1),
                "per_prompt_text_ms": round(t_text * 1e3 / group, 2),
                "per_prompt_ids_ms": round(t_ids * 1e3 / group, 2),
            }
            results.append(row)
            print(json.dumps(row), flush=True)

    print()
    header = (
        f"{'group':>6} {'tok/prompt':>11} {'round(text)':>12} {'round(ids)':>11} "
        f"{'tokenise':>9} {'saved':>7} {'ms/prompt text':>15} {'ids':>7}"
    )
    print(header)
    for row in results:
        print(
            f"{row['group']:>6} {row['prompt_tokens']:>11} {row['round_text_ms']:>11.1f}m "
            f"{row['round_ids_ms']:>10.1f}m {row['tokenise_ms']:>8.1f}m "
            f"{row['saved_ms']:>6.1f}m {row['per_prompt_text_ms']:>14.2f} "
            f"{row['per_prompt_ids_ms']:>7.2f}"
        )

    if args.out:
        Path(args.out).write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
