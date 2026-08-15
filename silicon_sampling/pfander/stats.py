"""Live throughput and legality statistics for a sampling run.

Everything here is computed from artifacts on disk, so it can be run against a
run that is still in progress without touching the GPU.

Three token counts matter and they differ by orders of magnitude, so the report
keeps them apart:

**decode tokens**
    tokens the model actually generated. This is the small number, and the one
    that maps to "tokens/s" in the usual sense.

**prompt tokens presented**
    the sum of every prompt's length over every call. Each respondent's
    transcript is re-presented in full ~85 times as it grows, so this is enormous
    — and almost entirely served from the KV cache.

**unique prefill tokens**
    what actually has to be computed: each respondent's final transcript, once.
    The ratio of presented to unique is the prefix cache's leverage, and it is
    the single number that decides whether this design is feasible at all.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .profiles import Profile, read_csv
from .run import session_for


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError:  # pragma: no cover - a torn final line
                    continue
    return rows


def legality(run_dir: Path, draws_per_call: int = 4) -> dict:
    """Illegal-generation rate, from the draw log and the answer log.

    Every rejected draw is logged, and every retry round is logged, so this is
    exact rather than sampled — with one small caveat noted in the result: for a
    slot that exhausted its retries and fell back to constrained decoding, only
    the fallback call's rejects are recorded, so its earlier rounds' rejects are
    missed. Those slots are rare and counted separately.
    """
    answers = _read_jsonl(run_dir / "answers.jsonl")
    draws = _read_jsonl(run_dir / "draws.jsonl")

    slots_asked = sum(record.get("n_asked", 0) for record in answers)
    retry_calls = sum(record.get("attempt", 0) for record in draws)
    fallbacks = [record for record in draws if record.get("attempt", 0) >= 4]
    calls = slots_asked + retry_calls
    total_draws = calls * draws_per_call
    rejected = sum(len(record.get("rejected", ())) for record in draws)

    by_slot: dict[str, int] = {}
    asked_by_slot: dict[str, int] = {}
    for record in draws:
        count = len(record.get("rejected", ()))
        if count:
            by_slot[record["slot"]] = by_slot.get(record["slot"], 0) + count
    for record in answers:
        for slot_id in record.get("answers", {}):
            asked_by_slot[slot_id] = asked_by_slot.get(slot_id, 0) + 1

    worst = sorted(
        (
            {
                "slot": slot_id,
                "rejected": count,
                "asked": asked_by_slot.get(slot_id, 0),
                "rejected_per_slot_asked": count
                / max(asked_by_slot.get(slot_id, 0), 1),
            }
            for slot_id, count in by_slot.items()
        ),
        key=lambda row: -row["rejected_per_slot_asked"],
    )

    return {
        "respondents": len(answers),
        "slots_asked": slots_asked,
        "calls": calls,
        "retry_calls": retry_calls,
        "draws": total_draws,
        "rejected_draws": rejected,
        "illegal_rate": rejected / total_draws if total_draws else 0.0,
        "slots_needing_a_retry": sum(
            1 for record in draws if record.get("attempt", 0) > 0
        ),
        "constrained_fallbacks": len(fallbacks),
        "worst_slots": worst[:15],
    }


def token_profile(
    profiles: list[Profile], answers_by_id: dict[str, dict], model: str, limit: int = 12
) -> dict:
    """Exact prompt-token accounting, replayed offline for a few respondents.

    The prompts are deterministic given the answers, so they can be rebuilt
    without the GPU and tokenized exactly.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)

    presented = 0
    unique = 0
    steps = 0
    replayed = 0
    for profile in profiles:
        record = answers_by_id.get(profile.profile_id)
        if record is None:
            continue
        session = session_for(profile)
        answers = record["answers"]
        while (step := session.next_prompt()) is not None:
            prompt, slot = step
            if slot.id not in answers:
                break
            presented += len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            steps += 1
            session.submit(slot, answers[slot.id])
        unique += len(
            tokenizer(session.transcript(), add_special_tokens=False)["input_ids"]
        )
        replayed += 1
        if replayed >= limit:
            break

    if not replayed:
        return {}
    return {
        "respondents_replayed": replayed,
        "steps_per_respondent": steps / replayed,
        "prompt_tokens_presented_per_respondent": presented / replayed,
        "unique_prefill_tokens_per_respondent": unique / replayed,
        "cache_leverage": presented / unique if unique else 0.0,
    }


def draw_lengths(run_dir: Path, model: str) -> dict:
    """Average generated length of a draw, measured on the logged draws."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    draws = _read_jsonl(run_dir / "draws.jsonl")
    texts = []
    for record in draws:
        if record.get("chosen"):
            texts.append(record["chosen"])
        texts.extend(record.get("rejected", ()))
    if not texts:
        return {}
    # +1 for the newline the model emitted to stop; vLLM strips the stop string
    # from the returned text but still had to generate it.
    lengths = [
        len(tokenizer(text, add_special_tokens=False)["input_ids"]) + 1
        for text in texts
    ]
    return {
        "logged_draws": len(lengths),
        "mean_tokens_per_draw": sum(lengths) / len(lengths),
        "max_tokens_per_draw": max(lengths),
    }


def measure_rate(run_dir: Path, seconds: float = 120.0) -> dict:
    """Instantaneous throughput, by watching the answer log grow."""
    path = run_dir / "answers.jsonl"

    def count() -> int:
        return sum(1 for _ in path.open(encoding="utf-8")) if path.exists() else 0

    start_count, start_time = count(), time.time()
    time.sleep(seconds)
    end_count, end_time = count(), time.time()
    elapsed = end_time - start_time
    done = end_count - start_count
    return {
        "window_seconds": round(elapsed, 1),
        "respondents_completed": done,
        "respondents_per_hour": done / elapsed * 3600 if elapsed else 0.0,
        "total_completed": end_count,
    }


def report(
    run_dir: Path,
    model: str = "Qwen/Qwen2.5-7B",
    window: float = 120.0,
    draws_per_call: int = 4,
) -> dict:
    """Everything, combined into the numbers a human actually wants."""
    legal = legality(run_dir, draws_per_call=draws_per_call)
    lengths = draw_lengths(run_dir, model)
    rate = measure_rate(run_dir, seconds=window)

    answers_by_id = {
        record["profile_id"]: record
        for record in _read_jsonl(run_dir / "answers.jsonl")[-40:]
    }
    profiles = [
        profile
        for profile in read_csv(run_dir / "profiles.csv")
        if profile.profile_id in answers_by_id
    ]
    tokens = token_profile(profiles, answers_by_id, model)

    per_second = (
        rate["respondents_per_hour"] / 3600 if rate["respondents_per_hour"] else 0.0
    )
    calls_per_respondent = (
        legal["calls"] / legal["respondents"] if legal["respondents"] else 0.0
    )
    draws_per_respondent = calls_per_respondent * draws_per_call
    mean_draw = lengths.get("mean_tokens_per_draw", 0.0)

    throughput = {
        "respondents_per_hour": rate["respondents_per_hour"],
        "decode_tokens_per_second": per_second * draws_per_respondent * mean_draw,
        "prompt_tokens_presented_per_second": per_second
        * tokens.get("prompt_tokens_presented_per_respondent", 0.0),
        "unique_prefill_tokens_per_second": per_second
        * tokens.get("unique_prefill_tokens_per_respondent", 0.0),
        "generation_calls_per_second": per_second * calls_per_respondent,
        "sequences_per_second": per_second * draws_per_respondent,
    }
    return {
        "rate": rate,
        "legality": legal,
        "draw_lengths": lengths,
        "tokens": tokens,
        "throughput": throughput,
    }


def format_report(result: dict) -> str:
    """The report as text."""
    rate, legal, tokens, through = (
        result["rate"],
        result["legality"],
        result["tokens"],
        result["throughput"],
    )
    lines = [
        f"Sampled {rate['total_completed']:,} respondents so far.",
        "",
        f"Throughput (measured over a {rate['window_seconds']:.0f} s window)",
        f"  respondents/hour                 {through['respondents_per_hour']:>12,.0f}",
        f"  generation calls/s               {through['generation_calls_per_second']:>12,.1f}",
        f"  sequences/s (calls x draws)      {through['sequences_per_second']:>12,.1f}",
        f"  decode tokens/s                  {through['decode_tokens_per_second']:>12,.0f}",
        f"  prompt tokens presented/s        {through['prompt_tokens_presented_per_second']:>12,.0f}",
        f"  unique prefill tokens/s          {through['unique_prefill_tokens_per_second']:>12,.0f}",
        "",
        "Per respondent",
        f"  response positions               {tokens.get('steps_per_respondent', 0):>12,.1f}",
        f"  prompt tokens presented          {tokens.get('prompt_tokens_presented_per_respondent', 0):>12,.0f}",
        f"  unique prefill tokens            {tokens.get('unique_prefill_tokens_per_respondent', 0):>12,.0f}",
        f"  prefix-cache leverage            {tokens.get('cache_leverage', 0):>12,.1f}x",
        "",
        "Legality",
        f"  draws                            {legal['draws']:>12,}",
        f"  illegal (rejected) draws         {legal['rejected_draws']:>12,}",
        f"  illegal rate                     {legal['illegal_rate']:>12.2%}",
        f"  slots needing >1 round           {legal['slots_needing_a_retry']:>12,}  of {legal['slots_asked']:,}",
        f"  constrained-decoding fallbacks   {legal['constrained_fallbacks']:>12,}",
        "",
        "Illegal draws per slot asked (worst 10)",
    ]
    for row in legal["worst_slots"][:10]:
        lines.append(
            f"  {row['slot']:<26} {row['rejected_per_slot_asked']:>6.2f}   ({row['rejected']:,} rejected / {row['asked']:,} asked)"
        )
    return "\n".join(lines)


def _normalise(text: str) -> str:
    """Lowercase, keep only alphanumerics — the loosest sane comparison."""
    return "".join(character for character in text.lower() if character.isalnum())


def near_misses(run_dir: Path, slots: dict) -> list[dict]:
    """Split rejected draws into *near misses* and genuine out-of-frame answers.

    This is the check that matters for bias. A rejected draw that would match a
    legal option under a looser comparison — different punctuation, a missing
    currency symbol — is the model giving the right answer in the wrong spelling,
    and rejecting it is dangerous: the failure rate then depends on *which* option
    was meant, and rejection sampling converts that asymmetry straight into a
    skewed distribution. A rejected draw that matches nothing is the model
    answering a different question, and rejecting it is correct.

    So a high near-miss share is a parser bug; a low one is the model being the
    model.
    """
    from ..survey.slots import ChoiceSlot, IntSlot

    draws = _read_jsonl(run_dir / "draws.jsonl")
    answers = _read_jsonl(run_dir / "answers.jsonl")
    asked: dict[str, int] = {}
    for record in answers:
        for slot_id in record.get("answers", {}):
            asked[slot_id] = asked.get(slot_id, 0) + 1

    stats: dict[str, dict] = {}
    for record in draws:
        slot = slots.get(record["slot"])
        if slot is None:
            continue
        entry = stats.setdefault(
            record["slot"], {"rejected": 0, "near": 0, "examples": []}
        )
        for text in record.get("rejected", ()):
            entry["rejected"] += 1
            near = False
            if isinstance(slot, ChoiceSlot):
                normalised = _normalise(text)
                near = any(
                    normalised.startswith(_normalise(option))
                    for option in slot.options
                    if option
                )
            elif isinstance(slot, IntSlot):
                digits = "".join(
                    character if character.isdigit() else " " for character in text
                ).split()
                near = any(slot.lo <= int(value) <= slot.hi for value in digits[:1])
            if near:
                entry["near"] += 1
                if len(entry["examples"]) < 3:
                    entry["examples"].append(text)

    rows = []
    for slot_id, entry in stats.items():
        if not entry["rejected"]:
            continue
        rows.append(
            {
                "slot": slot_id,
                "asked": asked.get(slot_id, 0),
                "rejected": entry["rejected"],
                "rejected_per_asked": entry["rejected"] / max(asked.get(slot_id, 0), 1),
                "near_misses": entry["near"],
                "near_miss_share": entry["near"] / entry["rejected"],
                "near_misses_per_asked": entry["near"] / max(asked.get(slot_id, 0), 1),
                "examples": entry["examples"],
            }
        )
    # Sorted by near misses *per slot asked*, not by share: the share says how
    # suspect a slot's rejections are, but only the rate says how much of the
    # distribution is actually exposed to them.
    return sorted(rows, key=lambda row: -row["near_misses_per_asked"])
