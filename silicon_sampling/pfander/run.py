"""The sampling run: profiles in, transcripts and answers out.

Restartable by construction.  ``answers.jsonl`` is the source of truth for what
is finished; a run that dies mid-group loses at most that group.  This matters
because the full run does not fit in one shell invocation.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from ..sampling.driver import DrawLog, SamplerConfig, run_group, token_budget
from ..sampling.engine import EngineConfig, VLLMEngine
from ..survey.session import Session
from ..survey.slots import Slot
from . import build, instrument, outcomes, templates
from .conditions import slug
from .profiles import Profile

BLOCKS_BY_KEY = {block.key: block for block in instrument.POST_RANDOMISED}

#: Raw draws are logged in full for this fraction of respondents; rejected draws
#: are always logged.  Logging every draw would mean a 1.5-million-line file.
DRAW_LOG_EVERY = 250


def session_for(profile: Profile) -> Session:
    """Build the session one profile walks through."""
    return build.make_session(
        profile.profile_id,
        profile.condition,
        code_name=profile.code_name,
        answers=dict(profile.prefilled),
        control_text=profile.control_text or None,
        consensus_order=(
            tuple(int(char) for char in profile.consensus_order)
            if profile.consensus_order
            else (1, 3, 2)
        ),
        post_order=(
            [BLOCKS_BY_KEY[key] for key in profile.post_order.split("|")]
            if profile.post_order
            else None
        ),
    )


def all_slots() -> dict[str, Slot]:
    """Every slot of every condition, keyed by id."""
    found: dict[str, Slot] = {}
    for condition in templates.conditions.CONDITIONS:
        for element in _walk_slots(templates.template_elements(condition)):
            found.setdefault(element.id, element)
    return found


def _walk_slots(elements):
    from ..survey.render import walk

    for event, payload in walk(elements):
        if event == "slot":
            yield payload


def fit_token_budgets(model: str) -> dict[str, int]:
    """Per-slot token budgets, measured with the model's own tokenizer."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    return {
        slot_id: token_budget(slot, tokenizer) for slot_id, slot in all_slots().items()
    }


def completed(answers_path: Path) -> set[str]:
    """Ids already written, so a restart can skip them."""
    if not answers_path.exists():
        return set()
    done = set()
    with answers_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["profile_id"])
                except (ValueError, KeyError):  # pragma: no cover - a torn final line
                    continue
    return done


class Runner:
    """Drives the sampler and writes everything to disk as it goes."""

    def __init__(
        self, out_dir: Path, engine_config: EngineConfig, sampler_config: SamplerConfig
    ) -> None:
        self.out = out_dir
        self.raw = out_dir / "raw"
        self.answers_path = out_dir / "answers.jsonl"
        self.draws_path = out_dir / "draws.jsonl"
        self.engine_config = engine_config
        self.sampler_config = sampler_config
        self.raw.mkdir(parents=True, exist_ok=True)
        self._by_id: dict[str, Profile] = {}
        self._answers = None
        self._draws = None

    # -- callbacks -------------------------------------------------------- #

    def _on_draw(self, session, slot, chosen, rejected, attempt) -> None:
        profile = self._by_id[session.answers["_profile_id"]]
        verbose = int(profile.profile_id[1:]) % DRAW_LOG_EVERY == 0
        if not rejected and not verbose and attempt == 0:
            return
        record = {
            "profile_id": profile.profile_id,
            "slot": slot.id,
            "attempt": attempt,
            "chosen": chosen,
        }
        if rejected:
            record["rejected"] = rejected
        self._draws.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _on_group_done(self, sessions: Sequence[Session], log: DrawLog) -> None:
        for session in sessions:
            profile = self._by_id[session.answers["_profile_id"]]
            transcript = session.transcript()
            path = self.raw / slug(profile.condition) / f"{profile.profile_id}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(transcript, encoding="utf-8")
            answers = {
                key: value
                for key, value in session.answers.items()
                if not key.startswith("_")
            }
            self._answers.write(
                json.dumps(
                    {
                        "profile_id": profile.profile_id,
                        "condition": profile.condition,
                        "code_name": profile.code_name,
                        "control_text": profile.control_text,
                        "consensus_order": profile.consensus_order,
                        "post_order": profile.post_order,
                        "n_asked": len(session.asked),
                        "answers": answers,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        self._answers.flush()
        self._draws.flush()

    # -- driving ---------------------------------------------------------- #

    def run(self, profiles: Sequence[Profile], resume: bool = True) -> dict:
        done = completed(self.answers_path) if resume else set()
        todo = [profile for profile in profiles if profile.profile_id not in done]
        print(
            f"[run] {len(todo)} of {len(profiles)} respondents to sample ({len(done)} already done)"
        )
        if not todo:
            return {"sampled": 0, "skipped": len(done)}

        if not self.sampler_config.max_tokens_by_slot:
            self.sampler_config.max_tokens_by_slot = fit_token_budgets(
                self.engine_config.model
            )
            widest = sorted(
                self.sampler_config.max_tokens_by_slot.items(),
                key=lambda item: -item[1],
            )[:3]
            print(f"[run] token budgets fitted; widest slots: {widest}")

        started = time.time()
        log = DrawLog()
        size = self.sampler_config.group_size
        with (
            self.answers_path.open("a", encoding="utf-8") as answers,
            self.draws_path.open("a", encoding="utf-8") as draws,
        ):
            self._answers, self._draws = answers, draws
            with VLLMEngine(self.engine_config) as engine:
                cache = engine.kv_cache_tokens()
                print(f"[run] KV cache: {cache} tokens; group size {size}")
                # Sessions are built one group at a time: holding 18,000 of them
                # at once costs hundreds of MB for no benefit.
                for start in range(0, len(todo), size):
                    chunk = todo[start : start + size]
                    sessions = []
                    for profile in chunk:
                        self._by_id[profile.profile_id] = profile
                        session = session_for(profile)
                        session.answers["_profile_id"] = profile.profile_id
                        sessions.append(session)
                    log.merge(
                        run_group(
                            engine,
                            sessions,
                            self.sampler_config,
                            [profile.seed for profile in chunk],
                            on_draw=self._on_draw,
                        )
                    )
                    self._on_group_done(sessions, log)
                    done_now = start + len(chunk)
                    rate = done_now / (time.time() - started) * 3600
                    print(
                        f"[run] {done_now}/{len(todo)} respondents | {rate:.0f}/h | "
                        f"eta {(len(todo) - done_now) / max(rate, 1e-9):.1f} h",
                        flush=True,
                    )

        elapsed = time.time() - started
        meta = {
            "model": self.engine_config.model,
            "sampled": len(todo),
            "skipped": len(done),
            "seconds": round(elapsed, 1),
            "respondents_per_hour": (
                round(len(todo) / elapsed * 3600, 1) if elapsed else None
            ),
            "kv_cache_tokens": cache,
            "engine": asdict(self.engine_config),
            "sampler": asdict(self.sampler_config),
            "draws": {
                "calls": log.calls,
                "draws": log.draws,
                "rejected": log.rejected,
                "rejection_rate": (
                    round(log.rejected / log.draws, 5) if log.draws else None
                ),
                "guided_fallbacks": log.guided_fallbacks,
                "forced": log.forced,
                "worst_slots": sorted(
                    log.rejected_by_slot.items(), key=lambda item: -item[1]
                )[:25],
            },
            "outcomes": list(outcomes.OUTCOMES),
        }
        (self.out / "run_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(
            f"[run] done in {elapsed / 60:.1f} min; rejection rate {meta['draws']['rejection_rate']}"
        )
        return meta
