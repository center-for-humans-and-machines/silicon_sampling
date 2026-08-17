"""A restartable sampling run: profiles in, transcripts and answers out.

Study-independent.  A study supplies four things — how to build a session from a
profile, where to file its transcript, the per-slot token budgets and the
worst-case transcript length — and gets back a run that survives being killed.

**Resumability is a property of the write order, not of a flag.**
``answers.jsonl`` is the single source of truth for what is finished. A
respondent's transcript is written *before* its answer record, so a kill between
the two costs a redo rather than a half-recorded respondent; the answer log is
`fsync`ed per group, so a killed machine loses at most one group; and a partial
final line is truncated before anything is appended, because appending onto an
unterminated record fuses two records into one unparseable string and loses both.
Seeds derive from the profile id, so a resumed run reproduces an uninterrupted
one exactly.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Protocol, Sequence

from ..survey.session import Session
from .driver import DrawLog, SamplerConfig, run_group
from .engine import EngineConfig, VLLMEngine
from .tokens import load_tokenizer

#: Raw draws are logged in full for one respondent in this many; rejected draws
#: are always logged.  Logging every draw would mean a million-line file.
DRAW_LOG_EVERY = 250


class Profile(Protocol):
    """What the runner needs of a respondent profile."""

    profile_id: str
    seed: int


def repair_jsonl(path: Path) -> int:
    """Drop a partial final line, returning the bytes removed."""
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open("rb+") as handle:
        handle.seek(-1, 2)
        if handle.read(1) == b"\n":
            return 0
        size = path.stat().st_size
        window = min(size, 1 << 20)
        handle.seek(size - window)
        tail = handle.read(window)
        cut = tail.rfind(b"\n")
        keep = size - window + cut + 1 if cut >= 0 else 0
        handle.truncate(keep)
        return size - keep


def completed(answers_path: Path) -> set[str]:
    """Ids already written, so a restart can skip them."""
    if not answers_path.exists():
        return set()
    done = set()
    with answers_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["profile_id"])
            except (ValueError, KeyError):  # pragma: no cover - a torn final line
                continue
    return done


class Runner:
    """Drives the sampler and writes everything to disk as it goes."""

    def __init__(
        self,
        out_dir: Path,
        engine_config: EngineConfig,
        sampler_config: SamplerConfig,
        *,
        session_for: Callable[[Profile], Session],
        record_for: Callable[[Profile, Session], dict],
        shard_for: Callable[[Profile], str],
        token_budgets: Callable[[str], dict],
        worst_case_tokens: Callable[[str, str], int],
    ) -> None:
        self.out = out_dir
        self.raw = out_dir / "raw"
        self.answers_path = out_dir / "answers.jsonl"
        self.draws_path = out_dir / "draws.jsonl"
        self.engine_config = engine_config
        self.sampler_config = sampler_config
        self.session_for = session_for
        self.record_for = record_for
        self.shard_for = shard_for
        self.token_budgets = token_budgets
        self.worst_case_tokens = worst_case_tokens
        self.raw.mkdir(parents=True, exist_ok=True)
        self._by_id: dict[str, Profile] = {}
        self._answers = None
        self._draws = None

    # -- callbacks -------------------------------------------------------- #

    def _on_draw(self, session, slot, chosen, rejected, attempt) -> None:
        profile = self._by_id[session.answers["_profile_id"]]
        verbose = _numeric_suffix(profile.profile_id) % DRAW_LOG_EVERY == 0
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

    def _on_group_done(self, sessions: Sequence[Session]) -> None:
        for session in sessions:
            profile = self._by_id[session.answers["_profile_id"]]
            # Transcript first: a kill between the two writes costs a redo, never
            # a respondent recorded as done without its transcript on disk.
            path = self.raw / self.shard_for(profile) / f"{profile.profile_id}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(session.transcript(), encoding="utf-8")
            self._answers.write(
                json.dumps(self.record_for(profile, session), ensure_ascii=False) + "\n"
            )
        for handle in (self._answers, self._draws):
            handle.flush()
            os.fsync(handle.fileno())

    # -- driving ---------------------------------------------------------- #

    def run(self, profiles: Sequence[Profile], resume: bool = True) -> dict:
        for path in (self.answers_path, self.draws_path):
            removed = repair_jsonl(path)
            if removed:
                print(
                    f"[run] recovered {path.name}: dropped {removed} B of a torn final record"
                )

        done = completed(self.answers_path) if resume else set()
        todo = [profile for profile in profiles if profile.profile_id not in done]
        print(
            f"[run] {len(todo)} of {len(profiles)} respondents to sample ({len(done)} already done)"
        )
        if not todo:
            return {"sampled": 0, "skipped": len(done)}

        if not self.sampler_config.max_tokens_by_slot:
            self.sampler_config.max_tokens_by_slot = self.token_budgets(
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
        tokenizer = (
            load_tokenizer(self.engine_config.model)
            if self.sampler_config.token_id_prompts
            else None
        )
        with (
            self.answers_path.open("a", encoding="utf-8") as answers,
            self.draws_path.open("a", encoding="utf-8") as draws,
        ):
            self._answers, self._draws = answers, draws
            with VLLMEngine(self.engine_config) as engine:
                cache = engine.kv_cache_tokens()
                auto = size <= 0
                print(
                    f"[run] KV cache: {cache} tokens; group size {'auto (per condition)' if auto else size}"
                )

                # Partition by shard (in practice, by condition) so a single
                # long-stimulus arm does not throttle the group size of the
                # short ones: each partition is sized to its own worst case.
                partitions: dict[str, list] = {}
                for profile in todo:
                    partitions.setdefault(self.shard_for(profile), []).append(profile)

                done_now = 0
                for shard, members in partitions.items():
                    if auto:
                        per_session = self.worst_case_tokens(
                            self.engine_config.model, shard
                        )
                        size = engine.group_size_for(per_session)
                        self.sampler_config.group_size = size
                        print(
                            f"[run] {shard}: longest transcript {per_session} tokens -> group size {size}"
                        )
                    for start in range(0, len(members), size):
                        chunk = members[start : start + size]
                        sessions = []
                        for profile in chunk:
                            self._by_id[profile.profile_id] = profile
                            session = self.session_for(profile)
                            session.answers["_profile_id"] = profile.profile_id
                            sessions.append(session)
                        log.merge(
                            run_group(
                                engine,
                                sessions,
                                self.sampler_config,
                                [profile.seed for profile in chunk],
                                on_draw=self._on_draw,
                                tokenizer=tokenizer,
                            )
                        )
                        self._on_group_done(sessions)
                        done_now += len(chunk)
                        rate = done_now / (time.time() - started) * 3600
                        print(
                            f"[run] {done_now}/{len(todo)} respondents | {rate:.0f}/h | eta {(len(todo) - done_now) / max(rate, 1e-9):.1f} h",
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
            "sampler": {
                key: value
                for key, value in asdict(self.sampler_config).items()
                if key != "max_tokens_by_slot"
            },
            "draws": {
                "calls": log.calls,
                "draws": log.draws,
                "rejected": log.rejected,
                "rejection_rate": (
                    round(log.rejected / log.draws, 5) if log.draws else None
                ),
                "structured_fallbacks": log.structured_fallbacks,
                "forced": log.forced,
                "worst_slots": sorted(
                    log.rejected_by_slot.items(), key=lambda item: -item[1]
                )[:25],
            },
        }
        (self.out / "run_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(
            f"[run] done in {elapsed / 60:.1f} min; rejection rate {meta['draws']['rejection_rate']}"
        )
        return meta


def _numeric_suffix(profile_id: str) -> int:
    digits = "".join(ch for ch in profile_id if ch.isdigit())
    return int(digits) if digits else 0
