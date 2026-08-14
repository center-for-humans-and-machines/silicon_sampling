"""Walk a batch of sessions through their slots, rejecting illegal draws.

Two constraints shape this loop.

**Sequential dependence.** A respondent's answer to question 41 is conditioned on
their answers to questions 1-40, so a session cannot be generated in one call.
Each slot is its own generation, with the whole transcript so far as the prompt.

**Prefix caching.** Those prompts are 6-7k tokens and grow by ~50 tokens a step, so
the work per step is trivial *if* the respondent's prefix is still in the GPU's KV
cache and catastrophic if it is not — a miss costs a full re-prefill. The cache
holds only so many transcripts at once, so sessions are processed in groups sized
to fit, and every session in a group is stepped in lockstep before moving on.

Rejection sampling asks for ``n`` independent continuations in one call: they
share the cached prefix, so four attempts cost barely more than one, and taking
the first legal one out of ``n`` i.i.d. draws is exactly rejection sampling from
the model's distribution restricted to the legal set.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from ..survey.session import Session
from ..survey.slots import ChoiceSlot, FreeTextSlot, IntSlot, Slot
from .engine import VLLMEngine


def token_budget(slot: Slot, tokenizer) -> int:
    """How many tokens a slot's longest legal answer needs, plus headroom.

    Sizing this by hand is a trap.  A budget that fits "Yes" but not
    "$100,000 to $167,999" does not merely waste retries — it rejects the long
    options *systematically*, so the recorded distribution collapses onto the
    short ones.  In a first calibration run that turned the income item into a
    barbell of the two shortest brackets, with the three middle brackets never
    once selected.
    """

    def length(text: str) -> int:
        return len(tokenizer(" " + str(text), add_special_tokens=False)["input_ids"])

    if isinstance(slot, ChoiceSlot):
        return max(length(option) for option in slot.options) + 4
    if isinstance(slot, IntSlot):
        return max(length(slot.lo), length(slot.hi)) + 3
    return slot.max_tokens


@dataclass
class SamplerConfig:
    #: Independent continuations per call.  Higher trades tokens for fewer rounds.
    draws_per_call: int = 4
    #: How many times to re-issue a call whose draws were all illegal.
    max_rounds: int = 4
    #: After that, fall back to grammar-constrained decoding for that one slot.
    use_structured_fallback: bool = True
    #: Sessions stepped in lockstep.  Bounded by KV cache capacity, not by VRAM.
    group_size: int = 24
    #: Per-slot token budgets, from :func:`token_budget`.  Empty means use each
    #: slot's own declared ``max_tokens``.
    max_tokens_by_slot: dict = field(default_factory=dict)

    def tokens_for(self, slot: Slot) -> int:
        return self.max_tokens_by_slot.get(slot.id, slot.max_tokens)


@dataclass
class DrawLog:
    """What the sampler did, for the diagnostics report."""

    calls: int = 0
    draws: int = 0
    rejected: int = 0
    structured_fallbacks: int = 0
    forced: int = 0
    seconds: float = 0.0
    rejected_by_slot: dict = field(default_factory=dict)
    fallback_slots: list = field(default_factory=list)

    def note_rejection(self, slot_id: str, count: int = 1) -> None:
        self.rejected += count
        self.rejected_by_slot[slot_id] = self.rejected_by_slot.get(slot_id, 0) + count

    def merge(self, other: "DrawLog") -> None:
        self.calls += other.calls
        self.draws += other.draws
        self.rejected += other.rejected
        self.structured_fallbacks += other.structured_fallbacks
        self.forced += other.forced
        self.seconds += other.seconds
        for slot_id, count in other.rejected_by_slot.items():
            self.rejected_by_slot[slot_id] = (
                self.rejected_by_slot.get(slot_id, 0) + count
            )
        self.fallback_slots += other.fallback_slots


def structured_for(engine: VLLMEngine, slot: Slot):
    """A grammar that can only emit legal answers, for the last-resort path."""
    if isinstance(slot, ChoiceSlot):
        return engine.structured_choice(list(slot.options))
    if isinstance(slot, IntSlot):
        return engine.structured_regex(
            "|".join(str(value) for value in range(slot.lo, slot.hi + 1))
        )
    if isinstance(slot, FreeTextSlot) and slot.pattern:
        return engine.structured_regex(slot.pattern)
    return None


def _default_value(slot: Slot):
    """Last resort when even constrained decoding fails; counted and reported."""
    if isinstance(slot, ChoiceSlot):
        return slot.options[0]
    if isinstance(slot, IntSlot):
        return (slot.lo + slot.hi) // 2
    return ""


def run_group(
    engine: VLLMEngine,
    sessions: Sequence[Session],
    config: SamplerConfig,
    seeds: Sequence[int],
    log: DrawLog | None = None,
    on_draw: Callable[[Session, Slot, str, list, int], None] | None = None,
) -> DrawLog:
    """Step a group of sessions to completion, in lockstep."""
    log = log or DrawLog()
    started = time.time()
    step = 0

    while True:
        pending: list[tuple[int, Session, Slot, str]] = []
        for index, session in enumerate(sessions):
            nxt = session.next_prompt()
            if nxt is not None:
                prompt, slot = nxt
                pending.append((index, session, slot, prompt))
        if not pending:
            break

        for attempt in range(config.max_rounds):
            if not pending:
                break
            params = [
                engine.params(
                    max_tokens=config.tokens_for(slot),
                    n=config.draws_per_call,
                    seed=seeds[index] + step * 1013 + attempt * 7919,
                )
                for index, _, slot, _ in pending
            ]
            results = engine.generate([prompt for *_, prompt in pending], params)
            log.calls += len(pending)
            log.draws += sum(len(group) for group in results)

            still: list[tuple[int, Session, Slot, str]] = []
            for (index, session, slot, prompt), draws in zip(pending, results):
                value, rejected = _first_legal(slot, draws)
                log.note_rejection(slot.id, len(rejected))
                if value is None:
                    still.append((index, session, slot, prompt))
                    continue
                if on_draw:
                    on_draw(session, slot, draws[len(rejected)], rejected, attempt)
                session.submit(slot, value)
            pending = still

        # Anything still unresolved gets a grammar that cannot go wrong.
        if pending and config.use_structured_fallback:
            params = []
            usable = []
            for index, session, slot, prompt in pending:
                structured = structured_for(engine, slot)
                if structured is None:
                    continue
                usable.append((index, session, slot, prompt))
                params.append(
                    engine.params(
                        max_tokens=config.tokens_for(slot),
                        n=1,
                        seed=seeds[index] + step,
                        structured=structured,
                    )
                )
            if usable:
                results = engine.generate([prompt for *_, prompt in usable], params)
                log.calls += len(usable)
                log.structured_fallbacks += len(usable)
                resolved = set()
                for (index, session, slot, _), draws in zip(usable, results):
                    value, rejected = _first_legal(slot, draws)
                    if value is None:
                        value = _default_value(slot)
                        log.forced += 1
                    log.fallback_slots.append(slot.id)
                    if on_draw:
                        on_draw(
                            session,
                            slot,
                            draws[0] if draws else "",
                            rejected,
                            config.max_rounds,
                        )
                    session.submit(slot, value)
                    resolved.add(id(session))
                pending = [item for item in pending if id(item[1]) not in resolved]

        for index, session, slot, _ in pending:
            log.forced += 1
            log.fallback_slots.append(slot.id)
            session.submit(slot, _default_value(slot))

        step += 1

    log.seconds += time.time() - started
    return log


def _first_legal(slot: Slot, draws: Iterable[str]):
    """The first legal parse among ``draws``, and the raw draws refused before it."""
    rejected: list[str] = []
    for draw in draws:
        value = slot.parse(draw)
        if value is not None:
            return value, rejected
        rejected.append(draw)
    return None, rejected


def run_sessions(
    engine: VLLMEngine,
    sessions: Sequence[Session],
    seeds: Sequence[int],
    config: SamplerConfig | None = None,
    on_group_done: Callable[[Sequence[Session], DrawLog], None] | None = None,
    on_draw: Callable[[Session, Slot, str, list, int], None] | None = None,
) -> DrawLog:
    """Run every session, one cache-sized group at a time."""
    config = config or SamplerConfig()
    total = DrawLog()
    for start in range(0, len(sessions), config.group_size):
        group = sessions[start : start + config.group_size]
        group_seeds = seeds[start : start + config.group_size]
        log = run_group(engine, group, config, group_seeds, on_draw=on_draw)
        total.merge(log)
        if on_group_done:
            on_group_done(group, log)
    return total
