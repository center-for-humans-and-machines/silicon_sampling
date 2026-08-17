"""A thin vLLM wrapper with the sampling settings this project needs.

Faithful sampling means an *untruncated* distribution: temperature 1.0, no
top-p/top-k cut, no repetition or presence penalties.  Any of those would reshape
the very distribution we are trying to measure.  The only shaping applied is
truncation at the first newline, which just stops the model from running past the
response line; the tokens before it are still drawn from the model's own
distribution.

The engine holds a large GPU allocation, so it is a context manager: leaving the
block tears the worker down.  A killed driver otherwise leaves an orphan
``VLLM::EngineCore`` holding the weights, and the next run fails to start.

The structured-output API moved between vLLM 0.11 and 0.23
(``guided_decoding`` -> ``structured_outputs``), and several ``LLM()`` keywords
became engine-arg passthroughs.  Both spellings are supported here so the package
runs on either.
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


def cache_root(fallback: Path | None = None) -> Path:
    """Where vLLM may write compiled graphs, and matplotlib its font cache.

    Compiled CUDA graphs must persist across restarts — recompiling costs minutes
    on every engine start, which a restartable run cannot afford — so this wants a
    writable path outside the repository.  ``SILICON_SAMPLING_CACHE`` names one
    explicitly, which is how the cluster points at a bound-in scratch directory;
    a container home is not always writable there.
    """
    candidates = []
    explicit = os.environ.get("SILICON_SAMPLING_CACHE")
    if explicit:
        candidates.append(Path(explicit))
    candidates.append(Path.home() / ".cache" / "silicon_sampling")
    if fallback is not None:
        candidates.append(fallback)
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".writable"
            probe.touch()
            probe.unlink()
            return candidate
        except OSError:
            continue
    raise SystemExit(
        "no writable cache directory; set SILICON_SAMPLING_CACHE to one"
    )  # pragma: no cover


def configure_runtime(root: Path, in_process: bool = True) -> None:
    """Environment vLLM needs before the engine is built.

    ``VLLM_CACHE_ROOT`` / ``TORCHINDUCTOR_CACHE_DIR``
        Keep compiled graphs on a writable path that survives container
        restarts.  Without a usable cache vLLM recompiles the model on every
        start, which a restartable run cannot afford.

    ``HF_HUB_CACHE`` / ``HF_HUB_OFFLINE=1``
        Point at whichever hub cache actually holds the weights, then forbid
        downloads.  Both halves are needed here: ``HF_HOME`` is set to a
        container-local path that carries only the small metadata files, while
        the 15 GB of weights live in the mounted cache — and the image's
        ``hf_xet`` is broken, so any attempted fetch raises instead of falling
        back.  Going offline also keeps a run reproducible.

    ``VLLM_ENABLE_V1_MULTIPROCESSING=0``
        Runs the engine in the calling process.  This began as a workaround —
        the separate ``EngineCore`` process reliably hung on this machine under
        vLLM 0.11 — and is kept because it also suits the workload: this driver
        issues tens of thousands of *tiny* generate calls, and in-process avoids
        an IPC round trip on each one.  It also puts engine errors in the
        caller's traceback instead of a silent stall.

        Only for a single GPU.  Tensor parallelism *is* the worker processes, so
        callers must pass ``in_process=False`` whenever
        ``tensor_parallel_size > 1`` or the engine cannot start.
    """
    (root / "vllm").mkdir(parents=True, exist_ok=True)
    (root / "inductor").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("VLLM_CACHE_ROOT", str(root / "vllm"))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(root / "inductor"))
    if in_process:
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    hub = resolve_hub_cache()
    if hub is not None:
        os.environ["HF_HUB_CACHE"] = str(hub)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


def hub_cache_candidates() -> list[Path]:
    """Places a hub cache might live, most explicit first."""
    candidates = []
    for value in (os.environ.get("HF_HUB_CACHE"), os.environ.get("HF_HOME")):
        if value:
            path = Path(value)
            candidates.append(path if path.name == "hub" else path / "hub")
    candidates.append(Path.home() / ".cache" / "huggingface" / "hub")
    return candidates


def resolve_hub_cache() -> Path | None:
    """The first hub cache that actually contains model weights.

    A metadata-only cache is worse than no cache: the config and tokenizer
    resolve, the engine starts, and it then fails on the missing safetensors —
    or, with downloads enabled, silently re-fetches 15 GB.  So the test is for
    weights, not for the directory existing.
    """
    seen = set()
    for candidate in hub_cache_candidates():
        if candidate in seen or not candidate.is_dir():
            continue
        seen.add(candidate)
        if any(candidate.glob("models--*/blobs/*")) and _has_weights(candidate):
            return candidate
    return None


def _has_weights(hub: Path) -> bool:
    for snapshot in hub.glob("models--*/snapshots/*"):
        for entry in snapshot.iterdir():
            if entry.suffix in (".safetensors", ".bin") and entry.resolve().is_file():
                return True
    return False


def _as_prompt(prompt):
    """vLLM's prompt form for either text or token ids."""
    return prompt if isinstance(prompt, str) else {"prompt_token_ids": list(prompt)}


@dataclass
class EngineConfig:
    model: str = "Qwen/Qwen2.5-7B"
    dtype: str = "bfloat16"
    #: Caps a single sequence.  Set just above the longest transcript: it does not
    #: change how much KV cache exists, but it does bound worst-case allocation.
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.92
    enable_prefix_caching: bool = True
    #: ``"auto"`` keeps the KV cache in ``dtype``; ``"fp8"`` roughly doubles how
    #: many respondent transcripts stay resident, at the cost of quantising
    #: attention — a real fidelity trade for a project measuring distributions.
    kv_cache_dtype: str = "auto"
    #: Skip torch.compile / CUDA-graph capture.  Eager costs a large factor on
    #: this workload (hundreds of tiny decode steps, launch-bound), so it is off.
    enforce_eager: bool = False
    #: Concurrent sequences the scheduler will run.  Must be at least
    #: ``group_size * draws_per_call`` or the driver's rounds get split in two.
    max_num_seqs: int = 256
    #: Token budget per scheduler step.  Each round prefills roughly
    #: ``group_size * 90`` new tokens, so the default is ample.
    max_num_batched_tokens: int | None = None
    #: GPUs to shard the weights across.  Anything above 1 needs vLLM's worker
    #: processes, so it is incompatible with the in-process engine — see
    #: :func:`configure_runtime`.
    tensor_parallel_size: int = 1
    #: Route the MoE experts across ranks instead of slicing every expert.  Each
    #: rank then streams whole expert matrices, which is the shape the decode
    #: kernels want; only meaningful for a mixture-of-experts model.
    enable_expert_parallel: bool = False
    seed: int = 0
    extra: dict = field(default_factory=dict)


class VLLMEngine:
    """Offline batch generation over a list of prompts."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self._llm = None
        self._structured = None

    def __enter__(self) -> "VLLMEngine":
        from vllm import LLM

        config = self.config
        kwargs = dict(
            model=config.model,
            dtype=config.dtype,
            max_model_len=config.max_model_len,
            gpu_memory_utilization=config.gpu_memory_utilization,
            enable_prefix_caching=config.enable_prefix_caching,
            kv_cache_dtype=config.kv_cache_dtype,
            enforce_eager=config.enforce_eager,
            max_num_seqs=config.max_num_seqs,
            seed=config.seed,
            **config.extra,
        )
        if config.max_num_batched_tokens:
            kwargs["max_num_batched_tokens"] = config.max_num_batched_tokens
        if config.tensor_parallel_size > 1:
            kwargs["tensor_parallel_size"] = config.tensor_parallel_size
        if config.enable_expert_parallel:
            kwargs["enable_expert_parallel"] = True
        self._llm = LLM(**kwargs)
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()

    def shutdown(self) -> None:
        if self._llm is None:
            return
        for attribute in ("engine_core", "llm_engine"):
            target = getattr(self._llm, attribute, None)
            shutdown = getattr(target, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception:  # pragma: no cover - best effort across versions
                    pass
        self._llm = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass

    # -- generation ------------------------------------------------------- #

    def params(self, *, max_tokens: int, n: int, seed: int, structured=None):
        """Sampling parameters: faithful to the model's own distribution."""
        from vllm import SamplingParams

        kwargs = dict(
            temperature=1.0,
            top_p=1.0,
            top_k=0,  # 0 disables the cut-off in current vLLM (-1 in older ones)
            repetition_penalty=1.0,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            max_tokens=max_tokens,
            n=n,
            seed=seed,
            stop=["\n"],
            detokenize=True,
        )
        if structured is not None:
            kwargs[self._structured_field()] = structured
        return SamplingParams(**kwargs)

    @staticmethod
    def _structured_field() -> str:
        from vllm import SamplingParams

        names = getattr(SamplingParams, "__annotations__", {})
        return (
            "structured_outputs" if "structured_outputs" in names else "guided_decoding"
        )

    @classmethod
    def _structured_params(cls, **kwargs):
        try:
            from vllm.sampling_params import StructuredOutputsParams as Params
        except ImportError:  # pragma: no cover - vLLM < 0.16
            from vllm.sampling_params import GuidedDecodingParams as Params
        return Params(**kwargs)

    @classmethod
    def structured_choice(cls, options: Sequence[str]):
        return cls._structured_params(choice=list(options))

    @classmethod
    def structured_regex(cls, pattern: str):
        return cls._structured_params(regex=pattern)

    def generate(
        self, prompts: Sequence[str] | Sequence[Sequence[int]], params: Sequence[object]
    ) -> list[list[str]]:
        """Return, per prompt, the text of each sampled continuation.

        Prompts may be text or token ids.  Ids skip vLLM's tokenisation of a
        transcript the caller has already tokenised — worth ~1 µs per prompt
        token per call, which is most of this pipeline's CPU time.
        """
        if not prompts:
            return []
        outputs = self._llm.generate(
            [_as_prompt(prompt) for prompt in prompts], list(params), use_tqdm=False
        )
        return [
            [completion.text for completion in output.outputs] for output in outputs
        ]

    # -- introspection ---------------------------------------------------- #

    def kv_cache_tokens(self) -> int | None:
        """How many tokens of KV cache the engine allocated, if it will say."""
        for getter in (self._cache_from_config, self._cache_from_scheduler):
            try:
                value = getter()
                if value:
                    return int(value)
            except Exception:  # pragma: no cover - informational only
                continue
        return None

    def _cache_from_config(self) -> int | None:
        cache_config = self._llm.llm_engine.vllm_config.cache_config
        blocks = getattr(cache_config, "num_gpu_blocks", None)
        return int(blocks) * int(cache_config.block_size) if blocks else None

    def _cache_from_scheduler(
        self,
    ) -> int | None:  # pragma: no cover - version dependent
        core = self._llm.llm_engine.engine_core
        for path in ("engine_core.scheduler", "scheduler"):
            target = core
            for part in path.split("."):
                target = getattr(target, part, None)
                if target is None:
                    break
            manager = getattr(target, "kv_cache_manager", None)
            if manager is not None:
                blocks = getattr(manager, "num_gpu_blocks", None)
                size = getattr(manager, "block_size", 16)
                if blocks:
                    return int(blocks) * int(size)
        return None

    def group_size_for(
        self, tokens_per_session: int, safety: float = 0.95, cap: int = 128
    ) -> int:
        """How many sessions can keep their transcripts resident at once.

        This is the number that decides throughput. Each session's prompt grows to
        ``tokens_per_session`` and must stay in the KV cache across all ~80 of its
        steps; the moment the working set exceeds the cache, evicted sessions pay
        a full re-prefill on their next step and the whole design collapses. The
        safety margin leaves room for the blocks in flight during a step.

        ``tokens_per_session`` should be the *worst-case* transcript length, not
        the typical one, so the margin on top of it can stay small.
        """
        cache = self.kv_cache_tokens()
        if not cache:
            return 16
        return max(1, min(cap, int(cache * safety // max(tokens_per_session, 1))))
