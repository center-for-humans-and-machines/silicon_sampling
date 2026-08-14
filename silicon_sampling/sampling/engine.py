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
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


def configure_runtime(root: Path) -> None:
    """Environment vLLM needs before the engine is built.

    Two settings, both load-bearing in this container:

    ``VLLM_CACHE_ROOT`` / ``TORCHINDUCTOR_CACHE_DIR``
        ``/home/claude/.cache`` is root-owned, so vLLM cannot write its compiled
        graphs and would recompile the model on every start.

    ``VLLM_ENABLE_V1_MULTIPROCESSING=0``
        The V1 engine's separate ``EngineCore`` process reliably hangs here right
        after the weights reach the GPU — reproduced on a bare vLLM script with
        none of this package involved. Running the engine in-process avoids it,
        and has the side benefit that engine errors surface in the caller's
        traceback instead of a silent stall.
    """
    (root / "vllm").mkdir(parents=True, exist_ok=True)
    (root / "inductor").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("VLLM_CACHE_ROOT", str(root / "vllm"))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(root / "inductor"))
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


@dataclass
class EngineConfig:
    model: str = "Qwen/Qwen2.5-7B"
    dtype: str = "bfloat16"
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.90
    enable_prefix_caching: bool = True
    #: ``"auto"`` keeps the KV cache in ``dtype``; ``"fp8"`` roughly doubles how
    #: many respondent transcripts stay resident, at a small precision cost.
    kv_cache_dtype: str = "auto"
    #: Skip torch.compile / CUDA-graph capture.  Costs some decode throughput but
    #: removes a multi-minute startup, which matters for a run that restarts often.
    enforce_eager: bool = False
    seed: int = 0
    extra: dict = field(default_factory=dict)


class VLLMEngine:
    """Offline batch generation over a list of prompts."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self._llm = None

    def __enter__(self) -> "VLLMEngine":
        from vllm import LLM

        config = self.config
        self._llm = LLM(
            model=config.model,
            dtype=config.dtype,
            max_model_len=config.max_model_len,
            gpu_memory_utilization=config.gpu_memory_utilization,
            enable_prefix_caching=config.enable_prefix_caching,
            kv_cache_dtype=config.kv_cache_dtype,
            enforce_eager=config.enforce_eager,
            seed=config.seed,
            **config.extra,
        )
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

    def params(self, *, max_tokens: int, n: int, seed: int, guided=None):
        """Sampling parameters: faithful to the model's own distribution."""
        from vllm import SamplingParams

        kwargs = dict(
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            repetition_penalty=1.0,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            max_tokens=max_tokens,
            n=n,
            seed=seed,
            stop=["\n"],
            detokenize=True,
        )
        if guided is not None:
            kwargs["guided_decoding"] = guided
        return SamplingParams(**kwargs)

    @staticmethod
    def guided_choice(options: Sequence[str]):
        from vllm.sampling_params import GuidedDecodingParams

        return GuidedDecodingParams(choice=list(options))

    @staticmethod
    def guided_regex(pattern: str):
        from vllm.sampling_params import GuidedDecodingParams

        return GuidedDecodingParams(regex=pattern)

    def generate(
        self, prompts: Sequence[str], params: Sequence[object]
    ) -> list[list[str]]:
        """Return, per prompt, the text of each sampled continuation."""
        if not prompts:
            return []
        outputs = self._llm.generate(list(prompts), list(params), use_tqdm=False)
        return [
            [completion.text for completion in output.outputs] for output in outputs
        ]

    def kv_cache_tokens(self) -> int | None:
        """How many tokens of KV cache the engine allocated, if it will say."""
        try:
            cache_config = self._llm.llm_engine.vllm_config.cache_config
            return int(cache_config.num_gpu_blocks) * int(cache_config.block_size)
        except Exception:  # pragma: no cover - informational only
            return None
