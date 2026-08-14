"""Driving a base LLM through fill-in-the-blank transcripts.

Study-independent: give it sessions and a model, get back filled transcripts.
"""

from __future__ import annotations

from .engine import EngineConfig, VLLMEngine
from .driver import DrawLog, SamplerConfig, run_sessions

__all__ = ["DrawLog", "EngineConfig", "SamplerConfig", "VLLMEngine", "run_sessions"]
