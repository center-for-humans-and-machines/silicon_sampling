"""Modules that import themselves on first use.

The sampling path has to run inside the Muse-Glimmer vLLM container, which ships
no pandas and no scipy.  Nothing on that path needs a DataFrame — profiles are read
with the stdlib ``csv`` module — but two modules on the import chain use pandas in
functions only the *analysis* path calls, and a module-level import there makes the
whole sampler unimportable in that container.

Rewriting every call site to import locally is the obvious fix and the wrong one:
it means editing a dozen function bodies, and a heuristic pass over them is exactly
how this project once split a ``@dataclass`` from its class and buried an import in
a nested function.  A proxy is one line per module and cannot land in the wrong
place.

Type annotations keep working because both affected modules use
``from __future__ import annotations``, so ``pd.DataFrame`` in a signature is never
evaluated.
"""

from __future__ import annotations

import importlib
from typing import Any


class LazyModule:
    """Stands in for a module until something actually touches it."""

    def __init__(self, name: str) -> None:
        self._lazy_name = name
        self._lazy_module: Any = None

    def _load(self) -> Any:
        if self._lazy_module is None:
            self._lazy_module = importlib.import_module(self._lazy_name)
        return self._lazy_module

    def __getattr__(self, attribute: str) -> Any:
        return getattr(self._load(), attribute)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        state = "loaded" if self._lazy_module is not None else "not yet imported"
        return f"<LazyModule {self._lazy_name!r} ({state})>"


def lazy_module(name: str) -> LazyModule:
    """A stand-in for ``import <name> as ...`` that defers the import."""
    return LazyModule(name)
