"""Where the project's data lives, which is not always beside the code.

On a laptop the repository and its data are one tree and none of this matters. On
a cluster they are not: the checkout sits under a home directory with a modest
quota, and the sampler writes gigabytes — a raw transcript tree runs about 900 MB
per 18,000-respondent run, and there are a dozen runs. Keeping outputs in the
checkout exhausted the home quota on DAIS and every job submission started failing
at ``mkdir``, which is a confusing way to discover a disk problem.

So the data root is configurable:

``SILICON_SAMPLING_DATA``
    Where study data lives. Defaults to ``<repo>/data``, so nothing changes for a
    single-machine setup.

**Reads fall back to the repository; writes do not.** The two kinds of thing under
``data/`` have different homes on a cluster: study *outputs* belong on scratch,
while *inputs* that ship with the project — the calibration datasets, a study's
Materials, the submission template — belong with the code. Rather than make every
call site classify itself, :func:`resolve` prefers the configured root and falls
back to the repository copy when the configured one does not exist. That way a
split layout works with no further annotation: on DAIS the study directories
resolve to scratch and ``data/calibration`` resolves to the checkout.

:func:`output` never falls back, because a write has to land somewhere definite.
"""

from __future__ import annotations

import os
from pathlib import Path

#: The repository root, two levels above this file.
REPO = Path(__file__).resolve().parents[1]

#: The data tree that ships with the checkout.
REPO_DATA = REPO / "data"

#: Environment variable naming an alternative data root.
DATA_ENV = "SILICON_SAMPLING_DATA"


def data_root() -> Path:
    """The configured data root, or the repository's own ``data/``."""
    configured = os.environ.get(DATA_ENV)
    return Path(configured).expanduser() if configured else REPO_DATA


def resolve(*parts: str | Path) -> Path:
    """A data path for *reading*, preferring the configured root.

    Falls back to the repository copy when the configured root has no such path,
    which is what lets a cluster keep bulky study outputs on scratch while the
    inputs that ship with the project stay beside the code.
    """
    primary = data_root().joinpath(*map(str, parts))
    if primary.exists():
        return primary
    fallback = REPO_DATA.joinpath(*map(str, parts))
    return fallback if fallback.exists() else primary


def output(*parts: str | Path) -> Path:
    """A data path for *writing*: always under the configured root, no fallback.

    A write needs one definite destination — silently splitting outputs across two
    roots depending on what happened to exist first is how a run ends up half in
    each.
    """
    return data_root().joinpath(*map(str, parts))


def describe() -> str:
    """One line naming the roots in force, for a run log."""
    root = data_root()
    where = "configured" if os.environ.get(DATA_ENV) else "repository default"
    return f"data root: {root} ({where}); fallback for reads: {REPO_DATA}"
