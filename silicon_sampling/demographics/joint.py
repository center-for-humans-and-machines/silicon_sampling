"""The joint distribution the moderators are drawn from, and how to draw it.

Two constraints pull in opposite directions.  Pfänder's preregistration publishes
gender x age and gender x race counts — those are its *recruitment quotas*, so a
submission that does not reproduce them is describing a different sample.  CCAM,
meanwhile, is the only thing that knows how education, income and party covary
with each other and with the demographics.  So the two are used for different
jobs: gender, age and race come from the quota, exactly and by integer count, and
CCAM supplies ``P(education, income, party | gender, age band, race)``.
``pfander.profiles`` keeps building the quota cells the way it always has; this
module never touches them, which is why the margins come out exact rather than
approximately exact.

**The conditional model is the maximum-entropy table consistent with every
two-way margin CCAM can estimate.**  Pfänder's six-way table has 3,840 cells and
CCAM offers 6,147 respondents, so its saturated version is noise: 1,938 cells are
occupied at all, the median occupied cell holds two respondents, and 449 cells
reach five.  Its two-way margins are a different matter — 8 to 30 cells each,
with median occupancies from 164 to 1,079.  Iterative proportional fitting to all
15 of them gives the unique table that reproduces every one and adds no
higher-order interaction: the same argument, and the same maximum-entropy answer,
that ``pfander.profiles`` already uses to turn two published two-way quota margins
into a three-way joint.

That choice was checked rather than assumed, and the check is in the repository —
``crossval.py``, runnable as ``python -m silicon_sampling.demographics.crossval``.
Five-fold held-out log-likelihood of a respondent's (education, income, party)
triple, in nats, weighted, folded by respondent, with an identical uniform floor
on every model so unseen triples are treated the same way::

    demographics ignored                         -4.2614
    cell counts shrunk to the pooled table       -4.2098   (best kappa 300)
    hierarchical backoff r -> a,r -> g,a,r       -4.1773   (best 500/1000/1500)
    all-two-way IPF                              -4.1411
    IPF without the moderator-moderator margins  -4.2554

Which fold a respondent lands in moves each of those by up to 0.01 nats and the
ranking not at all; ``crossval.FOLD_SEED`` is what to change to see that.

IPF wins, and it wins without a tuning parameter — the two shrinkage models were
each given their best hyperparameter *on the same folds they were scored on* and
still lost.  It wins for a legible reason, too: the education-income association
(0.114 nats of mutual information) dwarfs every association a demographic
variable has with a moderator (0.009 for gender, 0.031 for age band, 0.091 for
race, summed over the three), so a model that keeps the two-way structure and
throws away the rest keeps nearly everything there is.  Dropping only the three
moderator-moderator margins costs the whole gain and lands back at the
no-information baseline.

**Levels are calibrated separately from associations.**  A second IPF pass fixes
the drawn axes' marginals to their calendar-2024 values and the
gender x age band x race margin to the quota, starting from the stage-one table.
IPF is multiplicative, so this pass moves main effects and leaves every odds
ratio in the stage-one table untouched: the associations stay 2022-2024, the
levels become 2024, and the demographic composition becomes the study's.  Both
passes converge to a maximum margin error below 1e-13, and the fitted table is
strictly positive (smallest cell 1.9e-8, i.e. 3e-4 expected respondents in
18,000), so there are no structural zeros and no smoothing floor is needed.

**One table per study, and no silent fallback between them.**  The fitting
machinery is study-agnostic; the level strings are not, and they come from a
``codebook.Codebook``.  A level string the codebook does not know is an error
rather than a shrug: the only levels a sampler is allowed to answer with the
population average are the ones the codebook lists as ``collapsible``, which in
practice means a third gender option that CCAM cannot condition on.  This is the
one place where being strict costs something — a study that renames a level
breaks loudly instead of quietly drawing national averages — and loudly is the
whole point, because a national-average draw looks exactly like a working
sampler.

**The residual bias worth knowing about.**  The target for Pfänder is the
*national* distribution.  Pfänder recruits from a non-probability opt-in panel
quota-matched on age, gender and race only, and such panels are better educated
than the country.  If the realised human marginals are ever published, swapping
them in is a one-argument change — ``fit(targets=...)`` — and nothing else about
the model has to move.
"""

from __future__ import annotations

import bisect
import csv
import itertools
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from . import ccam
from .codebook import PFANDER, Codebook

TABLE_DIR = ccam.ROOT / "data" / "demographics"


def table_path(book: Codebook = PFANDER, directory: Path | None = None) -> Path:
    """Where a study's fitted table ships."""
    return (directory or TABLE_DIR) / book.table_name


@dataclass(frozen=True)
class Space:
    """The shape of one study's table: which axes, which levels, in what order."""

    book: Codebook
    axes: tuple[str, ...]
    dims: tuple[int, ...]
    given: tuple[int, ...]
    drawn: tuple[int, ...]
    pairs: tuple[tuple[int, int], ...]

    def levels(self, axis: str) -> tuple[str, ...]:
        return self.book.levels(axis)

    def position(self, axis: str, level: str) -> int | None:
        """Index of a level on an axis, or ``None`` when the axis does not have it."""
        if axis == "race":
            try:
                level = self.book.race(level)
            except KeyError:
                return None
        return {name: i for i, name in enumerate(self.levels(axis))}.get(level)

    @property
    def drawn_dims(self) -> tuple[int, ...]:
        return tuple(self.dims[axis] for axis in self.drawn)

    @property
    def drawn_axes(self) -> tuple[str, ...]:
        return tuple(self.axes[axis] for axis in self.drawn)


@lru_cache(maxsize=8)
def space(book: Codebook = PFANDER) -> Space:
    """The ``Space`` for a codebook, cached because it is pure and asked for often."""
    axes = book.axes
    return Space(
        book=book,
        axes=axes,
        dims=tuple(len(book.levels(axis)) for axis in axes),
        given=tuple(range(len(book.given))),
        drawn=tuple(range(len(book.given), len(axes))),
        pairs=tuple(itertools.combinations(range(len(axes)), 2)),
    )


def contingency(donors: pd.DataFrame, book: Codebook = PFANDER) -> np.ndarray:
    """The weighted table of a donor frame, normalised to sum to one."""
    shape = space(book)
    index = []
    for axis in shape.axes:
        lookup = {level: i for i, level in enumerate(shape.levels(axis))}
        positions = donors[axis].map(lookup)
        if positions.isna().any():
            unknown = sorted(set(donors.loc[positions.isna(), axis]))
            raise ValueError(
                f"{axis}: levels not in the {book.name} codebook: {unknown}"
            )
        index.append(positions.to_numpy(dtype=int))
    table = np.zeros(shape.dims)
    np.add.at(table, tuple(index), donors["weight"].to_numpy(dtype=float))
    return table / table.sum()


def ipf(
    seed: np.ndarray,
    margins: Sequence[tuple[tuple[int, ...], np.ndarray]],
    max_rounds: int = 500,
    tol: float = 1e-13,
) -> tuple[np.ndarray, float]:
    """Fit ``seed`` to a set of margins, returning the table and its worst error.

    Each margin is ``(axes it is defined over, target array)``.  Convergence is
    guaranteed when the targets are mutually consistent and the seed is positive
    on the targets' support; this module only ever passes such sets.
    """
    rank = seed.ndim
    dims = seed.shape
    table = seed / seed.sum()
    worst = float("inf")
    for _ in range(max_rounds):
        worst = 0.0
        for axes, target in margins:
            collapse = tuple(k for k in range(rank) if k not in axes)
            current = table.sum(axis=collapse)
            worst = max(worst, float(np.abs(current - target).max()))
            shape = [1] * rank
            for axis in axes:
                shape[axis] = dims[axis]
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(
                    current > 0, target / np.where(current > 0, current, 1.0), 0.0
                )
            table = table * ratio.reshape(shape)
        table = table / table.sum()
        if worst < tol:
            break
    return table, worst


def quota_margin(book: Codebook = PFANDER) -> np.ndarray:
    """The gender x age band x race margin implied by the preregistration quotas.

    Read straight off ``pfander.profiles.cell_counts`` so there is exactly one
    definition of the quotas in the repository, and imported inside the function
    so that ``pfander.profiles`` can import this module without a cycle.  The
    benchmark's five race levels fold onto CCAM's four here, which is only a
    relabelling of the Asian and Other slabs onto the same one.
    """
    from ..pfander.profiles import cell_counts

    shape = space(book)
    margin = np.zeros(tuple(shape.dims[axis] for axis in shape.given))
    for (gender, band, race), count in cell_counts().items():
        key = tuple(
            shape.position(axis, level)
            for axis, level in zip(("gender", "age_band", "race"), (gender, band, race))
        )
        if None in key:
            raise ValueError(f"quota cell {(gender, band, race)} is not in {book.name}")
        margin[key] += count
    return margin / margin.sum()


@dataclass(frozen=True)
class Fit:
    """A fitted table, with the diagnostics that say whether to trust it."""

    table: np.ndarray
    book: Codebook
    #: Worst absolute margin error of each IPF pass.
    structure_error: float
    level_error: float
    #: Donor rows behind the association structure.
    donor_rows: int
    smallest_cell: float

    @property
    def space(self) -> Space:
        return space(self.book)

    def conditional(self) -> np.ndarray:
        """``P(drawn axes | given axes)``."""
        return self.table / self.table.sum(axis=self.space.drawn, keepdims=True)

    def marginal(self, axis: str) -> pd.Series:
        """One axis's marginal, as a Series indexed by level."""
        shape = self.space
        position = shape.axes.index(axis)
        collapse = tuple(k for k in range(len(shape.axes)) if k != position)
        return pd.Series(
            self.table.sum(axis=collapse), index=shape.levels(axis), name=axis
        )


def fit(
    book: Codebook = PFANDER,
    donors: pd.DataFrame | None = None,
    targets: Mapping[str, Sequence[float]] | None = None,
    demographics: np.ndarray | None = None,
) -> Fit:
    """Fit the two-stage model.

    ``donors`` supplies the associations (default: CCAM waves 26-31), ``targets``
    the drawn axes' levels (default: CCAM calendar 2024) and ``demographics`` the
    gender x age band x race margin.  That last default is the one study-specific
    branch in the fitting path: Pfänder's composition is fixed by a published
    recruitment quota, so it is read from there, and any other study falls back to
    CCAM's own national composition unless the caller passes its own.
    """
    shape = space(book)
    donors = ccam.donor_table(book) if donors is None else donors
    observed = contingency(donors, book)
    structure, structure_error = ipf(
        np.ones(shape.dims),
        [
            (
                pair,
                observed.sum(
                    axis=tuple(k for k in range(len(shape.axes)) if k not in pair)
                ),
            )
            for pair in shape.pairs
        ],
    )
    levels = ccam.level_targets(book) if targets is None else targets
    if demographics is None:
        demographics = (
            quota_margin(book) if book is PFANDER else observed.sum(axis=shape.drawn)
        )
    calibrated, level_error = ipf(
        structure,
        [(shape.given, np.asarray(demographics, dtype=float))]
        + [
            ((axis,), np.asarray(levels[shape.axes[axis]], dtype=float))
            for axis in shape.drawn
        ],
    )
    return Fit(
        table=calibrated,
        book=book,
        structure_error=structure_error,
        level_error=level_error,
        donor_rows=int(len(donors)),
        smallest_cell=float(calibrated.min()),
    )


# --------------------------------------------------------------------------- #
# the shipped table
# --------------------------------------------------------------------------- #


def write_table(
    path: Path | str | None = None,
    model: Fit | None = None,
    book: Codebook = PFANDER,
) -> Path:
    """Write a fitted table to CSV, so sampling needs neither CCAM nor pyreadstat.

    The .sav is a 35,000-row SPSS file and ``pyreadstat`` is not part of the
    runtime image; the sampler should not depend on either.  This is the same
    move ``pfander.paths`` makes with the benchmark's shipped materials — read the
    authoritative source once, snapshot what the run needs.
    """
    model = fit(book) if model is None else model
    book = model.book
    shape = space(book)
    path = Path(path) if path is not None else table_path(book)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(shape.axes + ("probability",))
        for index in itertools.product(*(range(dim) for dim in shape.dims)):
            row = [
                shape.levels(axis)[position]
                for axis, position in zip(shape.axes, index)
            ]
            writer.writerow(row + [f"{model.table[index]:.12e}"])
    return path


def read_table(path: Path | str | None = None, book: Codebook = PFANDER) -> np.ndarray:
    """Read a table written by ``write_table`` back into array form."""
    shape = space(book)
    lookup = {
        axis: {level: i for i, level in enumerate(shape.levels(axis))}
        for axis in shape.axes
    }
    path = Path(path) if path is not None else table_path(book)
    table = np.zeros(shape.dims)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != shape.axes + ("probability",):
            raise ValueError(
                f"{path}: columns {reader.fieldnames} are not the {book.name} axes"
            )
        for row in reader:
            table[tuple(lookup[axis][row[axis]] for axis in shape.axes)] = float(
                row["probability"]
            )
    if not np.isclose(table.sum(), 1.0, atol=1e-9):
        raise ValueError(f"{path}: probabilities sum to {table.sum():.6f}, not 1")
    return table


class Sampler:
    """A fitted table, ready to answer ``draw`` for one quota cell at a time.

    Holds the flattened conditional and its cumulative sums, because a sampling
    run asks for 18,000 draws and rebuilding the cdf per draw would dominate the
    profile build.
    """

    def __init__(self, table: np.ndarray, book: Codebook = PFANDER) -> None:
        shape = space(book)
        if table.shape != shape.dims:
            raise ValueError(
                f"table shape {table.shape} is not the {book.name} shape {shape.dims}"
            )
        self.book = book
        self.space = shape
        self.table = table
        self.mass = table.sum(axis=shape.drawn)
        conditional = table / table.sum(axis=shape.drawn, keepdims=True)
        self.flat = conditional.reshape(*(shape.dims[a] for a in shape.given), -1)
        self.cumulative = np.cumsum(self.flat, axis=-1)

    # -- one cell --------------------------------------------------------- #

    def _positions(
        self, gender: str, age_band: str, race: str
    ) -> tuple[int | None, ...]:
        """Axis positions for one cell; ``None`` only where the codebook allows it."""
        out: list[int | None] = []
        for axis, level in zip(
            ("gender", "age_band", "race"), (gender, age_band, race)
        ):
            position = self.space.position(axis, level)
            if position is None and (axis, level) not in self.book.collapsible:
                raise ValueError(
                    f"{self.book.name}: {axis}={level!r} is not a codebook level; "
                    f"add it to the codebook, or to its ``collapsible`` set if CCAM "
                    f"genuinely cannot condition on it"
                )
            out.append(position)
        return tuple(out)

    def conditional(self, gender: str, age_band: str, race: str) -> np.ndarray:
        """``P(drawn axes)`` for one cell, flattened over the drawn combinations.

        A level the codebook marks collapsible — a third gender option, which the
        quotas never produce but the instrument allows — is averaged out against
        the table's own mass rather than raising, so the caller gets the
        population average over that axis.
        """
        flat, mass = self.flat, self.mass
        for axis, position in enumerate(self._positions(gender, age_band, race)):
            if position is not None:
                flat = np.take(flat, [position], axis=axis)
                mass = np.take(mass, [position], axis=axis)
        total = (flat.reshape(-1, flat.shape[-1]) * mass.reshape(-1, 1)).sum(axis=0)
        return total / total.sum()

    def draw(self, gender: str, age_band: str, race: str, rng) -> dict[str, str]:
        """One respondent's drawn moderators, for their cell.

        Consumes exactly one float from ``rng``.  That is a contract, not an
        implementation detail: ``pfander.profiles`` keys the draw on a profile id
        so a rebuilt profile reproduces it, and anything that consumed a variable
        number of floats would make the draw depend on the cell.
        """
        positions = self._positions(gender, age_band, race)
        if None in positions:
            cumulative = np.cumsum(self.conditional(gender, age_band, race))
        else:
            cumulative = self.cumulative[positions]
        position = bisect.bisect_left(
            cumulative.tolist(), rng.random() * float(cumulative[-1])
        )
        index = np.unravel_index(
            min(position, len(cumulative) - 1), self.space.drawn_dims
        )
        return {
            axis: self.space.levels(axis)[int(position)]
            for axis, position in zip(self.space.drawn_axes, index)
        }


@lru_cache(maxsize=8)
def shipped(book: Codebook = PFANDER, path: str | None = None) -> Sampler:
    """The study's ``Sampler``, from its shipped CSV, refitted when there is none.

    Refitting uses ``fit``'s defaults, which are right for Pfänder and merely
    plausible for anyone else — a study whose levels come from its own published
    marginals should build its ``Sampler`` explicitly rather than rely on this.
    """
    resolved = Path(path) if path is not None else table_path(book)
    table = read_table(resolved, book) if resolved.exists() else fit(book).table
    return Sampler(table, book)


def draw(
    gender: str,
    age_band: str,
    race: str,
    rng,
    book: Codebook = PFANDER,
    path: Path | str | None = None,
) -> dict[str, str]:
    """One respondent's drawn moderators, from the study's shipped table."""
    return shipped(book, str(path) if path is not None else None).draw(
        gender, age_band, race, rng
    )


def cell_conditional(
    gender: str,
    age_band: str,
    race: str,
    book: Codebook = PFANDER,
    path: Path | str | None = None,
) -> np.ndarray:
    """``P(drawn axes)`` for one cell, from the study's shipped table."""
    return shipped(book, str(path) if path is not None else None).conditional(
        gender, age_band, race
    )


def sample(
    cells: Mapping[tuple[str, str, str], int],
    seed: int = 20260814,
    book: Codebook = PFANDER,
    path: Path | str | None = None,
) -> pd.DataFrame:
    """Draw a whole sample: one row per respondent, every axis filled.

    ``cells`` maps (gender, age band, race) to a respondent count — exactly what
    ``pfander.profiles.cell_counts`` returns — so the three given axes come out at
    their published integer counts by construction.
    """
    import random

    sampler = shipped(book, str(path) if path is not None else None)
    rng = random.Random(seed)
    rows = []
    for (gender, band, race), count in sorted(cells.items()):
        for _ in range(count):
            row = {"gender": gender, "age_band": band, "race": race}
            row.update(sampler.draw(gender, band, race, rng))
            rows.append(row)
    frame = pd.DataFrame(rows)
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def marginals(frame: pd.DataFrame, book: Codebook = PFANDER) -> dict[str, pd.Series]:
    """Realised proportions of every moderator in a drawn sample."""
    shape = space(book)
    out: dict[str, pd.Series] = {}
    for axis in shape.axes:
        counts = frame[axis].value_counts()
        levels = None if axis == "race" else shape.levels(axis)
        counts = counts.reindex(levels) if levels else counts.sort_index()
        out[axis] = (counts / counts.sum()).fillna(0.0)
    return out
