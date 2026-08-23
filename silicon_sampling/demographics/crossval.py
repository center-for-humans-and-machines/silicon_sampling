"""Why the conditional model is all-two-way IPF and not something smoother.

``joint`` fits the maximum-entropy table consistent with every two-way CCAM
margin, and discards every higher-order interaction.  That is a modelling choice
with an obvious alternative — keep the saturated cell counts and shrink them
toward a coarser table — so it was cross-validated rather than argued.  This
module is that cross-validation, kept in the repository because a held-out
log-likelihood quoted in a docstring with no code behind it is an assertion, not
a result.

**What is scored.**  The held-out weighted mean log-likelihood, in nats, of a
respondent's drawn tuple — (education, income, party) for Pfänder — given their
gender, age band and race.  Folds are by respondent, not by donor row: the
crosswalk splits one respondent across up to four rows (see ``ccam.donor_table``),
and letting those rows straddle a fold would leak the respondent into their own
training set.  Every model gets the same uniform floor, so a tuple no model has
ever seen costs all of them the same and the comparison is about structure rather
than about who handles zeros more gracefully.

**The five models.**  ``ignored`` throws the demographics away and predicts the
pooled tuple distribution.  ``shrunk`` keeps the saturated cell counts and pulls
them toward that pooled table with one strength ``kappa``.  ``backoff`` climbs a
ladder — pooled, then race, then age x race, then gender x age x race — shrinking
each rung toward the one below with its own strength.  ``ipf`` is the shipped
model.  ``ipf_pairs_only`` is ``ipf`` with the moderator-moderator margins
removed, which isolates how much of the gain is the education-income association
rather than the demographics.

**The tradeoff to keep in view.**  The two shrinkage models are handed their best
hyperparameter *on the folds they are scored on*, so their numbers are optimistic
and IPF's is not.  That asymmetry is deliberate: it makes the comparison a
worst-case one for the model actually shipped, and IPF still wins.  What it means
is that these figures are a model-selection argument and not an estimate of
out-of-sample accuracy for any of the five.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import ccam, joint
from .codebook import PFANDER, Codebook, GENDER_FROM_CCAM, RACE_FROM_CCAM, age_band

#: Folds, and the seed that assigns respondents to them.  Five is enough that
#: each fold holds ~1,240 respondents and cheap enough to run in a minute.
FOLDS = 5
FOLD_SEED = 20260823

#: Weight on the uniform distribution over tuples, identical for every model.
#: Small enough not to move a well-estimated cell, large enough that an unseen
#: tuple costs a finite amount.
FLOOR = 1e-6

#: Shrinkage strengths searched for ``shrunk`` and for every rung of ``backoff``.
KAPPA_GRID = (10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000)


def respondent_table(
    book: Codebook = PFANDER,
    waves: Sequence[int] = ccam.STRUCTURE_WAVES,
    path: str | None = None,
) -> pd.DataFrame:
    """``ccam.donor_table`` plus the respondent id the folds are cut on.

    A separate builder rather than a column on ``donor_table``: the fitting path
    has no use for a case id, and carrying one there would invite someone to fold
    on donor rows, which is exactly the leak this module exists to avoid.
    """
    frame = ccam.load(path)
    rows = frame[frame["wave"].isin(list(waves))]
    kept = rows[
        rows["income"].between(1, 21) & rows["party"].isin(list(book.party_codes))
    ]
    columns = ["case_ID", "gender", "age_band", "race", *book.drawn, "weight"]
    records: list[tuple] = []
    for row in kept.itertuples():
        cell = (
            int(row.case_ID),
            GENDER_FROM_CCAM[int(row.gender)],
            age_band(row.age, book.age_bands),
            RACE_FROM_CCAM[int(row.race)],
        )
        source = {"education": row.educ, "income": row.income, "party": row.party}
        splits = [tuple(book.shares(axis, source[axis]).items()) for axis in book.drawn]
        for combination in itertools.product(*splits):
            weight = float(row.weight_aggregate)
            for _, share in combination:
                weight *= share
            records.append(cell + tuple(level for level, _ in combination) + (weight,))
    return pd.DataFrame(records, columns=columns)


@dataclass(frozen=True)
class Design:
    """A donor table reduced to integer positions, plus its fold assignment."""

    book: Codebook
    #: One column per axis, in ``space.axes`` order, then weight and fold.
    index: np.ndarray
    weight: np.ndarray
    fold: np.ndarray

    @property
    def dims(self) -> tuple[int, ...]:
        return joint.space(self.book).dims


def design(
    donors: pd.DataFrame, book: Codebook = PFANDER, folds: int = FOLDS
) -> Design:
    """Encode a respondent table into positions and assign respondents to folds."""
    import random

    shape = joint.space(book)
    columns = []
    for axis in shape.axes:
        lookup = {level: i for i, level in enumerate(shape.levels(axis))}
        columns.append(donors[axis].map(lookup).to_numpy(dtype=int))
    people = sorted(set(int(value) for value in donors["case_ID"]))
    random.Random(FOLD_SEED).shuffle(people)
    assignment = {person: i % folds for i, person in enumerate(people)}
    return Design(
        book=book,
        index=np.stack(columns, axis=1),
        weight=donors["weight"].to_numpy(dtype=float),
        fold=np.array([assignment[int(value)] for value in donors["case_ID"]]),
    )


def _counts(design: Design, rows: np.ndarray) -> np.ndarray:
    """The weighted contingency table of a row subset."""
    table = np.zeros(design.dims)
    np.add.at(table, tuple(design.index[rows].T), design.weight[rows])
    return table


def _normalise(table: np.ndarray, over: tuple[int, ...]) -> np.ndarray:
    """Turn counts into a conditional distribution over ``over``, uniform where empty."""
    total = table.sum(axis=over, keepdims=True)
    size = int(np.prod([table.shape[axis] for axis in over]))
    return np.where(total > 0, table / np.where(total > 0, total, 1.0), 1.0 / size)


def _score(
    design: Design, rows: np.ndarray, predicted: np.ndarray, drawn: tuple[int, ...]
) -> tuple[float, float]:
    """Weighted sum of log-likelihood over ``rows``, and the weight behind it."""
    size = int(np.prod([design.dims[axis] for axis in drawn]))
    floored = (1.0 - FLOOR) * predicted + FLOOR / size
    values = floored[tuple(design.index[rows].T)]
    weight = design.weight[rows]
    return float((weight * np.log(values)).sum()), float(weight.sum())


# --------------------------------------------------------------------------- #
# the five models: each returns P(drawn | given) as a full table
# --------------------------------------------------------------------------- #


def _ignored(counts: np.ndarray, given: tuple[int, ...], drawn: tuple[int, ...]):
    pooled = counts.sum(axis=given)
    pooled = pooled / pooled.sum()
    shape = [1] * counts.ndim
    for axis in drawn:
        shape[axis] = counts.shape[axis]
    return np.broadcast_to(pooled.reshape(shape), counts.shape)


def _ladder(
    counts: np.ndarray, given: tuple[int, ...], drawn: tuple[int, ...]
) -> list[np.ndarray]:
    """Cell counts at each rung of the backoff ladder, coarsest first."""
    rungs = []
    for depth in range(len(given) + 1):
        collapse = tuple(given[: len(given) - depth])
        table = counts.sum(axis=collapse, keepdims=True) if collapse else counts
        rungs.append(table)
    return rungs


def _shrink(
    counts: np.ndarray, prior: np.ndarray, drawn: tuple[int, ...], kappa: float
) -> np.ndarray:
    """``(n + kappa * prior) / (n + kappa)``, normalised over the drawn axes."""
    total = counts.sum(axis=drawn, keepdims=True)
    return (counts + kappa * prior) / (total + kappa)


def _ipf_conditional(
    counts: np.ndarray, pairs: Iterable[tuple[int, int]], drawn: tuple[int, ...]
) -> np.ndarray:
    observed = counts / counts.sum()
    rank = counts.ndim
    margins = [
        (pair, observed.sum(axis=tuple(k for k in range(rank) if k not in pair)))
        for pair in pairs
    ]
    table, _ = joint.ipf(np.ones(counts.shape), margins)
    return _normalise(table, drawn)


def compare(
    donors: pd.DataFrame | None = None,
    book: Codebook = PFANDER,
    folds: int = FOLDS,
    kappas: Sequence[int] = KAPPA_GRID,
) -> pd.DataFrame:
    """Held-out mean log-likelihood of every model, best hyperparameter included."""
    donors = respondent_table(book) if donors is None else donors
    plan = design(donors, book, folds)
    shape = joint.space(book)
    given, drawn = shape.given, shape.drawn
    modmod = tuple(pair for pair in shape.pairs if pair[0] in drawn)
    structure = tuple(pair for pair in shape.pairs if pair not in modmod)

    totals: dict[str, list[float]] = {}
    weights: dict[str, list[float]] = {}
    chosen: dict[str, object] = {}

    def add(name: str, value: float, weight: float) -> None:
        totals.setdefault(name, []).append(value)
        weights.setdefault(name, []).append(weight)

    per_fold_shrunk: dict[int, dict[int, tuple[float, float]]] = {}
    per_fold_backoff: dict[int, dict[tuple[int, ...], tuple[float, float]]] = {}

    for fold in range(folds):
        train = np.flatnonzero(plan.fold != fold)
        test = np.flatnonzero(plan.fold == fold)
        counts = _counts(plan, train)

        add("ignored", *_score(plan, test, _ignored(counts, given, drawn), drawn))
        add(
            "ipf",
            *_score(plan, test, _ipf_conditional(counts, shape.pairs, drawn), drawn),
        )
        add(
            "ipf_pairs_only",
            *_score(plan, test, _ipf_conditional(counts, structure, drawn), drawn),
        )

        pooled = _normalise(counts.sum(axis=given, keepdims=True), drawn)
        per_fold_shrunk[fold] = {
            kappa: _score(plan, test, _shrink(counts, pooled, drawn, kappa), drawn)
            for kappa in kappas
        }
        per_fold_backoff[fold] = _backoff_scores(
            plan, test, counts, given, drawn, kappas
        )

    best_kappa = _best(per_fold_shrunk, folds)
    chosen["shrunk"] = best_kappa
    for fold in range(folds):
        add("shrunk", *per_fold_shrunk[fold][best_kappa])

    best_rungs = _best(per_fold_backoff, folds)
    chosen["backoff"] = list(best_rungs)
    for fold in range(folds):
        add("backoff", *per_fold_backoff[fold][best_rungs])

    order = ["ignored", "shrunk", "backoff", "ipf", "ipf_pairs_only"]
    return pd.DataFrame(
        [
            {
                "model": name,
                "log_likelihood": sum(totals[name]) / sum(weights[name]),
                "hyperparameter": chosen.get(name, ""),
            }
            for name in order
        ]
    )


def _backoff_scores(
    plan: Design,
    test: np.ndarray,
    counts: np.ndarray,
    given: tuple[int, ...],
    drawn: tuple[int, ...],
    kappas: Sequence[int],
) -> dict[tuple[int, ...], tuple[float, float]]:
    """Every rung combination the coordinate search may ask about, scored once.

    The ladder has three rungs and the grid fourteen points, so the full product
    is 2,744 evaluations per fold — cheap enough to score exhaustively and much
    easier to defend than a search path.
    """
    rungs = _ladder(counts, given, drawn)
    pooled = _normalise(rungs[0], drawn)
    out: dict[tuple[int, ...], tuple[float, float]] = {}
    for first in kappas:
        level1 = _shrink(rungs[1], pooled, drawn, first)
        for second in kappas:
            level2 = _shrink(rungs[2], level1, drawn, second)
            for third in kappas:
                level3 = _shrink(rungs[3], level2, drawn, third)
                out[(first, second, third)] = _score(plan, test, level3, drawn)
    return out


def _best(per_fold: Mapping[int, Mapping], folds: int):
    """The hyperparameter with the best pooled held-out likelihood."""
    keys = list(per_fold[0])
    scored = {
        key: sum(per_fold[fold][key][0] for fold in range(folds))
        / sum(per_fold[fold][key][1] for fold in range(folds))
        for key in keys
    }
    return max(scored, key=lambda key: scored[key])


def mutual_information(
    donors: pd.DataFrame | None = None, book: Codebook = PFANDER
) -> pd.Series:
    """Mutual information, in nats, of each pair of axes under the donor table.

    The legible half of the model-selection argument: IPF wins because the
    association it keeps between the drawn axes is much larger than anything the
    demographics carry, and this is the number that says so.
    """
    donors = ccam.donor_table(book) if donors is None else donors
    observed = joint.contingency(donors, book)
    shape = joint.space(book)
    out = {}
    for first, second in shape.pairs:
        collapse = tuple(k for k in range(len(shape.axes)) if k not in (first, second))
        pair = observed.sum(axis=collapse)
        product = np.outer(pair.sum(axis=1), pair.sum(axis=0))
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(pair > 0, pair * np.log(pair / product), 0.0)
        out[f"{shape.axes[first]} x {shape.axes[second]}"] = float(term.sum())
    return pd.Series(out, name="nats")


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    from .codebook import study

    parser = argparse.ArgumentParser(prog="silicon_sampling.demographics.crossval")
    parser.add_argument("--study", default="pfander")
    args = parser.parse_args(list(argv) if argv is not None else None)
    book = study(args.study)

    table = compare(book=book)
    print(f"five-fold held-out weighted mean log-likelihood ({book.name}, nats)")
    for row in table.itertuples():
        detail = f"  best {row.hyperparameter}" if row.hyperparameter != "" else ""
        print(f"  {row.model:16s} {row.log_likelihood:8.4f}{detail}")
    print("\nmutual information between axes, nats")
    info = mutual_information(book=book).sort_values(ascending=False)
    for name, value in info.items():
        print(f"  {name:28s} {value:.4f}")
    return 0


if __name__ == "__main__":  # pragma: no cover - a CLI
    raise SystemExit(main())
