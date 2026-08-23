"""Command line for the demographic module: build the table, then check it.

python -m silicon_sampling.demographics.cli crosswalk   # every mapping decision
python -m silicon_sampling.demographics.cli fit          # refit and write the CSV
python -m silicon_sampling.demographics.cli check        # draw 18,000 and audit
python -m silicon_sampling.demographics.cli compare      # vs the model-generated runs
python -m silicon_sampling.demographics.cli arms         # control-arm cell sizes

Every command takes ``--study`` (default ``pfander``).  ``fit --study voelkel``
calibrates to Voelkel's published marginals rather than to CCAM's national ones,
which is why it reaches into that package for its targets.
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import pandas as pd

from ..models import LABELS, MODELS
from ..pfander import paths as pfander_paths
from . import ccam, joint
from .codebook import PFANDER, Codebook, study


def _rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def _book(args) -> Codebook:
    return study(getattr(args, "study", "pfander"))


def _model(book: Codebook) -> joint.Fit:
    """The fit for a study, with that study's own calibration targets."""
    if book is PFANDER:
        return joint.fit(book)
    from ..voelkel import profiles as voelkel_profiles

    return joint.fit(
        book,
        targets=voelkel_profiles.level_targets(),
        demographics=voelkel_profiles.given_margin(),
    )


def cmd_crosswalk(args) -> int:
    book = _book(args)
    counts = ccam.dropped(book)
    donors = ccam.donor_table(book)
    sizes = ccam.cell_sizes(donors)
    _rule("source")
    print(f"{ccam.CCAM_SAV}")
    print(f"codebook        {book.name}: axes {' x '.join(book.axes)}")
    print(
        f"structure waves {min(ccam.STRUCTURE_WAVES)}-{max(ccam.STRUCTURE_WAVES)}: "
        f"{counts['rows']:,} respondents"
    )
    print(f"level waves     {list(ccam.LEVEL_WAVES)}")
    print(
        f"dropped: {counts['party_refused']} party refusals, "
        f"{counts['income_out_of_range']} out-of-range income code(s)"
    )
    print(
        f"donor rows after splitting: {len(donors):,} "
        f"over {len(sizes)} gender x age-band x race cells "
        f"(min {sizes.min()}, median {int(sizes.median())})"
    )
    splits = book.split_codes
    print(
        f"CCAM categories spanning two {book.name} levels: {len(splits)}"
        + (
            f" ({', '.join(f'{axis} {code}' for axis, code in splits)})"
            if splits
            else ""
        )
    )
    for name, lines in ccam.crosswalk_notes(book).items():
        _rule(name)
        for line in lines:
            print(f"  {line}")
    _rule("weights")
    ratios = ccam.weight_ratios()
    for wave, ratio in ratios.items():
        print(f"  wave {int(wave)}  weight_aggregate / weight_wave = {ratio:.6f}")
    print(
        f"  range {ratios.min():.4f}-{ratios.max():.4f}, i.e. the two weights "
        f"reweight the waves against each other by "
        f"{100 * (ratios.max() / ratios.min() - 1):.1f}%"
    )
    return 0


def cmd_fit(args) -> int:
    book = _book(args)
    model = _model(book)
    path = joint.write_table(args.out, model, book)
    _rule(f"fit ({book.name})")
    print(f"donor rows                {model.donor_rows:,}")
    print(f"stage 1 max margin error  {model.structure_error:.2e}")
    print(f"stage 2 max margin error  {model.level_error:.2e}")
    print(f"smallest fitted cell      {model.smallest_cell:.2e}")
    print(f"written to                {path}")
    _rule("fitted marginals (%)")
    for axis in book.axes:
        print(f"  {axis}")
        for level, share in (100 * model.marginal(axis)).items():
            print(f"    {level:<40s} {share:6.2f}")
    return 0


def _margin(frame: pd.DataFrame, left: str, right: str) -> collections.Counter:
    return collections.Counter(zip(frame[left], frame[right]))


def cmd_check(args) -> int:
    from ..pfander.profiles import AGE_QUOTA, RACE_QUOTA, cell_counts

    book = _book(args)
    if book is not PFANDER:
        raise SystemExit("check audits the Pfänder quotas; no other study has any")
    cells = cell_counts(args.n)
    frame = joint.sample(cells, seed=args.seed, book=book)
    model = _model(book)
    donors = ccam.donor_table(book)
    ccam_own = {
        axis: ccam.marginal(donors, axis, book.levels(axis)) for axis in book.drawn
    }
    targets = ccam.level_targets(book)

    _rule(f"drawn sample: {len(frame):,} respondents")
    quota_share: collections.Counter = collections.Counter()
    for (gender, band, race), count in cells.items():
        quota_share[("gender", gender)] += count
        quota_share[("age_band", band)] += count
        quota_share[("race", race)] += count

    for axis in book.given:
        print(f"\n  {axis} (from the quotas)")
        print(f"    {'level':<40s} {'n':>6s} {'drawn %':>8s} {'quota %':>8s}")
        realised = frame[axis].value_counts()
        for level in sorted(realised.index, key=lambda name: -realised[name]):
            count = int(realised[level])
            print(
                f"    {level:<40s} {count:6d} {100 * count / len(frame):8.2f}"
                f" {100 * quota_share[(axis, level)] / args.n:8.2f}"
            )

    for axis in book.drawn:
        print(f"\n  {axis} (drawn from CCAM)")
        print(
            f"    {'level':<40s} {'n':>6s} {'drawn %':>8s} {'fitted %':>9s}"
            f" {'2024 %':>8s} {'22-24 %':>8s}"
        )
        realised = frame[axis].value_counts()
        for level in book.levels(axis):
            count = int(realised.get(level, 0))
            print(
                f"    {level:<40s} {count:6d} {100 * count / len(frame):8.2f}"
                f" {100 * model.marginal(axis)[level]:9.2f}"
                f" {100 * targets[axis][level]:8.2f}"
                f" {100 * ccam_own[axis][level]:8.2f}"
            )

    _rule("quota margins (the hard constraint)")
    apportioned_age: collections.Counter = collections.Counter()
    apportioned_race: collections.Counter = collections.Counter()
    for (gender, band, race), count in cells.items():
        apportioned_age[(gender, band)] += count
        apportioned_race[(gender, race)] += count
    published_age = collections.Counter(
        {
            (gender, band): count
            for band, _, male, female in AGE_QUOTA
            for gender, count in (("Male", male), ("Female", female))
        }
    )
    published_race = collections.Counter(
        {
            (gender, race): count
            for race, _, male, female in RACE_QUOTA
            for gender, count in (("Male", male), ("Female", female))
        }
    )

    worst = 0
    for name, drawn, apportioned, published in (
        (
            "gender x age",
            _margin(frame, "gender", "age_band"),
            apportioned_age,
            published_age,
        ),
        (
            "gender x race",
            _margin(frame, "gender", "race"),
            apportioned_race,
            published_race,
        ),
    ):
        exact = drawn == apportioned
        print(f"  {name}: drawn == cell_counts() apportionment -> {exact}")
        if not exact:
            raise SystemExit(f"{name}: the sampler perturbed the quota cells")
        for key in sorted(published):
            gap = drawn[key] - published[key]
            worst = max(worst, abs(gap))
            if gap:
                print(
                    f"    {key} published {published[key]:5d} drawn {drawn[key]:5d}"
                    f"  ({gap:+d})"
                )
    print(f"  worst deviation from the published quota tables: {worst} respondent(s)")
    if worst:
        print(
            "  NOTE: the two published tables disagree with each other by one "
            "respondent\n"
            "  (AGE_QUOTA gives 8,827 men, RACE_QUOTA gives 8,828), so no joint can\n"
            "  reproduce both exactly.  The residue is pfander.profiles's largest-\n"
            "  remainder apportionment, untouched by this module."
        )
    return 0


def cmd_arms(args) -> int:
    """Realised cell sizes in the arm the benchmark actually filters on.

    ``benchmark.scored`` drops a moderator level whose *control* arm holds fewer
    than 30 respondents, so a whole-sample marginal is the wrong thing to read: a
    level at 2% of 18,000 is 360 respondents and 40 control respondents, and only
    the second number decides whether it is scored.
    """
    from ..pfander.profiles import build
    from ..pfander.scoring import DESIGN

    book = _book(args)
    if book is not PFANDER:
        raise SystemExit("arms reads the Pfänder condition assignment")
    profiles = build(total=args.n, seed=args.seed, prefill=True)
    control = [profile for profile in profiles if profile.condition == "control"]
    floor = DESIGN.min_group_n
    _rule(f"control arm: {len(control):,} of {len(profiles):,}   floor {floor}")
    print(f"    {'moderator':<10s} {'level':<40s} {'control n':>9s} {'%':>6s} {'':>6s}")
    for axis in book.drawn:
        counts = collections.Counter(getattr(profile, axis) for profile in control)
        for level in book.levels(axis):
            count = counts[level]
            flag = "" if count >= floor else "  BELOW FLOOR"
            print(
                f"    {axis:<10s} {level:<40s} {count:9d}"
                f" {100 * count / len(control):6.2f}{flag}"
            )
    return 0


def cmd_compare(args) -> int:
    book = _book(args)
    model = _model(book)
    rows = []
    for run in args.runs or list(MODELS):
        path = pfander_paths.samples_dir(run) / "samples.csv"
        if not Path(path).exists():
            continue
        frame = pd.read_csv(path, low_memory=False, usecols=list(book.drawn))
        for axis in book.drawn:
            counts = frame[axis].value_counts()
            for level in book.levels(axis):
                rows.append(
                    {
                        "moderator": axis,
                        "level": level,
                        "run": LABELS.get(run, run),
                        "n": int(counts.get(level, 0)),
                        "percent": 100 * float(counts.get(level, 0)) / len(frame),
                    }
                )
    table = pd.DataFrame(rows)
    _rule("model-generated vs CCAM target (%)")
    print(
        f"    {'moderator':<10s} {'level':<40s} {'target':>7s} "
        + " ".join(f"{run:>18s}" for run in table["run"].unique())
    )
    for axis in book.drawn:
        for level in book.levels(axis):
            target = 100 * model.marginal(axis)[level]
            cells = table[(table.moderator == axis) & (table.level == level)]
            body = " ".join(
                f"{row.percent:9.2f} (n={row.n:5d})" for row in cells.itertuples()
            )
            print(f"    {axis:<10s} {level:<40s} {target:7.2f} {body}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="silicon_sampling.demographics")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, help: str):
        command = sub.add_parser(name, help=help)
        command.add_argument("--study", default="pfander")
        return command

    add("crosswalk", "print every CCAM -> study mapping decision")
    fit_parser = add("fit", "refit the joint and write the CSV")
    fit_parser.add_argument("--out", type=Path, default=None)
    check = add("check", "draw a sample and audit its margins")
    check.add_argument("--n", type=int, default=18000)
    check.add_argument("--seed", type=int, default=20260814)
    arms = add("arms", "control-arm cell sizes against the benchmark's floor")
    arms.add_argument("--n", type=int, default=18000)
    arms.add_argument("--seed", type=int, default=20260814)
    compare = add("compare", "target vs the model-generated runs")
    compare.add_argument("--runs", nargs="*", default=None)

    args = parser.parse_args(argv)
    return {
        "crosswalk": cmd_crosswalk,
        "fit": cmd_fit,
        "check": cmd_check,
        "arms": cmd_arms,
        "compare": cmd_compare,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
