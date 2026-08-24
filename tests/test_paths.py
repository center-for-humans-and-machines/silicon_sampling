"""The data root is configurable, because on a cluster it is not beside the code.

Keeping gigabytes of sampler output inside a home-quota checkout exhausted the
quota on DAIS and every job submission began failing at ``mkdir``. These tests pin
the behaviour that fixes it, including the split layout that made it awkward: study
outputs on scratch, inputs that ship with the project still beside the code.
"""

from __future__ import annotations

from silicon_sampling import paths as P


def test_default_root_is_the_repository(monkeypatch):
    monkeypatch.delenv(P.DATA_ENV, raising=False)
    assert P.data_root() == P.REPO_DATA


def test_the_environment_variable_moves_the_root(tmp_path, monkeypatch):
    monkeypatch.setenv(P.DATA_ENV, str(tmp_path))
    assert P.data_root() == tmp_path
    assert P.output("pfander", "silicon_sampling") == (
        tmp_path / "pfander" / "silicon_sampling"
    )


def test_reads_fall_back_to_the_repository_but_writes_do_not(tmp_path, monkeypatch):
    """The split layout: outputs on the configured root, shipped inputs in the repo.

    On DAIS the study directories were moved to scratch while ``data/calibration``
    stayed in the checkout, so a read has to try both while a write must land in
    exactly one place.
    """
    monkeypatch.setenv(P.DATA_ENV, str(tmp_path))
    (tmp_path / "OnScratch").mkdir()

    assert P.resolve("OnScratch") == tmp_path / "OnScratch"
    # calibration/ ships with the checkout and is not on the configured root
    assert P.resolve("calibration") == P.REPO_DATA / "calibration"
    # a write never falls back, even for a path that only exists in the repo
    assert P.output("calibration") == tmp_path / "calibration"


def test_a_path_absent_from_both_roots_resolves_to_the_configured_one(
    tmp_path, monkeypatch
):
    """A fresh study has to resolve somewhere, and that somewhere is the new root."""
    monkeypatch.setenv(P.DATA_ENV, str(tmp_path))
    assert P.resolve("NoSuchStudy") == tmp_path / "NoSuchStudy"


def test_every_study_package_honours_the_configured_root(tmp_path, monkeypatch):
    import importlib

    monkeypatch.setenv(P.DATA_ENV, str(tmp_path))
    for study, folder in (
        ("pfander", "pfander"),
        ("voelkel", "Voelkel"),
        ("icpc", "ICPC"),
        ("goldwert", "Goldwert"),
    ):
        module = importlib.reload(
            importlib.import_module(f"silicon_sampling.{study}.paths")
        )
        assert module.RUNS == tmp_path / folder / "silicon_sampling", study
    # leave the modules pointing at the real tree for the rest of the suite
    monkeypatch.delenv(P.DATA_ENV, raising=False)
    for study in ("pfander", "voelkel", "icpc", "goldwert"):
        importlib.reload(importlib.import_module(f"silicon_sampling.{study}.paths"))


def test_describe_names_the_root_in_force(monkeypatch):
    monkeypatch.delenv(P.DATA_ENV, raising=False)
    assert "repository default" in P.describe()
    monkeypatch.setenv(P.DATA_ENV, "/tmp/elsewhere")
    assert "configured" in P.describe() and "/tmp/elsewhere" in P.describe()
