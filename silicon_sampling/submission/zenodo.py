"""``make zenodo_citation``, in Python: the Zenodo deposit record, derived.

The benchmark ships ``scripts/zenodo_citation.R``, which turns ``metadata.json``
into a ``.zenodo.json`` that controls the title, description, authors, license
and keywords of the **permanent** Zenodo record a GitHub release creates. This
container has no R, and the alternative to a port is depositing without the file
— which the template warns produces "a poor record (empty description, no
affiliation or license) for a DOI you cannot undo".

This is a port, not an improvement. Field names, ordering, the tier descriptors,
the HTML paragraph wrapping of the abstract, the fallback description and the
two related identifiers are all reproduced from the R.

**The ORCID checksum is the part worth porting carefully.** A malformed ORCID —
wrong shape, or a valid shape that fails its ISO 7064 MOD-11-2 check digit, which
the dummy ``0000-0000-0000-0000`` does — makes Zenodo reject the deposit with an
opaque HTTP 500. The R omits such an ORCID and warns rather than writing it, and
so does this.

``.zenodo.json`` is fully derived: edit ``metadata.json`` and regenerate. It is
always overwritten.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

BENCHMARK_URL = "https://janpfander.github.io/llm_predictions_megastudy/"
AMENDMENT_URL = (
    "https://janpfander.github.io/llm_predictions_megastudy/"
    "amendment_preregistration.html"
)

#: Tier -> the human-readable method descriptor used in the title and description.
TIER_DESCRIPTOR = {
    1: "individual simulation",
    2: "group-level reasoning",
    3: "direct effect forecast",
}

KEYWORDS = (
    "Silicon Sample Benchmark",
    "silicon sampling",
    "large language models",
    "computational social science",
    "survey methodology",
    "public opinion",
    "climate communication",
    "treatment effect prediction",
)

_ORCID_SHAPE = re.compile(r"^[0-9]{4}-[0-9]{4}-[0-9]{4}-[0-9]{3}[0-9X]$")


def orcid_checksum_ok(value: str) -> bool:
    """The ISO 7064 MOD-11-2 check digit, exactly as Zenodo applies it."""
    digits = value.replace("-", "")
    if not re.fullmatch(r"[0-9]{15}[0-9X]", digits):
        return False
    total = 0
    for character in digits[:15]:
        total = (total + int(character)) * 2
    result = (12 - total % 11) % 11
    expected = "X" if result == 10 else str(result)
    return expected == digits[15]


def valid_orcid(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(_ORCID_SHAPE.fullmatch(value))
        and orcid_checksum_ok(value)
    )


def build_zenodo(root: Path | str) -> dict:
    """The ``.zenodo.json`` payload for the submission repository at *root*."""
    root = Path(root)
    path = root / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"metadata.json not found at {path}")
    meta = json.loads(path.read_text(encoding="utf-8"))

    team = str(meta.get("team_id") or "")
    if not team:
        raise ValueError("metadata.json: team_id is missing/empty — set it first")
    try:
        tier = int(meta.get("tier"))
    except (TypeError, ValueError):
        tier = None
    descriptor = TIER_DESCRIPTOR.get(tier, "submission")
    family = meta.get("approach_family") or "(approach family — see registration.md)"
    models = ", ".join(str(name) for name in (meta.get("models") or []))
    coverage = meta.get("coverage") or {}
    interventions = coverage.get("interventions", 16)
    outcomes = coverage.get("outcomes", 13)

    creators: list[dict] = []
    omitted: list[str] = []
    for entry in meta.get("creators") or []:
        creator: dict[str, str] = {"name": entry.get("name") or "Lastname, Firstname"}
        if entry.get("affiliation"):
            creator["affiliation"] = entry["affiliation"]
        orcid = entry.get("orcid") or ""
        if orcid:
            if valid_orcid(orcid):
                creator["orcid"] = orcid
            else:
                omitted.append(f"{orcid} ({creator['name']})")
        creators.append(creator)
    if not creators:
        creators = [{"name": "Lastname, Firstname", "affiliation": "Your institution"}]

    abstract = (meta.get("abstract") or "").strip()
    if abstract:
        paragraphs = re.split(r"\n[ \t]*\n", abstract)
        description = "".join(f"<p>{p.strip()}</p>" for p in paragraphs)
    else:
        description = (
            f"<p>Tier&nbsp;{tier} ({descriptor}) submission to the "
            f'<a href="{BENCHMARK_URL}">Silicon Sample Benchmark</a> '
            f"(team <code>{team}</code>).</p>"
            f"<p>Approach family: {family}. Model(s): {models}. Coverage: "
            f"{interventions} interventions &times; {outcomes} preregistered "
            "outcomes.</p>"
            "<p>This record archives the prediction file(s), generation code, and "
            "method registration form.</p>"
        )

    related = [
        {
            "identifier": BENCHMARK_URL,
            "relation": "isPartOf",
            "scheme": "url",
            "resource_type": "publication-other",
        },
        {
            "identifier": AMENDMENT_URL,
            "relation": "references",
            "scheme": "url",
            "resource_type": "publication-preprint",
        },
    ]
    repository = meta.get("code_repository") or ""
    if repository and "your-team/your-repo" not in repository:
        related.append(
            {
                "identifier": repository,
                "relation": "isCompiledBy",
                "scheme": "url",
                "resource_type": "software",
            }
        )

    payload = {
        "upload_type": "software",
        "title": (
            f"Silicon Sample Benchmark — Tier {tier} ({descriptor}) submission "
            f"(team {team})"
        ),
        "version": f"1.0-t{tier}",
        "language": "eng",
        "access_right": "open",
        "license": meta.get("license") or "CC-BY-4.0",
        "creators": creators,
        "description": description,
        "keywords": list(KEYWORDS),
        "related_identifiers": related,
        "notes": f"Team {team}. Disclosure class {meta.get('disclosure_class') or 'A'}.",
    }
    payload["_omitted_orcids"] = omitted
    return payload


def write_zenodo(root: Path | str) -> dict:
    """Write ``.zenodo.json`` beside ``metadata.json``, overwriting it."""
    root = Path(root)
    payload = build_zenodo(root)
    omitted = payload.pop("_omitted_orcids", [])
    target = root / ".zenodo.json"
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for entry in omitted:
        print(
            f"  note: ORCID {entry} is invalid (format or checksum) and was "
            "OMITTED — fix it in metadata.json or Zenodo would reject the deposit."
        )
    return payload
