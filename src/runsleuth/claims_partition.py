"""A program's record of what each network it trained used, and the errors the record shows.

The claims task asks a program to write partition.json beside its report: for every
network it trains (a fit), the rows of metadata.csv the network was trained on, the rows
any choice about it was made on, and the rows it was measured on for the estimate of each
use; the delivered network's fit is named delivered. read checks the record's form.
errors reads, in a well-formed record, for every estimate a report gives, the code labels
of the claims study that the record alone decides: group_leakage,
new_site_from_known_sites and selection_on_estimate_data, with the note
estimate_not_of_the_delivered_network.

The record is the program's own account of itself. Whether the program does what it
records is for the readers of its source; a record that contradicts the source is itself
a finding.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from runsleuth import claims_task

DELIVERED_FIT = "delivered"
ROLES = ("train", "select")


@dataclass(frozen=True, slots=True)
class Fit:
    """One network the program trained: the rows of each role, and of each use's estimate."""

    name: str
    train: frozenset[int]
    select: frozenset[int]
    estimates: Mapping[str, frozenset[int]]


def _rows(value: Any, count: int) -> frozenset[int] | None:
    """The rows a list names, or None unless it is a list of rows of the table."""
    if not isinstance(value, list):
        return None
    for row in value:
        if isinstance(row, bool) or not isinstance(row, int) or not 0 <= row < count:
            return None
    return frozenset(value)


def _fit(found: Any, number: int, uses: Sequence[str], count: int) -> tuple[Fit | None, list]:
    if not isinstance(found, dict) or not isinstance(found.get("name"), str) or not found["name"]:
        return None, [f"fit {number} has no name"]
    name, problems = found["name"], []
    roles = {}
    for role in ROLES:
        roles[role] = _rows(found.get(role), count)
        if roles[role] is None:
            problems.append(f"fit {name}: {role} is not a list of rows of the table")
    estimates = found.get("estimates")
    measured: dict[str, frozenset[int]] = {}
    if not isinstance(estimates, dict):
        problems.append(f"fit {name}: estimates is not an object")
    else:
        for use, rows in estimates.items():
            if use not in uses:
                problems.append(f"fit {name}: {use!r} is not a use")
                continue
            measured[use] = _rows(rows, count)
            if not measured[use]:
                problems.append(f"fit {name}: the rows of {use} are not a list of rows")
    if problems:
        return None, problems
    return Fit(name, roles["train"], roles["select"], measured), []


def read(path: Path, uses: Sequence[str], count: int) -> tuple[list[Fit] | None, list[str]]:
    """The fits of a partition record of a table of count rows, and what is wrong with it.

    The fits come back only when nothing is: every fit named once, its rows rows of the
    table, its estimates those of the uses, and one of them the delivered network's.
    """
    path = Path(path)
    if not path.is_file():
        return None, [f"no {claims_task.PARTITION}"]
    try:
        found = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError) as error:
        return None, [f"{claims_task.PARTITION} is not JSON: {str(error)[:200]}"]
    listed = found.get("fits") if isinstance(found, dict) else None
    if not isinstance(listed, list) or not listed:
        return None, ["fits is not a list of fits"]
    fits, problems = [], []
    for number, item in enumerate(listed):
        fit, wrong = _fit(item, number, uses, count)
        problems += wrong
        if fit is not None:
            fits.append(fit)
    names = [fit.name for fit in fits]
    problems += [
        f"two fits are named {name}" for name in sorted(set(names)) if names.count(name) > 1
    ]
    if not problems and DELIVERED_FIT not in names:
        problems.append(f"no fit is named {DELIVERED_FIT}")
    return (None if problems else fits), problems


def errors(
    fits: Sequence[Fit],
    table: Sequence[Mapping[str, str]],
    chosen: claims_task.Spec,
    estimated: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """For every use with an estimate, the errors the record shows in how it was measured.

    table is the given metadata, by row. An estimate measured on rows whose group (lesion or
    slide) a network trained on leaks; one for a new site measured on sites the network
    trained on is measured on known sites; one measured on rows that a choice about the
    measured network was made on is selected on its estimate data. An estimate no record
    measures is not recorded.
    """
    group = {int(line["row"]): line[chosen.group] for line in table}
    site = {int(line["row"]): line[chosen.site] for line in table}
    parts = {claim.name: claim.part for claim in chosen.claims}
    found: dict[str, dict[str, Any]] = {}
    for use in estimated:
        measuring = [fit for fit in fits if use in fit.estimates]
        if not measuring:
            found[use] = {"recorded": False}
            continue
        leaked, known, selected = [], [], []
        for fit in measuring:
            rows = fit.estimates[use]
            shared = {group[row] for row in rows} & {group[row] for row in fit.train}
            if shared:
                leaked.append(
                    {
                        "fit": fit.name,
                        "groups": len(shared),
                        "rows": sum(group[row] in shared for row in rows),
                        "of": len(rows),
                    }
                )
            both = {site[row] for row in rows} & {site[row] for row in fit.train}
            if parts[use] == "new_site" and both:
                known.append({"fit": fit.name, "sites": sorted(both, key=str)})
            if rows & fit.select:
                selected.append({"fit": fit.name, "rows": len(rows & fit.select), "of": len(rows)})
        found[use] = {
            "recorded": True,
            "fits": [fit.name for fit in measuring],
            "group_leakage": leaked,
            "new_site_from_known_sites": known if parts[use] == "new_site" else None,
            "selection_on_estimate_data": selected,
            "estimate_not_of_the_delivered_network": all(
                fit.name != DELIVERED_FIT for fit in measuring
            ),
        }
    return found
