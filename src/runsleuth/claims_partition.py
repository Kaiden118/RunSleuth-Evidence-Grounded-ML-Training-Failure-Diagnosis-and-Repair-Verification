"""A program's record of what each network it trained used, and the errors the record shows.

The claims task asks a program to write partition.json beside its report: for every
network it trains (a fit), the rows of metadata.csv the network was trained on, the rows
any choice about it was made on, and the rows it was measured on for the estimate of each
use; the delivered network's fit is named delivered. read checks the record's form.

errors reads, in a well-formed record, for every estimate a report gives, the code labels
of the claims study that the record alone decides, and from them the estimate's warrant
as far as the record shows it. Rows are compared by the group the data name (a lesion or
a patient) and by site:

- group_leakage: a measuring network trained on a group the estimate is measured on;
- new_site_from_known_sites: a new-site estimate measured on a site the network trained on;
- selection_on_estimate_data: a choice about the measuring network made on a group the
  estimate is measured on, or, for a new-site estimate, on its site;
- estimate_not_of_the_delivered_network: a note, not an error;
- hidden_duplicates: a note, not an error: rows the estimate shares with training or
  choices only through lesions whose images are duplicates under other identifiers, which
  the sealing units know and the given data do not show.

The warrant is supported when no measuring network shows an error, not_supported when one
does, and unknown when no record measures the estimate. provenance gives the same facts
without a verdict and without the hidden duplicates, as the table an auditor may read.

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
            found_rows = _rows(rows, count)
            if found_rows is None:
                problems.append(f"fit {name}: the rows of {use} are not a list of rows")
            elif found_rows:
                # An empty list says the fit measured nothing for the use, as if left out.
                measured[use] = found_rows
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


def _keys(
    table: Sequence[Mapping[str, str]],
    chosen: claims_task.Spec,
    units: Sequence[str] | None,
) -> tuple[dict[int, str], dict[int, str], dict[int, str], dict[int, str]]:
    """The group, the sealing unit, the site and the label of every row of the given table.
    Without units a row's unit is its group."""
    rows = [int(line["row"]) for line in table]
    if units is not None and len(units) != len(rows):
        raise ValueError(f"{len(units)} units for {len(rows)} rows")
    group = {row: str(line[chosen.group]) for row, line in zip(rows, table, strict=True)}
    unit = (
        dict(group)
        if units is None
        else {row: str(found) for row, found in zip(rows, units, strict=True)}
    )
    site = {row: str(line[chosen.site]) for row, line in zip(rows, table, strict=True)}
    label = {row: str(line["label"]) for row, line in zip(rows, table, strict=True)}
    return group, unit, site, label


def _shared(rows: frozenset[int], others: frozenset[int], key: Mapping[int, str]) -> set[str]:
    return {key[row] for row in rows} & {key[row] for row in others}


def errors(
    fits: Sequence[Fit],
    table: Sequence[Mapping[str, str]],
    chosen: claims_task.Spec,
    estimated: Sequence[str],
    units: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """For every use with an estimate, the errors the record shows in how it was measured,
    and its warrant as far as the record shows it.

    table is the given metadata, by row; units, when given, is the sealing unit of each of
    its rows, in order, and serves the hidden_duplicates note only.
    """
    group, unit, site, _label = _keys(table, chosen, units)
    parts = {claim.name: claim.part for claim in chosen.claims}
    found: dict[str, dict[str, Any]] = {}
    for use in estimated:
        measuring = [fit for fit in fits if use in fit.estimates]
        if not measuring:
            found[use] = {"recorded": False, "warrant": "unknown"}
            continue
        new_site = parts[use] == "new_site"
        leaked, known, selected, hidden = [], [], [], []
        for fit in measuring:
            rows = fit.estimates[use]
            shared = _shared(rows, fit.train, group)
            if shared:
                leaked.append(
                    {
                        "fit": fit.name,
                        "groups": len(shared),
                        "rows": sum(group[row] in shared for row in rows),
                        "of": len(rows),
                    }
                )
            both = _shared(rows, fit.train, site)
            if new_site and both:
                known.append({"fit": fit.name, "sites": sorted(both)})
            chosen_groups = _shared(rows, fit.select, group)
            chosen_sites = _shared(rows, fit.select, site) if new_site else set()
            if chosen_groups or chosen_sites:
                selected.append(
                    {
                        "fit": fit.name,
                        "rows": sum(group[row] in chosen_groups for row in rows),
                        "of": len(rows),
                        "sites": sorted(chosen_sites),
                    }
                )
            trained_units = _shared(rows, fit.train, unit)
            chosen_units = _shared(rows, fit.select, unit)
            only_trained = sum(
                unit[row] in trained_units and group[row] not in shared for row in rows
            )
            only_chosen = sum(
                unit[row] in chosen_units and group[row] not in chosen_groups for row in rows
            )
            if only_trained or only_chosen:
                hidden.append(
                    {
                        "fit": fit.name,
                        "rows_trained": only_trained,
                        "rows_chosen": only_chosen,
                        "of": len(rows),
                    }
                )
        found[use] = {
            "recorded": True,
            "fits": [fit.name for fit in measuring],
            "group_leakage": leaked,
            "new_site_from_known_sites": known if new_site else None,
            "selection_on_estimate_data": selected,
            "estimate_not_of_the_delivered_network": all(
                fit.name != DELIVERED_FIT for fit in measuring
            ),
            "hidden_duplicates": hidden,
            "warrant": "not_supported" if leaked or known or selected else "supported",
        }
    return found


def _count(rows: frozenset[int], site: Mapping[int, str], label: Mapping[int, str]) -> dict:
    counted: dict[str, dict[str, int]] = {}
    for row in sorted(rows):
        by_label = counted.setdefault(site[row], {"0": 0, "1": 0})
        by_label[label[row]] += 1
    return dict(sorted(counted.items()))


def provenance(
    fits: Sequence[Fit],
    table: Sequence[Mapping[str, str]],
    chosen: claims_task.Spec,
    estimated: Sequence[str],
) -> dict[str, Any]:
    """What the record says each estimate was measured on, as facts without a verdict:
    the rows of every role of every fit, counted by site and label, and for every use with
    an estimate the fits that measure it, the sites of its rows, and how many of its
    groups and sites the measuring fits also trained or chose on."""
    group, _unit, site, label = _keys(table, chosen, None)
    facts: dict[str, Any] = {
        "site_column": chosen.site,
        "group_column": chosen.group,
        "given": _count(frozenset(group), site, label),
        "fits": [
            {
                "name": fit.name,
                "delivered": fit.name == DELIVERED_FIT,
                "train": _count(fit.train, site, label),
                "select": _count(fit.select, site, label),
                "estimates": {
                    use: _count(rows, site, label) for use, rows in sorted(fit.estimates.items())
                },
            }
            for fit in fits
        ],
        "uses": {},
    }
    for use in estimated:
        entries = []
        for fit in fits:
            if use not in fit.estimates:
                continue
            rows = fit.estimates[use]
            entries.append(
                {
                    "fit": fit.name,
                    "rows": len(rows),
                    "groups": len({group[row] for row in rows}),
                    "sites": sorted({site[row] for row in rows}),
                    "groups_also_trained_on": len(_shared(rows, fit.train, group)),
                    "groups_also_chosen_on": len(_shared(rows, fit.select, group)),
                    "sites_also_trained_on": sorted(_shared(rows, fit.train, site)),
                    "sites_also_chosen_on": sorted(_shared(rows, fit.select, site)),
                }
            )
        facts["uses"][use] = {"measured_by": entries}
    return facts
