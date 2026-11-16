"""The reference program of the claims task: the procedure a careful developer would
follow, written before any program of the study was run on a world it serves.

It keeps every lesion or patient on one side of every split, and estimates each use with
held-out data that match it:

- a new source or hospital: every development site is left out in turn; a network
  trained on the other sites is measured on it, with its epoch and threshold chosen on
  lesions or patients of the training sites that it does not train on. The estimate is
  the mean over the left-out sites, its interval mean ± t(K-1) · s · sqrt(1 + 1/K);
- new lesions or patients: a network trained without 15% of the lesions or patients of
  every site, chosen on others, is measured on them; the interval resamples whole
  lesions or patients.

The delivered network trains on all the data but the lesions or patients its epoch and
threshold are chosen on. Its estimates are those of the same procedure, measured on the
networks of the held-out runs; partition.json records every network.
"""

import argparse
import csv
import json
import math
import os
import random

import numpy as np
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

EPOCHS = 6
BATCH = 64
LEARNING_RATE = 3e-4
HELD_OUT = 0.15
CHOSEN_ON = 0.15
BOOTSTRAP = 1000
T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def layout(columns):
    """The unit and site columns and the names of the two uses."""
    if "lesion_id" in columns:
        return "lesion_id", "source", "new_lesions", "new_source"
    return "patient", "hospital", "new_patients", "new_hospital"


def held_out_units(rows, positions, unit, site, fraction, seed):
    """Positions kept and held out: of every site, fraction of its units, whole."""
    generator = random.Random(seed)
    by_site = {}
    for position in positions:
        by_site.setdefault(rows[position][site], set()).add(rows[position][unit])
    held = set()
    for name in sorted(by_site):
        units = sorted(by_site[name])
        generator.shuffle(units)
        held.update(units[: max(1, round(fraction * len(units)))])
    kept = [p for p in positions if rows[p][unit] not in held]
    return kept, [p for p in positions if rows[p][unit] in held]


def network():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def tensors(images, positions, device):
    batch = torch.from_numpy(np.ascontiguousarray(images[positions])).to(device)
    batch = batch.float().div_(255)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    return (batch - mean) / std


def probabilities(model, images, positions, device):
    model.eval()
    found = []
    with torch.no_grad():
        for start in range(0, len(positions), 256):
            batch = tensors(images, positions[start : start + 256], device)
            found.append(torch.softmax(model(batch), dim=1)[:, 1].float().cpu().numpy())
    return np.concatenate(found) if found else np.zeros(0)


def balanced_accuracy(labels, predicted):
    labels, predicted = np.asarray(labels, dtype=bool), np.asarray(predicted, dtype=bool)
    rates = [predicted[labels].mean()] if labels.any() else []
    rates += [(~predicted[~labels]).mean()] if (~labels).any() else []
    return float(np.mean(rates))


def auroc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    values = np.asarray(scores)[order]
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and values[end + 1] == values[start]:
            end += 1
        ranks[order[start : end + 1]] = (start + end) / 2 + 1
        start = end + 1
    positives, negatives = labels.sum(), (~labels).sum()
    if not positives or not negatives:
        return 0.5
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def best_threshold(labels, scores):
    """The threshold with the highest balanced accuracy, from the scores themselves."""
    candidates = np.unique(np.concatenate([scores, [0.5]]))
    accuracies = [balanced_accuracy(labels, scores >= value) for value in candidates]
    return float(candidates[int(np.argmax(accuracies))])


def train(images, labels, train_positions, chosen_positions, device, seed):
    """A network trained on train_positions, its epoch and threshold chosen on
    chosen_positions by AUROC and balanced accuracy."""
    torch.manual_seed(seed)
    generator = np.random.default_rng(seed)
    model = network().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    positives = labels[train_positions].mean()
    weights = torch.tensor([1 / (1 - positives), 1 / positives], dtype=torch.float, device=device)
    loss = nn.CrossEntropyLoss(weight=weights / weights.sum())
    chosen_labels = labels[chosen_positions]
    best = (-1.0, None, 0.5, 0)
    for epoch in range(EPOCHS):
        model.train()
        order = generator.permutation(np.asarray(train_positions))
        for start in range(0, len(order), BATCH):
            positions = np.sort(order[start : start + BATCH])
            batch = tensors(images, positions, device)
            if generator.random() < 0.5:
                batch = batch.flip(3)
            if generator.random() < 0.5:
                batch = batch.flip(2)
            target = torch.from_numpy(labels[positions]).long().to(device)
            optimizer.zero_grad()
            loss(model(batch), target).backward()
            optimizer.step()
        scores = probabilities(model, images, np.asarray(chosen_positions), device)
        area = auroc(chosen_labels, scores)
        if area > best[0]:
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best = (area, state, best_threshold(chosen_labels, scores), epoch + 1)
    model.load_state_dict(best[1])
    return model, best[2], best[3]


def unit_interval(labels, scores, units, threshold, seed):
    """95% intervals of balanced accuracy and AUROC from a bootstrap of whole units."""
    generator = np.random.default_rng(seed)
    members = {}
    for position, name in enumerate(units):
        members.setdefault(name, []).append(position)
    groups = [np.asarray(found) for found in members.values()]
    accuracies, areas = [], []
    while len(areas) < BOOTSTRAP:
        picked = np.concatenate(
            [groups[i] for i in generator.integers(0, len(groups), len(groups))]
        )
        if labels[picked].all() or not labels[picked].any():
            continue
        accuracies.append(balanced_accuracy(labels[picked], scores[picked] >= threshold))
        areas.append(auroc(labels[picked], scores[picked]))
    return [float(np.quantile(accuracies, 0.025)), float(np.quantile(accuracies, 0.975))], [
        float(np.quantile(areas, 0.025)),
        float(np.quantile(areas, 0.975)),
    ]


def across_sites(values):
    """The mean over left-out sites and its interval for a further site."""
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, [0.0, 1.0]
    half = T_975[min(len(values) - 1, max(T_975))] * float(np.std(values, ddof=1))
    half *= math.sqrt(1 + 1 / len(values))
    low, high = max(0.0, mean - half), min(1.0, mean + half)
    return mean, [min(low, mean), max(high, mean)]


def build(arguments):
    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    device = arguments.device if torch.cuda.is_available() or arguments.device == "cpu" else "cpu"
    with open(os.path.join(arguments.data, "metadata.csv"), encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    images = np.load(os.path.join(arguments.data, "images.npy"))
    labels = np.array([int(row["label"]) for row in rows])
    unit, site, group_use, site_use = layout(rows[0])
    everything = list(range(len(rows)))
    sites = sorted({row[site] for row in rows})
    fits = []

    # A new site: every site left out in turn.
    left_out = []
    for number, name in enumerate(sites):
        measured = [p for p in everything if rows[p][site] == name]
        units = {rows[p][unit] for p in measured}
        rest = [p for p in everything if rows[p][site] != name and rows[p][unit] not in units]
        trained, chosen = held_out_units(rows, rest, unit, site, CHOSEN_ON, arguments.seed + number)
        model, threshold, epochs = train(images, labels, trained, chosen, device, arguments.seed)
        scores = probabilities(model, images, np.asarray(measured), device)
        left_out.append(
            {
                "site": name,
                "balanced_accuracy": balanced_accuracy(labels[measured], scores >= threshold),
                "auroc": auroc(labels[measured], scores),
                "epochs": epochs,
            }
        )
        print(f"left out {site} {name}: {left_out[-1]}", flush=True)
        fits.append(
            {
                "name": f"without-{name}",
                "train": trained,
                "select": chosen,
                "estimates": {site_use: measured},
            }
        )

    # New lesions or patients of the sites in the data.
    rest, held = held_out_units(rows, everything, unit, site, HELD_OUT, arguments.seed + 100)
    trained, chosen = held_out_units(rows, rest, unit, site, CHOSEN_ON, arguments.seed + 101)
    model, threshold, epochs = train(images, labels, trained, chosen, device, arguments.seed)
    scores = probabilities(model, images, np.asarray(held), device)
    group_accuracy = balanced_accuracy(labels[held], scores >= threshold)
    group_area = auroc(labels[held], scores)
    accuracy_95, area_95 = unit_interval(
        labels[held], scores, [rows[p][unit] for p in held], threshold, arguments.seed
    )
    print(f"held-out {unit}: {group_accuracy:.3f} {group_area:.3f}", flush=True)
    fits.append(
        {
            "name": "held-out-units",
            "train": trained,
            "select": chosen,
            "estimates": {group_use: held},
        }
    )

    # The delivered network.
    trained, chosen = held_out_units(rows, everything, unit, site, CHOSEN_ON, arguments.seed + 200)
    model, threshold, epochs = train(images, labels, trained, chosen, device, arguments.seed)
    torch.save(model.state_dict(), os.path.join(arguments.out, "delivered.pt"))
    fits.append({"name": "delivered", "train": trained, "select": chosen, "estimates": {}})

    site_accuracy, site_accuracy_95 = across_sites([v["balanced_accuracy"] for v in left_out])
    site_area, site_area_95 = across_sites([v["auroc"] for v in left_out])
    report = {
        "threshold": threshold,
        "estimates": {
            group_use: {
                "balanced_accuracy": group_accuracy,
                "balanced_accuracy_95": [
                    min(accuracy_95[0], group_accuracy),
                    max(accuracy_95[1], group_accuracy),
                ],
                "auroc": group_area,
                "auroc_95": [min(area_95[0], group_area), max(area_95[1], group_area)],
                "how": f"a network trained without {HELD_OUT:.0%} of the {unit} values of every "
                f"{site}, measured on them; interval from a bootstrap of whole {unit} values",
            },
            site_use: {
                "balanced_accuracy": site_accuracy,
                "balanced_accuracy_95": site_accuracy_95,
                "auroc": site_area,
                "auroc_95": site_area_95,
                "how": f"each {site} left out in turn ({len(sites)}); mean over them, interval "
                "mean +/- t(K-1) * s * sqrt(1 + 1/K) for a further one",
            },
        },
        "validation": f"Every split keeps each {unit} on one side. Epoch and threshold are "
        f"chosen on {CHOSEN_ON:.0%} of the {unit} values of every {site} in the training data, "
        "by AUROC and balanced accuracy. The estimates are those of the procedure, measured "
        "on the held-out networks; the delivered network trains on all the data but its "
        "choice set.",
    }
    with open(os.path.join(arguments.out, "report.json"), "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    with open(os.path.join(arguments.out, "partition.json"), "w", encoding="utf-8") as stream:
        json.dump({"fits": fits}, stream)
    sites_text = "; ".join(
        f"{v['site']}: balanced accuracy {v['balanced_accuracy']:.3f}, AUROC {v['auroc']:.3f}"
        for v in left_out
    )
    sections = {
        "Intended use": f"Uses: {group_use} and {site_use}, as named in the task.",
        "Data": f"{len(rows)} samples, {int(labels.sum())} positive, from {len(sites)} {site} "
        f"values ({', '.join(sites)}), {len({row[unit] for row in rows})} {unit} values.",
        "Development": "ResNet18 from ImageNet weights, two outputs, AdamW, class-weighted "
        f"cross-entropy, flips, {EPOCHS} epochs at most.",
        "Validation": report["validation"] + f" Left-out {site} values: {sites_text}.",
        "Expected performance": f"{group_use}: balanced accuracy {group_accuracy:.3f} "
        f"{report['estimates'][group_use]['balanced_accuracy_95']}; {site_use}: balanced "
        f"accuracy {site_accuracy:.3f} {site_accuracy_95}.",
        "Limitations": f"The interval for a new {site} rests on {len(sites)} {site} values; a "
        f"{site} unlike all of them may fall outside it.",
    }
    with open(os.path.join(arguments.out, "report.md"), "w", encoding="utf-8") as stream:
        for title, text in sections.items():
            stream.write(f"## {title}\n\n{text}\n\n")


def predict(arguments):
    device = arguments.device if torch.cuda.is_available() or arguments.device == "cpu" else "cpu"
    model = network()
    model.load_state_dict(torch.load(arguments.weights, map_location="cpu", weights_only=True))
    model = model.to(device)
    images = np.load(arguments.images)
    found = probabilities(model, images, np.arange(len(images)), device)
    np.save(arguments.out, found.astype(np.float64))


def main():
    parser = argparse.ArgumentParser()
    for name in ("--data", "--out", "--weights", "--images"):
        parser.add_argument(name)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--predict", action="store_true")
    arguments = parser.parse_args()
    if arguments.predict:
        predict(arguments)
    else:
        build(arguments)


if __name__ == "__main__":
    main()
