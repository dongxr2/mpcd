"""Minimal four-category illustration of the MPCD workflow.

The synthetic example represents a mirrored interface with four action regions.
It is intentionally independent of the tennis application and writes compact,
machine-readable summaries for the manuscript supplement.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import t


OUT = Path("method_outputs")
OUT.mkdir(exist_ok=True)
SEED = 20260627
CATEGORIES = ["upper_left", "upper_right", "lower_left", "lower_right"]
MIRROR_INDEX = np.array([1, 0, 3, 2])
SIGNATURE_INDEX = np.array([1, 3])


def simulate_group(
    rng: np.random.Generator, group: str, n_actors: int, center: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    counts = []
    successes = []
    action_effect = np.array([0.15, -0.10, 0.05, -0.12])
    for _ in range(n_actors):
        quality = rng.normal(0.0, 0.35)
        composition = rng.dirichlet(60 * center)
        n_events = int(np.clip(np.rint(np.exp(4.95 + 0.35 * quality + rng.normal(0, 0.25))), 80, 300))
        actor_counts = rng.multinomial(n_events, composition)
        group_signature = 0.12 if group == "Focal" else 0.0
        probability = expit(0.45 + quality + action_effect + group_signature * np.isin(np.arange(4), SIGNATURE_INDEX))
        counts.append(actor_counts)
        successes.append(rng.binomial(actor_counts, probability))
    return np.asarray(counts, dtype=float), np.asarray(successes, dtype=float)


def metrics(focal: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mirrored = reference[..., MIRROR_INDEX]
    residual = focal - mirrored
    tv = 0.5 * np.abs(residual).sum(axis=-1)
    focal_clr = np.log(focal) - np.log(focal).mean(axis=-1, keepdims=True)
    reference_clr = np.log(mirrored) - np.log(mirrored).mean(axis=-1, keepdims=True)
    distance = np.sqrt(((focal_clr - reference_clr) ** 2).sum(axis=-1))
    excess = focal[..., SIGNATURE_INDEX].sum(axis=-1) - mirrored[..., SIGNATURE_INDEX].sum(axis=-1)
    return tv, distance, excess


def compositions(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    event = counts.sum(axis=0) / counts.sum()
    actor = (counts / counts.sum(axis=1, keepdims=True)).mean(axis=0)
    return event, actor


def bootstrap_metrics(
    rng: np.random.Generator, focal: np.ndarray, reference: np.ndarray, repetitions: int = 5000
) -> dict[str, dict[str, tuple[float, float]]]:
    nf, nr = len(focal), len(reference)
    focal_share = focal / focal.sum(axis=1, keepdims=True)
    reference_share = reference / reference.sum(axis=1, keepdims=True)
    draws = {"Event average": [[], [], []], "Actor average": [[], [], []]}
    for start in range(0, repetitions, 250):
        size = min(250, repetitions - start)
        wf = rng.multinomial(nf, np.full(nf, 1 / nf), size=size)
        wr = rng.multinomial(nr, np.full(nr, 1 / nr), size=size)
        focal_event_counts = wf @ focal
        reference_event_counts = wr @ reference
        event_focal = focal_event_counts / focal_event_counts.sum(axis=1, keepdims=True)
        event_reference = reference_event_counts / reference_event_counts.sum(axis=1, keepdims=True)
        actor_focal = wf @ focal_share / nf
        actor_reference = wr @ reference_share / nr
        for estimand, pair in (
            ("Event average", (event_focal, event_reference)),
            ("Actor average", (actor_focal, actor_reference)),
        ):
            values = metrics(*pair)
            for index, value in enumerate(values):
                draws[estimand][index].append(value)
    result = {}
    for estimand, pieces in draws.items():
        result[estimand] = {}
        for name, values in zip(("total_variation", "aitchison_distance", "signature_excess"), pieces):
            combined = np.concatenate(values)
            result[estimand][name] = tuple(np.quantile(combined, [0.025, 0.975]))
    return result


def actor_outcome_contrast(counts: np.ndarray, successes: np.ndarray) -> np.ndarray:
    signature_counts = counts[:, SIGNATURE_INDEX].sum(axis=1)
    complement_index = np.array([0, 2])
    complement_counts = counts[:, complement_index].sum(axis=1)
    eligible = (signature_counts > 0) & (complement_counts > 0)
    signature_rate = successes[:, SIGNATURE_INDEX].sum(axis=1) / signature_counts
    complement_rate = successes[:, complement_index].sum(axis=1) / complement_counts
    return (signature_rate - complement_rate)[eligible]


def event_outcome_contrast(counts: np.ndarray, successes: np.ndarray) -> float:
    """Signature minus comparator outcome rate for the pooled event stream."""
    complement_index = np.array([0, 2])
    signature_rate = successes[:, SIGNATURE_INDEX].sum() / counts[:, SIGNATURE_INDEX].sum()
    complement_rate = successes[:, complement_index].sum() / counts[:, complement_index].sum()
    return float(signature_rate - complement_rate)


def bootstrap_event_outcome_difference(
    rng: np.random.Generator,
    focal_counts: np.ndarray,
    focal_successes: np.ndarray,
    reference_counts: np.ndarray,
    reference_successes: np.ndarray,
    repetitions: int = 5000,
) -> tuple[float, float]:
    """Actor-bootstrap interval for the pooled event-average outcome contrast."""
    complement_index = np.array([0, 2])
    nf, nr = len(focal_counts), len(reference_counts)
    draws = []
    for start in range(0, repetitions, 250):
        size = min(250, repetitions - start)
        wf = rng.multinomial(nf, np.full(nf, 1 / nf), size=size)
        wr = rng.multinomial(nr, np.full(nr, 1 / nr), size=size)
        focal_signature = (wf @ focal_successes[:, SIGNATURE_INDEX].sum(axis=1)) / (
            wf @ focal_counts[:, SIGNATURE_INDEX].sum(axis=1)
        )
        focal_comparator = (wf @ focal_successes[:, complement_index].sum(axis=1)) / (
            wf @ focal_counts[:, complement_index].sum(axis=1)
        )
        reference_signature = (wr @ reference_successes[:, SIGNATURE_INDEX].sum(axis=1)) / (
            wr @ reference_counts[:, SIGNATURE_INDEX].sum(axis=1)
        )
        reference_comparator = (wr @ reference_successes[:, complement_index].sum(axis=1)) / (
            wr @ reference_counts[:, complement_index].sum(axis=1)
        )
        draws.append((focal_signature - focal_comparator) - (reference_signature - reference_comparator))
    return tuple(np.quantile(np.concatenate(draws), [0.025, 0.975]))


def main() -> None:
    rng = np.random.default_rng(SEED)
    reference_counts, reference_successes = simulate_group(
        rng, "Reference", 160, np.array([0.36, 0.20, 0.26, 0.18])
    )
    focal_counts, focal_successes = simulate_group(
        rng, "Focal", 160, np.array([0.14, 0.40, 0.16, 0.30])
    )
    intervals = bootstrap_metrics(rng, focal_counts, reference_counts)
    summary_rows = []
    actor_components = None
    for estimand, focal, reference in (
        ("Event average", *[x[0] for x in (compositions(focal_counts), compositions(reference_counts))]),
        ("Actor average", *[x[1] for x in (compositions(focal_counts), compositions(reference_counts))]),
    ):
        values = metrics(focal, reference)
        for name, point in zip(("total_variation", "aitchison_distance", "signature_excess"), values):
            lo, hi = intervals[estimand][name]
            summary_rows.append(
                {"estimand": estimand, "metric": name, "estimate": float(point), "ci95_lo": lo, "ci95_hi": hi}
            )
        if estimand == "Actor average":
            actor_components = pd.DataFrame(
                {
                    "category": CATEGORIES,
                    "focal_composition": focal,
                    "mirrored_reference": reference[MIRROR_INDEX],
                    "mirror_residual": focal - reference[MIRROR_INDEX],
                }
            )

    focal_contrast = actor_outcome_contrast(focal_counts, focal_successes)
    reference_contrast = actor_outcome_contrast(reference_counts, reference_successes)
    estimate = focal_contrast.mean() - reference_contrast.mean()
    se = np.sqrt(focal_contrast.var(ddof=1) / len(focal_contrast) + reference_contrast.var(ddof=1) / len(reference_contrast))
    numerator = (focal_contrast.var(ddof=1) / len(focal_contrast) + reference_contrast.var(ddof=1) / len(reference_contrast)) ** 2
    denominator = (focal_contrast.var(ddof=1) / len(focal_contrast)) ** 2 / (len(focal_contrast) - 1)
    denominator += (reference_contrast.var(ddof=1) / len(reference_contrast)) ** 2 / (len(reference_contrast) - 1)
    critical = t.ppf(0.975, numerator / denominator)
    summary_rows.append(
        {
            "estimand": "Actor average",
            "metric": "outcome_contrast_difference",
            "estimate": estimate,
            "ci95_lo": estimate - critical * se,
            "ci95_hi": estimate + critical * se,
        }
    )
    event_difference = event_outcome_contrast(focal_counts, focal_successes)
    event_difference -= event_outcome_contrast(reference_counts, reference_successes)
    event_lo, event_hi = bootstrap_event_outcome_difference(
        rng, focal_counts, focal_successes, reference_counts, reference_successes
    )
    summary_rows.append(
        {
            "estimand": "Event average",
            "metric": "outcome_contrast_difference",
            "estimate": event_difference,
            "ci95_lo": event_lo,
            "ci95_hi": event_hi,
        }
    )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "toy_example_summary.csv", index=False)
    actor_components.to_csv(OUT / "toy_example_category_table.csv", index=False)
    print(summary.to_string(index=False))
    print("\nActor-average category decomposition")
    print(actor_components.to_string(index=False))


if __name__ == "__main__":
    main()
