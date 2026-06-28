"""Expanded Monte Carlo evaluation for mirror-paired contrast decomposition.

The simulation targets an actor-average difference of within-actor risk
contrasts.  It compares event pooling, ordinary independence-GEE estimation
with an actor sandwich variance, cluster-size weighted GEE, and the MPCD
actor-paired estimator.  Data are generated as binomial action-cell counts,
which keeps the study fast while preserving the relevant clustering and
informative action-allocation mechanisms.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm, t


OUT = Path("method_outputs")
OUT.mkdir(exist_ok=True)
SEED = 20260627


@dataclass(frozen=True)
class Scenario:
    name: str
    label: str
    n_reference: int
    n_focal: int
    mean_events: float
    log_sd: float
    total_informative: float
    allocation_informative: float
    minimum_cell: int


SCENARIOS = [
    Scenario("balanced", "Balanced, non-informative", 150, 150, 180, 0.00, 0.00, 0.00, 5),
    Scenario("unequal", "Unequal, non-informative", 300, 75, 180, 0.90, 0.00, 0.00, 5),
    Scenario("total_ics", "Informative total cluster size", 300, 75, 180, 0.80, 0.75, 0.00, 5),
    Scenario("action_ics", "Informative action allocation", 300, 75, 180, 0.55, 0.00, 1.00, 5),
    Scenario("combined", "Combined informativeness", 618, 76, 180, 0.90, 0.75, 1.00, 5),
    Scenario("sparse", "Sparse combined setting", 180, 60, 45, 0.85, 0.75, 1.00, 5),
]


def _group_data(rng: np.random.Generator, scenario: Scenario, group: int, delta: float) -> pd.DataFrame:
    n_actor = scenario.n_focal if group else scenario.n_reference
    quality = rng.normal(0.0, 1.0, n_actor)

    # Focal and reference groups have different action preferences, as in the
    # application. Group-specific dependence on quality makes event weighting
    # differ between groups when allocation is informative.
    log_mean = np.log(scenario.mean_events)
    log_total = (
        log_mean
        + scenario.log_sd * rng.normal(size=n_actor)
        + scenario.total_informative * (0.65 + 0.35 * group) * quality
        + 0.20 * group
    )
    total = np.maximum(8, np.rint(np.exp(log_total - 0.5 * scenario.log_sd**2))).astype(int)
    total = np.minimum(total, 4000)

    allocation_logit = (
        -0.35
        + 1.05 * group
        + scenario.allocation_informative * (0.55 + 0.60 * group) * quality
    )
    action_probability = expit(allocation_logit)
    n_signature = rng.binomial(total, action_probability)
    n_comparator = total - n_signature

    # The actor-specific baseline depends strongly on quality. The risk
    # contrast is constant within group, so the actor-average target is exactly
    # delta even if eligibility depends on the observed action counts.
    p_comparator = expit(0.72 + 0.62 * quality)
    p_signature = np.clip(p_comparator + 0.02 + delta * group, 0.02, 0.98)
    y_signature = rng.binomial(n_signature, p_signature)
    y_comparator = rng.binomial(n_comparator, p_comparator)

    data = pd.DataFrame(
        {
            "group": group,
            "n_s": n_signature,
            "n_c": n_comparator,
            "y_s": y_signature,
            "y_c": y_comparator,
            "p_s": p_signature,
            "p_c": p_comparator,
        }
    )
    return data[(data.n_s >= scenario.minimum_cell) & (data.n_c >= scenario.minimum_cell)].copy()


def _weighted_group_gap(data: pd.DataFrame, weights: np.ndarray) -> tuple[float, float]:
    """Weighted action-cell contrast and actor-sandwich variance."""
    ws = weights * data.n_s.to_numpy()
    wc = weights * data.n_c.to_numpy()
    ys = data.y_s.to_numpy()
    yc = data.y_c.to_numpy()
    p_s = np.sum(weights * ys) / np.sum(ws)
    p_c = np.sum(weights * yc) / np.sum(wc)
    gap = p_s - p_c
    influence = weights * (ys - data.n_s.to_numpy() * p_s) / np.sum(ws)
    influence -= weights * (yc - data.n_c.to_numpy() * p_c) / np.sum(wc)
    n = len(data)
    variance = (n / (n - 1)) * np.sum(influence**2) if n > 1 else np.nan
    return gap, variance


def _normal_result(estimate: float, se: float, truth: float) -> dict[str, float]:
    lo = estimate - 1.96 * se
    hi = estimate + 1.96 * se
    p_value = 2 * norm.sf(abs(estimate / se)) if se > 0 else np.nan
    return {"estimate": estimate, "se": se, "lo": lo, "hi": hi, "p": p_value, "cover": lo <= truth <= hi}


def _actor_result(left: np.ndarray, right: np.ndarray, truth: float) -> dict[str, float]:
    estimate = left.mean() - right.mean()
    variance = left.var(ddof=1) / len(left) + right.var(ddof=1) / len(right)
    se = np.sqrt(variance)
    numerator = variance**2
    denominator = ((left.var(ddof=1) / len(left)) ** 2 / (len(left) - 1)) + (
        (right.var(ddof=1) / len(right)) ** 2 / (len(right) - 1)
    )
    df = numerator / denominator
    critical = t.ppf(0.975, df)
    lo, hi = estimate - critical * se, estimate + critical * se
    p_value = 2 * t.sf(abs(estimate / se), df)
    return {"estimate": estimate, "se": se, "lo": lo, "hi": hi, "p": p_value, "cover": lo <= truth <= hi}


def _bootstrap_result(
    rng: np.random.Generator, left: np.ndarray, right: np.ndarray, truth: float, repetitions: int
) -> dict[str, float]:
    estimate = left.mean() - right.mean()
    left_boot = rng.choice(left, size=(repetitions, len(left)), replace=True).mean(axis=1)
    right_boot = rng.choice(right, size=(repetitions, len(right)), replace=True).mean(axis=1)
    boot = left_boot - right_boot
    lo, hi = np.quantile(boot, [0.025, 0.975])
    se = boot.std(ddof=1)
    p_value = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return {"estimate": estimate, "se": se, "lo": lo, "hi": hi, "p": min(p_value, 1.0), "cover": lo <= truth <= hi}


def one_replication(
    rng: np.random.Generator, scenario: Scenario, delta: float, bootstrap_repetitions: int
) -> list[dict[str, float | str]]:
    reference = _group_data(rng, scenario, 0, delta)
    focal = _group_data(rng, scenario, 1, delta)
    if min(len(reference), len(focal)) < 8:
        return []

    results: list[dict[str, float | str]] = []

    # Event-pooled point estimate with the conventional independent-binomial SE.
    cells = {}
    variances = {}
    for group, data in ((0, reference), (1, focal)):
        for action in ("s", "c"):
            n = data[f"n_{action}"].sum()
            p_hat = data[f"y_{action}"].sum() / n
            cells[group, action] = p_hat
            variances[group, action] = p_hat * (1 - p_hat) / n
    estimate = (cells[1, "s"] - cells[1, "c"]) - (cells[0, "s"] - cells[0, "c"])
    result = _normal_result(estimate, np.sqrt(sum(variances.values())), delta)
    results.append({"method": "Event pooled", **result})

    # Independence GEE point estimate with actor-cluster sandwich inference.
    gap_f, var_f = _weighted_group_gap(focal, np.ones(len(focal)))
    gap_r, var_r = _weighted_group_gap(reference, np.ones(len(reference)))
    result = _normal_result(gap_f - gap_r, np.sqrt(var_f + var_r), delta)
    results.append({"method": "GEE actor sandwich", **result})

    # Cluster-size weighted GEE, using inverse total actor observations.
    wf = 1.0 / (focal.n_s.to_numpy() + focal.n_c.to_numpy())
    wr = 1.0 / (reference.n_s.to_numpy() + reference.n_c.to_numpy())
    gap_f, var_f = _weighted_group_gap(focal, wf)
    gap_r, var_r = _weighted_group_gap(reference, wr)
    result = _normal_result(gap_f - gap_r, np.sqrt(var_f + var_r), delta)
    results.append({"method": "Cluster-weighted GEE", **result})

    # MPCD gives one signature-minus-comparator contrast to each actor.
    d_f = focal.y_s.to_numpy() / focal.n_s.to_numpy() - focal.y_c.to_numpy() / focal.n_c.to_numpy()
    d_r = reference.y_s.to_numpy() / reference.n_s.to_numpy() - reference.y_c.to_numpy() / reference.n_c.to_numpy()
    result = _actor_result(d_f, d_r, delta)
    results.append({"method": "MPCD Welch", **result})
    result = _bootstrap_result(rng, d_f, d_r, delta, bootstrap_repetitions)
    results.append({"method": "MPCD actor bootstrap", **result})

    for result in results:
        result["eligible_reference"] = len(reference)
        result["eligible_focal"] = len(focal)
    return results


def run(repetitions: int = 1000, bootstrap_repetitions: int = 499) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    records = []
    for scenario in SCENARIOS:
        for delta, effect_label in ((0.0, "Null"), (-0.05, "Dissociation")):
            for replication in range(1, repetitions + 1):
                rows = one_replication(rng, scenario, delta, bootstrap_repetitions)
                for row in rows:
                    records.append(
                        {
                            "scenario": scenario.name,
                            "scenario_label": scenario.label,
                            "effect": effect_label,
                            "truth": delta,
                            "replication": replication,
                            **row,
                        }
                    )
    result = pd.DataFrame(records)
    result.to_csv(OUT / "expanded_simulation_replicates.csv", index=False)
    return result


def summarize(result: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["scenario", "scenario_label", "effect", "truth", "method"]
    for key, data in result.groupby(keys, sort=False):
        truth = key[3]
        error = data.estimate - truth
        rows.append(
            {
                **dict(zip(keys, key)),
                "mean_estimate": data.estimate.mean(),
                "bias": error.mean(),
                "rmse": np.sqrt(np.mean(error**2)),
                "empirical_sd": data.estimate.std(ddof=1),
                "mean_se": data.se.mean(),
                "coverage": data.cover.mean(),
                "rejection_rate": (data.p < 0.05).mean(),
                "mean_eligible_reference": data.eligible_reference.mean(),
                "mean_eligible_focal": data.eligible_focal.mean(),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "expanded_simulation_summary.csv", index=False)
    return summary


def plot_summary(summary: pd.DataFrame) -> None:
    methods = ["Event pooled", "GEE actor sandwich", "Cluster-weighted GEE", "MPCD Welch", "MPCD actor bootstrap"]
    colors = dict(zip(methods, ["#b2182b", "#ef8a62", "#67a9cf", "#2166ac", "#542788"]))
    null = summary[summary.effect == "Null"]
    alt = summary[summary.effect == "Dissociation"]
    labels = [s.label for s in SCENARIOS]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
    for method in methods:
        q = null[null.method == method].set_index("scenario").reindex([s.name for s in SCENARIOS])
        axes[0].plot(x, q.rejection_rate, marker="o", linewidth=1.8, label=method, color=colors[method])
        q = alt[alt.method == method].set_index("scenario").reindex([s.name for s in SCENARIOS])
        axes[1].plot(x, q.coverage, marker="o", linewidth=1.8, label=method, color=colors[method])
    axes[0].axhline(0.05, color="black", linestyle="--", linewidth=1)
    axes[1].axhline(0.95, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Type-I error")
    axes[1].set_ylabel("95% interval coverage under dissociation")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=28, ha="right")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.2)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(OUT / "expanded_simulation_performance.png", dpi=320, bbox_inches="tight")
    fig.savefig(OUT / "expanded_simulation_performance.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    result = run()
    summary = summarize(result)
    plot_summary(summary)
    display = summary[
        ["scenario", "effect", "method", "bias", "rmse", "coverage", "rejection_rate"]
    ]
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
