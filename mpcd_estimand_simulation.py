"""Estimand-aligned Monte Carlo study for the MPCD framework.

The study treats the actor-average and event-average contrasts as distinct
scientific targets. Event estimators are evaluated against the event-average
truth, whereas MPCD and action-cell weighted GEE are evaluated against the
actor-average truth. Actor-null and event-null regimes provide symmetric
examples in which either target, but not necessarily the other, equals zero.
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
    quality_effect: float
    total_informative: float
    allocation_informative: float
    minimum_cell: int


SCENARIOS = [
    Scenario("balanced", "Balanced, non-informative", 150, 150, 180, 0.00, 0.00, 0.00, 0.00, 5),
    Scenario("unequal", "Unequal, non-informative", 300, 75, 180, 0.90, 1.00, 0.00, 0.00, 5),
    Scenario("total_ics", "Informative total size", 300, 75, 180, 0.80, 1.00, 0.75, 0.00, 5),
    Scenario("action_ics", "Informative action allocation", 300, 75, 180, 0.55, 1.00, 0.00, 1.00, 5),
    Scenario("combined", "Combined informativeness", 618, 76, 180, 0.90, 1.00, 0.75, 1.00, 5),
    Scenario("sparse", "Sparse combined setting", 180, 60, 45, 0.85, 1.00, 0.75, 1.00, 5),
]


def generate_structure(
    rng: np.random.Generator,
    scenario: Scenario,
    group: int,
    delta: float,
    n_actor: int,
) -> pd.DataFrame:
    quality = rng.normal(0.0, 1.0, n_actor)
    log_total = (
        np.log(scenario.mean_events)
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
    allocation = expit(allocation_logit)
    n_s = rng.binomial(total, allocation)
    n_c = total - n_s

    # The bounded baseline avoids clipping, so the actor-average contrast is
    # exactly delta. Informative action allocation can nevertheless separate
    # the event-average and actor-average targets through baseline quality.
    p_c = 0.58 + 0.16 * scenario.quality_effect * np.tanh(quality)
    p_s = p_c + 0.02 + delta * group
    if np.any((p_s <= 0) | (p_s >= 1)):
        raise ValueError("Probability outside the unit interval")

    data = pd.DataFrame({"group": group, "n_s": n_s, "n_c": n_c, "p_s": p_s, "p_c": p_c})
    return data[(data.n_s >= scenario.minimum_cell) & (data.n_c >= scenario.minimum_cell)].copy()


def add_outcomes(rng: np.random.Generator, data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["y_s"] = rng.binomial(result.n_s.to_numpy(), result.p_s.to_numpy())
    result["y_c"] = rng.binomial(result.n_c.to_numpy(), result.p_c.to_numpy())
    return result


def truths(reference: pd.DataFrame, focal: pd.DataFrame) -> tuple[float, float]:
    actor = (focal.p_s - focal.p_c).mean() - (reference.p_s - reference.p_c).mean()

    def event_gap(data: pd.DataFrame) -> float:
        signature = np.sum(data.n_s * data.p_s) / data.n_s.sum()
        comparator = np.sum(data.n_c * data.p_c) / data.n_c.sum()
        return signature - comparator

    event = event_gap(focal) - event_gap(reference)
    return float(actor), float(event)


def population_targets(population_size: int = 800_000) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 101)
    records = []
    for scenario in SCENARIOS:
        reference = generate_structure(rng, scenario, 0, 0.0, population_size)
        focal = generate_structure(rng, scenario, 1, 0.0, population_size)
        actor_zero, event_zero = truths(reference, focal)
        event_null_delta = -event_zero
        for regime, delta in (
            ("Actor null", 0.0),
            ("Event null", event_null_delta),
            ("Common alternative", -0.05),
        ):
            # Action counts and eligibility do not depend on delta. The focal
            # signature probability shifts additively, so the event target and
            # actor target both shift by delta from their values at delta zero.
            actor_truth = actor_zero + delta
            event_truth = event_zero + delta
            records.append(
                {
                    "scenario": scenario.name,
                    "scenario_label": scenario.label,
                    "regime": regime,
                    "delta": delta,
                    "actor_truth": actor_truth,
                    "event_truth": event_truth,
                    "estimand_divergence": event_truth - actor_truth,
                    "population_eligible_reference": len(reference),
                    "population_eligible_focal": len(focal),
                }
            )
    result = pd.DataFrame(records)
    result.to_csv(OUT / "estimand_population_targets.csv", index=False)
    return result


def event_estimate(data: pd.DataFrame) -> tuple[float, float, float]:
    p_s = data.y_s.sum() / data.n_s.sum()
    p_c = data.y_c.sum() / data.n_c.sum()
    gap = p_s - p_c
    naive_var = p_s * (1 - p_s) / data.n_s.sum() + p_c * (1 - p_c) / data.n_c.sum()
    influence = (data.y_s - data.n_s * p_s) / data.n_s.sum()
    influence -= (data.y_c - data.n_c * p_c) / data.n_c.sum()
    n = len(data)
    sandwich_var = (n / (n - 1)) * np.sum(influence.to_numpy() ** 2)
    return float(gap), float(naive_var), float(sandwich_var)


def actor_values(data: pd.DataFrame) -> np.ndarray:
    return data.y_s.to_numpy() / data.n_s.to_numpy() - data.y_c.to_numpy() / data.n_c.to_numpy()


def normal_interval(estimate: float, se: float) -> tuple[float, float, float]:
    lo, hi = estimate - 1.96 * se, estimate + 1.96 * se
    p = 2 * norm.sf(abs(estimate / se)) if se > 0 else np.nan
    return lo, hi, p


def welch_interval(left: np.ndarray, right: np.ndarray) -> tuple[float, float, float, float]:
    estimate = left.mean() - right.mean()
    v_left = left.var(ddof=1) / len(left)
    v_right = right.var(ddof=1) / len(right)
    se = np.sqrt(v_left + v_right)
    df = (v_left + v_right) ** 2 / (v_left**2 / (len(left) - 1) + v_right**2 / (len(right) - 1))
    critical = t.ppf(0.975, df)
    lo, hi = estimate - critical * se, estimate + critical * se
    p = 2 * t.sf(abs(estimate / se), df)
    return estimate, se, lo, hi, p


def bootstrap_distributions(
    rng: np.random.Generator,
    focal: pd.DataFrame,
    reference: pd.DataFrame,
    repetitions: int,
) -> tuple[np.ndarray, np.ndarray]:
    def weights(n: int) -> np.ndarray:
        return rng.multinomial(n, np.full(n, 1 / n), size=repetitions)

    wf = weights(len(focal))
    wr = weights(len(reference))

    def event_boot(data: pd.DataFrame, weight: np.ndarray) -> np.ndarray:
        ps = (weight @ data.y_s.to_numpy()) / (weight @ data.n_s.to_numpy())
        pc = (weight @ data.y_c.to_numpy()) / (weight @ data.n_c.to_numpy())
        return ps - pc

    event = event_boot(focal, wf) - event_boot(reference, wr)
    actor = wf @ actor_values(focal) / len(focal) - wr @ actor_values(reference) / len(reference)
    return event, actor


def append_result(
    rows: list[dict],
    method: str,
    target: str,
    estimate: float,
    se: float,
    lo: float,
    hi: float,
    p: float,
    actor_truth: float,
    event_truth: float,
) -> None:
    own_truth = actor_truth if target == "Actor average" else event_truth
    other_truth = event_truth if target == "Actor average" else actor_truth
    rows.append(
        {
            "method": method,
            "target": target,
            "estimate": estimate,
            "se": se,
            "lo": lo,
            "hi": hi,
            "p": p,
            "own_truth": own_truth,
            "other_truth": other_truth,
            "bias_own": estimate - own_truth,
            "bias_actor": estimate - actor_truth,
            "bias_event": estimate - event_truth,
            "cover_own": lo <= own_truth <= hi,
            "cover_actor": lo <= actor_truth <= hi,
            "cover_event": lo <= event_truth <= hi,
            "zero_reject": p < 0.05,
        }
    )


def one_replication(
    data_rng: np.random.Generator,
    bootstrap_rng: np.random.Generator,
    scenario: Scenario,
    delta: float,
    actor_truth: float,
    event_truth: float,
    bootstrap_repetitions: int,
) -> list[dict]:
    reference = add_outcomes(
        data_rng, generate_structure(data_rng, scenario, 0, delta, scenario.n_reference)
    )
    focal = add_outcomes(
        data_rng, generate_structure(data_rng, scenario, 1, delta, scenario.n_focal)
    )
    if min(len(reference), len(focal)) < 8:
        return []

    rows: list[dict] = []
    focal_gap, focal_naive, focal_sandwich = event_estimate(focal)
    reference_gap, reference_naive, reference_sandwich = event_estimate(reference)
    event_point = focal_gap - reference_gap

    se = np.sqrt(focal_naive + reference_naive)
    lo, hi, p = normal_interval(event_point, se)
    append_result(rows, "Event pooled, naive SE", "Event average", event_point, se, lo, hi, p, actor_truth, event_truth)

    se = np.sqrt(focal_sandwich + reference_sandwich)
    lo, hi, p = normal_interval(event_point, se)
    append_result(rows, "Event GEE, actor sandwich", "Event average", event_point, se, lo, hi, p, actor_truth, event_truth)

    d_f = actor_values(focal)
    d_r = actor_values(reference)
    actor_point, actor_se, lo, hi, p = welch_interval(d_f, d_r)
    append_result(rows, "MPCD Welch", "Actor average", actor_point, actor_se, lo, hi, p, actor_truth, event_truth)

    # A saturated GEE with inverse action-cell weights has the same point
    # estimate as MPCD. A normal reference distribution isolates the small
    # finite-sample difference from the Welch implementation.
    lo, hi, p = normal_interval(actor_point, actor_se)
    append_result(rows, "Action-cell weighted GEE", "Actor average", actor_point, actor_se, lo, hi, p, actor_truth, event_truth)

    event_boot, actor_boot = bootstrap_distributions(
        bootstrap_rng, focal, reference, bootstrap_repetitions
    )
    lo, hi = np.quantile(event_boot, [0.025, 0.975])
    p = min(1.0, 2 * min((event_boot <= 0).mean(), (event_boot >= 0).mean()))
    append_result(rows, "Event actor bootstrap", "Event average", event_point, event_boot.std(ddof=1), lo, hi, p, actor_truth, event_truth)

    lo, hi = np.quantile(actor_boot, [0.025, 0.975])
    p = min(1.0, 2 * min((actor_boot <= 0).mean(), (actor_boot >= 0).mean()))
    append_result(rows, "MPCD actor bootstrap", "Actor average", actor_point, actor_boot.std(ddof=1), lo, hi, p, actor_truth, event_truth)

    for row in rows:
        row["eligible_reference"] = len(reference)
        row["eligible_focal"] = len(focal)
        row["point_equivalence_error"] = abs(actor_point - (d_f.mean() - d_r.mean()))
    return rows


def run(repetitions: int = 1000, bootstrap_repetitions: int = 999) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = population_targets()
    data_rng = np.random.default_rng(SEED)
    bootstrap_rng = np.random.default_rng(SEED + 1)
    records = []
    scenario_map = {scenario.name: scenario for scenario in SCENARIOS}
    for target_row in targets.itertuples(index=False):
        scenario = scenario_map[target_row.scenario]
        for replication in range(1, repetitions + 1):
            result = one_replication(
                data_rng,
                bootstrap_rng,
                scenario,
                target_row.delta,
                target_row.actor_truth,
                target_row.event_truth,
                bootstrap_repetitions,
            )
            for row in result:
                records.append(
                    {
                        "scenario": scenario.name,
                        "scenario_label": scenario.label,
                        "regime": target_row.regime,
                        "delta": target_row.delta,
                        "actor_truth": target_row.actor_truth,
                        "event_truth": target_row.event_truth,
                        "replication": replication,
                        **row,
                    }
                )
    result = pd.DataFrame(records)
    result.to_csv(OUT / "estimand_simulation_replicates.csv", index=False)
    return targets, result


def summarize(result: pd.DataFrame) -> pd.DataFrame:
    keys = ["scenario", "scenario_label", "regime", "actor_truth", "event_truth", "method", "target"]
    rows = []
    for key, data in result.groupby(keys, sort=False):
        own_error = data.estimate - data.own_truth
        rows.append(
            {
                **dict(zip(keys, key)),
                "mean_estimate": data.estimate.mean(),
                "own_bias": own_error.mean(),
                "own_rmse": np.sqrt(np.mean(own_error**2)),
                "empirical_sd": data.estimate.std(ddof=1),
                "mean_se": data.se.mean(),
                "own_coverage": data.cover_own.mean(),
                "actor_target_coverage": data.cover_actor.mean(),
                "event_target_coverage": data.cover_event.mean(),
                "zero_rejection": data.zero_reject.mean(),
                "mean_eligible_reference": data.eligible_reference.mean(),
                "mean_eligible_focal": data.eligible_focal.mean(),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "estimand_simulation_summary.csv", index=False)
    return summary


def plot_results(targets: pd.DataFrame, summary: pd.DataFrame) -> None:
    order = [s.name for s in SCENARIOS]
    labels = [s.label for s in SCENARIOS]
    x = np.arange(len(order))
    actor_null = targets[targets.regime == "Actor null"].set_index("scenario").reindex(order)

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4))
    axes[0].plot(x, actor_null.actor_truth, "o-", color="#2166ac", label="Actor-average truth")
    axes[0].plot(x, actor_null.event_truth, "s-", color="#b2182b", label="Event-average truth")
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("True contrast under actor-null regime")
    axes[0].set_title("Estimand divergence")
    axes[0].legend(frameon=False, fontsize=8)

    own_methods = ["Event GEE, actor sandwich", "Event actor bootstrap", "MPCD Welch", "MPCD actor bootstrap"]
    colors = ["#ef8a62", "#b2182b", "#67a9cf", "#2166ac"]
    q = summary[summary.regime == "Actor null"]
    for method, color in zip(own_methods, colors):
        z = q[q.method == method].set_index("scenario").reindex(order)
        axes[1].plot(x, z.own_coverage, marker="o", color=color, label=method)
    axes[1].axhline(0.95, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Coverage of each method's own target")
    axes[1].set_title("Correct target alignment")
    axes[1].legend(frameon=False, fontsize=7)

    q_actor = summary[(summary.regime == "Actor null") & (summary.method == "Event GEE, actor sandwich")]
    q_event = summary[(summary.regime == "Event null") & (summary.method == "MPCD Welch")]
    q_actor = q_actor.set_index("scenario").reindex(order)
    q_event = q_event.set_index("scenario").reindex(order)
    axes[2].plot(x, q_actor.actor_target_coverage, "o-", color="#b2182b", label="Event GEE assessed at actor target")
    axes[2].plot(x, q_event.event_target_coverage, "s-", color="#2166ac", label="MPCD assessed at event target")
    axes[2].axhline(0.95, color="black", linestyle="--", linewidth=1)
    axes[2].set_ylabel("Coverage after target substitution")
    axes[2].set_title("Symmetric target misalignment")
    axes[2].legend(frameon=False, fontsize=7)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(-0.02 if ax is axes[0] else 0, 0.08 if ax is axes[0] else 1.02)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "estimand_alignment_simulation.png", dpi=320, bbox_inches="tight")
    fig.savefig(OUT / "estimand_alignment_simulation.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    targets, result = run()
    summary = summarize(result)
    plot_results(targets, summary)
    selected = summary[
        (summary.scenario.isin(["balanced", "action_ics", "combined", "sparse"]))
        & (summary.method.isin(["Event GEE, actor sandwich", "MPCD Welch", "Action-cell weighted GEE"]))
    ]
    print(
        selected[
            [
                "scenario",
                "regime",
                "method",
                "target",
                "own_bias",
                "own_rmse",
                "own_coverage",
                "actor_target_coverage",
                "event_target_coverage",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
