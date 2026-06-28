"""Robustness analyses for the professional-tennis MPCD application.

The script reuses the public Match Charting Project files cached by
``lefty_advantage_stage3.py``.  It creates a compact processed cache and then
reports event-average and actor-average placement compositions, actor-bootstrap
uncertainty for behavioral departures, threshold sensitivity for the outcome
contrast, alternative ace-like outcomes, and gender, surface, and decade strata.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


CACHE = Path("_tennis_cache")
OUT = Path("method_outputs")
OUT.mkdir(exist_ok=True)
PROCESSED = CACHE / "mpcd_processed_points.pkl"
SEED = 20260627
ZONES = ["deuce_wide", "deuce_middle", "deuce_t", "ad_wide", "ad_middle", "ad_t"]
MIRROR = {
    "deuce_wide": "ad_wide",
    "deuce_middle": "ad_middle",
    "deuce_t": "ad_t",
    "ad_wide": "deuce_wide",
    "ad_middle": "deuce_middle",
    "ad_t": "deuce_t",
}
BH = {"deuce_t", "ad_wide"}
FH = {"deuce_wide", "ad_t"}
SCORE = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4, "ad": 4}


def build_processed() -> pd.DataFrame:
    if PROCESSED.exists():
        cached = pd.read_pickle(PROCESSED)
        if {"returner", "match_id"}.issubset(cached.columns):
            return cached

    metadata = []
    points = []
    for gender in ("m", "w"):
        meta = pd.read_csv(CACHE / f"charting-{gender}-matches.csv", low_memory=False)
        meta["gender"] = gender.upper()
        metadata.append(
            meta[
                ["match_id", "Player 1", "Player 2", "Pl 1 hand", "Pl 2 hand", "Date", "Surface", "gender"]
            ]
        )
        for era in ("to-2009", "2010s", "2020s"):
            path = CACHE / f"charting-{gender}-points-{era}.csv"
            point = pd.read_csv(
                path,
                usecols=["match_id", "Pt", "Gm#", "Svr", "1st", "2nd", "PtWinner", "Pts"],
                low_memory=False,
            )
            points.append(point)

    meta = pd.concat(metadata, ignore_index=True).drop_duplicates("match_id").set_index("match_id")
    data = pd.concat(points, ignore_index=True)
    data = data.dropna(subset=["match_id", "Gm#", "Svr", "PtWinner"]).copy()
    data["Svr"] = pd.to_numeric(data.Svr, errors="coerce")
    data["PtWinner"] = pd.to_numeric(data.PtWinner, errors="coerce")
    data["Pt"] = pd.to_numeric(data.Pt, errors="coerce")
    data = data.dropna(subset=["Svr", "PtWinner"])
    data["Svr"] = data.Svr.astype(np.int8)
    data["PtWinner"] = data.PtWinner.astype(np.int8)

    data = data.sort_values(["match_id", "Gm#", "Pt"])
    within_game = data.groupby(["match_id", "Gm#"], sort=False).cumcount()
    sequential_court = np.where(within_game % 2 == 0, "deuce", "ad")
    score_parts = data.Pts.fillna("").astype(str).str.split("-", n=1, expand=True)
    if score_parts.shape[1] == 2:
        first_score = score_parts[0].map(SCORE)
        second_score = score_parts[1].str.strip().map(SCORE)
        parsed = first_score.notna() & second_score.notna()
        score_court = np.where((first_score.fillna(0) + second_score.fillna(0)) % 2 == 0, "deuce", "ad")
        court = np.where(parsed, score_court, sequential_court)
    else:
        court = sequential_court

    first = data["1st"].fillna("").astype(str).str.strip()
    second = data["2nd"].fillna("").astype(str).str.strip()
    has_second = ~second.isin(["", "nan", "NaN"])
    serve = first.where(~has_second, second)
    direction = serve.str[:1].map({"4": "wide", "5": "middle", "6": "t"})

    joined = data[["match_id", "Svr", "PtWinner"]].join(meta, on="match_id")
    server_is_one = joined.Svr.eq(1)
    server_name = joined["Player 1"].where(server_is_one, joined["Player 2"])
    returner_name = joined["Player 2"].where(server_is_one, joined["Player 1"])
    server_hand = joined["Pl 1 hand"].where(server_is_one, joined["Pl 2 hand"])
    returner_hand = joined["Pl 2 hand"].where(server_is_one, joined["Pl 1 hand"])
    year = pd.to_numeric(joined.Date.astype(str).str[:4], errors="coerce")

    processed = pd.DataFrame(
        {
            "match_id": joined.match_id.astype("category"),
            "server": server_name.astype("category"),
            "returner": returner_name.astype("category"),
            "hand": server_hand.astype("category"),
            "returner_hand": returner_hand.astype("category"),
            "zone": (pd.Series(court, index=data.index) + "_" + direction).astype("category"),
            "first": (~has_second).to_numpy(),
            "won": joined.PtWinner.eq(joined.Svr).to_numpy(dtype=np.int8),
            "ace_like": ((serve.str.len() <= 2) & serve.str[-1:].isin(["*", "#"])).to_numpy(dtype=np.int8),
            "gender": joined.gender.astype("category"),
            "surface": joined.Surface.fillna("Unknown").astype("category"),
            "year": year.to_numpy(),
        }
    )
    processed = processed[
        processed.returner_hand.eq("R") & processed.hand.isin(["L", "R"]) & processed.zone.isin(ZONES)
    ].copy()
    processed["decade"] = (processed.year // 10 * 10).astype("Int64")
    processed.to_pickle(PROCESSED)
    return processed


def clr(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.log(values) - np.log(values).mean()


def _behavior_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, np.ndarray]:
    """Return mirror-departure metrics for one row or a matrix of rows."""
    mirror_index = np.array([ZONES.index(MIRROR[z]) for z in ZONES])
    right_mirror = right[..., mirror_index]
    residual = left - right_mirror
    left_clr = np.log(left) - np.log(left).mean(axis=-1, keepdims=True)
    right_clr = np.log(right_mirror) - np.log(right_mirror).mean(axis=-1, keepdims=True)
    signature = np.array([ZONES.index(z) for z in ZONES if z in BH])
    return {
        "total_variation": 0.5 * np.abs(residual).sum(axis=-1),
        "aitchison_distance": np.sqrt(((left_clr - right_clr) ** 2).sum(axis=-1)),
        "signature_excess": left[..., signature].sum(axis=-1) - right_mirror[..., signature].sum(axis=-1),
    }


def behavior_bootstrap(
    actor_counts: pd.DataFrame, repetitions: int = 5000, chunk_size: int = 250
) -> pd.DataFrame:
    """Stratified actor bootstrap for event- and actor-average mirror metrics."""
    rng = np.random.default_rng(SEED + 101)
    counts = {}
    shares = {}
    for hand in ("L", "R"):
        matrix = actor_counts.loc[hand, ZONES].to_numpy(dtype=float)
        counts[hand] = matrix
        shares[hand] = matrix / matrix.sum(axis=1, keepdims=True)

    draws = {estimand: {metric: [] for metric in ("total_variation", "aitchison_distance", "signature_excess")}
             for estimand in ("Event average", "Actor average")}
    for start in range(0, repetitions, chunk_size):
        size = min(chunk_size, repetitions - start)
        weights = {
            hand: rng.multinomial(len(counts[hand]), np.full(len(counts[hand]), 1 / len(counts[hand])), size=size)
            for hand in ("L", "R")
        }
        event = {}
        actor = {}
        for hand in ("L", "R"):
            event_counts = weights[hand] @ counts[hand]
            event[hand] = event_counts / event_counts.sum(axis=1, keepdims=True)
            actor[hand] = weights[hand] @ shares[hand] / len(counts[hand])
        for estimand, composition in (("Event average", event), ("Actor average", actor)):
            metrics = _behavior_metrics(composition["L"], composition["R"])
            for metric, values in metrics.items():
                draws[estimand][metric].append(values)

    rows = []
    for estimand, metrics in draws.items():
        for metric, pieces in metrics.items():
            values = np.concatenate(pieces)
            lo, hi = np.quantile(values, [0.025, 0.975])
            rows.append(
                {
                    "estimand": estimand,
                    "metric": metric,
                    "bootstrap_se": values.std(ddof=1),
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                    "repetitions": repetitions,
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "tennis_behavior_bootstrap.csv", index=False)
    return result


def composition_summary(data: pd.DataFrame) -> pd.DataFrame:
    event_counts = data.groupby(["hand", "zone"], observed=True).size().unstack(fill_value=0).reindex(columns=ZONES)
    event_share = event_counts.div(event_counts.sum(axis=1), axis=0)

    actor_counts = data.groupby(["hand", "server", "zone"], observed=True).size().unstack(fill_value=0).reindex(columns=ZONES, fill_value=0)
    actor_share = actor_counts.div(actor_counts.sum(axis=1), axis=0)
    actor_average = actor_share.groupby(level="hand", observed=True).mean().reindex(columns=ZONES)

    uncertainty = behavior_bootstrap(actor_counts)
    interval = uncertainty.set_index(["estimand", "metric"])
    rows = []
    for estimand, share in (("Event average", event_share), ("Actor average", actor_average)):
        left = share.loc["L", ZONES].to_numpy()
        right_mirror = np.array([share.loc["R", MIRROR[z]] for z in ZONES])
        tv = 0.5 * np.abs(left - right_mirror).sum()
        aitchison = np.sqrt(np.sum((clr(left) - clr(right_mirror)) ** 2))
        signature_left = sum(share.loc["L", z] for z in BH)
        signature_reference = sum(share.loc["R", MIRROR[z]] for z in BH)
        rows.append(
            {
                "estimand": estimand,
                "n_left_actors": actor_share.loc["L"].shape[0],
                "n_right_actors": actor_share.loc["R"].shape[0],
                "total_variation": tv,
                "aitchison_distance": aitchison,
                "left_signature_share": signature_left,
                "mirrored_reference_share": signature_reference,
                "signature_excess": signature_left - signature_reference,
            }
        )
        for metric in ("total_variation", "aitchison_distance", "signature_excess"):
            rows[-1][f"{metric}_lo"] = interval.loc[(estimand, metric), "ci95_lo"]
            rows[-1][f"{metric}_hi"] = interval.loc[(estimand, metric), "ci95_hi"]
        for zone, p_left, p_reference in zip(ZONES, left, right_mirror):
            rows[-1][f"residual_{zone}"] = p_left - p_reference
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "tennis_composition_estimands.csv", index=False)
    actor_share.reset_index().to_csv(OUT / "tennis_actor_compositions.csv", index=False)
    return result


def eligibility_profile(data: pd.DataFrame, threshold: int = 50) -> pd.DataFrame:
    """Describe placement selectivity among eligible and excluded left-handed servers."""
    corner = data[data["first"] & data.zone.isin(BH | FH)].copy()
    corner["target"] = np.where(corner.zone.isin(BH), "BH", "FH")
    counts = (
        corner.groupby(["server", "hand", "target"], observed=True)
        .size()
        .unstack("target", fill_value=0)
        .reindex(columns=["BH", "FH"], fill_value=0)
    )
    left = counts[counts.index.get_level_values("hand") == "L"].copy()
    left["total"] = left.BH + left.FH
    left["signature_share"] = left.BH / left.total
    eligible = left.BH.ge(threshold) & left.FH.ge(threshold)
    groups = {
        "Eligible in primary analysis": eligible,
        "Excluded, all": ~eligible,
        "Excluded with at least 100 corner serves": (~eligible) & left.total.ge(2 * threshold),
        "Excluded because forehand cell below 50 only": left.BH.ge(threshold) & left.FH.lt(threshold),
        "Excluded because backhand cell below 50 only": left.BH.lt(threshold) & left.FH.ge(threshold),
        "Excluded because both cells below 50": left.BH.lt(threshold) & left.FH.lt(threshold),
    }
    rows = []
    for label, select in groups.items():
        subset = left[select]
        rows.append(
            {
                "group": label,
                "n_left_servers": len(subset),
                "mean_signature_share": subset.signature_share.mean(),
                "median_signature_share": subset.signature_share.median(),
                "pooled_signature_share": subset.BH.sum() / subset.total.sum() if len(subset) else np.nan,
                "median_corner_serves": subset.total.median(),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "tennis_eligibility_profile.csv", index=False)
    return result


def actor_contrasts(data: pd.DataFrame, threshold: int, outcome: str = "won") -> pd.DataFrame:
    corner = data[data["first"] & data.zone.isin(BH | FH)].copy()
    corner["target"] = np.where(corner.zone.isin(BH), "BH", "FH")
    aggregate = corner.groupby(["server", "hand", "target"], observed=True)[outcome].agg(["sum", "count"])
    sums = aggregate["sum"].unstack("target")
    counts = aggregate["count"].unstack("target")
    valid = counts.BH.ge(threshold) & counts.FH.ge(threshold)
    result = pd.DataFrame(
        {
            "hand": sums.index.get_level_values("hand").to_numpy(),
            "n_bh": counts.BH.to_numpy(),
            "n_fh": counts.FH.to_numpy(),
            "rate_bh": (sums.BH / counts.BH).to_numpy(),
            "rate_fh": (sums.FH / counts.FH).to_numpy(),
        },
        index=sums.index.get_level_values("server"),
    )
    result = result[valid.to_numpy()].copy()
    result["gap"] = result.rate_bh - result.rate_fh
    return result


def bootstrap_difference(
    rng: np.random.Generator, left: np.ndarray, right: np.ndarray, repetitions: int = 5000
) -> tuple[float, float, float]:
    estimate = left.mean() - right.mean()
    boot_left = rng.choice(left, size=(repetitions, len(left)), replace=True).mean(axis=1)
    boot_right = rng.choice(right, size=(repetitions, len(right)), replace=True).mean(axis=1)
    lo, hi = np.quantile(boot_left - boot_right, [0.025, 0.975])
    return estimate, lo, hi


def threshold_sensitivity(data: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for outcome in ("won", "ace_like"):
        for threshold in (20, 30, 50, 75, 100):
            actors = actor_contrasts(data, threshold, outcome)
            left = actors.loc[actors.hand == "L", "gap"].to_numpy()
            right = actors.loc[actors.hand == "R", "gap"].to_numpy()
            estimate, lo, hi = bootstrap_difference(rng, left, right)
            rows.append(
                {
                    "outcome": outcome,
                    "threshold": threshold,
                    "n_left": len(left),
                    "n_right": len(right),
                    "left_gap": left.mean(),
                    "right_gap": right.mean(),
                    "mpcd": estimate,
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                    "welch_p": ttest_ind(left, right, equal_var=False).pvalue,
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "tennis_threshold_sensitivity.csv", index=False)
    return result


def stratum_summary(data: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 1)
    rows = []
    specifications = [
        ("Gender", "gender", ["M", "W"]),
        ("Surface", "surface", ["Hard", "Clay", "Grass"]),
        ("Decade", "decade", [1970, 1980, 1990, 2000, 2010, 2020]),
    ]
    for stratum_type, column, levels in specifications:
        for level in levels:
            actors = actor_contrasts(data[data[column].eq(level)], threshold=20, outcome="won")
            left = actors.loc[actors.hand == "L", "gap"].to_numpy()
            right = actors.loc[actors.hand == "R", "gap"].to_numpy()
            if min(len(left), len(right)) < 8:
                continue
            estimate, lo, hi = bootstrap_difference(rng, left, right, repetitions=3000)
            rows.append(
                {
                    "stratum_type": stratum_type,
                    "stratum": level,
                    "threshold": 20,
                    "n_left": len(left),
                    "n_right": len(right),
                    "left_gap": left.mean(),
                    "right_gap": right.mean(),
                    "mpcd": estimate,
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "tennis_stratified_sensitivity.csv", index=False)
    return result


def weighting_comparison(data: pd.DataFrame) -> pd.DataFrame:
    corner = data[data["first"] & data.zone.isin(BH | FH)].copy()
    corner["target"] = np.where(corner.zone.isin(BH), "BH", "FH")
    aggregate = corner.groupby(["server", "hand", "target"], observed=True).won.agg(["sum", "count"])
    rows = []
    for method in ("Event average", "Cluster-size weighted", "Actor paired"):
        gaps = {}
        for hand in ("L", "R"):
            q = aggregate.xs(hand, level="hand")
            sums = q["sum"].unstack("target")
            counts = q["count"].unstack("target")
            valid = counts.BH.ge(5) & counts.FH.ge(5)
            sums, counts = sums[valid], counts[valid]
            if method == "Event average":
                gap = sums.BH.sum() / counts.BH.sum() - sums.FH.sum() / counts.FH.sum()
            elif method == "Cluster-size weighted":
                weight = 1 / (counts.BH + counts.FH)
                gap = (weight * sums.BH).sum() / (weight * counts.BH).sum()
                gap -= (weight * sums.FH).sum() / (weight * counts.FH).sum()
            else:
                gap = (sums.BH / counts.BH - sums.FH / counts.FH).mean()
            gaps[hand] = gap
        rows.append({"method": method, "left_gap": gaps["L"], "right_gap": gaps["R"], "contrast": gaps["L"] - gaps["R"]})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "tennis_weighting_comparison.csv", index=False)
    return result


def crossed_actor_returner_bootstrap(
    data: pd.DataFrame, threshold: int = 50, repetitions: int = 3000
) -> pd.DataFrame:
    """Bayesian pigeonhole bootstrap over server and returner identities."""
    corner = data[data["first"] & data.zone.isin(BH | FH)].copy()
    corner["target"] = np.where(corner.zone.isin(BH), "BH", "FH")
    totals = corner.groupby(["server", "hand", "target"], observed=True).size().unstack("target")
    eligible = totals[(totals.BH >= threshold) & (totals.FH >= threshold)].reset_index()[["server", "hand"]]
    corner = corner.merge(eligible, on=["server", "hand"], how="inner")
    cells = (
        corner.groupby(["server", "hand", "returner", "target"], observed=True)
        .won.agg(["sum", "count"])
        .reset_index()
    )

    server_table = eligible.drop_duplicates("server").reset_index(drop=True)
    server_index = {name: i for i, name in enumerate(server_table.server)}
    returners = pd.Index(cells.returner.unique())
    returner_index = {name: i for i, name in enumerate(returners)}
    target_index = cells.target.map({"BH": 0, "FH": 1}).to_numpy(dtype=int)
    s_idx = cells.server.map(server_index).to_numpy(dtype=int)
    r_idx = cells.returner.map(returner_index).to_numpy(dtype=int)
    cell_index = 2 * s_idx + target_index
    y = cells["sum"].to_numpy(dtype=float)
    n = cells["count"].to_numpy(dtype=float)
    hand_left = server_table.hand.eq("L").to_numpy()
    hand_right = server_table.hand.eq("R").to_numpy()

    rng = np.random.default_rng(SEED + 77)
    estimates = np.empty(repetitions)
    for b in range(repetitions):
        server_weight = rng.exponential(size=len(server_table))
        returner_weight = rng.exponential(size=len(returners))
        observation_weight = returner_weight[r_idx]
        weighted_y = np.bincount(cell_index, weights=observation_weight * y, minlength=2 * len(server_table))
        weighted_n = np.bincount(cell_index, weights=observation_weight * n, minlength=2 * len(server_table))
        rates = (weighted_y / weighted_n).reshape(len(server_table), 2)
        gaps = rates[:, 0] - rates[:, 1]
        left_mean = np.sum(server_weight[hand_left] * gaps[hand_left]) / np.sum(server_weight[hand_left])
        right_mean = np.sum(server_weight[hand_right] * gaps[hand_right]) / np.sum(server_weight[hand_right])
        estimates[b] = left_mean - right_mean

    ordinary = actor_contrasts(data, threshold, "won")
    point = ordinary.loc[ordinary.hand == "L", "gap"].mean() - ordinary.loc[ordinary.hand == "R", "gap"].mean()
    lo, hi = np.quantile(estimates, [0.025, 0.975])
    result = pd.DataFrame(
        [
            {
                "threshold": threshold,
                "n_servers": len(server_table),
                "n_returners": len(returners),
                "point_estimate": point,
                "crossed_bootstrap_se": estimates.std(ddof=1),
                "ci95_lo": lo,
                "ci95_hi": hi,
                "repetitions": repetitions,
            }
        ]
    )
    result.to_csv(OUT / "tennis_crossed_bootstrap.csv", index=False)
    pd.DataFrame({"replicate": np.arange(1, repetitions + 1), "estimate": estimates}).to_csv(
        OUT / "tennis_crossed_bootstrap_replicates.csv", index=False
    )
    return result


def plot_sensitivity(thresholds: pd.DataFrame, strata: pd.DataFrame) -> None:
    point = thresholds[thresholds.outcome == "won"].copy()
    forest = strata.copy()
    forest["label"] = forest.stratum_type + ": " + forest.stratum.astype(str)
    y = np.arange(len(forest))

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.8), gridspec_kw={"width_ratios": [1, 1.35]})
    axes[0].errorbar(
        point.threshold,
        point.mpcd,
        yerr=[point.mpcd - point.ci95_lo, point.ci95_hi - point.mpcd],
        fmt="o-",
        color="#2166ac",
        capsize=3,
    )
    axes[0].axhline(0, color="black", linewidth=1, linestyle="--")
    axes[0].set_xlabel("Minimum first serves in each target set")
    axes[0].set_ylabel("MPCD payoff contrast")
    axes[0].set_title("Threshold sensitivity")
    axes[0].grid(alpha=0.2)

    axes[1].errorbar(
        forest.mpcd,
        y,
        xerr=[forest.mpcd - forest.ci95_lo, forest.ci95_hi - forest.mpcd],
        fmt="o",
        color="#b2182b",
        capsize=3,
    )
    axes[1].axvline(0, color="black", linewidth=1, linestyle="--")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(forest.label)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("MPCD payoff contrast")
    axes[1].set_title("Gender, surface, and decade strata")
    axes[1].grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "tennis_sensitivity.png", dpi=320, bbox_inches="tight")
    fig.savefig(OUT / "tennis_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    data = build_processed()
    composition = composition_summary(data)
    eligibility = eligibility_profile(data)
    thresholds = threshold_sensitivity(data)
    strata = stratum_summary(data)
    weighting = weighting_comparison(data)
    crossed = crossed_actor_returner_bootstrap(data)
    plot_sensitivity(thresholds, strata)
    print("Composition estimands")
    print(composition.to_string(index=False))
    print("\nEligibility profile for left-handed servers")
    print(eligibility.to_string(index=False))
    print("\nThreshold sensitivity")
    print(thresholds.to_string(index=False))
    print("\nStratified sensitivity")
    print(strata.to_string(index=False))
    print("\nWeighting comparison")
    print(weighting.to_string(index=False))
    print("\nCrossed server-returner bootstrap")
    print(crossed.to_string(index=False))


if __name__ == "__main__":
    main()
