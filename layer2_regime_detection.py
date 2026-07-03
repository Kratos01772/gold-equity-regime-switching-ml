from __future__ import annotations

import contextlib
import io
import math
import os
import pickle
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from hmmlearn.hmm import GaussianHMM
from scipy.stats import f_oneway
from sklearn.preprocessing import StandardScaler


INPUT_PATH = Path("data") / "gold_equity_master_data.csv"
OUTPUT_DATA_PATH = Path("data") / "regime_labeled_data.csv"
MODEL_PATH = Path("models") / "hmm_model.pkl"
SCALER_PATH = Path("models") / "hmm_scaler.pkl"
VISUALIZATION_DIR = Path("visualizations")

RANDOM_STATES = [42, 0, 7, 123]
COMPONENT_OPTIONS = [3, 4]

HMM_FEATURE_ALIASES = {
    "nifty_return": "nifty_return",
    "india_vix": "india_vix",
    "bank_spread": "bank_corwin_schultz_spread",
    "gold_nifty_corr_30d": "gold_nifty_corr_30d",
    "nifty_vol": "nifty_volatility_20d",
}

DISPLAY_TO_SOURCE = HMM_FEATURE_ALIASES.copy()
SOURCE_TO_DISPLAY = {value: key for key, value in DISPLAY_TO_SOURCE.items()}

REGIME_COLOR_MAP = {
    "Crisis": "#c0392b",
    "Late_Cycle": "#d68910",
    "Expansion": "#1e8449",
    "Recovery": "#2874a6",
}


@dataclass
class ModelRun:
    n_components: int
    random_state: int
    model: GaussianHMM
    scaler: StandardScaler
    scaled_features: np.ndarray
    hidden_states: np.ndarray
    bic: float
    log_likelihood: float
    diagonal_mean: float
    balance_score: float
    regime_proportions: Dict[int, float]
    working_data: pd.DataFrame


def log_progress(message: str) -> None:
    print(f"[INFO] {message}")


def safe_print(message: str) -> None:
    try:
        print(message)
    except UnicodeEncodeError:
        fallback = message.encode("ascii", errors="ignore").decode("ascii")
        print(fallback.strip() or "All validations passed")


def load_and_prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    log_progress("Loading Layer 1 master dataset")
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH.resolve()}")

    df = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    log_progress("Creating HMM feature aliases and handling missing values")
    for display_name, source_name in DISPLAY_TO_SOURCE.items():
        if source_name not in df.columns:
            raise KeyError(f"Required source column '{source_name}' is missing from Layer 1 output.")
        df[display_name] = df[source_name]

    hmm_features = list(HMM_FEATURE_ALIASES.keys())
    working = df.copy()
    working[hmm_features] = working[hmm_features].ffill()
    working = working.dropna(subset=hmm_features).copy()

    if working.empty:
        raise ValueError("No rows remain after HMM feature forward-fill and dropna.")

    return df, working, hmm_features


def compute_bic(model: GaussianHMM, scaled_features: np.ndarray) -> float:
    n_samples, n_features = scaled_features.shape
    log_likelihood = model.score(scaled_features)
    n_params = model.n_components * (n_features + n_features**2 + 1)
    bic = -2 * log_likelihood * n_samples + n_params * math.log(n_samples)
    return bic


def compute_balance_score(hidden_states: np.ndarray, n_components: int) -> tuple[float, Dict[int, float]]:
    counts = pd.Series(hidden_states).value_counts(normalize=True).reindex(range(n_components), fill_value=0.0)
    proportions = counts.to_dict()
    probabilities = counts.to_numpy(dtype=float)

    non_zero_probabilities = probabilities[probabilities > 0]
    if non_zero_probabilities.size == 0:
        return 0.0, proportions

    entropy = -np.sum(non_zero_probabilities * np.log(non_zero_probabilities))
    normalized_entropy = float(entropy / np.log(n_components))
    return normalized_entropy, proportions


def fit_candidate_models(working: pd.DataFrame, hmm_features: List[str]) -> List[ModelRun]:
    log_progress("Training Gaussian HMM candidates across component counts and random seeds")

    X = working[hmm_features].to_numpy(dtype=float)
    candidates: List[ModelRun] = []

    for n_components in COMPONENT_OPTIONS:
        for random_state in RANDOM_STATES:
            log_progress(f"Fitting HMM with n_components={n_components}, random_state={random_state}")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = GaussianHMM(
                n_components=n_components,
                covariance_type="full",
                n_iter=500,
                tol=1e-4,
                random_state=random_state,
            )
            with (
                warnings.catch_warnings(),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                warnings.simplefilter("ignore")
                model.fit(X_scaled)
                hidden_states = model.predict(X_scaled)
                log_likelihood = model.score(X_scaled)
            bic = compute_bic(model, X_scaled)
            diagonal_mean = float(np.mean(np.diag(model.transmat_)))
            balance_score, regime_proportions = compute_balance_score(hidden_states, n_components)

            candidates.append(
                ModelRun(
                    n_components=n_components,
                    random_state=random_state,
                    model=model,
                    scaler=scaler,
                    scaled_features=X_scaled,
                    hidden_states=hidden_states,
                    bic=bic,
                    log_likelihood=log_likelihood,
                    diagonal_mean=diagonal_mean,
                    balance_score=balance_score,
                    regime_proportions=regime_proportions,
                    working_data=working.copy(),
                )
            )

    return candidates


def select_best_model(candidates: List[ModelRun]) -> ModelRun:
    log_progress("Selecting best HMM run using balance-first screening and BIC tie-breaks")

    eligible = [run for run in candidates if run.diagonal_mean > 0.75]
    pool = eligible if eligible else candidates

    if not eligible:
        log_progress("No candidate satisfied diagonal > 0.75, falling back to all models and selecting by balance/BIC.")

    best_run = sorted(
        pool,
        key=lambda run: (
            -run.balance_score,
            run.bic,
            -run.diagonal_mean,
            -min(run.regime_proportions.values()),
        ),
    )[0]

    print("\nCandidate model summary:")
    summary_rows = []
    for run in sorted(candidates, key=lambda candidate: (candidate.n_components, candidate.random_state)):
        proportions_display = ", ".join(
            f"{state}:{share:.1%}" for state, share in run.regime_proportions.items()
        )
        summary_rows.append(
            {
                "n_components": run.n_components,
                "random_state": run.random_state,
                "bic": round(run.bic, 2),
                "diag_mean": round(run.diagonal_mean, 4),
                "balance_score": round(run.balance_score, 4),
                "regime_distribution": proportions_display,
            }
        )
    print(pd.DataFrame(summary_rows).to_string(index=False))

    print(
        f"\nSelected model: n_components={best_run.n_components}, "
        f"random_state={best_run.random_state}, BIC={best_run.bic:.2f}, "
        f"diag_mean={best_run.diagonal_mean:.4f}, balance_score={best_run.balance_score:.4f}"
    )
    return best_run


def build_regime_statistics(data: pd.DataFrame) -> pd.DataFrame:
    stats = (
        data.groupby("regime", dropna=False)
        .agg(
            avg_return=("nifty_return", "mean"),
            std_return=("nifty_return", "std"),
            avg_vix=("india_vix", "mean"),
            avg_spread=("bank_spread", "mean"),
            count=("regime", "size"),
        )
        .reset_index()
    )
    stats["sharpe"] = np.where(
        stats["std_return"] > 0,
        (stats["avg_return"] * 252) / (stats["std_return"] * np.sqrt(252)),
        np.nan,
    )
    return stats


def assign_regime_labels(stats: pd.DataFrame) -> Dict[int, str]:
    labels: Dict[int, str] = {}
    remaining_states = set(stats["regime"].tolist())

    vix_sorted = stats.sort_values("avg_vix", ascending=False)
    crisis_state = int(vix_sorted.iloc[0]["regime"])
    labels[crisis_state] = "Crisis"
    remaining_states.discard(crisis_state)

    if len(vix_sorted) > 1:
        late_cycle_state = int(vix_sorted.iloc[1]["regime"])
        labels[late_cycle_state] = "Late_Cycle"
        remaining_states.discard(late_cycle_state)

    expansion_candidates = stats[(stats["avg_return"] > 0) & (stats["regime"].isin(remaining_states))]
    if not expansion_candidates.empty:
        expansion_state = int(expansion_candidates.sort_values("avg_vix", ascending=True).iloc[0]["regime"])
        labels[expansion_state] = "Expansion"
        remaining_states.discard(expansion_state)
    elif remaining_states:
        lowest_vix_remaining = int(
            stats[stats["regime"].isin(remaining_states)].sort_values("avg_vix", ascending=True).iloc[0]["regime"]
        )
        labels[lowest_vix_remaining] = "Expansion"
        remaining_states.discard(lowest_vix_remaining)

    for regime in sorted(remaining_states):
        labels[int(regime)] = "Recovery"

    return labels


def merge_predictions_back(
    base_data: pd.DataFrame,
    best_run: ModelRun,
    label_map: Dict[int, str],
) -> pd.DataFrame:
    log_progress("Merging best-model regime predictions back to the full dataset")

    predicted = best_run.working_data.copy()
    predicted["regime"] = best_run.hidden_states.astype(int)
    predicted["regime_label"] = predicted["regime"].map(label_map)

    merged = base_data.merge(
        predicted[["date", "regime", "regime_label"]],
        on="date",
        how="left",
    )
    return merged


def run_validations(data: pd.DataFrame, model: GaussianHMM, stats: pd.DataFrame) -> List[str]:
    log_progress("Running regime validation checks")

    failed_checks: List[str] = []
    labeled = data.dropna(subset=["regime", "regime_label"]).copy()

    regime_distribution = labeled["regime_label"].value_counts(normalize=True).sort_index()
    regime_distribution_check = bool((regime_distribution > 0.05).all())
    print(f"\nValidation 1 - each regime > 5% of data: {regime_distribution_check}")
    if not regime_distribution_check:
        failed_checks.append("Each regime has > 5% of data")

    diagonal_mean = float(np.mean(np.diag(model.transmat_)))
    diagonal_check = diagonal_mean > 0.75
    print(f"Validation 2 - transition diagonal mean > 0.75: {diagonal_check} (value={diagonal_mean:.4f})")
    if not diagonal_check:
        failed_checks.append("Transition matrix diagonal mean > 0.75")

    grouped_vix = [group["india_vix"].to_numpy() for _, group in labeled.groupby("regime") if len(group) > 1]
    if len(grouped_vix) >= 2:
        anova_result = f_oneway(*grouped_vix)
        print(
            f"Validation 3 - ANOVA on india_vix across regimes: "
            f"F-stat={anova_result.statistic:.4f}, p-value={anova_result.pvalue:.6g}"
        )
    else:
        anova_result = None
        print("Validation 3 - ANOVA on india_vix across regimes: insufficient distinct regimes for test")
        failed_checks.append("ANOVA on india_vix across regimes")

    crisis_window = labeled.loc[
        (labeled["date"] >= pd.Timestamp("2020-03-01")) & (labeled["date"] <= pd.Timestamp("2020-04-30"))
    ]
    if crisis_window.empty:
        crisis_check = False
        crisis_mode_label = "No data"
    else:
        crisis_mode_label = crisis_window["regime_label"].mode().iat[0]
        crisis_check = crisis_mode_label == "Crisis"
    print(
        f"Validation 4 - dominant regime between 2020-03-01 and 2020-04-30 is Crisis: "
        f"{crisis_check} (observed={crisis_mode_label})"
    )
    if not crisis_check:
        failed_checks.append("2020-03-01 to 2020-04-30 dominant regime should be Crisis")

    distribution_table = stats.copy()
    distribution_table["regime_label"] = distribution_table["regime"].map(
        labeled[["regime", "regime_label"]].drop_duplicates().set_index("regime")["regime_label"]
    )
    distribution_table["pct"] = distribution_table["count"] / distribution_table["count"].sum() * 100
    distribution_table = distribution_table[
        ["regime_label", "count", "pct", "avg_vix", "avg_return", "sharpe"]
    ].sort_values(["avg_vix", "avg_return"], ascending=[False, False])

    print("\nRegime distribution table:")
    print(distribution_table.round(6).to_string(index=False))

    return failed_checks


def save_artifacts(best_run: ModelRun, labeled_data: pd.DataFrame) -> None:
    log_progress("Saving labeled data, HMM model, and scaler artifacts")
    OUTPUT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    labeled_data.to_csv(OUTPUT_DATA_PATH, index=False)

    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(best_run.model, model_file)

    with SCALER_PATH.open("wb") as scaler_file:
        pickle.dump(best_run.scaler, scaler_file)


def plot_regime_timeline(data: pd.DataFrame) -> None:
    log_progress("Saving regime timeline plot")
    plot_df = data.dropna(subset=["regime_label", "nifty_close"]).copy()
    if plot_df.empty:
        raise ValueError("No labeled rows available for regime timeline plot.")

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(plot_df["date"], plot_df["nifty_close"], color="#1f2933", linewidth=1.5, label="Nifty Close")

    label_order = [label for label in REGIME_COLOR_MAP if label in plot_df["regime_label"].unique()]
    regime_codes = plot_df["regime_label"].astype("category")
    plot_df["regime_change"] = plot_df["regime_label"].ne(plot_df["regime_label"].shift())
    segment_ids = plot_df["regime_change"].cumsum()

    for _, segment in plot_df.groupby(segment_ids):
        label = segment["regime_label"].iat[0]
        color = REGIME_COLOR_MAP.get(label, "#95a5a6")
        ax.axvspan(segment["date"].iloc[0], segment["date"].iloc[-1], color=color, alpha=0.18)

    events = {
        "2020-03-20": "COVID",
        "2022-06-01": "Rate hikes",
    }
    for event_date, label in events.items():
        timestamp = pd.Timestamp(event_date)
        ax.axvline(timestamp, color="black", linestyle="--", linewidth=1.0)
        ax.text(timestamp, ax.get_ylim()[1], label, rotation=90, va="top", ha="right", fontsize=9)

    legend_handles = [plt.Line2D([0], [0], color="#1f2933", linewidth=1.5, label="Nifty Close")]
    legend_handles.extend(
        [
            plt.Rectangle((0, 0), 1, 1, color=REGIME_COLOR_MAP[label], alpha=0.35, label=label)
            for label in label_order
        ]
    )

    ax.legend(handles=legend_handles, loc="upper left", frameon=True)
    ax.set_title("Nifty Regime Timeline")
    ax.set_xlabel("Date")
    ax.set_ylabel("Nifty Close")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(VISUALIZATION_DIR / "regime_timeline.png", dpi=150)
    plt.close(fig)


def plot_regime_boxplot(data: pd.DataFrame) -> None:
    log_progress("Saving regime boxplot panel")
    plot_df = data.dropna(subset=["regime_label"]).copy()
    label_order = [label for label in REGIME_COLOR_MAP if label in plot_df["regime_label"].unique()]

    plot_columns = [
        ("india_vix", "VIX"),
        ("nifty_return", "Nifty Return"),
        ("bank_spread", "Bank Spread"),
        ("gold_nifty_corr_30d", "Gold-Nifty Corr"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    palette = {label: REGIME_COLOR_MAP[label] for label in label_order}

    for axis, (column, title) in zip(axes, plot_columns):
        sns.boxplot(
            data=plot_df,
            x="regime_label",
            y=column,
            hue="regime_label",
            order=label_order,
            palette=palette,
            dodge=False,
            legend=False,
            ax=axis,
        )
        axis.set_title(title)
        axis.set_xlabel("Regime")
        axis.set_ylabel(title)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(alpha=0.15)

    fig.tight_layout()
    fig.savefig(VISUALIZATION_DIR / "regime_boxplot.png", dpi=150)
    plt.close(fig)


def plot_transition_heatmap(model: GaussianHMM, label_map: Dict[int, str]) -> None:
    log_progress("Saving transition probability heatmap")
    state_order = list(range(model.n_components))
    labels = [f"{state}: {label_map.get(state, 'Unknown')}" for state in state_order]
    transition_df = pd.DataFrame(model.transmat_, index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(transition_df, annot=True, fmt=".2f", cmap="YlOrRd", cbar=True, ax=ax)
    ax.set_title("Regime Transition Probability Matrix")
    ax.set_xlabel("To Regime")
    ax.set_ylabel("From Regime")
    fig.tight_layout()
    fig.savefig(VISUALIZATION_DIR / "transition_heatmap.png", dpi=150)
    plt.close(fig)


def main() -> None:
    log_progress("Initializing Layer 2 regime detection pipeline")
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001 - keep script portable across consoles
            pass
    sns.set_theme(style="whitegrid")
    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

    warnings.filterwarnings("ignore", message=".*KMeans is known to have a memory leak on Windows with MKL.*")
    warnings.filterwarnings("ignore", message=".*Passing `palette` without assigning `hue` is deprecated.*")
    warnings.filterwarnings("ignore", message=".*Model is not converging.*")

    base_data, working_data, hmm_features = load_and_prepare_data()
    candidates = fit_candidate_models(working_data, hmm_features)
    best_run = select_best_model(candidates)

    labeled_working = best_run.working_data.copy()
    labeled_working["regime"] = best_run.hidden_states.astype(int)
    labeled_working["bank_spread"] = labeled_working["bank_spread"]
    labeled_working["nifty_vol"] = labeled_working["nifty_vol"]

    regime_stats = build_regime_statistics(labeled_working)
    label_map = assign_regime_labels(regime_stats)
    labeled_data = merge_predictions_back(base_data, best_run, label_map)

    save_artifacts(best_run, labeled_data)

    plot_regime_timeline(labeled_data)
    plot_regime_boxplot(labeled_data)
    plot_transition_heatmap(best_run.model, label_map)

    validation_data = labeled_data.dropna(subset=["regime", "regime_label"]).copy()
    validation_data["regime"] = validation_data["regime"].astype(int)
    validation_data["bank_spread"] = validation_data["bank_spread"]
    failed_checks = run_validations(validation_data, best_run.model, regime_stats)

    if not failed_checks:
        safe_print("\n✅ All validations passed")
    else:
        print("\nValidation failures:")
        for failed_check in failed_checks:
            print(f"- {failed_check}")

    print(f"\nSaved labeled dataset to: {OUTPUT_DATA_PATH.resolve()}")
    print(f"Saved HMM model to: {MODEL_PATH.resolve()}")
    print(f"Saved scaler to: {SCALER_PATH.resolve()}")
    print(f"Saved visualizations to: {VISUALIZATION_DIR.resolve()}")


if __name__ == "__main__":
    main()
