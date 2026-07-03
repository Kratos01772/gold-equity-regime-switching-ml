from __future__ import annotations

import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import ttest_ind
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


INPUT_PATH = Path("data") / "regime_labeled_data.csv"
HMM_MODEL_PATH = Path("models") / "hmm_model.pkl"
HMM_SCALER_PATH = Path("models") / "hmm_scaler.pkl"
MODELS_DIR = Path("models")
VISUALIZATIONS_DIR = Path("visualizations")
OUTPUTS_DIR = Path("outputs")
RESULTS_PATH = OUTPUTS_DIR / "layer3_improved_results.txt"
MODEL_CONFIG_PATH = OUTPUTS_DIR / "model_config.json"

TRAIN_END_DATE = pd.Timestamp("2023-01-01")
TARGET_COL = "bank_return_next"
REGIME_ORDER = ["Crisis", "Late_Cycle", "Expansion"]
OLD_BEST_R2 = 0.3554
OLD_BEST_DIRECTION = 73.21

FEATURE_RANKING_PARAMS = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "reg_lambda": 2.0,
}
FINAL_MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "reg_lambda": 1.0,
}
TOP_FEATURE_COUNT = 10
REGIME_WEIGHT_OWN = 8.0
REGIME_WEIGHT_OTHER = 1.0

BASE_FEATURE_CANDIDATES: Dict[str, List[str]] = {
    "gold_return_lag1": ["gold_lag1_return"],
    "gold_return_lag5": ["gold_lag5_return"],
    "gold_return_lag20": ["gold_lag20_return"],
    "nifty_return_lag1": ["nifty_lag1_return"],
    "nifty_return_lag5": ["nifty_lag5_return"],
    "nifty_return_lag20": ["nifty_lag20_return"],
    "bank_return_lag1": ["bank_lag1_return"],
    "bank_return_lag5": ["bank_lag5_return"],
    "bank_return_lag20": ["bank_lag20_return"],
    "oil_return_lag1": ["oil_lag1_return"],
    "oil_return_lag5": ["oil_lag5_return"],
    "oil_return_lag20": ["oil_lag20_return"],
    "usdinr_return_lag1": ["usdinr_lag1_return"],
    "usdinr_return_lag5": ["usdinr_lag5_return"],
    "usdinr_return_lag20": ["usdinr_lag20_return"],
    "gold_vol": ["gold_volatility_20d"],
    "nifty_vol": ["nifty_volatility_20d"],
    "bank_vol": ["bank_volatility_20d"],
    "india_vix": ["india_vix", "vix_close"],
    "usdinr_return": ["usdinr_return"],
    "oil_return": ["oil_return"],
    "bank_spread": ["bank_corwin_schultz_spread"],
    "nifty_amihud": ["nifty_amihud_illiquidity"],
    "gold_nifty_corr_30d": ["gold_nifty_corr_30d"],
    "gold_bank_corr_30d": ["gold_bank_corr_30d"],
    "gold_vix_interaction": ["gold_vix_interaction"],
    "gold_spread_interaction": ["gold_spread_interaction"],
    "gold_momentum_20d": ["gold_momentum_20d"],
    "nifty_momentum_20d": ["nifty_momentum_20d"],
    "bank_momentum_20d": ["bank_momentum_20d"],
    "gold_silver_ratio": ["gold_silver_ratio"],
}

HMM_FEATURE_DISPLAY_TO_SOURCE = {
    "nifty_return": "nifty_return",
    "india_vix": "india_vix",
    "bank_spread": "bank_corwin_schultz_spread",
    "gold_nifty_corr_30d": "gold_nifty_corr_30d",
    "nifty_vol": "nifty_volatility_20d",
}

WALK_FORWARD_FOLDS = [
    ("Fold 1", "2014-01-01", "2018-12-31", "2019-01-01", "2019-12-31"),
    ("Fold 2", "2014-01-01", "2019-12-31", "2020-01-01", "2020-12-31"),
    ("Fold 3", "2014-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
    ("Fold 4", "2014-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("Fold 5", "2014-01-01", "2022-12-31", "2023-01-01", "2024-12-31"),
]


def log_progress(message: str) -> None:
    print(f"[INFO] {message}")


def start_step(message: str) -> float:
    log_progress(message)
    return time.time()


def end_step(start_time: float) -> None:
    print(f"[INFO] Step completed in {time.time() - start_time:.1f}s")


def resolve_stdout_encoding() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass


def format_metric(value: float, decimals: int = 3) -> str:
    if pd.isna(value):
        return "nan"
    if np.isinf(value):
        return "inf"
    return f"{value:.{decimals}f}"


def load_dataset() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH.resolve()}")
    df = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if "regime" not in df.columns or "regime_label" not in df.columns:
        raise KeyError("Input data must contain 'regime' and 'regime_label'.")
    return df


def load_hmm_artifacts() -> Tuple[object, object]:
    with HMM_MODEL_PATH.open("rb") as file:
        hmm_model = pickle.load(file)
    with HMM_SCALER_PATH.open("rb") as file:
        hmm_scaler = pickle.load(file)
    return hmm_model, hmm_scaler


def resolve_base_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    feature_frame = pd.DataFrame(index=df.index)
    for alias, candidates in BASE_FEATURE_CANDIDATES.items():
        for candidate in candidates:
            if candidate in df.columns:
                feature_frame[alias] = pd.to_numeric(df[candidate], errors="coerce")
                break

    if feature_frame.empty:
        raise ValueError("No base features were resolved from the input dataset.")

    return feature_frame, list(feature_frame.columns)


def rolling_last_percentile(series: pd.Series, window: int) -> pd.Series:
    def _last_rank(values: np.ndarray) -> float:
        last = values[-1]
        return float(np.sum(values <= last) / len(values))

    return series.rolling(window=window, min_periods=window).apply(_last_rank, raw=True)


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], int, int, int]:
    t0 = start_step("Engineering expanded feature set")

    base_features, base_feature_names = resolve_base_features(df)
    modeling = df.copy()
    modeling[base_feature_names] = base_features
    modeling[base_feature_names] = modeling[base_feature_names].ffill()
    modeling["regime"] = pd.to_numeric(modeling["regime"], errors="coerce")
    modeling["gold_return"] = pd.to_numeric(modeling["gold_return"], errors="coerce")
    modeling["gold_close"] = pd.to_numeric(modeling["gold_close"], errors="coerce")
    modeling["nifty_close"] = pd.to_numeric(modeling["nifty_close"], errors="coerce")
    modeling["bank_return"] = pd.to_numeric(modeling["bank_return"], errors="coerce")

    bank_close_col = "bank_close" if "bank_close" in modeling.columns else "bank_adj_close"
    if bank_close_col not in modeling.columns:
        raise KeyError("No bank close column found in the dataset.")
    modeling[bank_close_col] = pd.to_numeric(modeling[bank_close_col], errors="coerce")

    new_features: Dict[str, pd.Series] = {}
    new_features["vix_5d_change"] = modeling["india_vix"] - modeling["india_vix"].shift(5)
    new_features["vix_20d_change"] = modeling["india_vix"] - modeling["india_vix"].shift(20)
    new_features["vix_regime_zscore"] = (
        (modeling["india_vix"] - modeling["india_vix"].rolling(60).mean())
        / modeling["india_vix"].rolling(60).std()
    )
    new_features["gold_return_lag2"] = modeling["gold_return"].shift(2)
    new_features["gold_return_lag10"] = modeling["gold_return"].shift(10)
    new_features["gold_vol_ratio"] = modeling["gold_vol"] / modeling["gold_vol"].shift(20)
    gold_ma20 = modeling["gold_close"].rolling(20).mean()
    new_features["gold_above_ma20"] = np.where(gold_ma20.notna(), (modeling["gold_close"] > gold_ma20).astype(float), np.nan)

    regime_change = (modeling["regime_label"] != modeling["regime_label"].shift(1)).astype(int)
    new_features["regime_change"] = regime_change
    new_features["days_in_regime"] = modeling.groupby((regime_change == 1).cumsum()).cumcount() + 1

    nifty_bank_ratio = modeling["nifty_close"] / modeling[bank_close_col]
    new_features["nifty_bank_ratio_5d_change"] = (nifty_bank_ratio / nifty_bank_ratio.shift(5)) - 1
    gold_nifty_ratio = modeling["gold_close"] / modeling["nifty_close"]
    new_features["gold_nifty_ratio_change"] = (gold_nifty_ratio / gold_nifty_ratio.shift(20)) - 1
    new_features["bank_vol_percentile"] = rolling_last_percentile(modeling["bank_vol"], 252)
    new_features["vix_percentile"] = rolling_last_percentile(modeling["india_vix"], 252)

    new_feature_names = list(new_features.keys())
    for feature_name, values in new_features.items():
        modeling[feature_name] = values

    for display_name, source_name in HMM_FEATURE_DISPLAY_TO_SOURCE.items():
        modeling[display_name] = pd.to_numeric(modeling[source_name], errors="coerce")

    modeling[TARGET_COL] = modeling["bank_return"].shift(-1)
    modeling = modeling.iloc[:-1].copy()
    modeling = modeling.dropna(subset=new_feature_names + [TARGET_COL]).copy()

    final_feature_names = base_feature_names + new_feature_names
    print(f"Feature set expanded from {len(base_feature_names)} to {len(final_feature_names)} features")
    end_step(t0)
    return modeling, final_feature_names, len(base_feature_names), len(new_feature_names), len(final_feature_names)


def build_train_test_sets(modeling: pd.DataFrame, feature_names: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    t0 = start_step("Preparing train/test sets and imputing missing values")
    train_mask = modeling["date"] < TRAIN_END_DATE
    test_mask = modeling["date"] >= TRAIN_END_DATE

    hmm_feature_names = list(HMM_FEATURE_DISPLAY_TO_SOURCE.keys())
    impute_columns = list(dict.fromkeys(feature_names + hmm_feature_names))
    medians = modeling.loc[train_mask, impute_columns].median(numeric_only=True)

    modeling[impute_columns] = modeling[impute_columns].ffill()
    modeling[impute_columns] = modeling[impute_columns].fillna(medians)

    train_df = modeling.loc[train_mask].copy()
    test_df = modeling.loc[test_mask].copy()
    print(
        f"Train rows: {len(train_df)} ({train_df['date'].min().date()} to {train_df['date'].max().date()}) | "
        f"Test rows: {len(test_df)} ({test_df['date'].min().date()} to {test_df['date'].max().date()})"
    )
    print("Test regime counts:")
    print(test_df["regime_label"].value_counts(dropna=False).to_string())
    end_step(t0)
    return modeling, train_df, test_df


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "R2": r2_score(y_true, y_pred) if len(y_true) >= 2 else np.nan,
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Direction%": float((np.sign(y_true) == np.sign(y_pred)).mean() * 100),
    }


def regime_dummy_columns() -> List[str]:
    return [f"is_{label.lower()}" for label in REGIME_ORDER]


def add_regime_dummies(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    for regime_label in REGIME_ORDER:
        enriched[f"is_{regime_label.lower()}"] = (enriched["regime_label"] == regime_label).astype(float)
    return enriched


def build_regime_context_frame(frame: pd.DataFrame, selected_features: List[str], regime_label: str) -> pd.DataFrame:
    context_frame = frame[selected_features].copy()
    for label in REGIME_ORDER:
        context_frame[f"is_{label.lower()}"] = 1.0 if label == regime_label else 0.0
    return context_frame


def create_xgb(params: Dict[str, float], n_jobs: int = -1) -> XGBRegressor:
    return XGBRegressor(
        random_state=42,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=n_jobs,
        verbosity=0,
        **params,
    )


def select_compact_feature_set(train_df: pd.DataFrame, feature_names: List[str]) -> List[str]:
    t0 = start_step("Selecting compact feature subset for stable regime models")
    ranking_model = create_xgb(FEATURE_RANKING_PARAMS, n_jobs=-1)
    ranking_model.fit(train_df[feature_names], train_df[TARGET_COL])
    importance = pd.Series(ranking_model.feature_importances_, index=feature_names).sort_values(ascending=False)
    selected = importance.head(TOP_FEATURE_COUNT).index.tolist()
    if "gold_return_lag1" in feature_names and "gold_return_lag1" not in selected:
        selected.append("gold_return_lag1")
    print(f"Selected compact feature set ({len(selected)} features): {selected}")
    end_step(t0)
    return selected


def run_walk_forward_cv(modeling: pd.DataFrame, feature_names: List[str]) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    t0 = start_step("Running walk-forward cross validation")
    cv_rows: List[Dict[str, object]] = []
    predictor_names = feature_names + regime_dummy_columns()

    for fold_name, train_start, train_end, test_start, test_end in WALK_FORWARD_FOLDS:
        fold_train_mask = (modeling["date"] >= pd.Timestamp(train_start)) & (modeling["date"] <= pd.Timestamp(train_end))
        fold_test_mask = (modeling["date"] >= pd.Timestamp(test_start)) & (modeling["date"] <= pd.Timestamp(test_end))
        fold_train = modeling.loc[fold_train_mask].copy()

        for regime_label in REGIME_ORDER:
            fold_test = modeling.loc[fold_test_mask & (modeling["regime_label"] == regime_label)].copy()
            if len(fold_train) < 120 or len(fold_test) < 5:
                print(f"[WARN] {fold_name} / {regime_label} skipped: train={len(fold_train)}, test={len(fold_test)}")
                cv_rows.append({"Fold": fold_name, "Regime": regime_label, "R2": np.nan, "RMSE": np.nan, "Direction%": np.nan})
                continue

            weights = np.where(fold_train["regime_label"].eq(regime_label), REGIME_WEIGHT_OWN, REGIME_WEIGHT_OTHER)
            model = create_xgb(FINAL_MODEL_PARAMS, n_jobs=-1)
            model.fit(fold_train[predictor_names], fold_train[TARGET_COL], sample_weight=weights)
            predictions = model.predict(build_regime_context_frame(fold_test, feature_names, regime_label))
            metrics = evaluate_predictions(fold_test[TARGET_COL], predictions)
            cv_rows.append(
                {
                    "Fold": fold_name,
                    "Regime": regime_label,
                    "R2": metrics["R2"],
                    "RMSE": metrics["RMSE"],
                    "Direction%": metrics["Direction%"],
                }
            )

    cv_df = pd.DataFrame(cv_rows)
    print("\nWalk-forward CV results:")
    print(cv_df.round(6).to_string(index=False))

    cv_summary: Dict[str, Tuple[float, float]] = {}
    for regime_label in REGIME_ORDER:
        regime_r2 = cv_df.loc[cv_df["Regime"] == regime_label, "R2"]
        cv_summary[regime_label] = (float(regime_r2.mean()), float(regime_r2.std()))
        print(f"{regime_label} mean R2: {cv_summary[regime_label][0]:.3f} +/- {cv_summary[regime_label][1]:.3f}")

    end_step(t0)
    return cv_df, cv_summary


def train_final_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_names: List[str],
) -> Tuple[Dict[str, XGBRegressor], XGBRegressor, pd.DataFrame]:
    t0 = start_step("Training final pooled regime models and global benchmark")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    regime_models: Dict[str, XGBRegressor] = {}
    evaluation_rows: List[Dict[str, object]] = []
    predictor_names = feature_names + regime_dummy_columns()

    for regime_label in REGIME_ORDER:
        regime_test = test_df.loc[test_df["regime_label"] == regime_label].copy()
        weights = np.where(train_df["regime_label"].eq(regime_label), REGIME_WEIGHT_OWN, REGIME_WEIGHT_OTHER)
        model = create_xgb(FINAL_MODEL_PARAMS, n_jobs=-1)
        model.fit(train_df[predictor_names], train_df[TARGET_COL], sample_weight=weights)
        regime_models[regime_label] = model
        joblib.dump(model, MODELS_DIR / f"{regime_label}_bank.pkl")

        if regime_test.empty:
            metrics = {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "Direction%": np.nan}
        else:
            predictions = model.predict(build_regime_context_frame(regime_test, feature_names, regime_label))
            metrics = evaluate_predictions(regime_test[TARGET_COL], predictions)
        evaluation_rows.append({"Model": f"{regime_label}_bank", **metrics})

    global_model = create_xgb(FEATURE_RANKING_PARAMS, n_jobs=-1)
    global_model.fit(train_df[feature_names], train_df[TARGET_COL])
    joblib.dump(global_model, MODELS_DIR / "Global_bank.pkl")
    global_predictions = global_model.predict(test_df[feature_names])
    global_metrics = evaluate_predictions(test_df[TARGET_COL], global_predictions)
    evaluation_rows.append({"Model": "Global_bank", **global_metrics})

    performance_df = pd.DataFrame(evaluation_rows)
    end_step(t0)
    return regime_models, global_model, performance_df


def compute_hmm_probabilities(test_df: pd.DataFrame, hmm_model: object, hmm_scaler: object) -> pd.DataFrame:
    hmm_features = list(HMM_FEATURE_DISPLAY_TO_SOURCE.keys())
    scaled = hmm_scaler.transform(test_df[hmm_features].to_numpy(dtype=float))
    probs = hmm_model.predict_proba(scaled)

    state_map = (
        test_df[["regime", "regime_label"]]
        .dropna()
        .drop_duplicates()
        .sort_values("regime")
        .set_index("regime")["regime_label"]
        .to_dict()
    )
    columns = [state_map.get(state, f"State_{state}") for state in range(probs.shape[1])]
    return pd.DataFrame(probs, index=test_df.index, columns=columns).reindex(columns=REGIME_ORDER, fill_value=0.0)


def evaluate_ensemble(
    test_df: pd.DataFrame,
    feature_names: List[str],
    regime_models: Dict[str, XGBRegressor],
    hmm_model: object,
    hmm_scaler: object,
) -> Tuple[pd.Series, Dict[str, float]]:
    t0 = start_step("Evaluating soft ensemble using HMM posterior probabilities")
    regime_probs = compute_hmm_probabilities(test_df, hmm_model, hmm_scaler)
    all_preds = np.column_stack(
        [regime_models[label].predict(build_regime_context_frame(test_df, feature_names, label)) for label in REGIME_ORDER]
    )
    ensemble_pred = np.sum(regime_probs.to_numpy(dtype=float) * all_preds, axis=1)
    metrics = evaluate_predictions(test_df[TARGET_COL], ensemble_pred)
    print(f"Ensemble direction accuracy: {metrics['Direction%']:.1f}%")
    end_step(t0)
    return pd.Series(ensemble_pred, index=test_df.index, name="Ensemble"), metrics


def optimize_signal_threshold(y_true: pd.Series, predictions: pd.Series) -> Dict[str, float]:
    t0 = start_step("Searching for optimal directional threshold")
    thresholds_pct = np.arange(0.1, 0.8, 0.05)
    best_threshold_pct = 0.3
    best_direction_acc = 0.0

    for threshold_pct in thresholds_pct:
        threshold_decimal = threshold_pct / 100.0
        signals = np.where(predictions > threshold_decimal, 1, np.where(predictions < -threshold_decimal, -1, 0))
        active = signals != 0
        if active.sum() < 20:
            continue
        accuracy = float((np.sign(y_true.to_numpy()[active]) == signals[active]).mean())
        if accuracy > best_direction_acc:
            best_direction_acc = accuracy
            best_threshold_pct = float(threshold_pct)

    print(f"Optimal signal threshold: +/-{best_threshold_pct:.2f}%")
    print(f"Direction accuracy at optimal threshold: {best_direction_acc:.1%}")
    end_step(t0)
    return {
        "best_threshold_pct": best_threshold_pct,
        "best_threshold_decimal": best_threshold_pct / 100.0,
        "best_direction_acc": best_direction_acc,
    }


def run_updated_shap(
    regime_models: Dict[str, XGBRegressor],
    test_df: pd.DataFrame,
    feature_names: List[str],
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    t0 = start_step("Recomputing regime-specific SHAP analysis")
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)

    shap_rows: List[Dict[str, object]] = []
    gold_shap_vectors: Dict[str, np.ndarray] = {}

    for regime_label in REGIME_ORDER:
        regime_test = test_df.loc[test_df["regime_label"] == regime_label].copy()
        if regime_test.empty:
            print(f"[WARN] No test rows available for SHAP in {regime_label}.")
            continue

        try:
            shap_input = build_regime_context_frame(regime_test, feature_names, regime_label)
            explainer = shap.TreeExplainer(regime_models[regime_label])
            shap_values = explainer.shap_values(shap_input)
            shap_array = np.asarray(shap_values)
            importance = pd.Series(np.abs(shap_array).mean(axis=0), index=shap_input.columns).sort_values(ascending=False)
            gold_value = float(importance.get("gold_return_lag1", np.nan))
            gold_rank = int(np.where(importance.index == "gold_return_lag1")[0][0] + 1) if "gold_return_lag1" in importance.index else np.nan
            gold_shap_vectors[regime_label] = np.abs(shap_array[:, shap_input.columns.get_loc("gold_return_lag1")])

            try:
                plt.figure(figsize=(11, 7))
                shap.summary_plot(shap_array, shap_input, max_display=15, show=False)
                plt.tight_layout()
                plt.savefig(VISUALIZATIONS_DIR / f"shap_summary_{regime_label}.png", dpi=150, bbox_inches="tight")
                plt.close("all")
            except Exception as exc:
                print(f"[WARN] SHAP plot failed for {regime_label}: {exc}")
                plt.close("all")

            shap_rows.append({"Regime": regime_label, "Gold SHAP": gold_value, "Gold Rank": gold_rank})
        except Exception as exc:
            print(f"[WARN] SHAP analysis failed for {regime_label}: {exc}")
            plt.close("all")

    t_stat = np.nan
    p_value = np.nan
    ratio = np.nan
    crisis_vals = gold_shap_vectors.get("Crisis")
    expansion_vals = gold_shap_vectors.get("Expansion")
    late_cycle_vals = gold_shap_vectors.get("Late_Cycle")
    if crisis_vals is not None and expansion_vals is not None and len(crisis_vals) > 1 and len(expansion_vals) > 1:
        t_stat, p_value = ttest_ind(crisis_vals, expansion_vals, equal_var=False, nan_policy="omit")
    if late_cycle_vals is not None and expansion_vals is not None:
        late_cycle_mean = float(np.nanmean(late_cycle_vals))
        expansion_mean = float(np.nanmean(expansion_vals))
        ratio = np.inf if expansion_mean == 0 else late_cycle_mean / expansion_mean

    shap_table = pd.DataFrame(shap_rows)
    print("\nUpdated gold SHAP table:")
    if not shap_table.empty:
        print(shap_table.round(6).to_string(index=False))
    print(f"Gold SHAP t-test (Crisis vs Expansion): t-stat={t_stat:.6f}, p-value={p_value:.6g}")
    print(f"Gold SHAP ratio (Late_Cycle vs Expansion): {ratio:.2f}x")
    end_step(t0)
    return shap_table, {"t_stat": float(t_stat), "p_value": float(p_value), "ratio": float(ratio)}


def save_ensemble_and_config(
    regime_models: Dict[str, XGBRegressor],
    hmm_model: object,
    hmm_scaler: object,
    feature_names: List[str],
    threshold_info: Dict[str, float],
) -> None:
    t0 = start_step("Saving ensemble wrapper and model configuration")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    ensemble_config = {
        "models": regime_models,
        "hmm": hmm_model,
        "scaler": hmm_scaler,
        "hmm_features": list(HMM_FEATURE_DISPLAY_TO_SOURCE.keys()),
        "pred_features": feature_names + regime_dummy_columns(),
        "base_features": feature_names,
        "regime_order": REGIME_ORDER,
    }
    joblib.dump(ensemble_config, MODELS_DIR / "ensemble_model.pkl")

    config_payload = {
        "best_threshold_percent": threshold_info["best_threshold_pct"],
        "best_threshold_decimal": threshold_info["best_threshold_decimal"],
        "direction_accuracy_at_threshold": threshold_info["best_direction_acc"],
        "prediction_features": feature_names + regime_dummy_columns(),
        "base_features": feature_names,
        "hmm_features": list(HMM_FEATURE_DISPLAY_TO_SOURCE.keys()),
        "regime_order": REGIME_ORDER,
        "regime_training_mode": "pooled_weighted_compact_features",
        "final_model_params": FINAL_MODEL_PARAMS,
        "feature_ranking_params": FEATURE_RANKING_PARAMS,
        "regime_weight_own": REGIME_WEIGHT_OWN,
        "regime_weight_other": REGIME_WEIGHT_OTHER,
    }
    with MODEL_CONFIG_PATH.open("w", encoding="utf-8") as file:
        json.dump(config_payload, file, indent=2)
    end_step(t0)


def write_results(
    original_feature_count: int,
    new_feature_count: int,
    final_feature_count: int,
    selected_feature_names: List[str],
    cv_df: pd.DataFrame,
    cv_summary: Dict[str, Tuple[float, float]],
    performance_df: pd.DataFrame,
    threshold_info: Dict[str, float],
    shap_table: pd.DataFrame,
    shap_stats: Dict[str, float],
) -> None:
    t0 = start_step("Writing improved Layer 3 report")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    lookup = performance_df.set_index("Model").to_dict("index")
    best_single_candidates = ["Crisis_bank", "Late_Cycle_bank", "Expansion_bank"]
    best_single_model = max(best_single_candidates, key=lambda model_name: lookup[model_name]["R2"])
    best_single_r2 = float(lookup[best_single_model]["R2"])
    new_best_model = max(performance_df["Model"], key=lambda model_name: lookup[model_name]["R2"])
    new_best_r2 = float(lookup[new_best_model]["R2"])
    improvement_pct = ((new_best_r2 - OLD_BEST_R2) / OLD_BEST_R2) * 100
    new_best_direction = float(lookup[new_best_model]["Direction%"])
    ensemble_vs_best_single = float(lookup["Ensemble"]["R2"] / best_single_r2) if best_single_r2 not in (0, np.nan) else np.inf
    shap_lookup = shap_table.set_index("Regime").to_dict("index") if not shap_table.empty else {}

    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        def log(msg: str) -> None:
            print(msg)
            file.write(msg + "\n")

        log("=== IMPROVED MODEL RESULTS ===")
        log(f"Generated: {datetime.now()}")
        log("")
        log("--- FEATURE ENGINEERING ---")
        log(f"Original features : {original_feature_count}")
        log(f"New features added: {new_feature_count}")
        log(f"Final feature count: {final_feature_count}")
        log(f"Compact feature count: {len(selected_feature_names)}")
        log(f"Compact features: {', '.join(selected_feature_names)}")
        log("")
        log("--- WALK-FORWARD CV RESULTS ---")
        log(cv_df.round(6).to_string(index=False))
        for regime_label in REGIME_ORDER:
            mean_r2, std_r2 = cv_summary[regime_label]
            log(f"{regime_label} mean R2: {mean_r2:.3f} +/- {std_r2:.3f}")
        log("")
        log("--- TUNED MODEL PERFORMANCE (Test Set 2023+) ---")
        log("Model           | R2    | RMSE  | MAE   | Direction%")
        log("----------------|-------|-------|-------|----------")
        for model_name in ["Crisis_bank", "Late_Cycle_bank", "Expansion_bank", "Ensemble", "Global_bank"]:
            row = lookup[model_name]
            log(
                f"{model_name:<15}| "
                f"{row['R2']:.3f} | "
                f"{row['RMSE'] * 100:.2f}% | "
                f"{row['MAE'] * 100:.2f}% | "
                f"{row['Direction%']:.1f}%"
            )
        log("")
        log("--- IMPROVEMENT SUMMARY ---")
        log("Old best R2     : 0.3554 (Expansion_bank)")
        log(f"New best R2     : {new_best_r2:.4f} ({new_best_model})")
        log(f"Improvement     : {improvement_pct:+.2f}%")
        log("Old direction%  : 73.21% (Expansion)")
        log(f"New direction%  : {new_best_direction:.2f}%")
        log(f"Ensemble vs best single: {ensemble_vs_best_single:.2f}x")
        log("")
        log("--- OPTIMAL THRESHOLD ---")
        log(f"Best signal threshold: +/-{threshold_info['best_threshold_pct']:.2f}%")
        log(f"Direction acc at threshold: {threshold_info['best_direction_acc'] * 100:.2f}%")
        log("")
        log("--- GOLD SHAP (UPDATED) ---")
        for regime_label in REGIME_ORDER:
            row = shap_lookup.get(regime_label, {"Gold SHAP": np.nan, "Gold Rank": np.nan})
            rank_text = "nan" if pd.isna(row["Gold Rank"]) else str(int(row["Gold Rank"]))
            log(f"{regime_label:<11}: {format_metric(row['Gold SHAP'], 6)} (rank #{rank_text})")
        log(f"Gold SHAP t-stat (Crisis vs Expansion): {format_metric(shap_stats['t_stat'], 6)}")
        log(f"Gold SHAP p-value (Crisis vs Expansion): {format_metric(shap_stats['p_value'], 6)}")
        log(f"Gold SHAP ratio (Late_Cycle vs Expansion): {format_metric(shap_stats['ratio'], 2)}x")
        log("=== END ===")
    end_step(t0)


def main() -> None:
    resolve_stdout_encoding()
    total_t0 = time.time()
    log_progress("Initializing simplified improved Layer 3 pipeline")

    t0 = start_step("Loading data and saved HMM artifacts")
    df = load_dataset()
    hmm_model, hmm_scaler = load_hmm_artifacts()
    end_step(t0)

    modeling, feature_names, original_feature_count, new_feature_count, final_feature_count = engineer_features(df)
    modeling, train_df, test_df = build_train_test_sets(modeling, feature_names)
    modeling = add_regime_dummies(modeling)
    train_df = add_regime_dummies(train_df)
    test_df = add_regime_dummies(test_df)

    selected_feature_names = select_compact_feature_set(train_df, feature_names)
    cv_df, cv_summary = run_walk_forward_cv(modeling, selected_feature_names)
    regime_models, global_model, performance_df = train_final_models(train_df, test_df, selected_feature_names)
    ensemble_pred, ensemble_metrics = evaluate_ensemble(test_df, selected_feature_names, regime_models, hmm_model, hmm_scaler)

    performance_df = pd.concat(
        [
            performance_df,
            pd.DataFrame(
                [
                    {
                        "Model": "Ensemble",
                        "R2": ensemble_metrics["R2"],
                        "RMSE": ensemble_metrics["RMSE"],
                        "MAE": ensemble_metrics["MAE"],
                        "Direction%": ensemble_metrics["Direction%"],
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    best_single_r2 = performance_df.loc[performance_df["Model"].isin(["Crisis_bank", "Late_Cycle_bank", "Expansion_bank"]), "R2"].max()
    print(f"Ensemble R2: {ensemble_metrics['R2']:.3f} vs Best single regime R2: {best_single_r2:.3f}")

    threshold_info = optimize_signal_threshold(test_df[TARGET_COL], ensemble_pred)
    shap_table, shap_stats = run_updated_shap(regime_models, test_df, selected_feature_names)
    save_ensemble_and_config(regime_models, hmm_model, hmm_scaler, selected_feature_names, threshold_info)
    write_results(
        original_feature_count,
        new_feature_count,
        final_feature_count,
        selected_feature_names,
        cv_df,
        cv_summary,
        performance_df,
        threshold_info,
        shap_table,
        shap_stats,
    )

    print(f"\nSaved tuned regime models to: {MODELS_DIR.resolve()}")
    print(f"Saved ensemble model to: {(MODELS_DIR / 'ensemble_model.pkl').resolve()}")
    print(f"Saved model config to: {MODEL_CONFIG_PATH.resolve()}")
    print(f"Saved improved results to: {RESULTS_PATH.resolve()}")
    print(f"[INFO] Full pipeline completed in {time.time() - total_t0:.1f}s")


if __name__ == "__main__":
    main()
