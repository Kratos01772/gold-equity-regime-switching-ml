from __future__ import annotations

import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler  # noqa: F401 - required for pickle loading
from xgboost import XGBRegressor  # noqa: F401 - required for joblib loading


INPUT_PATH = Path("data") / "regime_labeled_data.csv"
MODELS_DIR = Path("models")
VISUALIZATIONS_DIR = Path("visualizations")
OUTPUTS_DIR = Path("outputs")
FORECAST_PLOT_PATH = VISUALIZATIONS_DIR / "price_forecast.png"
FORECAST_TXT_PATH = OUTPUTS_DIR / "layer5_forecast.txt"

FORECAST_HORIZON = 30
SHORT_FORECAST_HORIZON = 5
REGIME_REFRESH_DAYS = 5
RECENT_CONTEXT_DAYS = 60
TRAIN_END_DATE = pd.Timestamp("2023-01-01")
TEST_START_DATE = pd.Timestamp("2023-01-01")
BAND_MULTIPLIER = 2.5
VALIDATION_ANCHOR_STEP = 30

FEATURE_CANDIDATES: Dict[str, List[str]] = {
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

HMM_FEATURE_SOURCES = {
    "nifty_return": "nifty_return",
    "india_vix": "india_vix",
    "bank_spread": "bank_corwin_schultz_spread",
    "gold_nifty_corr_30d": "gold_nifty_corr_30d",
    "nifty_vol": "nifty_volatility_20d",
}

REGIME_COLORS = {
    "Crisis": "#c0392b",
    "Late_Cycle": "#d68910",
    "Expansion": "#1e8449",
}


def log_progress(message: str) -> None:
    print(f"[INFO] {message}")


def safe_print(message: str) -> None:
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode("ascii", errors="ignore").decode("ascii"))


def format_pct(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value * 100:.2f}%"


def format_inr(value: float) -> str:
    return f"₹{value:,.2f}"


def load_dataset() -> pd.DataFrame:
    log_progress("Loading regime-labeled data")
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH.resolve()}")

    df = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def resolve_feature_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    feature_frame = pd.DataFrame(index=df.index)
    for alias, candidates in FEATURE_CANDIDATES.items():
        for candidate in candidates:
            if candidate in df.columns:
                feature_frame[alias] = df[candidate]
                break

    if feature_frame.empty:
        raise ValueError("No forecast features could be resolved from the dataset.")

    return feature_frame, list(feature_frame.columns)


def prepare_history_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], str]:
    log_progress("Preparing model feature frame and regime/HMM aliases")
    feature_frame, selected_features = resolve_feature_frame(df)
    history = df.copy()
    for feature_name in selected_features:
        history[feature_name] = feature_frame[feature_name]

    for alias, source in HMM_FEATURE_SOURCES.items():
        history[alias] = history[source]

    history["regime"] = pd.to_numeric(history["regime"], errors="coerce")
    history[selected_features] = history[selected_features].ffill()
    hmm_cols = list(HMM_FEATURE_SOURCES.keys())
    history[hmm_cols] = history[hmm_cols].ffill()

    price_col = "bank_close" if "bank_close" in history.columns else "bank_adj_close"
    if price_col not in history.columns:
        raise KeyError("No bank close price column found.")

    return history, selected_features, price_col


def get_regime_maps(history: pd.DataFrame) -> Tuple[Dict[int, str], Dict[str, int]]:
    regime_map = (
        history.dropna(subset=["regime", "regime_label"])
        .drop_duplicates(subset=["regime"])
        .sort_values("regime")[["regime", "regime_label"]]
    )
    id_to_label = {int(row["regime"]): row["regime_label"] for _, row in regime_map.iterrows()}
    label_to_id = {label: regime_id for regime_id, label in id_to_label.items()}
    return id_to_label, label_to_id


def load_artifacts() -> Tuple[object, object, Dict[str, object], object | None]:
    log_progress("Loading saved HMM, scaler, and bank models")
    with (MODELS_DIR / "hmm_model.pkl").open("rb") as model_file:
        hmm_model = pickle.load(model_file)
    with (MODELS_DIR / "hmm_scaler.pkl").open("rb") as scaler_file:
        hmm_scaler = pickle.load(scaler_file)

    regime_models: Dict[str, object] = {}
    for regime_label in ["Crisis", "Late_Cycle", "Expansion"]:
        path = MODELS_DIR / f"{regime_label}_bank.pkl"
        try:
            regime_models[regime_label] = joblib.load(path)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Could not load {path.name}: {exc}")

    global_model = None
    global_path = MODELS_DIR / "Global_bank.pkl"
    try:
        global_model = joblib.load(global_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Could not load {global_path.name}: {exc}")

    return hmm_model, hmm_scaler, regime_models, global_model


def build_training_statistics(
    history: pd.DataFrame,
    selected_features: List[str],
) -> Tuple[Dict[str, pd.Series], pd.Series, Dict[str, float], float]:
    training = history.loc[history["date"] < TRAIN_END_DATE].copy()
    if training.empty:
        raise ValueError("Training subset before 2023-01-01 is empty.")

    training[selected_features] = training[selected_features].ffill()
    training[selected_features] = training[selected_features].fillna(training[selected_features].median(numeric_only=True))
    training[list(HMM_FEATURE_SOURCES.keys())] = training[list(HMM_FEATURE_SOURCES.keys())].ffill()
    training[list(HMM_FEATURE_SOURCES.keys())] = training[list(HMM_FEATURE_SOURCES.keys())].fillna(
        training[list(HMM_FEATURE_SOURCES.keys())].median(numeric_only=True)
    )

    stats_columns = list(dict.fromkeys(selected_features + list(HMM_FEATURE_SOURCES.keys())))
    regime_feature_means = {
        regime_label: group[stats_columns].mean(numeric_only=True)
        for regime_label, group in training.groupby("regime_label")
    }
    global_feature_means = training[stats_columns].mean(numeric_only=True)

    regime_volatility = training.groupby("regime_label")["bank_return"].std().to_dict()
    global_volatility = float(training["bank_return"].std())
    return regime_feature_means, global_feature_means, regime_volatility, global_volatility


def fill_feature_row(
    row: pd.Series,
    regime_label: str,
    regime_feature_means: Dict[str, pd.Series],
    global_feature_means: pd.Series,
) -> pd.Series:
    filled = row.copy()
    regime_means = regime_feature_means.get(regime_label, global_feature_means)
    for column in filled.index:
        if pd.isna(filled[column]):
            fallback = regime_means.get(column, global_feature_means.get(column, np.nan))
            filled[column] = fallback
    return filled


def predict_regime_from_window(
    window: pd.DataFrame,
    hmm_model: object,
    hmm_scaler: object,
    id_to_label: Dict[int, str],
    regime_feature_means: Dict[str, pd.Series],
    global_feature_means: pd.Series,
) -> Tuple[int, str]:
    hmm_columns = list(HMM_FEATURE_SOURCES.keys())
    hmm_window = window[hmm_columns].copy().apply(pd.to_numeric, errors="coerce").ffill()
    for column in hmm_columns:
        if column in global_feature_means.index:
            hmm_window[column] = hmm_window[column].fillna(global_feature_means[column])
    scaled = hmm_scaler.transform(hmm_window.to_numpy(dtype=float))
    state = int(hmm_model.predict(scaled)[-1])
    return state, id_to_label.get(state, f"State_{state}")


def update_synthetic_row(
    history: pd.DataFrame,
    forecast_date: pd.Timestamp,
    predicted_return: float,
    predicted_price: float,
    regime_label: str,
    label_to_id: Dict[str, int],
    selected_features: List[str],
    regime_feature_means: Dict[str, pd.Series],
    global_feature_means: pd.Series,
    price_col: str,
) -> pd.Series:
    new_row = history.iloc[-1].copy()
    new_row["date"] = forecast_date
    new_row["regime_label"] = regime_label
    new_row["regime"] = label_to_id.get(regime_label, np.nan)
    new_row[price_col] = predicted_price
    if "bank_adj_close" in new_row.index:
        new_row["bank_adj_close"] = predicted_price
    new_row["bank_return"] = predicted_return
    new_row["bank_lag1_return"] = predicted_return

    bank_returns = pd.concat([history["bank_return"], pd.Series([predicted_return])], ignore_index=True)
    bank_prices = pd.concat([history[price_col], pd.Series([predicted_price])], ignore_index=True)

    new_row["bank_lag5_return"] = bank_returns.iloc[-5] if len(bank_returns) >= 5 else np.nan
    new_row["bank_lag20_return"] = bank_returns.iloc[-20] if len(bank_returns) >= 20 else np.nan
    if len(bank_returns) >= 20:
        new_row["bank_vol"] = bank_returns.iloc[-20:].std(ddof=0) * np.sqrt(252)
    else:
        new_row["bank_vol"] = np.nan
    if len(bank_prices) >= 21 and bank_prices.iloc[-21] > 0:
        new_row["bank_momentum_20d"] = (predicted_price - bank_prices.iloc[-21]) / bank_prices.iloc[-21]
    else:
        new_row["bank_momentum_20d"] = np.nan

    regime_means = regime_feature_means.get(regime_label, global_feature_means)
    for feature in selected_features:
        if feature.startswith("bank_"):
            continue
        new_row[feature] = regime_means.get(feature, global_feature_means.get(feature, np.nan))

    for hmm_feature in HMM_FEATURE_SOURCES.keys():
        new_row[hmm_feature] = regime_means.get(hmm_feature, global_feature_means.get(hmm_feature, np.nan))

    feature_row = fill_feature_row(new_row[selected_features], regime_label, regime_feature_means, global_feature_means)
    for feature in selected_features:
        new_row[feature] = feature_row[feature]

    return new_row


def generate_forecast(
    history: pd.DataFrame,
    selected_features: List[str],
    price_col: str,
    id_to_label: Dict[int, str],
    label_to_id: Dict[str, int],
    hmm_model: object,
    hmm_scaler: object,
    regime_models: Dict[str, object],
    global_model: object | None,
    regime_feature_means: Dict[str, pd.Series],
    global_feature_means: pd.Series,
    regime_volatility: Dict[str, float],
    global_volatility: float,
    horizon: int = FORECAST_HORIZON,
    forecast_dates: pd.DatetimeIndex | None = None,
    announce_current_regime: bool = True,
) -> Tuple[pd.DataFrame, str]:
    if announce_current_regime:
        log_progress("Detecting current regime from the latest 60-row context")
    recent_context = history.tail(RECENT_CONTEXT_DAYS).copy()
    _, current_regime = predict_regime_from_window(
        recent_context,
        hmm_model,
        hmm_scaler,
        id_to_label,
        regime_feature_means,
        global_feature_means,
    )
    if announce_current_regime:
        print(f"Current detected regime: {current_regime} as of {history['date'].iloc[-1].date()}")

    synthetic_history = history.copy()
    if forecast_dates is None:
        forecast_dates = pd.bdate_range(start=history["date"].iloc[-1] + pd.Timedelta(days=1), periods=horizon)
    else:
        horizon = len(forecast_dates)
    last_price = float(history[price_col].iloc[-1])
    forecast_rows: List[Dict[str, object]] = []
    regime_for_step = current_regime

    for step_idx, forecast_date in enumerate(forecast_dates):
        if step_idx > 0 and step_idx % REGIME_REFRESH_DAYS == 0:
            recent_window = synthetic_history.tail(RECENT_CONTEXT_DAYS).copy()
            _, regime_for_step = predict_regime_from_window(
                recent_window,
                hmm_model,
                hmm_scaler,
                id_to_label,
                regime_feature_means,
                global_feature_means,
            )

        model = regime_models.get(regime_for_step, global_model)
        if model is None:
            raise ValueError("No regime-specific or global bank model is available for forecasting.")

        feature_row = fill_feature_row(
            synthetic_history.iloc[-1][selected_features],
            regime_for_step,
            regime_feature_means,
            global_feature_means,
        )
        predicted_return = float(model.predict(pd.DataFrame([feature_row], columns=selected_features))[0])
        predicted_price = float(last_price * np.exp(predicted_return))

        vol = regime_volatility.get(regime_for_step, global_volatility)
        step_number = step_idx + 1
        band_width = BAND_MULTIPLIER * vol * np.sqrt(step_number)
        upper_band = float(predicted_price * np.exp(band_width))
        lower_band = float(predicted_price * np.exp(-band_width))

        forecast_rows.append(
            {
                "date": forecast_date,
                "predicted_return": predicted_return,
                "predicted_price": predicted_price,
                "regime_used": regime_for_step,
                "confidence_band_upper": upper_band,
                "confidence_band_lower": lower_band,
            }
        )

        next_row = update_synthetic_row(
            synthetic_history,
            forecast_date,
            predicted_return,
            predicted_price,
            regime_for_step,
            label_to_id,
            selected_features,
            regime_feature_means,
            global_feature_means,
            price_col,
        )
        synthetic_history = pd.concat([synthetic_history, next_row.to_frame().T], ignore_index=True)
        last_price = predicted_price

    return pd.DataFrame(forecast_rows), current_regime


def run_forecast_validation(
    full_history: pd.DataFrame,
    selected_features: List[str],
    price_col: str,
    id_to_label: Dict[int, str],
    label_to_id: Dict[str, int],
    hmm_model: object,
    hmm_scaler: object,
    regime_models: Dict[str, object],
    global_model: object | None,
    regime_feature_means: Dict[str, pd.Series],
    global_feature_means: pd.Series,
    regime_volatility: Dict[str, float],
    global_volatility: float,
    horizon: int,
) -> Dict[str, float]:
    log_progress(f"Running rolling {horizon}-day forecast validation across the 2023-2024 test window")

    validation_data = full_history.copy().reset_index(drop=True)
    test_indices = validation_data.index[validation_data["date"] >= TEST_START_DATE].tolist()
    anchor_indices = test_indices[::VALIDATION_ANCHOR_STEP]

    direction_hits: List[float] = []
    band_hits: List[float] = []
    price_errors: List[float] = []

    for anchor_idx in anchor_indices:
        future_slice = validation_data.iloc[anchor_idx + 1 : anchor_idx + 1 + horizon].copy()
        if len(future_slice) < horizon:
            continue

        history_slice = validation_data.iloc[: anchor_idx + 1].copy()
        forecast_df, _ = generate_forecast(
            history_slice,
            selected_features,
            price_col,
            id_to_label,
            label_to_id,
            hmm_model,
            hmm_scaler,
            regime_models,
            global_model,
            regime_feature_means,
            global_feature_means,
            regime_volatility,
            global_volatility,
            horizon=horizon,
            forecast_dates=pd.DatetimeIndex(future_slice["date"]),
            announce_current_regime=False,
        )

        anchor_price = float(history_slice[price_col].iloc[-1])
        actual_end_price = float(future_slice[price_col].iloc[-1])
        predicted_end_price = float(forecast_df["predicted_price"].iloc[-1])

        actual_change = (actual_end_price / anchor_price) - 1.0
        predicted_change = (predicted_end_price / anchor_price) - 1.0
        direction_hits.append(float(np.sign(actual_change) == np.sign(predicted_change)))

        actual_prices = future_slice[price_col].to_numpy(dtype=float)
        lower_band = forecast_df["confidence_band_lower"].to_numpy(dtype=float)
        upper_band = forecast_df["confidence_band_upper"].to_numpy(dtype=float)
        band_hits.extend(((actual_prices >= lower_band) & (actual_prices <= upper_band)).astype(float).tolist())

        price_errors.append(abs(predicted_end_price - actual_end_price) / actual_end_price)

    summary = {
        "horizon": horizon,
        "direction_accuracy": float(np.mean(direction_hits) * 100) if direction_hits else np.nan,
        "confidence_band_hit_rate": float(np.mean(band_hits) * 100) if band_hits else np.nan,
        "mean_absolute_price_error": float(np.mean(price_errors) * 100) if price_errors else np.nan,
        "validation_windows": len(direction_hits),
    }

    print(f"\nForecast validation summary ({horizon}-day):")
    print(f"Forecast direction accuracy ({horizon}-day): {summary['direction_accuracy']:.2f}%")
    print(f"Confidence band hit rate: {summary['confidence_band_hit_rate']:.2f}% of actual prices fell within bands")
    print(f"Mean absolute price error: {summary['mean_absolute_price_error']:.2f}%")
    return summary


def save_forecast_plot(actual_history: pd.DataFrame, forecast_df: pd.DataFrame, price_col: str) -> None:
    log_progress("Saving 30-day price forecast visualization")
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)

    actual_tail = actual_history.tail(120).copy()
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(14, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax1.plot(actual_tail["date"], actual_tail[price_col], color="#1f77b4", linewidth=2.0, label="Actual bank close")
    ax1.plot(forecast_df["date"], forecast_df["predicted_price"], color="#f39c12", linewidth=3.0, alpha=0.55, label="Forecast")
    ax1.fill_between(
        forecast_df["date"],
        forecast_df["confidence_band_lower"],
        forecast_df["confidence_band_upper"],
        color="#f39c12",
        alpha=0.3,
        label="Confidence band",
    )

    last_actual_date = actual_tail["date"].iloc[-1]
    last_actual_price = actual_tail[price_col].iloc[-1]
    segment_dates = [last_actual_date] + forecast_df["date"].tolist()
    segment_prices = [last_actual_price] + forecast_df["predicted_price"].tolist()
    segment_regimes = forecast_df["regime_used"].tolist()

    for idx, regime_label in enumerate(segment_regimes):
        ax1.plot(
            segment_dates[idx : idx + 2],
            segment_prices[idx : idx + 2],
            color=REGIME_COLORS.get(regime_label, "#f39c12"),
            linewidth=3.2,
        )

    ax1.axvline(forecast_df["date"].iloc[0], linestyle="--", color="black", linewidth=1.1)
    ax1.text(forecast_df["date"].iloc[0], ax1.get_ylim()[1], "Forecast start", rotation=90, va="top", ha="right")
    ax1.set_title("Nifty Bank Index — 30-Day Price Forecast")
    ax1.set_ylabel("Index Level")
    ax1.grid(alpha=0.2)
    legend_handles = [
        plt.Line2D([0], [0], color="#1f77b4", linewidth=2.0, label="Actual bank close"),
        plt.Line2D([0], [0], color="#f39c12", linewidth=3.0, label="Forecast"),
        plt.Rectangle((0, 0), 1, 1, color="#f39c12", alpha=0.3, label="Confidence band"),
    ]
    legend_handles.extend(
        [
            plt.Line2D([0], [0], color=color, linewidth=3.0, label=label)
            for label, color in REGIME_COLORS.items()
        ]
    )
    ax1.legend(handles=legend_handles, loc="upper left")

    bar_colors = np.where(forecast_df["predicted_return"] >= 0, "#2e7d32", "#c62828")
    ax2.bar(forecast_df["date"], forecast_df["predicted_return"] * 100, color=bar_colors, width=0.8)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("Predicted Daily Returns (%)")
    ax2.set_ylabel("%")
    ax2.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(FORECAST_PLOT_PATH, dpi=150)
    plt.close(fig)


def write_forecast_results(
    history: pd.DataFrame,
    forecast_df: pd.DataFrame,
    price_col: str,
    current_regime: str,
    validation_summary_30d: Dict[str, float],
    validation_summary_5d: Dict[str, float],
) -> None:
    log_progress("Writing Layer 5 forecast report")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    last_known_date = history["date"].iloc[-1].date()
    last_known_price = float(history[price_col].iloc[-1])
    end_price = float(forecast_df["predicted_price"].iloc[-1])
    expected_change = (end_price / last_known_price) - 1.0
    regime_counts = forecast_df["regime_used"].value_counts()
    most_likely_regime = regime_counts.idxmax()

    with FORECAST_TXT_PATH.open("w", encoding="utf-8") as file:
        def log(msg: str) -> None:
            print(msg)
            file.write(msg + "\n")

        log("=== LAYER 5 — PRICE FORECAST ===")
        log(f"Generated: {datetime.now()}")
        log(f"Last known date    : {last_known_date}")
        log(f"Last known price   : {format_inr(last_known_price)}")
        log(f"Current regime     : {current_regime}")
        log("")
        log("--- 30-DAY FORECAST TABLE ---")
        log("Date       | Regime     | Pred Return | Pred Price | Lower Band | Upper Band")
        log("-----------|------------|-------------|------------|------------|----------")
        for _, row in forecast_df.iterrows():
            log(
                f"{row['date'].date()} | "
                f"{row['regime_used']:<10} | "
                f"{format_pct(float(row['predicted_return'])):>11} | "
                f"{format_inr(float(row['predicted_price'])):>10} | "
                f"{format_inr(float(row['confidence_band_lower'])):>10} | "
                f"{format_inr(float(row['confidence_band_upper'])):>10}"
            )

        log("")
        log("--- SUMMARY ---")
        log(f"Forecast end price : {format_inr(end_price)}")
        log(f"Expected change    : {format_pct(expected_change)} over 30 days")
        log(
            "Regime breakdown   : "
            f"{int(regime_counts.get('Crisis', 0))} days Crisis, "
            f"{int(regime_counts.get('Late_Cycle', 0))} days Late_Cycle, "
            f"{int(regime_counts.get('Expansion', 0))} days Expansion"
        )
        log(f"Most likely regime : {most_likely_regime}")
        log("")
        log("--- VALIDATION SUMMARY ---")
        log(f"Forecast direction accuracy (30-day): {validation_summary_30d['direction_accuracy']:.2f}%")
        log(
            "Confidence band hit rate (30-day): "
            f"{validation_summary_30d['confidence_band_hit_rate']:.2f}% of actual prices fell within bands"
        )
        log(f"Mean absolute price error (30-day): {validation_summary_30d['mean_absolute_price_error']:.2f}%")
        log(f"Validation windows (30-day)      : {int(validation_summary_30d['validation_windows'])}")
        log(f"Forecast direction accuracy (5-day): {validation_summary_5d['direction_accuracy']:.2f}%")
        log(
            "Confidence band hit rate (5-day) : "
            f"{validation_summary_5d['confidence_band_hit_rate']:.2f}% of actual prices fell within bands"
        )
        log(f"Mean absolute price error (5-day) : {validation_summary_5d['mean_absolute_price_error']:.2f}%")
        log(f"Validation windows (5-day)       : {int(validation_summary_5d['validation_windows'])}")
        log("")
        log("=== IMPORTANT CAVEATS ===")
        log("- This is a statistical model forecast, not financial advice")
        log("- Prediction error compounds over multi-step horizon")
        log("- Confidence bands widen with each step")
        log("- Model trained on data up to 2022-12-31")
        log("")
        log("=== END ===")


def print_limitations_box() -> None:
    safe_print("\n⚠️  FORECAST LIMITATIONS:")
    safe_print("- Multi-step forecasts compound errors — Day 30 uncertainty is much higher than Day 1")
    safe_print("- This model was not designed for live trading")
    safe_print("- Past regime patterns may not repeat")
    safe_print("- Use confidence bands, not point estimates, for any decisions")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    log_progress("Initializing Layer 5 forward forecasting pipeline")
    history, selected_features, price_col = prepare_history_frame(load_dataset())
    id_to_label, label_to_id = get_regime_maps(history)
    hmm_model, hmm_scaler, regime_models, global_model = load_artifacts()
    regime_feature_means, global_feature_means, regime_volatility, global_volatility = build_training_statistics(
        history,
        selected_features,
    )

    forecast_df, current_regime = generate_forecast(
        history,
        selected_features,
        price_col,
        id_to_label,
        label_to_id,
        hmm_model,
        hmm_scaler,
        regime_models,
        global_model,
        regime_feature_means,
        global_feature_means,
        regime_volatility,
        global_volatility,
        horizon=FORECAST_HORIZON,
    )

    validation_summary_30d = run_forecast_validation(
        history,
        selected_features,
        price_col,
        id_to_label,
        label_to_id,
        hmm_model,
        hmm_scaler,
        regime_models,
        global_model,
        regime_feature_means,
        global_feature_means,
        regime_volatility,
        global_volatility,
        horizon=FORECAST_HORIZON,
    )

    validation_summary_5d = run_forecast_validation(
        history,
        selected_features,
        price_col,
        id_to_label,
        label_to_id,
        hmm_model,
        hmm_scaler,
        regime_models,
        global_model,
        regime_feature_means,
        global_feature_means,
        regime_volatility,
        global_volatility,
        horizon=SHORT_FORECAST_HORIZON,
    )

    save_forecast_plot(history, forecast_df, price_col)
    write_forecast_results(
        history,
        forecast_df,
        price_col,
        current_regime,
        validation_summary_30d,
        validation_summary_5d,
    )
    print_limitations_box()

    print(f"\nSaved forecast plot to: {FORECAST_PLOT_PATH.resolve()}")
    print(f"Saved forecast report to: {FORECAST_TXT_PATH.resolve()}")


if __name__ == "__main__":
    main()
