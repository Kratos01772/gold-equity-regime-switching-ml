from __future__ import annotations

import json
import math
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import yfinance as yf
from pandas.errors import PerformanceWarning


matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=PerformanceWarning)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "regime_labeled_data.csv"
ENSEMBLE_PATH = BASE_DIR / "models" / "ensemble_model.pkl"
HMM_MODEL_PATH = BASE_DIR / "models" / "hmm_model.pkl"
HMM_SCALER_PATH = BASE_DIR / "models" / "hmm_scaler.pkl"
CONFIG_PATH = BASE_DIR / "outputs" / "model_config.json"
OUTPUT_TEXT_PATH = BASE_DIR / "outputs" / "tomorrows_prediction.txt"
OUTPUT_JSON_PATH = BASE_DIR / "outputs" / "prediction_latest.json"
CHART_PATH = BASE_DIR / "visualizations" / "tomorrow_prediction.png"

TRAIN_END_DATE = pd.Timestamp("2023-01-01")
REGIME_ORDER = ["Crisis", "Late_Cycle", "Expansion"]
ASSET_TICKERS = {
    "gold": "GC=F",
    "nifty": "^NSEI",
    "bank": "^NSEBANK",
    "it": "^CNXIT",
    "vix": "^INDIAVIX",
    "usdinr": "USDINR=X",
    "oil": "CL=F",
    "silver": "SI=F",
}
OHLCV_FIELDS = ["open", "high", "low", "close", "adj_close", "volume"]
BOX_WIDTH = 58


def log(message: str) -> None:
    print(message)


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass


def load_historical() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    return df


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_yf_columns(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    data = raw.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    data = data.rename(columns=rename_map)
    for field in OHLCV_FIELDS:
        if field not in data.columns:
            data[field] = np.nan
    if data["adj_close"].isna().all():
        data["adj_close"] = data["close"]
    data.index = pd.to_datetime(data.index).tz_localize(None).normalize()
    return data[OHLCV_FIELDS]


def get_historical_asset_frame(history: pd.DataFrame, asset: str, rows: int = 400) -> pd.DataFrame:
    columns = [f"{asset}_{field}" for field in OHLCV_FIELDS if f"{asset}_{field}" in history.columns]
    asset_df = history[["date"] + columns].tail(rows).copy()
    asset_df = asset_df.set_index("date")
    asset_df.index = pd.to_datetime(asset_df.index).normalize()
    return asset_df


def fetch_asset_data(asset: str, ticker: str, history: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    historical_asset = get_historical_asset_frame(history, asset, rows=400)
    try:
        raw = yf.download(ticker, period="90d", interval="1d", auto_adjust=False, progress=False)
        live = normalize_yf_columns(raw)
        if len(live) < 30:
            raise ValueError(f"Fetched only {len(live)} rows")
        live = live.rename(columns={field: f"{asset}_{field}" for field in OHLCV_FIELDS})
        combined = historical_asset.copy()
        union_index = combined.index.union(live.index)
        combined = combined.reindex(union_index)
        combined.update(live)
        return combined.sort_index(), True
    except Exception:
        log(f"⚠️ Live fetch failed for {asset}, using historical data")
        fallback = get_historical_asset_frame(history, asset, rows=90)
        combined = historical_asset.copy()
        combined.update(fallback)
        return combined.sort_index(), False


def build_market_dataset(history: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    frames: List[pd.DataFrame] = []
    live_successes = []
    for asset, ticker in ASSET_TICKERS.items():
        asset_df, success = fetch_asset_data(asset, ticker, history)
        frames.append(asset_df)
        live_successes.append(success)

    combined = pd.concat(frames, axis=1).sort_index()
    combined = combined.reset_index().rename(columns={"index": "date"})
    combined["date"] = pd.to_datetime(combined["date"])
    return combined, any(live_successes)


def rolling_last_percentile(series: pd.Series, window: int) -> pd.Series:
    def _last_rank(values: np.ndarray) -> float:
        last = values[-1]
        return float(np.sum(values <= last) / len(values))

    return series.rolling(window=window, min_periods=window).apply(_last_rank, raw=True)


def corwin_schultz_spread(high: pd.Series, low: pd.Series) -> pd.Series:
    ratio = (high / low).replace([np.inf, -np.inf], np.nan)
    beta = 0.5 * np.log(ratio.clip(lower=1e-12))
    exp_beta = np.exp(beta)
    spread = 2 * (exp_beta - 1) / (1 + exp_beta)
    return spread.replace([np.inf, -np.inf], np.nan)


def amihud_illiquidity(return_series: pd.Series, volume: pd.Series) -> pd.Series:
    denom = (volume / 1_000_000).replace(0, np.nan)
    illiq = return_series.abs() / denom
    return illiq.replace([np.inf, -np.inf], np.nan)


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy().sort_values("date").reset_index(drop=True)

    for asset in ASSET_TICKERS:
        close_col = f"{asset}_close"
        if close_col not in df.columns:
            continue
        returns = np.log(df[close_col] / df[close_col].shift(1))
        df[f"{asset}_return"] = returns
        df[f"{asset}_volatility_20d"] = returns.rolling(20).std() * np.sqrt(252)
        df[f"{asset}_momentum_20d"] = (df[close_col] - df[close_col].shift(20)) / df[close_col].shift(20)
        df[f"{asset}_lag1_return"] = returns.shift(1)
        df[f"{asset}_lag2_return"] = returns.shift(2)
        df[f"{asset}_lag5_return"] = returns.shift(5)
        df[f"{asset}_lag10_return"] = returns.shift(10)
        df[f"{asset}_lag20_return"] = returns.shift(20)

        high_col = f"{asset}_high"
        low_col = f"{asset}_low"
        volume_col = f"{asset}_volume"
        if high_col in df.columns and low_col in df.columns:
            df[f"{asset}_corwin_schultz_spread"] = corwin_schultz_spread(df[high_col], df[low_col])
        if volume_col in df.columns:
            df[f"{asset}_amihud_illiquidity"] = amihud_illiquidity(returns, df[volume_col])

    df["gold_nifty_corr_30d"] = df["gold_return"].rolling(30).corr(df["nifty_return"])
    df["gold_bank_corr_30d"] = df["gold_return"].rolling(30).corr(df["bank_return"])

    df["india_vix"] = df["vix_close"]
    df["bank_spread"] = df["bank_corwin_schultz_spread"]
    df["nifty_vol"] = df["nifty_volatility_20d"]
    df["gold_vix_interaction"] = df["gold_return"] * df["india_vix"]
    df["gold_spread_interaction"] = df["gold_return"] * df["bank_corwin_schultz_spread"]
    df["gold_silver_ratio"] = df["gold_close"] / df["silver_close"]

    df["vix_5d_change"] = df["india_vix"] - df["india_vix"].shift(5)
    df["vix_20d_change"] = df["india_vix"] - df["india_vix"].shift(20)
    df["vix_regime_zscore"] = (df["india_vix"] - df["india_vix"].rolling(60).mean()) / df["india_vix"].rolling(60).std()
    df["gold_vol"] = df["gold_volatility_20d"]
    df["bank_vol"] = df["bank_volatility_20d"]
    df["gold_return_lag1"] = df["gold_lag1_return"]
    df["bank_return_lag5"] = df["bank_lag5_return"]
    df["gold_vol_ratio"] = df["gold_volatility_20d"] / df["gold_volatility_20d"].shift(20)
    df["gold_above_ma20"] = (df["gold_close"] > df["gold_close"].rolling(20).mean()).astype(float)

    nifty_bank_ratio = df["nifty_close"] / df["bank_close"]
    df["nifty_bank_ratio"] = nifty_bank_ratio
    df["nifty_bank_ratio_5d_change"] = (nifty_bank_ratio / nifty_bank_ratio.shift(5)) - 1

    gold_nifty_ratio = df["gold_close"] / df["nifty_close"]
    df["gold_nifty_ratio_change"] = (gold_nifty_ratio / gold_nifty_ratio.shift(20)) - 1
    df["bank_vol_percentile"] = rolling_last_percentile(df["bank_volatility_20d"], 252)
    df["vix_percentile"] = rolling_last_percentile(df["india_vix"], 252)

    df = df.ffill(limit=5)
    return df


def load_state_mapping(history: pd.DataFrame) -> Dict[int, str]:
    mapping = (
        history[["regime", "regime_label"]]
        .dropna()
        .drop_duplicates()
        .sort_values("regime")
    )
    return {int(row["regime"]): str(row["regime_label"]) for _, row in mapping.iterrows()}


def attach_regime_context(df: pd.DataFrame, hmm_model: object, hmm_scaler: object, state_to_label: Dict[int, str], hmm_features: List[str]) -> pd.DataFrame:
    enriched = df.copy()
    valid = enriched[hmm_features].ffill().dropna()
    if valid.empty:
        enriched["regime_state_live"] = np.nan
        enriched["regime_label_live"] = pd.Series(index=enriched.index, dtype="object")
        enriched["regime_change"] = np.nan
        enriched["days_in_regime"] = np.nan
        return enriched

    scaled = hmm_scaler.transform(valid.to_numpy(dtype=float))
    states = hmm_model.predict(scaled)
    labels = [state_to_label.get(int(state), f"State_{int(state)}") for state in states]
    enriched["regime_state_live"] = np.nan
    enriched["regime_label_live"] = pd.Series(index=enriched.index, dtype="object")
    enriched.loc[valid.index, "regime_state_live"] = states
    enriched.loc[valid.index, "regime_label_live"] = labels
    enriched["regime_label_live"] = enriched["regime_label_live"].ffill()
    enriched["regime_change"] = (enriched["regime_label_live"] != enriched["regime_label_live"].shift(1)).astype(float)
    group_keys = (enriched["regime_change"].fillna(0) == 1).cumsum()
    days = enriched.groupby(group_keys).cumcount() + 1
    enriched["days_in_regime"] = np.where(enriched["regime_label_live"].notna(), days, np.nan)
    return enriched


def make_bar(prob: float, width: int = 10) -> str:
    filled = int(prob * width)
    return "█" * filled + "░" * (width - filled)


def confidence_label(prob: float) -> str:
    if prob >= 0.7:
        return "High"
    if prob >= 0.45:
        return "Med"
    return "Low"


def next_business_day(last_date: pd.Timestamp) -> pd.Timestamp:
    return pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=1)[0]


def format_box_line(text: str = "") -> str:
    inner = f" {text}"
    return f"║{inner.ljust(BOX_WIDTH)}║"


def build_output_box(payload: dict) -> str:
    lines = [
        "╔══════════════════════════════════════════════════════════╗",
        "║         NIFTY BANK — TOMORROW'S PREDICTION               ║",
        f"║         Generated: {payload['generated_date']} at {payload['generated_time']} IST{' ' * max(0, 16 - len(payload['generated_time']))}║",
        "╠══════════════════════════════════════════════════════════╣",
        format_box_line(),
        format_box_line(f"TODAY'S CLOSE      : ₹{payload['current_price']:>10,.2f}"),
        format_box_line(f"PREDICTED TOMORROW : ₹{payload['predicted_price']:>10,.2f}"),
        format_box_line(f"EXPECTED CHANGE    : {payload['predicted_return_pct']:>+10.2f}%"),
        format_box_line(),
        format_box_line(f"CONFIDENCE BAND    : ₹{payload['lower']:,.0f} — ₹{payload['upper']:,.0f}"),
        format_box_line(),
        "╠══════════════════════════════════════════════════════════╣",
        format_box_line(f"SIGNAL             : {payload['signal_display']}"),
        format_box_line(f"CURRENT REGIME     : {payload['regime']}"),
        format_box_line(f"REGIME CONFIDENCE  : {payload['regime_confidence']:.1%} ({payload['regime_confidence_label']})"),
        "╠══════════════════════════════════════════════════════════╣",
        format_box_line("MODEL BREAKDOWN    :"),
        format_box_line(f"  Crisis model     : {payload['model_breakdown']['Crisis']:+.3f}%"),
        format_box_line(f"  Late_Cycle model : {payload['model_breakdown']['Late_Cycle']:+.3f}%"),
        format_box_line(f"  Expansion model  : {payload['model_breakdown']['Expansion']:+.3f}%"),
        format_box_line(f"  Ensemble (final) : {payload['model_breakdown']['Ensemble']:+.3f}%"),
        "╠══════════════════════════════════════════════════════════╣",
        format_box_line("REGIME PROBABILITIES:"),
        format_box_line(f"  Crisis    [{payload['bars']['Crisis']}] {payload['regime_probabilities']['Crisis']:.1%}"),
        format_box_line(f"  Late_Cycle[{payload['bars']['Late_Cycle']}] {payload['regime_probabilities']['Late_Cycle']:.1%}"),
        format_box_line(f"  Expansion [{payload['bars']['Expansion']}] {payload['regime_probabilities']['Expansion']:.1%}"),
        "╠══════════════════════════════════════════════════════════╣",
        format_box_line("TOP 3 FACTORS DRIVING PREDICTION:"),
    ]
    for idx, item in enumerate(payload["top_features"], start=1):
        lines.append(format_box_line(f"  {idx}. {item['name']:<20} : {item['direction_symbol']} {item['magnitude']:.3f}%"))
    lines.extend(
        [
            "╠══════════════════════════════════════════════════════════╣",
            format_box_line("⚠️  NOT FINANCIAL ADVICE — Academic ML Project"),
            "╚══════════════════════════════════════════════════════════╝",
        ]
    )
    return "\n".join(lines)


def get_regime_vol(history: pd.DataFrame, regime_name: str) -> float:
    train = history.loc[history["date"] < TRAIN_END_DATE].copy()
    regime_returns = train.loc[train["regime_label"] == regime_name, "bank_return"].dropna()
    if len(regime_returns) < 20:
        regime_returns = train["bank_return"].dropna()
    vol = float(regime_returns.std())
    return vol if math.isfinite(vol) and vol > 0 else float(train["bank_return"].dropna().std())


def plot_prediction(last_30: pd.DataFrame, tomorrow_date: pd.Timestamp, predicted_price: float, lower: float, upper: float, signal: str) -> None:
    CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(last_30["date"], last_30["bank_close"], color="#1f77b4", linewidth=2, label="Actual close")
    ax.errorbar(
        [tomorrow_date],
        [predicted_price],
        yerr=[[predicted_price - lower], [upper - predicted_price]],
        fmt="*",
        color="#ff7f0e",
        markersize=15,
        capsize=6,
        linewidth=2,
        label="Tomorrow prediction",
    )
    ax.axhline(lower, color="#ffb366", linestyle="--", linewidth=1)
    ax.axhline(upper, color="#ffb366", linestyle="--", linewidth=1)
    ax.annotate(
        f"Tomorrow\n₹{predicted_price:,.0f}\n{signal}",
        xy=(tomorrow_date, predicted_price),
        xytext=(10, 15),
        textcoords="offset points",
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#ff7f0e"},
    )
    ax.set_title(f"Nifty Bank — Tomorrow's Prediction ({tomorrow_date.date()})")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_outputs(text_box: str, json_payload: dict) -> None:
    OUTPUT_TEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_TEXT_PATH.write_text(text_box, encoding="utf-8")
    with OUTPUT_JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(json_payload, file, indent=2, ensure_ascii=False)


def main() -> None:
    configure_stdout()
    history = load_historical()
    config = load_json(CONFIG_PATH)
    state_to_label = load_state_mapping(history)
    regime_order = config.get("regime_order", REGIME_ORDER)
    threshold_decimal = float(config.get("best_threshold_decimal", 0.003))
    ensemble_artifact = joblib.load(ENSEMBLE_PATH) if ENSEMBLE_PATH.exists() else None

    with HMM_MODEL_PATH.open("rb") as file:
        hmm_model = pickle.load(file)
    with HMM_SCALER_PATH.open("rb") as file:
        hmm_scaler = pickle.load(file)

    log("Fetching latest market data...")
    market_data, live_available = build_market_dataset(history)
    if not live_available:
        last_known_date = history["date"].max().date()
        log(f"⚠️ Using last known data from {last_known_date} — live fetch unavailable")

    log("Engineering live features...")
    featured = engineer_features(market_data)
    hmm_features = config.get("hmm_features", ["nifty_return", "india_vix", "bank_spread", "gold_nifty_corr_30d", "nifty_vol"])
    featured = attach_regime_context(featured, hmm_model, hmm_scaler, state_to_label, hmm_features)

    base_features = config.get("base_features") or (ensemble_artifact or {}).get("base_features") or []
    if not base_features:
        raise ValueError("No prediction feature list found in outputs/model_config.json or ensemble artifact.")

    required_cols = ["date", "bank_close"] + base_features + hmm_features
    usable = featured.dropna(subset=required_cols).copy()
    if usable.empty:
        raise ValueError("Unable to build a complete feature row for prediction.")

    stability = usable.dropna(subset=hmm_features).tail(30).copy()
    scaled_window = hmm_scaler.transform(stability[hmm_features].to_numpy(dtype=float))
    hard_states = hmm_model.predict(scaled_window)
    prob_matrix = hmm_model.predict_proba(scaled_window)
    hard_state = int(hard_states[-1])
    current_regime = state_to_label.get(hard_state, f"State_{hard_state}")

    regime_probabilities = {label: 0.0 for label in regime_order}
    last_probs = prob_matrix[-1]
    for state_idx, prob in enumerate(last_probs):
        label = state_to_label.get(int(state_idx), f"State_{int(state_idx)}")
        if label in regime_probabilities:
            regime_probabilities[label] = float(prob)
    regime_conf = regime_probabilities.get(current_regime, 0.0)

    today_row = usable.iloc[[-1]].copy()
    current_price = float(today_row["bank_close"].iloc[0])
    latest_date = pd.Timestamp(today_row["date"].iloc[0])
    tomorrow_date = next_business_day(latest_date)

    model_breakdown_raw: Dict[str, float] = {}
    regime_models: Dict[str, object] = {}
    for regime in regime_order:
        model_path = BASE_DIR / "models" / f"{regime}_bank.pkl"
        if ensemble_artifact and "models" in ensemble_artifact and regime in ensemble_artifact["models"]:
            regime_models[regime] = ensemble_artifact["models"][regime]
        elif model_path.exists():
            regime_models[regime] = joblib.load(model_path)
        else:
            raise FileNotFoundError(f"Required regime model not found: {model_path}")

    log("Generating tomorrow prediction...")
    for regime in regime_order:
        context = today_row[base_features].copy()
        for label in regime_order:
            context[f"is_{label.lower()}"] = 1.0 if label == regime else 0.0
        pred = float(regime_models[regime].predict(context)[0])
        model_breakdown_raw[regime] = pred

    if ensemble_artifact is None:
        log(f"⚠️ Ensemble not found, using {current_regime}_bank model")
        predicted_return = model_breakdown_raw[current_regime]
    else:
        predicted_return = sum(regime_probabilities.get(regime, 0.0) * model_breakdown_raw[regime] for regime in regime_order)

    if predicted_return > threshold_decimal:
        signal = "BUY"
        signal_display = "BUY 📈"
    elif predicted_return < -threshold_decimal:
        signal = "SELL"
        signal_display = "SELL 📉"
    else:
        signal = "HOLD"
        signal_display = "HOLD ➡️"

    regime_vol = get_regime_vol(history, current_regime)
    predicted_price = current_price * math.exp(predicted_return)
    upper = current_price * math.exp(predicted_return + 1.96 * regime_vol)
    lower = current_price * math.exp(predicted_return - 1.96 * regime_vol)

    current_model = regime_models[current_regime]
    current_context = today_row[base_features].copy()
    for label in regime_order:
        current_context[f"is_{label.lower()}"] = 1.0 if label == current_regime else 0.0

    try:
        explainer = shap.TreeExplainer(current_model)
        shap_values = np.asarray(explainer.shap_values(current_context))[0]
        shap_series = pd.Series(shap_values, index=current_context.columns)
        top_factors = []
        for feature_name, shap_value in shap_series.abs().sort_values(ascending=False).head(3).items():
            signed_value = shap_series[feature_name]
            top_factors.append(
                {
                    "name": feature_name,
                    "shap_value": float(signed_value),
                    "direction": "up" if signed_value >= 0 else "down",
                    "direction_symbol": "↑" if signed_value >= 0 else "↓",
                    "magnitude": abs(float(signed_value)) * 100,
                }
            )
    except Exception:
        top_factors = []
        for feature_name in base_features[:3]:
            value = float(today_row[feature_name].iloc[0]) if pd.notna(today_row[feature_name].iloc[0]) else 0.0
            top_factors.append(
                {
                    "name": feature_name,
                    "shap_value": value,
                    "direction": "up" if value >= 0 else "down",
                    "direction_symbol": "↑" if value >= 0 else "↓",
                    "magnitude": abs(value),
                }
            )

    generated_at = datetime.now()
    box_payload = {
        "generated_date": generated_at.strftime("%Y-%m-%d"),
        "generated_time": generated_at.strftime("%H:%M:%S"),
        "current_price": current_price,
        "predicted_price": predicted_price,
        "predicted_return_pct": predicted_return * 100,
        "lower": lower,
        "upper": upper,
        "signal_display": signal_display,
        "regime": current_regime,
        "regime_confidence": regime_conf,
        "regime_confidence_label": confidence_label(regime_conf),
        "model_breakdown": {
            "Crisis": model_breakdown_raw.get("Crisis", 0.0) * 100,
            "Late_Cycle": model_breakdown_raw.get("Late_Cycle", 0.0) * 100,
            "Expansion": model_breakdown_raw.get("Expansion", 0.0) * 100,
            "Ensemble": predicted_return * 100,
        },
        "regime_probabilities": regime_probabilities,
        "bars": {label: make_bar(regime_probabilities.get(label, 0.0)) for label in regime_order},
        "top_features": top_factors,
    }
    text_box = build_output_box(box_payload)
    print(text_box)

    output_json = {
        "generated_at": generated_at.isoformat(),
        "current_price": current_price,
        "predicted_price": predicted_price,
        "predicted_return_pct": predicted_return * 100,
        "signal": signal,
        "regime": current_regime,
        "regime_confidence": regime_conf,
        "regime_probabilities": regime_probabilities,
        "confidence_band": {"lower": lower, "upper": upper},
        "top_features": [
            {"name": item["name"], "shap_value": item["shap_value"], "direction": item["direction"]} for item in top_factors
        ],
    }
    save_outputs(text_box, output_json)

    last_30 = usable[["date", "bank_close"]].tail(30).copy()
    plot_prediction(last_30, tomorrow_date, predicted_price, lower, upper, signal)

    log("✅ Prediction complete. See outputs/tomorrows_prediction.txt")
    log("📊 Chart saved to visualizations/tomorrow_prediction.png")


if __name__ == "__main__":
    main()
