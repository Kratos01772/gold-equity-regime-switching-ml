from __future__ import annotations

import json
import pickle
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pandas.io.formats.style import Styler

import predict_tomorrow as pt

try:
    import shap
    SHAP_IMPORT_ERROR = None
except Exception as exc:  # noqa: BLE001
    shap = None
    SHAP_IMPORT_ERROR = str(exc)


BASE_DIR = Path(r"C:\Users\Krishna Shetty\Desktop\Ml_project\1.0")
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
VIZ_DIR = BASE_DIR / "visualizations"
OUTPUTS_DIR = BASE_DIR / "outputs"

DATA_PATH = DATA_DIR / "regime_labeled_data.csv"
LAYER3_RESULTS_PATH = OUTPUTS_DIR / "layer3_improved_results.txt"
LAYER3_RESULTS_FALLBACK = OUTPUTS_DIR / "layer3_results.txt"
LAYER4_RESULTS_PATH = OUTPUTS_DIR / "layer4_results.txt"
LAYER5_FORECAST_PATH = OUTPUTS_DIR / "layer5_forecast.txt"
PREDICTION_JSON_PATH = OUTPUTS_DIR / "prediction_latest.json"
MODEL_CONFIG_PATH = OUTPUTS_DIR / "model_config.json"
ENSEMBLE_PATH = MODELS_DIR / "ensemble_model.pkl"

REGIME_COLORS = {
    "Crisis": "#d9534f",
    "Late_Cycle": "#f0ad4e",
    "Expansion": "#2ca25f",
}
REGIME_BG_COLORS = {
    "Crisis": "rgba(217, 83, 79, 0.10)",
    "Late_Cycle": "rgba(240, 173, 78, 0.12)",
    "Expansion": "rgba(44, 162, 95, 0.10)",
}
REGIME_INTERPRETATION = {
    "Crisis": "⚠️ Liquidity stress dominates. Bank spread is key.",
    "Late_Cycle": "📡 Gold is a leading indicator in this regime.",
    "Expansion": "📈 Momentum drives returns. Gold provides structure.",
}
LIVE_TICKERS = {
    "gold": "GC=F",
    "nifty": "^NSEI",
    "bank": "^NSEBANK",
    "it": "^CNXIT",
    "vix": "^INDIAVIX",
    "usdinr": "USDINR=X",
    "oil": "CL=F",
    "silver": "SI=F",
}


st.set_page_config(
    page_title="Gold-Equity Regime ML Dashboard",
    page_icon="📈",
    layout="wide",
)


def emoji(text: str, fallback: str) -> str:
    try:
        text.encode("utf-8")
        return text
    except Exception:
        return fallback


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(ttl=3600)
def load_regime_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    return df


@st.cache_data(ttl=3600)
def fetch_live_market_data() -> Tuple[pd.DataFrame, bool, str]:
    history = load_regime_data()
    market_data, live_available = pt.build_market_dataset(history)
    market_data = market_data.sort_values("date").tail(260).reset_index(drop=True)
    return market_data, live_available, datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@st.cache_resource
def load_artifacts() -> Dict[str, object]:
    artifacts: Dict[str, object] = {"errors": []}

    try:
        artifacts["config"] = load_json(MODEL_CONFIG_PATH)
    except Exception as exc:  # noqa: BLE001
        artifacts["errors"].append(f"Config load failed: {exc}")
        artifacts["config"] = {}

    for model_name in ["Crisis_bank.pkl", "Late_Cycle_bank.pkl", "Expansion_bank.pkl", "Global_bank.pkl"]:
        try:
            artifacts[model_name] = joblib.load(MODELS_DIR / model_name)
        except Exception as exc:  # noqa: BLE001
            artifacts["errors"].append(f"{model_name} load failed: {exc}")
            artifacts[model_name] = None

    try:
        artifacts["ensemble"] = joblib.load(ENSEMBLE_PATH)
    except Exception as exc:  # noqa: BLE001
        artifacts["errors"].append(f"ensemble_model.pkl load failed: {exc}")
        artifacts["ensemble"] = None

    try:
        with (MODELS_DIR / "hmm_model.pkl").open("rb") as file:
            artifacts["hmm_model"] = pickle.load(file)
    except Exception as exc:  # noqa: BLE001
        artifacts["errors"].append(f"hmm_model.pkl load failed: {exc}")
        artifacts["hmm_model"] = None

    try:
        with (MODELS_DIR / "hmm_scaler.pkl").open("rb") as file:
            artifacts["hmm_scaler"] = pickle.load(file)
    except Exception as exc:  # noqa: BLE001
        artifacts["errors"].append(f"hmm_scaler.pkl load failed: {exc}")
        artifacts["hmm_scaler"] = None

    return artifacts


def parse_layer5_forecast(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    rows: List[Dict[str, object]] = []
    in_table = False
    for line in lines:
        if line.startswith("--- 30-DAY FORECAST TABLE ---"):
            in_table = True
            continue
        if in_table and line.startswith("--- SUMMARY ---"):
            break
        if not in_table or "|" not in line or line.startswith("Date") or line.startswith("---"):
            continue

        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 6:
            continue

        def clean_num(text: str) -> float:
            cleaned = re.sub(r"[^0-9.\-+]", "", text)
            return float(cleaned) if cleaned else np.nan

        try:
            rows.append(
                {
                    "date": pd.to_datetime(parts[0]),
                    "regime": parts[1],
                    "predicted_return_pct": clean_num(parts[2]),
                    "predicted_price": clean_num(parts[3]),
                    "lower_band": clean_num(parts[4]),
                    "upper_band": clean_num(parts[5]),
                }
            )
        except Exception:
            continue

    return pd.DataFrame(rows)


def parse_gold_shap_values() -> Dict[str, float]:
    path = LAYER3_RESULTS_PATH if LAYER3_RESULTS_PATH.exists() else LAYER3_RESULTS_FALLBACK
    if not path.exists():
        return {"Crisis": 0.000085, "Late_Cycle": 0.000562, "Expansion": 0.000249}

    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"^(Crisis|Late_Cycle|Expansion)\s*:\s*([0-9.]+)", text, flags=re.MULTILINE)
    values = {name: float(value) for name, value in matches}
    if values:
        return values
    return {"Crisis": 0.000085, "Late_Cycle": 0.000562, "Expansion": 0.000249}


def compute_regime_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = (
        df.dropna(subset=["regime_label"])
        .groupby("regime_label")
        .agg(
            Count=("regime_label", "size"),
            Avg_VIX=("india_vix", "mean"),
            Avg_Return=("bank_return", "mean"),
            Std_Return=("bank_return", "std"),
        )
        .reindex(["Crisis", "Late_Cycle", "Expansion"])
        .reset_index()
        .rename(columns={"regime_label": "Regime"})
    )
    stats["%"] = stats["Count"] / stats["Count"].sum() * 100
    stats["Sharpe"] = (stats["Avg_Return"] * 252) / (stats["Std_Return"] * np.sqrt(252))
    return stats[["Regime", "Count", "%", "Avg_VIX", "Avg_Return", "Sharpe"]]


def style_regime_table(df: pd.DataFrame) -> Styler:
    def row_style(row: pd.Series) -> List[str]:
        if row["Regime"] == "Crisis":
            return ["background-color: rgba(217, 83, 79, 0.12)"] * len(row)
        if row["Regime"] == "Expansion":
            return ["background-color: rgba(44, 162, 95, 0.12)"] * len(row)
        return ["background-color: rgba(240, 173, 78, 0.12)"] * len(row)

    styled = df.style.apply(row_style, axis=1).format(
        {
            "%": "{:.1f}%",
            "Avg_VIX": "{:.2f}",
            "Avg_Return": "{:.3%}",
            "Sharpe": "{:.2f}",
        }
    )
    return styled


def style_prediction_table(df: pd.DataFrame) -> Styler:
    def signal_style(col: pd.Series) -> List[str]:
        styles = []
        for value in col:
            if value == "BUY":
                styles.append("background-color: rgba(44, 162, 95, 0.18)")
            elif value == "SELL":
                styles.append("background-color: rgba(217, 83, 79, 0.18)")
            else:
                styles.append("background-color: rgba(108, 117, 125, 0.12)")
        return styles

    return df.style.apply(signal_style, subset=["Signal"]).format(
        {"Predicted Return": "{:+.2f}%", "Predicted Price": "₹{:,.0f}"}
    )


def build_price_chart(actual_df: pd.DataFrame, forecast_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    actual_df = actual_df.sort_values("date").copy()
    forecast_df = forecast_df.sort_values("date").copy()
    if not actual_df.empty and not forecast_df.empty:
        forecast_df = forecast_df.loc[forecast_df["date"] > actual_df["date"].max()].copy()

    fig.add_trace(
        go.Scatter(
            x=actual_df["date"],
            y=actual_df["bank_close"],
            mode="lines",
            name="Actual",
            line={"color": "#1f77b4", "width": 2.5},
            hovertemplate="%{x|%Y-%m-%d}<br>₹%{y:,.0f}<extra></extra>",
        )
    )

    if not forecast_df.empty:
        fig.add_trace(
            go.Scatter(
                x=forecast_df["date"],
                y=forecast_df["upper_band"],
                mode="lines",
                line={"color": "rgba(255,127,14,0)"},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_df["date"],
                y=forecast_df["lower_band"],
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(255,127,14,0.15)",
                line={"color": "rgba(255,127,14,0)"},
                name="Confidence band",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_df["date"],
                y=forecast_df["predicted_price"],
                mode="lines",
                name="Forecast",
                line={"color": "#ff7f0e", "dash": "dash", "width": 3},
                hovertemplate="%{x|%Y-%m-%d}<br>₹%{y:,.0f}<extra></extra>",
            )
        )

        forecast_start = forecast_df["date"].iloc[0]
        fig.add_vline(x=forecast_start, line_dash="dot", line_color="#ff7f0e")
        fig.add_annotation(x=forecast_start, y=float(actual_df["bank_close"].max()), text="Forecast start", showarrow=False, yshift=12)

    regime_series = actual_df[["date", "regime_label_live"]].dropna().copy()
    if not regime_series.empty:
        regime_series["block"] = (regime_series["regime_label_live"] != regime_series["regime_label_live"].shift()).cumsum()
        for _, block in regime_series.groupby("block"):
            regime = block["regime_label_live"].iloc[0]
            fig.add_vrect(
                x0=block["date"].iloc[0],
                x1=block["date"].iloc[-1],
                fillcolor=REGIME_BG_COLORS.get(regime, "rgba(0,0,0,0.04)"),
                line_width=0,
                layer="below",
            )

    fig.update_layout(
        title="Nifty Bank Index — Price History & 30-Day Forecast",
        template="plotly_white",
        hovermode="x unified",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    fig.update_yaxes(title_text="Price")
    return fig


def build_feature_chart(contributions: pd.DataFrame) -> go.Figure:
    if contributions.empty:
        fig = go.Figure()
        fig.update_layout(
            title="What's driving tomorrow's prediction",
            template="plotly_dark",
            height=420,
            annotations=[
                {
                    "text": "Contribution data unavailable",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                    "font": {"size": 16},
                }
            ],
            margin={"l": 20, "r": 20, "t": 60, "b": 20},
        )
        return fig

    chart_df = contributions.sort_values("abs_value").tail(8).copy()
    chart_df["color"] = np.where(chart_df["value"] >= 0, "#2ca25f", "#d9534f")
    fig = go.Figure(
        go.Bar(
            x=chart_df["value"],
            y=chart_df["feature"],
            orientation="h",
            marker_color=chart_df["color"],
            text=[f"{v:+.4f}" for v in chart_df["value"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="What's driving tomorrow's prediction",
        template="plotly_dark",
        height=420,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        xaxis_title="SHAP contribution",
        yaxis_title="",
    )
    return fig


def build_contribution_fallback() -> pd.DataFrame:
    cached = load_json(PREDICTION_JSON_PATH)
    top_features = cached.get("top_features", [])
    if not top_features:
        return pd.DataFrame(columns=["feature", "value", "abs_value"])
    rows = []
    for item in top_features:
        value = float(item.get("shap_value", 0.0))
        rows.append({"feature": item.get("name", "feature"), "value": value, "abs_value": abs(value)})
    return pd.DataFrame(rows).sort_values("abs_value", ascending=False)


def build_live_context_prediction(
    history_df: pd.DataFrame,
    artifacts: Dict[str, object],
    sector: str = "bank",
) -> Tuple[Dict[str, object], pd.DataFrame]:
    config = artifacts.get("config", {})
    hmm_model = artifacts.get("hmm_model")
    hmm_scaler = artifacts.get("hmm_scaler")
    if hmm_model is None or hmm_scaler is None:
        raise ValueError("HMM artifacts are unavailable.")

    market_data, live_available, fetch_time = fetch_live_market_data()
    featured = pt.engineer_features(market_data)
    state_mapping = pt.load_state_mapping(history_df)
    hmm_features = config.get("hmm_features", ["nifty_return", "india_vix", "bank_spread", "gold_nifty_corr_30d", "nifty_vol"])
    featured = pt.attach_regime_context(featured, hmm_model, hmm_scaler, state_mapping, hmm_features)

    base_features = config.get("base_features") or (artifacts.get("ensemble") or {}).get("base_features") or []
    required = ["date", "bank_close"] + base_features + hmm_features
    usable = featured.dropna(subset=required).copy()
    if usable.empty:
        raise ValueError("No usable live feature rows available after engineering.")

    stability_window = usable.dropna(subset=hmm_features).tail(30).copy()
    scaled = hmm_scaler.transform(stability_window[hmm_features].to_numpy(dtype=float))
    hard_states = hmm_model.predict(scaled)
    probs = hmm_model.predict_proba(scaled)
    hard_state = int(hard_states[-1])
    current_regime = state_mapping.get(hard_state, f"State_{hard_state}")

    regime_probabilities = {label: 0.0 for label in config.get("regime_order", ["Crisis", "Late_Cycle", "Expansion"])}
    for idx, prob in enumerate(probs[-1]):
        label = state_mapping.get(int(idx), f"State_{int(idx)}")
        if label in regime_probabilities:
            regime_probabilities[label] = float(prob)

    row = usable.iloc[[-1]].copy()
    current_price = float(row["bank_close"].iloc[0])

    if sector.lower() != "bank":
        st.warning("Only Bank live forecasting is available in the saved artifacts. Showing Bank forecast.")

    model_lookup = {
        "Crisis": artifacts.get("Crisis_bank.pkl"),
        "Late_Cycle": artifacts.get("Late_Cycle_bank.pkl"),
        "Expansion": artifacts.get("Expansion_bank.pkl"),
    }
    breakdown: Dict[str, float] = {}
    for regime, model in model_lookup.items():
        if model is None:
            breakdown[regime] = np.nan
            continue
        context = row[base_features].copy()
        for label in ["Crisis", "Late_Cycle", "Expansion"]:
            context[f"is_{label.lower()}"] = 1.0 if label == regime else 0.0
        breakdown[regime] = float(model.predict(context)[0])

    threshold_decimal = float(config.get("best_threshold_decimal", 0.003))
    ensemble = artifacts.get("ensemble")
    if ensemble is None:
        st.warning(f"Ensemble not found, using {current_regime}_bank model")
        predicted_return = breakdown.get(current_regime, 0.0)
    else:
        predicted_return = 0.0
        for regime, prob in regime_probabilities.items():
            value = breakdown.get(regime, np.nan)
            if pd.notna(value):
                predicted_return += prob * value

    if predicted_return > threshold_decimal:
        signal, signal_display, delta_color = "BUY", "BUY 📈", "normal"
    elif predicted_return < -threshold_decimal:
        signal, signal_display, delta_color = "SELL", "SELL 📉", "inverse"
    else:
        signal, signal_display, delta_color = "HOLD", "HOLD ➡️", "off"

    predicted_price = current_price * float(np.exp(predicted_return))
    regime_vol = pt.get_regime_vol(history_df, current_regime)
    upper = current_price * float(np.exp(predicted_return + 1.96 * regime_vol))
    lower = current_price * float(np.exp(predicted_return - 1.96 * regime_vol))

    transition_prob = float(getattr(hmm_model, "transmat_", np.eye(3))[hard_state, hard_state])
    confidence_prob = regime_probabilities.get(current_regime, 0.0)
    confidence_text = pt.confidence_label(confidence_prob)

    contributions = pd.DataFrame(columns=["feature", "value", "abs_value"])
    current_model = model_lookup.get(current_regime)
    if current_model is not None and shap is not None:
        try:
            context = row[base_features].copy()
            for label in ["Crisis", "Late_Cycle", "Expansion"]:
                context[f"is_{label.lower()}"] = 1.0 if label == current_regime else 0.0
            explainer = shap.TreeExplainer(current_model)
            shap_values = np.asarray(explainer.shap_values(context))[0]
            contributions = (
                pd.DataFrame({"feature": context.columns, "value": shap_values})
                .assign(abs_value=lambda d: d["value"].abs())
                .sort_values("abs_value", ascending=False)
            )
        except Exception:
            contributions = pd.DataFrame(columns=["feature", "value", "abs_value"])

    result = {
        "fetch_time": fetch_time,
        "live_available": live_available,
        "current_regime": current_regime,
        "transition_prob": transition_prob,
        "signal": signal,
        "signal_display": signal_display,
        "delta_color": delta_color,
        "predicted_return": predicted_return,
        "predicted_price": predicted_price,
        "current_price": current_price,
        "upper": upper,
        "lower": lower,
        "confidence_prob": confidence_prob,
        "confidence_text": confidence_text,
        "regime_probabilities": regime_probabilities,
        "model_breakdown": breakdown,
        "latest_date": pd.Timestamp(row["date"].iloc[0]),
        "contributions": contributions,
    }
    return result, usable


def fallback_forecast_from_prediction(prediction: Dict[str, object], horizon: int) -> pd.DataFrame:
    start = pd.Timestamp(prediction["latest_date"]) + pd.Timedelta(days=1)
    dates = pd.bdate_range(start=start, periods=horizon)
    rows = []
    last_price = float(prediction["current_price"])
    regime = str(prediction["current_regime"])
    regime_vol = (np.log(float(prediction["upper"]) / float(prediction["lower"])) / (2 * 1.96)) if prediction["lower"] > 0 else 0.02
    pred_return = float(prediction["predicted_return"])
    for step, forecast_date in enumerate(dates, start=1):
        last_price = last_price * np.exp(pred_return)
        band = 1.96 * regime_vol * np.sqrt(step)
        rows.append(
            {
                "date": forecast_date,
                "regime": regime,
                "predicted_return_pct": pred_return * 100,
                "predicted_price": last_price,
                "lower_band": last_price * np.exp(-band),
                "upper_band": last_price * np.exp(band),
            }
        )
    return pd.DataFrame(rows)


def build_signal_forecast(prediction: Dict[str, object], horizon: int) -> pd.DataFrame:
    horizon = max(1, int(horizon))
    return fallback_forecast_from_prediction(prediction, horizon)


def render_dashboard() -> None:
    history_df = load_regime_data()
    artifacts = load_artifacts()
    for error in artifacts.get("errors", []):
        st.error(error)

    st.markdown(
        """
        <style>
            .stMetric { padding: 1rem; border-radius: 12px;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.2); }
            [data-testid="metric-container"] {
                background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
                border: 1px solid rgba(148, 163, 184, 0.18);
                border-radius: 14px;
                padding: 1rem;
                min-height: 150px;
            }
            [data-testid="metric-container"] label,
            [data-testid="metric-container"] p,
            [data-testid="metric-container"] div {
                color: #f8fafc !important;
            }
            [data-testid="metric-container"] [data-testid="stMetricValue"] {
                color: #ffffff !important;
                font-size: 2.2rem !important;
                font-weight: 700 !important;
            }
            [data-testid="metric-container"] [data-testid="stMetricDelta"] {
                color: #86efac !important;
                font-size: 1rem !important;
            }
            [data-testid="metric-container"] [data-testid="stMetricLabel"] {
                color: #cbd5e1 !important;
                font-weight: 600 !important;
            }
            .main-header { font-size: 2rem; font-weight: 600;
                           background: linear-gradient(90deg, #667eea, #764ba2);
                           -webkit-background-clip: text; color: transparent; }
            .sub-header { color: #94a3b8; font-size: 1rem; margin-top: -0.5rem; }
            .pill { padding: 0.2rem 0.6rem; border-radius: 999px; font-weight: 600; display: inline-block; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    top_left, top_right = st.columns([0.85, 0.15])
    with top_left:
        st.markdown(f'<div class="main-header">{emoji("📈", "")} Gold-Equity Regime Switching ML</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Live Market Intelligence Dashboard</div>', unsafe_allow_html=True)
    with top_right:
        if st.button(f"{emoji('🔄', '')} Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    with st.sidebar:
        st.header(f"{emoji('⚙️', 'Settings')} Controls")
        horizon = st.slider("Forecast horizon (days)", 1, 30, 5)
        sector = st.selectbox("Sector", ["Bank", "IT"])
        if st.checkbox("Show raw regime data"):
            st.dataframe(history_df.tail(30), use_container_width=True)

        st.header(f"{emoji('📋', 'Info')} Model Info")
        st.write("HMM: 3-state Gaussian")
        st.write("Predictor: XGBoost (regime-specific)")
        st.write("Training data: 2014–2022")
        st.write("Test window: 2023–2024")
        st.write("Features: 40+")

        st.header(f"{emoji('⚠️', 'Note')} Disclaimer")
        st.warning("This is an academic ML project. Not financial advice.")

    try:
        prediction, usable_live = build_live_context_prediction(history_df, artifacts, sector=sector.lower())
    except Exception as exc:  # noqa: BLE001
        st.error(f"Live prediction failed: {exc}")
        cached_prediction = load_json(PREDICTION_JSON_PATH)
        if not cached_prediction:
            st.stop()
        prediction = {
            "fetch_time": cached_prediction.get("generated_at", "cached"),
            "live_available": False,
            "current_regime": cached_prediction.get("regime", "Unknown"),
            "transition_prob": 0.0,
            "signal": cached_prediction.get("signal", "HOLD"),
            "signal_display": cached_prediction.get("signal", "HOLD"),
            "delta_color": "off",
            "predicted_return": cached_prediction.get("predicted_return_pct", 0.0) / 100.0,
            "predicted_price": cached_prediction.get("predicted_price", np.nan),
            "current_price": cached_prediction.get("current_price", np.nan),
            "upper": cached_prediction.get("confidence_band", {}).get("upper", np.nan),
            "lower": cached_prediction.get("confidence_band", {}).get("lower", np.nan),
            "confidence_prob": cached_prediction.get("regime_confidence", 0.0),
            "confidence_text": pt.confidence_label(cached_prediction.get("regime_confidence", 0.0)),
            "regime_probabilities": cached_prediction.get("regime_probabilities", {"Crisis": 0.0, "Late_Cycle": 0.0, "Expansion": 0.0}),
            "model_breakdown": {"Crisis": np.nan, "Late_Cycle": np.nan, "Expansion": np.nan},
            "latest_date": history_df["date"].max(),
        }
        usable_live = history_df.tail(120).copy()
        usable_live["regime_label_live"] = usable_live["regime_label"]

    st.caption(f"Data as of: {prediction['fetch_time']}")
    if not prediction["live_available"]:
        st.warning("Using cached data — live fetch failed")

    metric_cols = st.columns(4)
    current_regime = prediction["current_regime"]
    regime_color = REGIME_COLORS.get(current_regime, "#6c757d")
    metric_cols[0].metric(
        f"{emoji('🎯', '')} Current Market Regime",
        current_regime,
        f"Transition prob: {prediction['transition_prob']:.1%}",
    )
    metric_cols[0].markdown(
        f"<span class='pill' style='background:{regime_color}; color:white;'>{current_regime}</span>",
        unsafe_allow_html=True,
    )
    metric_cols[1].metric(
        f"{emoji('📊', '')} Tomorrow's Signal",
        prediction["signal_display"],
        f"Predicted return: {prediction['predicted_return']:+.2%}",
        delta_color=prediction["delta_color"],
    )
    metric_cols[2].metric(
        f"{emoji('💰', '')} Predicted Bank Price",
        f"₹{prediction['predicted_price']:,.0f}",
        f"{(prediction['predicted_price'] / prediction['current_price'] - 1):+.2%} from ₹{prediction['current_price']:,.0f}",
    )
    metric_cols[3].metric(
        f"{emoji('🎲', '')} Signal Confidence",
        prediction["confidence_text"],
        f"HMM prob: {prediction['confidence_prob'] * 100:.1f}%",
    )

    forecast_df = build_signal_forecast(prediction, 30)
    actual_prices = usable_live[["date", "bank_close", "regime_label_live"]].dropna(subset=["bank_close"]).tail(120).copy()
    st.plotly_chart(build_price_chart(actual_prices, forecast_df), use_container_width=True)

    left_col, right_col = st.columns([0.6, 0.4])
    with left_col:
        st.subheader(f"{emoji('📊', '')} Regime Detection")
        regime_counts = history_df["regime_label"].dropna().value_counts().reindex(["Crisis", "Late_Cycle", "Expansion"]).fillna(0)
        pie_df = pd.DataFrame({"Regime": regime_counts.index, "Count": regime_counts.values})
        pie_fig = px.pie(
            pie_df,
            names="Regime",
            values="Count",
            color="Regime",
            color_discrete_map=REGIME_COLORS,
            hole=0.45,
        )
        pie_fig.update_traces(textinfo="percent+label")
        st.plotly_chart(pie_fig, use_container_width=True)

        regime_stats = compute_regime_stats(history_df)
        st.dataframe(style_regime_table(regime_stats), use_container_width=True)

        timeline_path = VIZ_DIR / "regime_timeline.png"
        if timeline_path.exists():
            st.image(str(timeline_path), caption="Historical regime classification", use_container_width=True)

    with right_col:
        st.subheader(f"{emoji('🔮', '')} Prediction Breakdown")
        if shap is None:
            st.warning(f"SHAP is not installed in this environment. Feature contribution chart is limited. Error: {SHAP_IMPORT_ERROR}")
        contribution_df = prediction.get("contributions", pd.DataFrame())
        if contribution_df is None or contribution_df.empty:
            contribution_df = build_contribution_fallback()
        st.plotly_chart(build_feature_chart(contribution_df), use_container_width=True)
        st.info(REGIME_INTERPRETATION.get(prediction["current_regime"], "Model interpretation unavailable."))

    with st.expander(f"{emoji('📉', '')} Model Performance & Backtesting Results"):
        perf_cols = st.columns(2)
        with perf_cols[0]:
            perf_df = pd.DataFrame(
                [
                    {"Model": "Regime model", "R²": "0.000", "RMSE": "0.79%", "Direction%": "52.2%", "Sharpe": "6.33", "Max DD": "-11.94%"},
                    {"Model": "Global model", "R²": "-0.072", "RMSE": "0.89%", "Direction%": "52.2%", "Sharpe": "1.91", "Max DD": "-4.26%"},
                    {"Model": "Buy & Hold", "R²": "—", "RMSE": "—", "Direction%": "53.93%", "Sharpe": "0.58", "Max DD": "-11.89%"},
                ]
            )
            st.dataframe(perf_df, use_container_width=True)
        with perf_cols[1]:
            returns_path = VIZ_DIR / "cumulative_returns.png"
            if returns_path.exists():
                st.image(str(returns_path), caption="Backtest cumulative returns", use_container_width=True)

    with st.expander(f"{emoji('🔬', '')} SHAP Feature Importance by Regime"):
        st.write("Gold's predictive role changes dramatically across regimes")
        if shap is None:
            st.warning(f"SHAP is optional for this dashboard. Install it to enable live SHAP-based contributions. Error: {SHAP_IMPORT_ERROR}")
        shap_cols = st.columns(3)
        shap_images = [
            ("Crisis", "Crisis: Gold rank #11 — liquidity dominates"),
            ("Late_Cycle", "Late_Cycle: Gold rank #1 — direct fear signal"),
            ("Expansion", "Expansion: Gold rank #4 — structural signal"),
        ]
        for col, (regime, caption) in zip(shap_cols, shap_images):
            image_path = VIZ_DIR / f"shap_summary_{regime}.png"
            if image_path.exists():
                col.image(str(image_path), caption=caption, use_container_width=True)

        gold_shap = parse_gold_shap_values()
        shap_bar = go.Figure(
            go.Bar(
                x=list(gold_shap.keys()),
                y=list(gold_shap.values()),
                marker_color=[REGIME_COLORS[k] for k in gold_shap.keys()],
                text=[f"{v:.6f}" for v in gold_shap.values()],
                textposition="outside",
            )
        )
        shap_bar.update_layout(
            title="Gold's Predictive Power by Market Regime",
            template="plotly_white",
            yaxis_title="Mean absolute SHAP value",
            xaxis_title="Regime",
        )
        st.plotly_chart(shap_bar, use_container_width=True)

    forecast_preview = build_signal_forecast(prediction, horizon).head(5).copy()
    if forecast_preview.empty:
        forecast_preview = fallback_forecast_from_prediction(prediction, 5).head(5)
    forecast_preview["Signal"] = np.where(
        forecast_preview["predicted_return_pct"] > 0.3,
        "BUY",
        np.where(forecast_preview["predicted_return_pct"] < -0.3, "SELL", "HOLD"),
    )
    forecast_preview = forecast_preview.rename(
        columns={
            "date": "Date",
            "regime": "Regime",
            "predicted_return_pct": "Predicted Return",
            "predicted_price": "Predicted Price",
        }
    )
    st.subheader("5-Day Forecast Table")
    st.dataframe(style_prediction_table(forecast_preview[["Date", "Regime", "Predicted Return", "Predicted Price", "Signal"]]), use_container_width=True)


render_dashboard()

# To run: streamlit run dashboard.py
# Opens automatically at http://localhost:8501
