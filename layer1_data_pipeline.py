from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats.mstats import winsorize


TICKERS: Dict[str, str] = {
    "gold": "GC=F",
    "nifty": "^NSEI",
    "bank": "^NSEBANK",
    "it": "^CNXIT",
    "metal": "NIFTYMETAL.NS",
    "fmcg": "NIFTYFMCG.NS",
    "vix": "^INDIAVIX",
    "usdinr": "USDINR=X",
    "oil": "CL=F",
    "silver": "SI=F",
}

FEATURE_ASSETS = ["gold", "nifty", "bank", "it", "metal", "fmcg", "oil", "silver", "usdinr"]
CORRELATION_ASSETS = ["nifty", "bank", "it", "metal"]
START_DATE = "2014-01-01"
END_DATE = "2024-12-31"
MASTER_CALENDAR = pd.date_range(start=START_DATE, end=END_DATE, freq="B")
OUTPUT_PATH = Path("data") / "gold_equity_master_data.csv"


def log_progress(message: str) -> None:
    print(f"[INFO] {message}")


def fetch_single_ticker(asset_name: str, ticker: str, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    log_progress(f"Fetching {asset_name} ({ticker}) from Yahoo Finance")

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            df = yf.download(
                tickers=ticker,
                start=START_DATE,
                end=END_DATE,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )

        if df.empty:
            raise ValueError("Received an empty dataframe from yfinance.")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        expected_columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        df = df.reindex(columns=expected_columns)
        df.index = pd.to_datetime(df.index)
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()
        df = df.reindex(calendar)

        renamed = {
            "Open": f"{asset_name}_open",
            "High": f"{asset_name}_high",
            "Low": f"{asset_name}_low",
            "Close": f"{asset_name}_close",
            "Adj Close": f"{asset_name}_adj_close",
            "Volume": f"{asset_name}_volume",
        }
        return df.rename(columns=renamed)
    except Exception as exc:  # noqa: BLE001 - keep pipeline resilient ticker-by-ticker
        log_progress(f"Fetch failed for {asset_name} ({ticker}): {exc}. Filling with NaNs.")
        nan_frame = pd.DataFrame(index=calendar)
        for suffix in ["open", "high", "low", "close", "adj_close", "volume"]:
            nan_frame[f"{asset_name}_{suffix}"] = np.nan
        return nan_frame


def fetch_market_data() -> pd.DataFrame:
    log_progress("Starting raw market data fetch")
    frames = [fetch_single_ticker(asset_name, ticker, MASTER_CALENDAR) for asset_name, ticker in TICKERS.items()]
    market_data = pd.concat(frames, axis=1)
    market_data.index.name = "date"
    return market_data


def compute_corwin_schultz_spread(high: pd.Series, low: pd.Series) -> pd.Series:
    safe_high = high.astype(float).where(high > 0)
    safe_low = low.astype(float).where(low > 0)
    ratio = (safe_high / safe_low).where((safe_high.notna()) & (safe_low.notna()))
    log_hl = np.log(ratio.where(ratio > 0))
    exp_term = np.exp(0.5 * log_hl)
    spread = 2 * (exp_term - 1) / (1 + exp_term)
    return spread.replace([np.inf, -np.inf], np.nan)


def winsorize_series(series: pd.Series, lower_limit: float = 0.01, upper_limit: float = 0.01) -> pd.Series:
    non_null = series.dropna()
    if non_null.empty:
        return series

    winsorized_values = winsorize(non_null.to_numpy(), limits=[lower_limit, upper_limit])
    result = series.copy()
    result.loc[non_null.index] = np.asarray(winsorized_values, dtype=float)
    return result


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    log_progress("Engineering return, volatility, momentum, lag, spread, illiquidity, and interaction features")

    features = data.copy()

    nifty_volume_millions = features["nifty_volume"].replace(0, np.nan) / 1_000_000

    for asset in FEATURE_ASSETS:
        close_col = f"{asset}_close"
        high_col = f"{asset}_high"
        low_col = f"{asset}_low"

        asset_close = features[close_col].astype(float)
        asset_high = features[high_col].astype(float)
        asset_low = features[low_col].astype(float)

        return_col = f"{asset}_return"
        price_ratio = (asset_close / asset_close.shift(1)).where((asset_close > 0) & (asset_close.shift(1) > 0))
        features[return_col] = np.log(price_ratio)
        features[f"{asset}_volatility_20d"] = features[return_col].rolling(window=20, min_periods=20).std() * np.sqrt(252)
        features[f"{asset}_momentum_20d"] = (asset_close - asset_close.shift(20)) / asset_close.shift(20)
        features[f"{asset}_lag1_return"] = features[return_col].shift(1)
        features[f"{asset}_lag5_return"] = features[return_col].shift(5)
        features[f"{asset}_lag20_return"] = features[return_col].shift(20)
        features[f"{asset}_corwin_schultz_spread"] = compute_corwin_schultz_spread(asset_high, asset_low)
        features[f"{asset}_amihud_illiquidity"] = np.abs(features[return_col]) / nifty_volume_millions

    for asset in CORRELATION_ASSETS:
        features[f"gold_{asset}_corr_30d"] = (
            features["gold_return"]
            .rolling(window=30, min_periods=30)
            .corr(features[f"{asset}_return"])
        )

    features["india_vix"] = features["vix_close"]
    features["gold_vix_interaction"] = features["gold_return"] * features["india_vix"]
    features["gold_spread_interaction"] = features["gold_return"] * features["bank_corwin_schultz_spread"]
    features["gold_silver_ratio"] = features["gold_close"] / features["silver_close"]

    return features


def apply_data_quality(data: pd.DataFrame) -> pd.DataFrame:
    log_progress("Applying forward-fill, missing-data filtering, winsorization, and rolling-window cleanup")

    cleaned = data.copy()
    cleaned = cleaned.ffill(limit=5)

    # Exclude permanently missing columns from the row-level missing filter so a single
    # failed ticker does not force the entire dataset to be dropped.
    active_columns = cleaned.columns[cleaned.notna().any(axis=0)]
    active_view = cleaned[active_columns]
    missing_threshold = int(np.floor(active_view.shape[1] * 0.2))
    cleaned = cleaned.loc[active_view.isna().sum(axis=1) <= missing_threshold].copy()

    return_columns = [
        column
        for column in cleaned.columns
        if column.endswith("_return")
        or column.endswith("_lag1_return")
        or column.endswith("_lag5_return")
        or column.endswith("_lag20_return")
    ]
    for column in return_columns:
        cleaned[column] = winsorize_series(cleaned[column])

    cleaned = cleaned.iloc[20:].copy()
    return cleaned


def build_summary_table(data: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "mean": data.mean(numeric_only=True),
            "std": data.std(numeric_only=True),
            "min": data.min(numeric_only=True),
            "max": data.max(numeric_only=True),
            "missing%": data.isna().mean() * 100,
        }
    )
    return summary


def main() -> None:
    log_progress("Initializing Layer 1 data pipeline")

    raw_data = fetch_market_data()
    feature_data = engineer_features(raw_data)
    final_data = apply_data_quality(feature_data)

    log_progress("Saving master dataset to disk")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_data.to_csv(OUTPUT_PATH)

    log_progress("Building and printing summary statistics")
    summary = build_summary_table(final_data)
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 240):
        print("\nSummary statistics by column:")
        print(summary.round(6).to_string())

    print(f"\nFinal dataframe shape: {final_data.shape}")
    print(f"Saved master dataset to: {OUTPUT_PATH.resolve()}")

    assert final_data.shape[0] > 2000, f"Expected more than 2000 rows, found {final_data.shape[0]}"
    assert final_data.shape[1] > 45, f"Expected more than 45 columns, found {final_data.shape[1]}"

    log_progress("Layer 1 data pipeline completed successfully")


if __name__ == "__main__":
    main()
