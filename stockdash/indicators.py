from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    return result.fillna(50)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean().bfill()


def add_indicators(df: pd.DataFrame, benchmark: pd.DataFrame | None = None) -> pd.DataFrame:
    out = df.copy().sort_values("date").reset_index(drop=True)
    c = out["close"].astype(float)
    v = out["volume"].astype(float)

    out["ret_1d"] = c.pct_change() * 100
    out["ret_5d"] = c.pct_change(5) * 100
    out["ret_20d"] = c.pct_change(20) * 100
    out["ma20"] = c.rolling(20).mean()
    out["ma50"] = c.rolling(50).mean()
    out["ma200"] = c.rolling(200).mean()
    out["ema20"] = c.ewm(span=20, adjust=False).mean()
    out["ema50"] = c.ewm(span=50, adjust=False).mean()
    out["rsi14"] = rsi(c, 14)

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    std20 = c.rolling(20).std()
    out["bb_upper"] = out["ma20"] + 2 * std20
    out["bb_lower"] = out["ma20"] - 2 * std20
    out["atr14"] = atr(out, 14)
    out["atr_pct"] = out["atr14"] / c.replace(0, np.nan) * 100
    out["vol_ma20"] = v.rolling(20).mean()
    out["volume_ratio20"] = v / out["vol_ma20"].replace(0, np.nan)

    # Use prior rolling levels so today's close can actually break them.
    out["resistance20"] = out["high"].shift(1).rolling(20).max()
    out["support20"] = out["low"].shift(1).rolling(20).min()
    out["breakout20"] = c > out["resistance20"]
    out["breakdown20"] = c < out["support20"]
    out["volatility20"] = c.pct_change().rolling(20).std() * np.sqrt(252) * 100

    if benchmark is not None and len(benchmark) > 25:
        b = benchmark.copy().sort_values("date")[["date", "close"]].rename(columns={"close": "bench_close"})
        merged = out[["date", "close"]].merge(b, on="date", how="left")
        stock_r20 = merged["close"].pct_change(20) * 100
        bench_r20 = merged["bench_close"].pct_change(20) * 100
        out["relative_strength20"] = (stock_r20 - bench_r20).values
    else:
        out["relative_strength20"] = 0.0

    return out
