from __future__ import annotations

import pandas as pd

from .config import DEFAULT_WEIGHTS, SIGNAL_THRESHOLDS, ScoreWeights


def _safe(v, default=0.0):
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def market_regime(index_df: pd.DataFrame, stock_latest: pd.DataFrame | None = None) -> dict:
    x = index_df.iloc[-1]
    score = 0
    reasons: list[str] = []

    close = _safe(x.get("close"))
    ema20 = _safe(x.get("ema20"))
    ma50 = _safe(x.get("ma50"))
    ma200 = _safe(x.get("ma200"))
    rsi = _safe(x.get("rsi14"), 50)
    ret20 = _safe(x.get("ret_20d"))

    if close > ema20:
        score += 20; reasons.append("Chỉ số đang nằm trên EMA20")
    if ema20 > ma50:
        score += 18; reasons.append("EMA20 đang nằm trên MA50")
    if ma50 > ma200:
        score += 17; reasons.append("MA50 đang nằm trên MA200")
    if 52 <= rsi <= 72:
        score += 15; reasons.append("Động lượng RSI đang ở vùng tích cực")
    elif rsi > 72:
        score += 8; reasons.append("Động lượng mạnh nhưng RSI đã ở vùng cao")
    if ret20 > 0:
        score += 10; reasons.append("Lợi suất 20 phiên đang dương")

    breadth = 50.0
    if stock_latest is not None and not stock_latest.empty and "change_pct" in stock_latest.columns:
        breadth = float((stock_latest["change_pct"] > 0).mean() * 100)
        score += max(0, min(20, breadth / 5))
        if breadth >= 60:
            reasons.append("Độ rộng thị trường tích cực")
        elif breadth <= 40:
            reasons.append("Độ rộng thị trường yếu")

    score = int(round(max(0, min(100, score))))
    if score >= 72:
        label, risk = "TÍCH CỰC", "TRUNG BÌNH"
    elif score >= 50:
        label, risk = "TRUNG LẬP", "TRUNG BÌNH"
    else:
        label, risk = "PHÒNG THỦ", "CAO"

    return {
        "score": score,
        "label": label,
        "risk": risk,
        "breadth": round(breadth, 1),
        "reasons": reasons[:4],
    }


def score_stock(row: pd.Series, market_score: int, weights: ScoreWeights = DEFAULT_WEIGHTS) -> dict:
    close = _safe(row.get("close"))
    ema20 = _safe(row.get("ema20"))
    ma50 = _safe(row.get("ma50"))
    ma200 = _safe(row.get("ma200"))
    rsi = _safe(row.get("rsi14"), 50)
    macd = _safe(row.get("macd"))
    macd_signal = _safe(row.get("macd_signal"))
    ret20 = _safe(row.get("ret_20d"))
    volr = _safe(row.get("volume_ratio20"), 1)
    rs20 = _safe(row.get("relative_strength20"))
    atrp = _safe(row.get("atr_pct"), 3)
    change = _safe(row.get("change_pct", row.get("ret_1d", 0)))
    breakout = bool(row.get("breakout20", False))
    breakdown = bool(row.get("breakdown20", False))

    reasons: list[str] = []

    market_component = weights.market * market_score / 100

    trend_raw = 0.0
    if close > ema20: trend_raw += 5; reasons.append("Giá trên EMA20")
    if ema20 > ma50: trend_raw += 5; reasons.append("EMA20 trên MA50")
    if ma50 > ma200: trend_raw += 7; reasons.append("MA50 trên MA200")
    if ret20 > 0: trend_raw += 4
    if rs20 > 0: trend_raw += 4; reasons.append("Mạnh hơn VN-Index")
    trend_component = weights.trend * trend_raw / 25

    momentum_raw = 0.0
    if 50 <= rsi <= 68: momentum_raw += 9; reasons.append(f"RSI {rsi:.0f} tích cực")
    elif 40 <= rsi < 50 or 68 < rsi <= 75: momentum_raw += 5
    elif rsi > 75: momentum_raw += 2
    if macd > macd_signal: momentum_raw += 7; reasons.append("MACD cho tín hiệu tăng")
    if macd > 0: momentum_raw += 4
    momentum_component = weights.momentum * momentum_raw / 20

    volume_raw = 0.0
    if volr >= 2: volume_raw += 8; reasons.append(f"Khối lượng gấp {volr:.1f} lần TB20")
    elif volr >= 1.3: volume_raw += 6; reasons.append(f"Khối lượng gấp {volr:.1f} lần TB20")
    elif volr >= 0.9: volume_raw += 3
    if breakout: volume_raw += 7; reasons.append("Vượt đỉnh 20 phiên")
    elif change > 1.5 and volr >= 1.2: volume_raw += 4
    volume_component = weights.volume * min(volume_raw, 15) / 15

    risk_raw = 10.0
    if atrp > 6: risk_raw -= 4
    elif atrp > 4.5: risk_raw -= 2
    if rsi > 80: risk_raw -= 2
    if breakdown: risk_raw = min(risk_raw, 2); reasons.append("Đã thủng hỗ trợ 20 phiên")
    risk_component = weights.risk * max(0, risk_raw) / 10

    attention_raw = min(4, abs(change) / 1.5) + min(4, max(0, volr - 1) * 2) + (2 if breakout or breakdown else 0)
    attention_component = weights.attention * min(10, attention_raw) / 10

    total = market_component + trend_component + momentum_component + volume_component + risk_component + attention_component

    if breakdown:
        total = min(total, 42)
    if market_score < 38:
        total = min(total, 68)

    score = int(round(max(0, min(100, total))))
    signal = signal_from_score(score)
    if breakdown and signal in {"MUA MẠNH", "MUA", "THEO DÕI"}:
        signal = "GIẢM TỶ TRỌNG"

    confidence = int(round(min(95, max(50, 52 + abs(score - 50) * 0.75))))
    reason = " + ".join(reasons[:4]) if reasons else "Các tín hiệu kỹ thuật đang đan xen"

    return {
        "score": score,
        "signal": signal,
        "confidence": confidence,
        "reason": reason,
        "attention": round(attention_component, 1),
        "components": {
            "Thị trường": round(market_component, 1),
            "Xu hướng": round(trend_component, 1),
            "Động lượng": round(momentum_component, 1),
            "Khối lượng": round(volume_component, 1),
            "Rủi ro": round(risk_component, 1),
            "Mức chú ý": round(attention_component, 1),
        },
    }


def signal_from_score(score: int) -> str:
    for label, threshold in SIGNAL_THRESHOLDS.items():
        if score >= threshold:
            return label
    return "BÁN"


