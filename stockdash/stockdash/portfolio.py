from __future__ import annotations

import numpy as np
import pandas as pd


def _num(v, default=np.nan):
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def _pct_distance(price, level):
    p, l = _num(price), _num(level)
    if pd.isna(p) or pd.isna(l) or l == 0:
        return np.nan
    return (p / l - 1) * 100


def portfolio_recommendation(r: pd.Series) -> tuple[str, str]:
    close = _num(r.get("close"))
    avg = _num(r.get("avg_cost"))
    pnl_pct = _num(r.get("pnl_pct"), 0)
    support = _num(r.get("support20"))
    resistance = _num(r.get("resistance20"))
    rsi = _num(r.get("rsi14"), 50)
    volr = _num(r.get("volume_ratio20"), 1)
    score = int(_num(r.get("score"), 0))
    signal = str(r.get("signal", "") or "")
    breakdown = bool(r.get("breakdown20", False))
    breakout = bool(r.get("breakout20", False))
    ema20 = _num(r.get("ema20"))
    ma50 = _num(r.get("ma50"))
    price_only = bool(r.get("price_only", False))

    if pd.isna(close):
        return (
            "CHƯA CÓ GIÁ THẬT",
            "Chưa nhận được giá thị trường cho mã này nên hệ thống không tính lãi/lỗ và không đưa ra hành động. Hãy bấm CẬP NHẬT GIÁ hoặc kết nối nguồn dữ liệu thật rồi thử lại.",
        )

    if price_only:
        pnl_text = f"Đang {'lãi' if pnl_pct >= 0 else 'lỗ'} {abs(pnl_pct):.1f}% so với giá vốn" if not pd.isna(avg) else "Đã nhận được giá thị trường hiện tại"
        return (
            "ĐÃ CÓ GIÁ THẬT - CHỜ PHÂN TÍCH KỸ THUẬT",
            f"{pnl_text}. Giá hiện tại đã được cập nhật từ bảng giá Vnstock bằng chế độ tiết kiệm request; RSI, xu hướng, hỗ trợ và kháng cự chưa được dùng cho khuyến nghị này. Nhập API key Vnstock miễn phí rồi bấm CẬP NHẬT TOÀN BỘ để có phân tích đầy đủ.",
        )

    reasons = []
    if not pd.isna(avg):
        reasons.append(f"vị thế hiện đang {'lãi' if pnl_pct >= 0 else 'lỗ'} {abs(pnl_pct):.1f}% so với giá vốn")
    if not pd.isna(ema20) and not pd.isna(ma50):
        if close > ema20 > ma50:
            reasons.append("giá đang trên EMA20 và EMA20 trên MA50, tức xu hướng ngắn hạn còn tích cực")
        elif close < ema20 and ema20 < ma50:
            reasons.append("giá đang dưới EMA20 và EMA20 dưới MA50, tức xu hướng ngắn hạn đang yếu")
        elif close > ema20:
            reasons.append("giá đang trên EMA20 nhưng cấu trúc xu hướng chưa thật sự đồng thuận")
        else:
            reasons.append("giá đang dưới EMA20 nên lực giá ngắn hạn chưa tốt")
    if rsi >= 75:
        reasons.append(f"RSI {rsi:.0f} ở vùng cao, cần đề phòng mua đuổi")
    elif rsi >= 52:
        reasons.append(f"RSI {rsi:.0f} cho thấy động lượng khá tích cực")
    elif rsi < 40:
        reasons.append(f"RSI {rsi:.0f} cho thấy động lượng yếu")
    if volr >= 1.5:
        reasons.append(f"khối lượng hiện tại cao khoảng {volr:.1f} lần mức trung bình 20 phiên")
    if breakout:
        reasons.append("giá vừa vượt vùng kháng cự 20 phiên")
    if breakdown:
        reasons.append("giá đã thủng vùng hỗ trợ 20 phiên")

    support_dist = _pct_distance(close, support)
    resistance_dist = _pct_distance(resistance, close)
    if not pd.isna(support_dist):
        reasons.append(f"giá đang cao hơn hỗ trợ khoảng {support_dist:.1f}%")
    if not pd.isna(resistance_dist):
        reasons.append(f"kháng cự gần nhất cao hơn giá hiện tại khoảng {resistance_dist:.1f}%")

    if breakdown or (signal in {"BÁN", "GIẢM TỶ TRỌNG"} and score < 45):
        rec = "ƯU TIÊN GIẢM TỶ TRỌNG / QUẢN TRỊ CẮT LỖ"
    elif pnl_pct <= -7 and score < 55:
        rec = "HẠ TỶ TRỌNG, KHÔNG BÌNH QUÂN GIÁ XUỐNG"
    elif pnl_pct >= 15 and score >= 55:
        rec = "NẮM GIỮ, DỜI MỐC BẢO VỆ LỢI NHUẬN"
    elif signal in {"MUA", "MUA MẠNH"} and score >= 72:
        rec = "NẮM GIỮ; CHỈ GIA TĂNG KHI GIÁ XÁC NHẬN"
    elif signal == "THEO DÕI":
        rec = "NẮM GIỮ / THEO DÕI, CHƯA CẦN MUA ĐUỔI"
    else:
        rec = "NẮM GIỮ VÀ THEO DÕI HỖ TRỢ"

    explanation = "; ".join(reasons[:6]).capitalize() + "."
    return rec, explanation


def portfolio_action_now(r: pd.Series) -> str:
    """Hành động ngắn gọn tại thời điểm dữ liệu hiện tại.

    Chỉ trả kết luận GIỮ TIẾP hoặc BÁN/GIẢM khi đã có chỉ báo kỹ thuật.
    Nếu mới có giá quote mà chưa có lịch sử, hệ thống nói rõ là chưa đủ dữ liệu
    thay vì suy đoán từ riêng lãi/lỗ.
    """
    close = _num(r.get("close"))
    if pd.isna(close):
        return "CHƯA CÓ GIÁ"
    if bool(r.get("price_only", False)):
        return "CHƯA ĐỦ DỮ LIỆU KỸ THUẬT"

    score = int(_num(r.get("score"), 0))
    signal = str(r.get("signal", "") or "").upper().strip()
    pnl_pct = _num(r.get("pnl_pct"), 0)
    breakdown = bool(r.get("breakdown20", False))
    ema20 = _num(r.get("ema20"))
    ma50 = _num(r.get("ma50"))

    if breakdown or signal == "BÁN":
        return "BÁN / GIẢM TỶ TRỌNG"
    if signal == "GIẢM TỶ TRỌNG" and score < 55:
        return "BÁN / GIẢM TỶ TRỌNG"
    if pnl_pct <= -7 and score < 55:
        return "BÁN / GIẢM TỶ TRỌNG"
    if pd.notna(ema20) and pd.notna(ma50) and close < ema20 < ma50 and score < 50:
        return "BÁN / GIẢM TỶ TRỌNG"
    return "GIỮ TIẾP"


def aggregate_portfolio(lots: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    if lots is None or lots.empty:
        return pd.DataFrame()

    x = lots.copy()
    for c in ["buy_price", "quantity", "fee"]:
        if c not in x.columns:
            x[c] = 0
        x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0)
    x["ticker"] = x.get("ticker", "").astype(str).str.upper().str.strip()
    x = x[(x["ticker"] != "") & (x["ticker"] != "NAN") & (x["quantity"] > 0)]
    if x.empty:
        return pd.DataFrame()

    # Fee được giữ để tương thích dữ liệu cũ, nhưng giao diện V1.3 không yêu cầu nhập phí.
    x["cost_value"] = x["buy_price"] * x["quantity"] + x["fee"]
    grouped = x.groupby("ticker", as_index=False).agg(
        quantity=("quantity", "sum"),
        total_cost=("cost_value", "sum"),
        lots=("ticker", "size"),
    )
    grouped["avg_cost"] = grouped["total_cost"] / grouped["quantity"]

    cols = [
        "ticker", "close", "change_pct", "price_time", "score", "signal", "confidence",
        "support20", "resistance20", "reason", "rsi14", "volume_ratio20", "ema20", "ma50",
        "ma200", "breakout20", "breakdown20", "ret_20d", "relative_strength20", "price_only",
    ]
    if market is None or market.empty or "ticker" not in market.columns:
        m = pd.DataFrame(columns=cols)
    else:
        m = market[[c for c in cols if c in market.columns]].copy()
        for c in cols:
            if c not in m.columns:
                m[c] = np.nan
        m = m[cols]
    out = grouped.merge(m, on="ticker", how="left")
    out["market_value"] = out["close"] * out["quantity"]
    out["pnl"] = out["market_value"] - out["total_cost"]
    out["pnl_pct"] = np.where(out["total_cost"] > 0, out["pnl"] / out["total_cost"] * 100, np.nan)

    recs = out.apply(portfolio_recommendation, axis=1)
    out["recommendation"] = [x[0] for x in recs]
    out["explanation"] = [x[1] for x in recs]
    out["action_now"] = out.apply(portfolio_action_now, axis=1)
    return out.sort_values("market_value", ascending=False, na_position="last").reset_index(drop=True)
