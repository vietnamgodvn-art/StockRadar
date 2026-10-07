"""Giao diện Stock Radar mới: một trang HTML tự chứa (bản đồ nhiệt / bảng tín hiệu / danh mục / cửa sổ
chi tiết có biểu đồ), được nạp dữ liệu THẬT từ chính bộ xử lý của Dashboard.

Python chỉ chuẩn bị dữ liệu (JSON); mọi thao tác xem (đổi tab, mở mã, lọc, đổi khung nến) chạy
phía trình duyệt nên không phải chạy lại trang.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from ..news_service import all_cached, combine, has_ai_key, model_name
from ..portfolio import aggregate_portfolio
from ..storage import load_portfolio
from ..ui_common import (
    _chart_epoch, _chart_interval_payload, _clean_history, buy_action_now, combined_trend_projection,
    decision_explanation, embed_html, general_recommendation, portfolio_exit_plan, safe_num, short_reason,
)
from .analysis import (
    _CANONICAL_META, _display_exchange, _ensure_confidence_value as _conf_market, _extract_index_snapshot,
    _finite, _fmt_volume, _index_change, _trend_bucket, _trend_rec,
)
from .portfolio import _ensure_confidence_value as _conf_port

_TEMPLATE = Path(__file__).resolve().parent.parent / "radar_v2.html"
_CALIB = Path(__file__).resolve().parent.parent / "calibration.json"
TOP_N = 10
_OVERLAYS = ["EMA20", "MA50", "MA200"]


# ---------------------------------------------------------------- tiện ích
def _clean(o):
    """Đưa mọi giá trị về kiểu JSON an toàn (NaN/inf -> None, numpy -> python)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    return o


def _spark(hist, n: int = 24) -> list:
    if not isinstance(hist, pd.DataFrame) or hist.empty or "close" not in hist.columns:
        return []
    c = pd.to_numeric(hist["close"], errors="coerce").dropna().tail(n)
    return [round(float(x), 2) for x in c] if len(c) >= 3 else []


def _title(text: str) -> str:
    text = str(text or "").strip()
    return text[:1] + text[1:].lower() if text else ""


# ---------------------------------------------------------------- chỉ số sàn
def _indices(bundle) -> list[dict]:
    specs = [
        ("VNINDEX", "HOSE", "HSX · VN-INDEX"),
        ("HNXINDEX", "HNX", "HNX · HNX-INDEX"),
        ("UPCOMINDEX", "UPCOM", "UPCoM · UPCoM-INDEX"),
    ]
    wide = getattr(bundle, "market_breadth", None) or {}
    out = []
    for idx, wide_key, label in specs:
        obj = getattr(bundle, "indexes", {}).get(idx) if bundle is not None else None
        proxy = isinstance(obj, pd.DataFrame) and not obj.empty and "ĐẠI DIỆN" in str(obj.iloc[-1].get("price_time", "")).upper()
        snap = _extract_index_snapshot(obj)
        value, point, pct = snap["value"], snap["point"], snap["pct"]
        if (value is None or point is None or pct is None) and isinstance(obj, pd.DataFrame) and not obj.empty and not proxy:
            value, point, pct = _index_change(obj)
        stat = wide.get(wide_key, {}) if isinstance(wide, dict) else {}
        coverage = float(stat.get("coverage", 0.0) or 0.0) if stat else 0.0
        if stat and int(stat.get("quoted", 0) or 0) > 0:
            up, down, flat = int(stat.get("up", 0) or 0), int(stat.get("down", 0) or 0), int(stat.get("flat", 0) or 0)
            vol = _finite(stat.get("volume"))
        else:
            up = down = flat = 0
            vol = None
        n = up + down + flat
        ratio = up / n if n else None
        status = ("API chỉ số chưa sẵn sàng" if proxy else (
            "Tích cực" if ratio is not None and ratio >= .55 else (
                "Thận trọng" if ratio is not None and ratio <= .40 else (
                    "Trung lập" if ratio is not None else "Chưa đủ dữ liệu"))))
        series = []
        if isinstance(obj, pd.DataFrame) and not obj.empty and "close" in obj.columns and not proxy:
            c = pd.to_numeric(obj["close"], errors="coerce").dropna().tail(40)
            series = [round(float(x), 2) for x in c] if len(c) >= 3 else []
        out.append({
            "label": label, "value": value, "point": point, "pct": pct, "up": up, "down": down, "flat": flat,
            "vol": _fmt_volume(vol), "coverage": int(round(coverage * 100)) if coverage > 0 else 0,
            "status": status, "series": series,
        })
    return out


# ---------------------------------------------------------------- Top mã nổi bật
def _top_rows(view: pd.DataFrame, regime: dict, histories: dict) -> list[dict]:
    rows = []
    if not isinstance(view, pd.DataFrame) or view.empty:
        return rows
    for _, r in view.head(TOP_N).iterrows():
        t = str(r.get("ticker", "") or "").upper().strip()
        if not t:
            continue
        ex = str(r.get("exchange", "") or "")
        if t in _CANONICAL_META and not ex:
            ex = _CANONICAL_META[t][0]
        rec = _trend_rec(r, regime)
        cls = "g" if rec.startswith("NÊN MUA") else ("y" if rec.startswith("CHỜ") else ("n" if "CHƯA ĐỦ" in rec else "r"))
        rows.append({
            "t": t, "ex": _display_exchange(ex), "p": _finite(r.get("close")), "chg": _finite(r.get("change_pct")),
            "vol": _finite(r.get("volume")), "conf": _conf_market(r), "rec": cls, "recText": _title(rec),
            "trend": _trend_bucket(r), "spark": _spark(histories.get(t)),
        })
    return rows


# ---------------------------------------------------------------- danh mục
def _portfolio_payload(portfolio: pd.DataFrame, histories: dict, is_live: bool) -> dict:
    if portfolio is None or portfolio.empty:
        return {"rows": []}
    x = portfolio.copy()
    action = x.get("action_now", pd.Series([""] * len(x), index=x.index)).astype(str).str.upper().str.strip()
    sell = action.str.contains("BÁN|GIẢM", regex=True, na=False)
    hold = action.str.contains("GIỮ", regex=True, na=False) & ~sell
    x["_conf"] = x.apply(_conf_port, axis=1)
    x["_pnl"] = pd.to_numeric(x.get("pnl_pct"), errors="coerce")
    x["_score"] = pd.to_numeric(x.get("score"), errors="coerce")

    priced_mask = x["close"].notna() if is_live else pd.Series(False, index=x.index)
    priced = int(priced_mask.sum())
    total = len(x)
    total_cost = float(x["total_cost"].sum())
    priced_cost = float(x.loc[priced_mask, "total_cost"].sum()) if priced else float("nan")
    total_value = float(x.loc[priced_mask, "market_value"].sum()) if priced else float("nan")
    pnl = total_value - priced_cost if priced else float("nan")
    pnl_pct = pnl / priced_cost * 100 if priced and priced_cost else float("nan")

    rows = []
    for i, r in x.iterrows():
        a = str(action.loc[i])
        if bool(sell.loc[i]):
            cls = "s"
        elif bool(hold.loc[i]):
            cls = "g"
        else:
            cls = "n"
        live = is_live and pd.notna(r.get("close"))
        rows.append({
            "t": str(r["ticker"]), "qty": _finite(r.get("quantity")), "cost": _finite(r.get("avg_cost")),
            "totalCost": _finite(r.get("total_cost")),
            "p": _finite(r.get("close")) if live else None,
            "chg": _finite(r.get("change_pct")) if live else None,
            "pnl": _finite(r.get("pnl")) if live else None, "pnlPct": _finite(r.get("pnl_pct")) if live else None,
            "value": _finite(r.get("market_value")) if live else None,
            "conf": r["_conf"], "rec": cls, "recText": _title(a or "—"), "spark": _spark(histories.get(str(r["ticker"]))),
        })

    # --- phần tổng quan (cùng logic với giao diện cũ)
    sell_n, hold_n = int(sell.sum()), int(hold.sum())
    wait_n = total - sell_n - hold_n
    sell_ratio = sell_n / max(total, 1)
    hold_ratio = hold_n / max(total, 1)
    known = bool(np.isfinite(pnl_pct))
    pp = x["_pnl"].dropna()
    conf = pd.to_numeric(x["_conf"], errors="coerce").dropna()
    avg_conf = int(round(float(conf.mean()))) if not conf.empty else 0
    if (known and pnl_pct <= -15) or sell_ratio >= 0.50:
        state, color = "PHÒNG THỦ · ƯU TIÊN HẠ RỦI RO", "#e11d48"
        desc = (f"Danh mục hiện {'âm' if pnl_pct < 0 else 'tăng'} khoảng <b>{abs(pnl_pct):.1f}%</b> so với giá vốn. "
                f"Có <b>{sell_n}/{total}</b> mã đang ở trạng thái Bán/Giảm tỷ trọng. "
                "Ưu tiên bảo toàn vốn và xử lý mã yếu trước, chưa bình quân giá xuống toàn danh mục.") if known else (
            f"Có <b>{sell_n}/{total}</b> mã đang ở trạng thái Bán/Giảm tỷ trọng. Ưu tiên xử lý mã yếu trước và hạn chế tăng thêm rủi ro.")
    elif (known and pnl_pct < -5) or sell_ratio >= 0.35:
        state, color = "THẬN TRỌNG · ƯU TIÊN TÁI CƠ CẤU", "#d97706"
        desc = (f"Danh mục đang {'âm' if pnl_pct < 0 else 'tăng'} khoảng <b>{abs(pnl_pct):.1f}%</b>; {sell_n} mã cần giảm rủi ro và "
                f"{hold_n} mã có thể tiếp tục giữ. Nên tái cơ cấu theo chất lượng tín hiệu thay vì xử lý đồng loạt.") if known else (
            f"{sell_n} mã cần giảm rủi ro và {hold_n} mã có thể tiếp tục giữ. Ưu tiên tái cơ cấu theo chất lượng tín hiệu.")
    elif known and pnl_pct >= 0 and hold_ratio >= 0.60:
        state, color = "TÍCH CỰC · ƯU TIÊN GIỮ MÃ KHỎE", "#059669"
        desc = (f"Danh mục đang tăng khoảng <b>{abs(pnl_pct):.1f}%</b>; phần lớn mã vẫn ở trạng thái giữ. "
                "Ưu tiên giữ mã khỏe, dời mốc bảo vệ lợi nhuận và chỉ tăng tỷ trọng khi giá/khối lượng xác nhận.")
    else:
        state, color = "CÂN BẰNG · ƯU TIÊN CHỌN LỌC", "#4f46e5"
        pre = f"P/L toàn danh mục khoảng <b>{pnl_pct:+.1f}%</b>. " if known else ""
        desc = pre + f"Hiện có {sell_n} mã Bán/Giảm, {hold_n} mã Giữ và {wait_n} mã Chờ. Nên hành động theo từng mã, không tăng tỷ trọng đồng loạt."

    def names(mask, kind):
        z = x.loc[mask, ["ticker", "_pnl", "_score", "_conf"]].copy()
        if z.empty:
            return "—"
        if kind == "sell":
            z = z.sort_values(["_pnl", "_conf"], ascending=[True, False], na_position="last")
        elif kind == "hold":
            z = z.sort_values(["_score", "_conf"], ascending=[False, False], na_position="last")
        else:
            z = z.sort_values(["_conf", "_score"], ascending=[False, False], na_position="last")
        v = z["ticker"].astype(str).str.upper().tolist()[:5]
        return " · ".join(v) if v else "—"

    return {
        "rows": rows,
        "sum": {"cost": total_cost, "value": total_value if priced else None, "pnl": pnl if priced else None,
                "pnlPct": pnl_pct if priced else None, "count": total, "priced": priced},
        "state": state, "stateColor": color, "desc": desc, "avgConf": avg_conf,
        "win": int((pp > 0).sum()), "loss": int((pp < 0).sum()),
        "coverage": f"{priced}/{total} mã đã có giá" if total else "—",
        "names": {"sell": names(sell, "sell"), "hold": names(hold, "hold"), "wait": names(~(sell | hold), "wait")},
    }


# ---------------------------------------------------------------- chi tiết từng mã
def _detail(ticker: str, market: pd.DataFrame, histories: dict, regime: dict, prow, hourly, is_live: bool) -> dict | None:
    if ticker not in histories or market is None or market.empty or "ticker" not in market.columns:
        return None
    m = market.copy()
    m["ticker"] = m["ticker"].astype(str).str.upper().str.strip()
    m = m.drop_duplicates("ticker", keep="last").set_index("ticker")
    if ticker not in m.index:
        return None
    row = m.loc[ticker]
    hist = _clean_history(histories[ticker])
    if hist.empty:
        return None

    decision = decision_explanation(row, regime, histories[ticker])
    plan = None
    if prow is not None:
        rec = str(prow.get("recommendation", "") or "")
        expl = str(prow.get("explanation", "") or "")
        pnl_pct = safe_num(prow.get("pnl_pct"))
        plan = portfolio_exit_plan(row, prow)
    else:
        rec, expl = general_recommendation(row, regime), short_reason(row)
        pnl_pct = np.nan

    date_value = row.get("date")
    date_text = pd.to_datetime(date_value).strftime("%d/%m/%Y") if pd.notna(date_value) else "—"
    exch = _display_exchange(row.get("exchange", ""))

    def _p(key):
        return None if prow is None or pd.isna(safe_num(prow.get(key))) else float(safe_num(prow.get(key)))

    info = {
        "price": None if pd.isna(safe_num(row.get("close"))) else safe_num(row.get("close")),
        "change_pct": safe_num(row.get("change_pct"), 0),
        "sector": str(row.get("sector", "") or ""), "exchange": exch,
        "score": int(safe_num(row.get("score"), 0)), "confidence": int(safe_num(row.get("confidence"), 0)),
        "buy_action": buy_action_now(row, regime),
        "rsi": round(safe_num(row.get("rsi14"), 50), 1), "volume_ratio": round(safe_num(row.get("volume_ratio20"), 0), 2),
        "recommendation": rec, "explanation": expl, "date_text": date_text,
        "source": str(row.get("price_time", "") or ""), "demo_warning": not is_live,
        "pnl_pct": None if pd.isna(pnl_pct) else float(pnl_pct), "pnl_amount": _p("pnl"),
        "avg_cost": _p("avg_cost"), "quantity": _p("quantity"), "market_value": _p("market_value"),
        "why": decision.get("why", ""), "entry": decision.get("entry", ""), "invalid": decision.get("invalid", ""),
        "portfolio_plan": plan,
    }

    base = hist.tail(1100)
    iv = {
        "1D": _chart_interval_payload(base, "1D", _OVERLAYS),
        "1W": _chart_interval_payload(hist, "1W", _OVERLAYS),
        "1M": _chart_interval_payload(hist, "1M", _OVERLAYS),
        "1H": _chart_interval_payload(hourly, "1H", _OVERLAYS) if isinstance(hourly, pd.DataFrame) and not hourly.empty else {"t": []},
    }
    proj = dict(combined_trend_projection(hist, horizon=30) or {})
    pts = []
    for p in proj.get("points", []) or []:
        try:
            pts.append({"time": _chart_epoch(pd.Timestamp(p.get("time"))), "value": p.get("value")})
        except Exception:
            continue
    proj["points"] = pts

    support, resistance = safe_num(row.get("support20")), safe_num(row.get("resistance20"))
    cost = safe_num(prow.get("avg_cost")) if prow is not None else np.nan
    return {
        "info": info, "iv": iv, "proj": proj,
        "support": None if pd.isna(support) else float(support),
        "resistance": None if pd.isna(resistance) else float(resistance),
        "cost": None if pd.isna(cost) else float(cost),
    }


# ---------------------------------------------------------------- vào cửa chính
def render_radar_v2(
    bundle, market: pd.DataFrame, top: pd.DataFrame, histories: dict, regime: dict, is_live: bool,
    portfolio_path: str, hourly_histories: dict | None = None, open_ticker: str | None = None,
    dark: bool = False, price_note: str = "",
) -> None:
    hourly_histories = hourly_histories or {}
    lots = load_portfolio(portfolio_path)
    portfolio = aggregate_portfolio(lots, market)

    top_rows = _top_rows(top, regime, histories)
    port = _portfolio_payload(portfolio, histories, is_live)

    wanted: list[str] = [r["t"] for r in top_rows] + [r["t"] for r in port.get("rows", [])]
    if open_ticker:
        wanted.append(open_ticker)
    pmap = portfolio.set_index("ticker") if not portfolio.empty else pd.DataFrame()
    detail: dict = {}
    for t in dict.fromkeys(wanted):  # giữ thứ tự, bỏ trùng
        try:
            prow = pmap.loc[t] if not portfolio.empty and t in pmap.index else None
            d = _detail(t, market, histories, regime, prow, hourly_histories.get(t), is_live)
            if d:
                detail[t] = d
        except Exception:
            continue

    # Nhận định tin tức do AI (đọc từ bộ nhớ đệm; không gọi AI tại đây) + ghép với khuyến nghị kỹ thuật.
    tech_cls = {r["t"]: r["rec"] for r in top_rows}
    tech_cls.update({r["t"]: r["rec"] for r in port.get("rows", [])})
    news = {}
    for tk, item in all_cached(detail.keys()).items():
        news[tk] = {
            "ts": item.get("ts"), "model": item.get("model"), "stale": bool(item.get("stale")),
            "result": item.get("result"), "headlines": item.get("headlines", []),
            "combined": combine(tech_cls.get(tk, "n"), item),
        }

    try:
        calib = json.loads(_CALIB.read_text(encoding="utf-8"))
    except Exception:
        calib = None
    data = {
        "calib": calib, "ai": {"enabled": has_ai_key(), "model": model_name()}, "news": news,
        "dark": bool(dark), "note": price_note, "indices": _indices(bundle),
        "regime": ({k: regime.get(k) for k in ("score", "label", "risk", "breadth")} if regime else None),
        "top": top_rows, "port": port, "detail": detail, "open": (open_ticker or "").upper() or None, "tab": "market",
    }
    payload = json.dumps(_clean(data), ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    doc = _TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", payload, 1)

    embed_html(doc, 900)
