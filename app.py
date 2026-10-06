from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import html
import os
import sys
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
load_dotenv(BASE_DIR / ".env", override=True)

from stockdash.auth import require_password
from stockdash.config import APP_NAME, DEFAULT_UNIVERSE, PORTFOLIO_CSV, RUNTIME_CACHE_PATH
from stockdash.engine import build_market_tables
from stockdash.providers import MarketBundle, SSIProvider, VnstockProvider, register_vnstock_key
from stockdash.storage import load_portfolio
from stockdash.indicators import add_indicators
from stockdash.scoring import score_stock
from stockdash.search_service import load_symbol_catalog, fetch_single_history, fetch_single_timeframe
from stockdash.ui_common import DARK_CSS, RADAR_CSS, apply_css, icon
from stockdash.views.analysis import render_analysis
from stockdash.views.portfolio import render_portfolio, render_portfolio_manager
from stockdash.views.radar_v2 import TOP_N as RADAR_TOP_N, render_radar_v2

VERSION = "1.9.0"
# Stock Radar V1.9.0 — giao diện Bento nổi khối (stockdash/views/radar_v2.py + stockdash/radar_v2.html)
PORTFOLIO_PATH = str(PORTFOLIO_CSV)
UNIVERSE_TICKERS = {str(t).upper() for t, _, _ in DEFAULT_UNIVERSE}
VN_TZ = timezone(timedelta(hours=7))
DEV_MODE = os.getenv("DASHBOARD_DEV_MODE", "0") == "1"
VNSTOCK_LOGIN_URL = "https://vnstocks.com/login"
FULL_HISTORY_FROM = date(2000, 1, 1)

CLASSIC_UI = str(st.query_params.get("ui", "")).lower() == "classic"
st.set_page_config(page_title=APP_NAME, page_icon=":material/radar:", layout="wide",
                   initial_sidebar_state="expanded" if CLASSIC_UI else "collapsed")
apply_css()
require_password()


def _now_vn() -> datetime:
    return datetime.now(VN_TZ).replace(tzinfo=None)


def _env_values() -> dict[str, str]:
    env_path = BASE_DIR / ".env"
    values: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                values[k.strip()] = v.strip()
    return values


def _save_env(**updates: str):
    values = _env_values()
    for k, v in updates.items():
        values[k] = str(v).strip()
        os.environ[k] = str(v).strip()
    (BASE_DIR / ".env").write_text(
        "\n".join(f"{k}={v}" for k, v in values.items()) + "\n",
        encoding="utf-8",
    )


def has_ssi_keys() -> bool:
    return bool(os.getenv("SSI_CONSUMER_ID", "").strip() and os.getenv("SSI_CONSUMER_SECRET", "").strip())


def has_vnstock_key() -> bool:
    return bool(os.getenv("VNSTOCK_API_KEY", "").strip())


def save_ssi_keys(consumer_id: str, consumer_secret: str):
    _save_env(
        SSI_CONSUMER_ID=consumer_id,
        SSI_CONSUMER_SECRET=consumer_secret,
        SSI_BASE_URL=os.getenv("SSI_BASE_URL", "https://fc-data.ssi.com.vn/api/v2/Market"),
    )


def save_vnstock_key(api_key: str):
    _save_env(VNSTOCK_API_KEY=api_key)


def held_symbols() -> list[str]:
    lots = load_portfolio(PORTFOLIO_PATH)
    if lots.empty:
        return []
    s = lots["ticker"].fillna("").astype(str).str.upper().str.strip()
    return [x for x in s.unique().tolist() if x and x != "NAN"]


def get_provider(mode: str):
    extra = held_symbols()
    if mode == "DỮ LIỆU THẬT - SSI":
        return SSIProvider(extra_symbols=extra)
    if mode == "DỮ LIỆU THẬT - VNSTOCK":
        return VnstockProvider(extra_symbols=extra)
    raise RuntimeError("Stock Radar chỉ sử dụng dữ liệu thị trường thật.")


def _save_runtime_cache() -> None:
    """Lưu trạng thái dữ liệu thật để mở lại Dashboard không bị trắng sau restart/hotfix."""
    try:
        RUNTIME_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        bundle = st.session_state.get("bundle")
        payload = {
            "saved_at": _now_vn(),
            "market": st.session_state.get("market", pd.DataFrame()),
            "top30": st.session_state.get("top30", pd.DataFrame()),
            "regime": st.session_state.get("regime", {}),
            "histories": st.session_state.get("histories", {}),
            "mode_loaded": st.session_state.get("mode_loaded"),
            "live_quotes": st.session_state.get("live_quotes", pd.DataFrame()),
            "quote_updated_at": st.session_state.get("quote_updated_at"),
            "quick_indexes": st.session_state.get("quick_indexes", {}),
            "market_breadth": st.session_state.get("market_breadth", {}),
            "selected_mode": st.session_state.get("selected_mode"),
            "symbol_catalog": st.session_state.get("symbol_catalog", pd.DataFrame()),
            "bundle_metadata": getattr(bundle, "metadata", pd.DataFrame()) if bundle is not None else pd.DataFrame(),
            "bundle_indexes": getattr(bundle, "indexes", {}) if bundle is not None else {},
            "bundle_source": getattr(bundle, "source", "CACHE DỮ LIỆU THẬT") if bundle is not None else "CACHE DỮ LIỆU THẬT",
            "bundle_updated_at": getattr(bundle, "updated_at", None) if bundle is not None else None,
        }
        tmp = RUNTIME_CACHE_PATH.with_suffix(".tmp")
        with tmp.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(RUNTIME_CACHE_PATH)
    except Exception:
        # Cache chỉ là lớp phục hồi; không được làm hỏng chức năng cập nhật chính.
        pass


def _load_runtime_cache() -> bool:
    """Khôi phục lần cập nhật gần nhất từ ổ đĩa. Trả True nếu có dữ liệu dùng được."""
    if not RUNTIME_CACHE_PATH.exists():
        return False
    try:
        with RUNTIME_CACHE_PATH.open("rb") as f:
            payload = pickle.load(f)
        market = payload.get("market", pd.DataFrame())
        live_quotes = payload.get("live_quotes", pd.DataFrame())
        histories = payload.get("histories", {})
        has_any = (isinstance(market, pd.DataFrame) and not market.empty) or (isinstance(live_quotes, pd.DataFrame) and not live_quotes.empty) or bool(histories)
        if not has_any:
            return False
        st.session_state["market"] = market if isinstance(market, pd.DataFrame) else pd.DataFrame()
        st.session_state["top30"] = payload.get("top30", pd.DataFrame())
        st.session_state["regime"] = payload.get("regime", {"label":"CHƯA CẬP NHẬT","score":0,"breadth":0.0,"risk":"CHƯA CÓ DỮ LIỆU"})
        st.session_state["histories"] = histories if isinstance(histories, dict) else {}
        st.session_state["mode_loaded"] = payload.get("mode_loaded")
        st.session_state["live_quotes"] = live_quotes if isinstance(live_quotes, pd.DataFrame) else pd.DataFrame()
        st.session_state["quote_updated_at"] = payload.get("quote_updated_at")
        st.session_state["quick_indexes"] = payload.get("quick_indexes", {}) or {}
        st.session_state["market_breadth"] = payload.get("market_breadth", {}) or {}
        st.session_state["selected_mode"] = payload.get("selected_mode") or st.session_state.get("selected_mode")
        st.session_state["symbol_catalog"] = payload.get("symbol_catalog", pd.DataFrame())
        metadata = payload.get("bundle_metadata", pd.DataFrame())
        indexes = payload.get("bundle_indexes", {}) or {}
        source = str(payload.get("bundle_source", "CACHE DỮ LIỆU THẬT") or "CACHE DỮ LIỆU THẬT")
        updated_at = payload.get("bundle_updated_at") or payload.get("saved_at") or _now_vn()
        st.session_state["bundle"] = MarketBundle(
            histories=st.session_state["histories"],
            metadata=metadata if isinstance(metadata, pd.DataFrame) else pd.DataFrame(),
            indexes=indexes, source=source, updated_at=updated_at,
            market_breadth=st.session_state.get("market_breadth", {}) or {},
        )
        st.session_state["cache_restored"] = True
        return True
    except Exception:
        return False


def refresh_data(mode: str):
    provider = get_provider(mode)
    bundle = provider.fetch()
    market, regime, histories = build_market_tables(bundle)
    scan_market = market[market.get("sector", "") != "Danh mục đang nắm giữ"].copy()
    top30 = scan_market.head(30).copy()
    st.session_state["bundle"] = bundle
    st.session_state["market"] = market
    st.session_state["top30"] = top30
    st.session_state["regime"] = regime
    st.session_state["histories"] = histories
    st.session_state["mode_loaded"] = mode
    try:
        catalog = load_symbol_catalog(force=True)
    except Exception:
        catalog = st.session_state.get("symbol_catalog", pd.DataFrame())
    st.session_state["symbol_catalog"] = catalog

    breadth = getattr(bundle, "market_breadth", None) or {}
    if mode == "DỮ LIỆU THẬT - VNSTOCK" and isinstance(provider, VnstockProvider) and isinstance(catalog, pd.DataFrame) and not catalog.empty:
        try:
            _, breadth = provider.market_wide_snapshot(catalog)
        except Exception:
            breadth = {}
    st.session_state["market_breadth"] = breadth or {}
    try:
        bundle.market_breadth = breadth or {}
    except Exception:
        pass

    st.session_state.pop("live_quotes", None)
    st.session_state.pop("quote_updated_at", None)
    _save_runtime_cache()

def refresh_quotes_only() -> int:
    """Đồng bộ nhanh: giá rổ theo dõi + breadth/KL trên toàn bộ 3 sàn."""
    provider = VnstockProvider(extra_symbols=held_symbols())
    try:
        # Đồng bộ breadth toàn sàn cần danh sách niêm yết hiện tại, không dùng cache catalog quá cũ.
        # Danh sách niêm yết đổi rất ít trong ngày: tải lại khi cache quá 1 ngày, không phải mỗi lần bấm.
        fresh_catalog = load_symbol_catalog(max_age_days=1)
        if isinstance(fresh_catalog, pd.DataFrame) and not fresh_catalog.empty:
            st.session_state["symbol_catalog"] = fresh_catalog
    except Exception:
        pass
    if not isinstance(st.session_state.get("symbol_catalog"), pd.DataFrame):
        st.session_state["symbol_catalog"] = pd.DataFrame()
    catalog = st.session_state.get("symbol_catalog")

    quotes = pd.DataFrame()
    breadth = {}
    if isinstance(catalog, pd.DataFrame) and not catalog.empty:
        try:
            all_quotes, breadth = provider.market_wide_snapshot(catalog)
            wanted = {str(x[0]).upper().strip() for x in provider.universe}
            if isinstance(all_quotes, pd.DataFrame) and not all_quotes.empty:
                quotes = all_quotes[all_quotes["ticker"].astype(str).str.upper().isin(wanted)].copy()
        except Exception:
            quotes = pd.DataFrame()
            breadth = {}
    if quotes is None or quotes.empty:
        quotes = provider.quote_snapshot()
    if quotes is None or quotes.empty:
        raise RuntimeError("Nguồn Vnstock chưa trả về bảng giá hiện tại.")

    st.session_state["live_quotes"] = quotes
    st.session_state["market_breadth"] = breadth or {}
    st.session_state["quote_updated_at"] = _now_vn()
    try:
        st.session_state["quick_indexes"] = provider.quick_indexes()
    except Exception:
        st.session_state["quick_indexes"] = {}
    _save_runtime_cache()
    return len(quotes)

def _quick_market_from_quotes(quotes: pd.DataFrame, technical_market: pd.DataFrame, histories: dict, catalog: pd.DataFrame | None):
    """Dựng lớp hiển thị nhanh từ quote hiện tại + các mã đã tải riêng.

    Quan trọng: mã được tìm ngoài Top 20/30 có thể không nằm trong batch quote ban đầu.
    Nếu mã đó đã được tải lịch sử bằng load_symbol_on_demand(), phải ghép row kỹ thuật
    vào dataset hiển thị để panel bên phải mở đúng mã thay vì lặp tải vô hạn.
    """
    quotes = quotes.copy() if isinstance(quotes, pd.DataFrame) else pd.DataFrame()
    technical_market = technical_market.copy() if isinstance(technical_market, pd.DataFrame) else pd.DataFrame()

    canonical = {str(t).upper(): {"exchange": e, "sector": s} for t, e, s in DEFAULT_UNIVERSE}
    cat = catalog.copy() if isinstance(catalog, pd.DataFrame) else pd.DataFrame()
    cat_map = {}
    if not cat.empty and "ticker" in cat.columns:
        cat["ticker"] = cat["ticker"].astype(str).str.upper().str.strip()
        cat = cat.drop_duplicates("ticker")
        for _, rr in cat.iterrows():
            cat_map[str(rr["ticker"])] = {
                "exchange": str(rr.get("exchange", "") or ""),
                "company_name": str(rr.get("company_name", "") or ""),
                "sector": str(rr.get("sector", "") or ""),
            }

    tech_map = {}
    if not technical_market.empty and "ticker" in technical_market.columns:
        for _, rr in technical_market.iterrows():
            t = str(rr.get("ticker", "") or "").upper().strip()
            if t:
                tech_map[t] = rr.to_dict()

    rows = []
    seen = set()
    if not quotes.empty:
        for _, q in quotes.iterrows():
            ticker = str(q.get("ticker", "") or "").upper().strip()
            if not ticker:
                continue
            seen.add(ticker)
            base = dict(tech_map.get(ticker, {})) if ticker in histories else {}
            meta = canonical.get(ticker, {})
            cmeta = cat_map.get(ticker, {})
            base.update({
                "ticker": ticker,
                "exchange": str(q.get("exchange", "") or cmeta.get("exchange", "") or meta.get("exchange", "")),
                "sector": str(base.get("sector", "") or cmeta.get("sector", "") or meta.get("sector", "") or "—"),
                "company_name": str(base.get("company_name", "") or cmeta.get("company_name", "") or ticker),
                "close": pd.to_numeric(q.get("close"), errors="coerce"),
                "change_pct": pd.to_numeric(q.get("change_pct"), errors="coerce"),
                "volume": pd.to_numeric(q.get("volume"), errors="coerce"),
                "date": pd.to_datetime(q.get("date"), errors="coerce"),
                "price_time": str(q.get("price_time", "") or q.get("received_at", "") or ""),
                "received_at": str(q.get("received_at", "") or ""),
                "price_only": ticker not in histories,
            })
            if ticker not in histories:
                for c in [
                    "rsi14", "score", "confidence", "support20", "resistance20",
                    "volume_ratio20", "ema20", "ma50", "ma200", "ret_20d",
                    "relative_strength20", "rank_score",
                ]:
                    base[c] = np.nan
                base["signal"] = ""
                base["reason"] = ""
            rows.append(base)

    # Ghép các mã đã tải on-demand nhưng không có trong batch quote ban đầu.
    # fetch_single_history() đã overlay snapshot riêng của mã, nên close/change_pct ở row
    # kỹ thuật là dữ liệu mới nhất mà nguồn trả khi người dùng chọn mã đó.
    for ticker, tr in tech_map.items():
        if ticker in seen or ticker not in histories:
            continue
        row = dict(tr)
        cmeta = cat_map.get(ticker, {})
        meta = canonical.get(ticker, {})
        row["ticker"] = ticker
        row["exchange"] = str(row.get("exchange", "") or cmeta.get("exchange", "") or meta.get("exchange", ""))
        row["sector"] = str(row.get("sector", "") or cmeta.get("sector", "") or meta.get("sector", "") or "—")
        row["company_name"] = str(row.get("company_name", "") or cmeta.get("company_name", "") or ticker)
        row["close"] = pd.to_numeric(row.get("close"), errors="coerce")
        row["change_pct"] = pd.to_numeric(row.get("change_pct"), errors="coerce")
        row["volume"] = pd.to_numeric(row.get("volume"), errors="coerce")
        row["price_only"] = False
        rows.append(row)
        seen.add(ticker)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["_move_abs"] = pd.to_numeric(out.get("change_pct"), errors="coerce").abs().fillna(-1)
    out = out.sort_values(["_move_abs", "ticker"], ascending=[False, True]).drop(columns="_move_abs").reset_index(drop=True)
    return out

def _quick_regime(quotes: pd.DataFrame) -> dict:
    if quotes is None or quotes.empty:
        return {"label": "CHẾ ĐỘ NHANH", "score": 50, "breadth": 0.0, "risk": "CHƯA ĐÁNH GIÁ"}
    chg = pd.to_numeric(quotes.get("change_pct"), errors="coerce")
    valid = chg.dropna()
    breadth = float((valid > 0).mean() * 100) if len(valid) else 0.0
    return {
        "label": "GIÁ THẬT - NHANH",
        "score": 50,
        "breadth": breadth,
        "risk": "CHƯA QUÉT TOÀN BỘ",
    }




def _single_symbol_meta(ticker: str) -> dict:
    ticker = str(ticker).upper().strip()
    meta = {"ticker": ticker, "exchange": "", "sector": "", "company_name": ticker}
    catalog = st.session_state.get("symbol_catalog")
    if isinstance(catalog, pd.DataFrame) and not catalog.empty and "ticker" in catalog.columns:
        try:
            cat = catalog.copy()
            cat["ticker"] = cat["ticker"].astype(str).str.upper().str.strip()
            if ticker in set(cat["ticker"]):
                r = cat.set_index("ticker").loc[ticker]
                meta.update({
                    "exchange": str(r.get("exchange", "") or ""),
                    "company_name": str(r.get("company_name", ticker) or ticker),
                    "sector": str(r.get("sector", "") or ""),
                })
        except Exception:
            pass
    for t, exch, sector in DEFAULT_UNIVERSE:
        if t == ticker:
            meta["exchange"] = meta["exchange"] or exch
            meta["sector"] = meta["sector"] or sector
            break
    return meta


def _latest_quote_snapshot(ticker: str) -> dict | None:
    quotes = st.session_state.get("live_quotes")
    if not isinstance(quotes, pd.DataFrame) or quotes.empty or "ticker" not in quotes.columns:
        return None
    q = quotes.copy()
    q["ticker"] = q["ticker"].astype(str).str.upper().str.strip()
    if ticker not in set(q["ticker"]):
        return None
    r = q[q["ticker"] == ticker].iloc[-1]
    return {
        "date": pd.to_datetime(r.get("date"), errors="coerce"),
        "time": str(r.get("price_time", "") or r.get("received_at", "") or "Cập nhật gần nhất"),
        "close": pd.to_numeric(r.get("close"), errors="coerce"),
        "open": pd.to_numeric(r.get("open"), errors="coerce"),
        "high": pd.to_numeric(r.get("high"), errors="coerce"),
        "low": pd.to_numeric(r.get("low"), errors="coerce"),
        "volume": pd.to_numeric(r.get("volume"), errors="coerce"),
        "change_pct": pd.to_numeric(r.get("change_pct"), errors="coerce"),
        "exchange": str(r.get("exchange", "") or ""),
    }


def _benchmark_indicators():
    """Chỉ báo VN-Index dùng làm benchmark; tính một lần cho mỗi bundle thay vì mỗi mã."""
    bundle_now = st.session_state.get("bundle")
    indexes = getattr(bundle_now, "indexes", None) or {}
    vnindex = indexes.get("VNINDEX")
    if not isinstance(vnindex, pd.DataFrame) or vnindex.empty:
        return None
    cached = st.session_state.get("_benchmark_cache")
    if cached and cached[0] is vnindex:
        return cached[1]
    try:
        calc = add_indicators(vnindex)
    except Exception:
        calc = None
    st.session_state["_benchmark_cache"] = (vnindex, calc)
    return calc


def _store_symbol_analysis(ticker: str, hist_raw: pd.DataFrame, meta: dict, full_history: bool = True) -> None:
    """Tính chỉ báo, chấm điểm và ghép một mã vào market/histories của phiên."""
    regime_now = dict(st.session_state.get("regime", {"score": 50}) or {"score": 50})
    if int(regime_now.get("score", 0) or 0) <= 0 and isinstance(st.session_state.get("live_quotes"), pd.DataFrame):
        regime_now = _quick_regime(st.session_state.get("live_quotes"))
    calc = add_indicators(hist_raw, benchmark=_benchmark_indicators())
    row = calc.iloc[-1].to_dict()
    row["ticker"] = ticker
    row["exchange"] = meta.get("exchange", "")
    row["sector"] = meta.get("sector", "") or "Tìm kiếm toàn thị trường"
    src_chg = row.get("source_change_pct")
    row["change_pct"] = float(src_chg) if pd.notna(src_chg) else float(row.get("ret_1d", 0) or 0)
    scored = score_stock(pd.Series(row), int(regime_now.get("score", 50)))
    row.update({k: v for k, v in scored.items() if k != "components"})
    row["components"] = scored.get("components", {})
    row["rank_score"] = round(scored.get("score", 0) * 0.72 + scored.get("attention", 0) * 2.8, 1)
    row["company_name"] = meta.get("company_name", ticker)

    market_now = st.session_state.get("market", pd.DataFrame()).copy()
    if not market_now.empty and "ticker" in market_now.columns:
        market_now = market_now[market_now["ticker"].astype(str) != ticker]
    st.session_state["market"] = pd.concat([market_now, pd.DataFrame([row])], ignore_index=True)
    hists = dict(st.session_state.get("histories", {}))
    hists[ticker] = calc
    st.session_state["histories"] = hists
    # Đánh dấu đã có lịch sử dài để view không tải lại lần hai.
    if full_history:
        loaded = set(st.session_state.get("full_history_loaded_v1619k", []) or [])
        loaded.add(ticker)
        st.session_state["full_history_loaded_v1619k"] = sorted(loaded)


@st.cache_data(ttl=6 * 3600, show_spinner=False, max_entries=400)
def _cached_recent_history(ticker: str, to_date_iso: str, exchange: str, sector: str) -> pd.DataFrame:
    """Lịch sử ngày ~3 năm của một mã, nhớ 6 giờ (dùng chung mọi phiên).

    ĐỒNG BỘ lặp lại không phải tải lại N mã danh mục: nguồn Guest chỉ 20 request/phút
    nên 18 mã tải lại mỗi lần là nguyên nhân chính khiến ĐỒNG BỘ chậm.
    """
    to_date = datetime.fromisoformat(to_date_iso).date()
    p = VnstockProvider(universe=[(ticker, exchange, sector)], extra_symbols=[ticker])
    df = p._stock_history(ticker, to_date - timedelta(days=1100), to_date)
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError(f"Không có lịch sử cho {ticker}")  # lỗi thì không bị cache
    return df


def analyze_symbol_and_store(ticker: str, mode_now: str | None = None, prefer_live_quote: bool = False) -> str:
    """Tải lịch sử thật của một mã, tính chỉ báo và ghi vào session_state.

    prefer_live_quote=True (dùng ngay sau ĐỒNG BỘ): tận dụng bảng giá vừa nhận để
    overlay vào lịch sử, tránh thêm 1 request snapshot cho mỗi mã danh mục, và chỉ
    tải ~3 năm (đủ cho MA200/RSI). Lịch sử dài được tải khi người dùng mở mã đó —
    tải lịch sử từ 2000 cho cả danh mục trong lúc ĐỒNG BỘ làm vượt hạn mức Guest.
    """
    ticker = str(ticker).upper().strip()
    if not ticker:
        raise ValueError("Mã cổ phiếu không hợp lệ")
    mode_now = mode_now or st.session_state.get("selected_mode", "DỮ LIỆU THẬT - VNSTOCK")
    meta = _single_symbol_meta(ticker)

    if prefer_live_quote and mode_now == "DỮ LIỆU THẬT - VNSTOCK":
        snap = _latest_quote_snapshot(ticker)
        if snap and pd.notna(snap.get("close")):
            p = VnstockProvider(universe=[(ticker, meta.get("exchange", ""), meta.get("sector", ""))], extra_symbols=[ticker])
            to_date = _now_vn().date()
            base = _cached_recent_history(ticker, to_date.isoformat(), meta.get("exchange", ""), meta.get("sector", "")).copy()
            hist_raw = p._overlay_snapshot(base, snap)
            if snap.get("exchange"):
                meta["exchange"] = str(snap.get("exchange"))
            _store_symbol_analysis(ticker, hist_raw, meta, full_history=False)
            return ticker

    hist_raw, meta2 = fetch_single_history(mode_now, ticker)
    meta.update({k: v for k, v in (meta2 or {}).items() if v})
    _store_symbol_analysis(ticker, hist_raw, meta)
    return ticker


def preload_portfolio_technical(mode_now: str) -> tuple[int, int, list[str]]:
    symbols = held_symbols()
    if not symbols:
        return 0, 0, []
    ok = 0
    failed = []
    for ticker in symbols:
        try:
            analyze_symbol_and_store(ticker, mode_now=mode_now, prefer_live_quote=True)
            ok += 1
        except Exception:
            failed.append(ticker)
    if ok:
        _save_runtime_cache()
    return len(symbols), ok, failed


def load_symbol_on_demand(ticker: str):
    """Tải riêng một mã ngoài Top 30 mà không quét lại toàn bộ thị trường."""
    if str(ticker).strip():
        analyze_symbol_and_store(ticker)
        _save_runtime_cache()


def load_hourly_on_demand(ticker: str):
    """Tải dữ liệu 1 giờ cho đúng mã đang mở.

    Nhớ cả kết quả RỖNG trong phiên: nguồn không có 1H thì không gọi lại mỗi lần
    Streamlit chạy lại trang (mỗi cú click), tránh chậm và tốn hạn mức request.
    """
    ticker = str(ticker).upper().strip()
    if not ticker:
        return pd.DataFrame()
    cache = st.session_state.setdefault("hourly_histories", {})
    if ticker in cache:
        old = cache.get(ticker)
        return old if isinstance(old, pd.DataFrame) else pd.DataFrame()
    mode_now = st.session_state.get("selected_mode", "DỮ LIỆU THẬT - VNSTOCK")
    try:
        df = fetch_single_timeframe(mode_now, ticker, "1H")
    except Exception:
        df = pd.DataFrame()
    cache[ticker] = df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    st.session_state["hourly_histories"] = cache
    return cache[ticker]


def _init_empty_state():
    now = _now_vn()
    st.session_state["bundle"] = MarketBundle(
        histories={}, metadata=pd.DataFrame(columns=["ticker", "exchange", "sector"]),
        indexes={}, source="CHƯA CÓ DỮ LIỆU THẬT", updated_at=now,
    )
    st.session_state["market"] = pd.DataFrame()
    st.session_state["top30"] = pd.DataFrame()
    st.session_state["regime"] = {"label": "CHƯA CẬP NHẬT", "score": 0, "breadth": 0.0, "risk": "CHƯA CÓ DỮ LIỆU"}
    st.session_state["histories"] = {}
    st.session_state["mode_loaded"] = None
    st.session_state["symbol_catalog"] = pd.DataFrame()
    st.session_state["market_breadth"] = {}
    st.session_state["hourly_histories"] = {}

def ensure_data():
    # Không xóa dữ liệu chỉ vì đổi số phiên bản. Hotfix trước đây làm mất toàn bộ
    # session_state sau khi restart, khiến Dashboard mở lên trắng.
    if "market" not in st.session_state:
        if not _load_runtime_cache():
            _init_empty_state()
    if "hourly_histories" not in st.session_state:
        st.session_state["hourly_histories"] = {}
    st.session_state["app_version"] = VERSION


def _is_technical_live(bundle_obj) -> bool:
    source = str(getattr(bundle_obj, "source", "") or "")
    return source.startswith("VNSTOCK") or source.startswith("SSI")


def _last_real_data_at() -> datetime | None:
    """Thời điểm nhận dữ liệu thật gần nhất (bảng giá hoặc quét toàn bộ)."""
    stamps = []
    quote_at = st.session_state.get("quote_updated_at")
    if isinstance(quote_at, datetime):
        stamps.append(quote_at)
    bundle_obj = st.session_state.get("bundle")
    if _is_technical_live(bundle_obj) and isinstance(getattr(bundle_obj, "updated_at", None), datetime):
        stamps.append(bundle_obj.updated_at)
    return max(stamps) if stamps else None


def _data_age_days() -> int | None:
    last = _last_real_data_at()
    return None if last is None else (_now_vn().date() - last.date()).days


def _overlay_live_quotes(df: pd.DataFrame, quotes: pd.DataFrame) -> pd.DataFrame:
    """Ghi giá/%/KL mới nhất từ ĐỒNG BỘ lên bảng của lần quét toàn bộ trước đó.

    Chỉ báo kỹ thuật giữ nguyên từ lần quét toàn bộ; chỉ phần giá được làm mới.
    """
    if not isinstance(df, pd.DataFrame) or df.empty or "ticker" not in df.columns:
        return df
    if not isinstance(quotes, pd.DataFrame) or quotes.empty or "ticker" not in quotes.columns:
        return df
    q = quotes.copy()
    q["ticker"] = q["ticker"].astype(str).str.upper().str.strip()
    q = q.drop_duplicates("ticker", keep="last").set_index("ticker")
    out = df.copy()
    tickers = out["ticker"].astype(str).str.upper().str.strip()
    for col in ("close", "change_pct", "volume"):
        if col in q.columns:
            vals = pd.to_numeric(tickers.map(q[col]), errors="coerce")
            mask = vals.notna()
            out.loc[mask, col] = vals[mask]
    if "price_time" in q.columns:
        vals = tickers.map(q["price_time"])
        mask = vals.notna() & (vals.astype(str).str.strip() != "")
        out.loc[mask, "price_time"] = vals[mask]
    return out

ensure_data()

with st.sidebar:
    st.title("TRUNG TÂM ĐIỀU KHIỂN")
    if DEV_MODE:
        st.markdown(
            '<div class="dev-banner"><b>CHẾ ĐỘ CHỈNH SỬA NHANH:</b> Dashboard tự nạp lại khi mã nguồn thay đổi.</div>',
            unsafe_allow_html=True,
        )

    default_source = "DỮ LIỆU THẬT - SSI" if has_ssi_keys() else "DỮ LIỆU THẬT - VNSTOCK"
    mode_options = ["DỮ LIỆU THẬT - VNSTOCK", "DỮ LIỆU THẬT - SSI"]
    mode = st.radio(
        "Nguồn dữ liệu",
        mode_options,
        index=mode_options.index(st.session_state.get("selected_mode", default_source))
        if st.session_state.get("selected_mode", default_source) in mode_options else 0,
        help="Chọn nguồn rồi dùng nút CẬP NHẬT GIÁ hoặc CẬP NHẬT TOÀN BỘ ở đầu trang.",
    )
    st.session_state["selected_mode"] = mode

    # Mở Dashboard khi chưa có dữ liệu, hoặc dữ liệu khôi phục từ cache đã sang ngày khác:
    # tự ĐỒNG BỘ một lần để không hiển thị giá cũ như giá hiện tại. Chỉ thử 1 lần/session.
    no_market = not isinstance(st.session_state.get("market"), pd.DataFrame) or st.session_state.get("market").empty
    no_quotes = not isinstance(st.session_state.get("live_quotes"), pd.DataFrame) or st.session_state.get("live_quotes").empty
    age_days = _data_age_days()
    stale = age_days is not None and age_days >= 1
    if mode == "DỮ LIỆU THẬT - VNSTOCK" and ((no_market and no_quotes) or stale) and not st.session_state.get("auto_quick_attempted"):
        st.session_state["auto_quick_attempted"] = True
        try:
            with st.spinner("Đang tự đồng bộ bảng giá toàn sàn..."):
                refresh_quotes_only()
            st.session_state["update_message"] = (
                f"Dữ liệu lưu trước đó đã cũ {age_days} ngày; đã tự đồng bộ bảng giá mới." if stale
                else "Đã tự khôi phục bảng giá nhanh sau khi mở Dashboard."
            )
        except Exception as e:
            st.session_state["auto_sync_error"] = str(e)

    if mode == "DỮ LIỆU THẬT - VNSTOCK":
        if has_vnstock_key():
            st.success("Vnstock: đã có API key Cộng đồng · tối đa 60 request/phút.")
            st.caption("Có thể dùng CẬP NHẬT TOÀN BỘ để quét lịch sử, tính RSI/MA/khuyến nghị cho toàn rổ.")
        else:
            st.info("Vnstock Guest: 20 request/phút. ĐỒNG BỘ dùng vài request (danh sách mã, bảng giá 3 sàn, 3 chỉ số).")
            st.caption("Quét toàn bộ 48 mã cần API key Cộng đồng miễn phí để tránh lỗi giới hạn request.")
        st.caption("Gói Cộng đồng của Vnstock trả tối đa 8 năm lịch sử ngày và 1 năm dữ liệu giờ.")
    elif mode == "DỮ LIỆU THẬT - SSI":
        st.caption("Nguồn SSI FastConnect. Cần mã định danh kết nối và khóa bí mật.")

    with st.expander("VNSTOCK - API KEY MIỄN PHÍ", expanded=(mode == "DỮ LIỆU THẬT - VNSTOCK" and not has_vnstock_key())):
        st.caption("Khách không đăng ký bị giới hạn 20 request/phút. API key Cộng đồng miễn phí nâng lên 60 request/phút và chỉ cần nhập một lần.")
        if hasattr(st, "link_button"):
            st.link_button("MỞ TRANG LẤY API KEY", VNSTOCK_LOGIN_URL, width="stretch", icon=":material/open_in_new:")
        # Không đưa key đã lưu ngược ra trình duyệt (ô password vẫn gửi giá trị thật về client).
        vn_key = st.text_input(
            "API key Vnstock",
            value="",
            type="password",
            placeholder="Đã lưu key · dán key mới để thay" if has_vnstock_key() else "Dán API key vào đây",
            key="vnstock_key_input",
        )
        if st.button("LƯU API KEY VNSTOCK", width="stretch", icon=":material/key:"):
            if not vn_key.strip():
                st.error("Chưa nhập API key.")
            else:
                try:
                    # Xác thực trước khi lưu; không gọi dữ liệu thị trường.
                    if not register_vnstock_key(vn_key.strip()):
                        raise RuntimeError("Vnstock không chấp nhận key này.")
                    save_vnstock_key(vn_key.strip())
                    st.success("Đã lưu và xác thực API key. Bây giờ có thể bấm CẬP NHẬT TOÀN BỘ.")
                except Exception as e:
                    st.error(f"API key chưa xác thực được: {e}")

    with st.expander("KẾT NỐI SSI (TÙY CHỌN)", expanded=False):
        cid = st.text_input("Mã định danh kết nối SSI", value=os.getenv("SSI_CONSUMER_ID", ""), key="ssi_cid")
        secret = st.text_input(
            "Khóa bí mật SSI", value="", type="password", key="ssi_secret",
            placeholder="Đã lưu khóa · nhập khóa mới để thay" if has_ssi_keys() else "",
        )
        if st.button("LƯU KẾT NỐI SSI", width="stretch", icon=":material/link:"):
            if not cid.strip() or not secret.strip():
                st.error("Cần nhập đủ mã định danh kết nối và khóa bí mật SSI.")
            else:
                save_ssi_keys(cid, secret)
                st.success("Đã lưu khóa SSI.")

    st.divider()
    full_clicked = st.button("CẬP NHẬT TOÀN BỘ", width="stretch", icon=":material/query_stats:", help="Tải lịch sử để tính RSI, MA, hỗ trợ/kháng cự và khuyến nghị cho toàn rổ.")
    st.markdown("**Cách hiểu điểm**")
    st.caption("Thị trường 20 · Xu hướng 25 · Động lượng 20 · Khối lượng 15 · Rủi ro 10 · Mức chú ý 10")
    st.caption("Khuyến nghị là mô hình hỗ trợ quyết định, không phải lệnh giao dịch tự động.")

bundle = st.session_state["bundle"]
market = st.session_state["market"]
top30 = st.session_state["top30"]
regime = st.session_state["regime"]
histories = st.session_state["histories"]

def _chips_html(rows) -> str:
    return '<div class="nm-chips">' + "".join(f'<span class="nm-chip nm-{kind}"><i></i><span>{body}</span></span>' for kind, body in rows) + "</div>"


if CLASSIC_UI:
    header_l, sync_col, full_col = st.columns([6.0, 1.3, 1.9], vertical_alignment="center")
else:
    # Giao diện mới: MỘT thanh trên duy nhất giống bản mẫu (thương hiệu · tìm mã · Danh mục · tối · ĐỒNG BỘ · CẬP NHẬT).
    st.markdown(RADAR_CSS, unsafe_allow_html=True)
    header_l, c_search, c_pf, c_dark, sync_col, full_col = st.columns([2.0, 3.4, 1.35, 1.55, 1.25, 1.9], vertical_alignment="center")
    status_ph = st.empty()
mode_text_top = "Chỉnh sửa nhanh" if DEV_MODE else "Sử dụng hằng ngày"
header_ph = header_l.empty()  # điền sau, khi đã biết các thông báo trạng thái (gộp chung 1 thẻ)
status_rows: list[tuple[str, str]] = []
with sync_col:
    quick_clicked = st.button("ĐỒNG BỘ", width="stretch", type="primary", icon=":material/sync:", help="Cập nhật giá, % thay đổi, khối lượng khớp hôm nay và 3 chỉ số thị trường. Dùng ít request.")
with full_col:
    full_clicked = full_clicked or st.button("CẬP NHẬT TOÀN BỘ", width="stretch", icon=":material/query_stats:", key="full_top_btn", help="Tải lịch sử để tính RSI, MA, hỗ trợ/kháng cự và khuyến nghị cho toàn rổ.")

if quick_clicked:
    try:
        if mode == "DỮ LIỆU THẬT - VNSTOCK":
            with st.spinner("Đang lấy bảng giá hiện tại bằng chế độ tiết kiệm request..."):
                nquotes = refresh_quotes_only()
            total_hold, ok_hold, failed_hold = preload_portfolio_technical(mode)
            extra = ""
            if total_hold:
                extra = f" Đã tự cập nhật kỹ thuật {ok_hold}/{total_hold} mã trong danh mục để cột khuyến nghị hiển thị ngay."
                if failed_hold:
                    preview = ', '.join(failed_hold[:4])
                    extra += f" Chưa tải được: {preview}{'...' if len(failed_hold) > 4 else ''}."
            st.session_state["update_message"] = (
                f"Đã đồng bộ giá hiện tại ({nquotes} mã). Giá, % thay đổi, khối lượng khớp hôm nay và danh mục đã được cập nhật.{extra}"
            )
            st.rerun()
        elif mode == "DỮ LIỆU THẬT - SSI":
            if not has_ssi_keys():
                st.error("Chưa có khóa SSI. Hãy nhập khóa ở thanh bên hoặc dùng Vnstock.")
            else:
                with st.spinner("Đang cập nhật dữ liệu SSI..."):
                    refresh_data(mode)
                st.session_state["update_message"] = "Đã cập nhật dữ liệu SSI."
                st.rerun()
    except Exception as e:
        st.error(f"Cập nhật giá thất bại: {e}")

if full_clicked:
    try:
        if mode == "DỮ LIỆU THẬT - VNSTOCK" and not has_vnstock_key():
            st.warning("Tài khoản Vnstock Guest chỉ có 20 request/phút nên không đủ để quét 48 mã. Hãy lấy API key Cộng đồng miễn phí ở thanh bên (60 request/phút), hoặc dùng CẬP NHẬT GIÁ để cập nhật danh mục ngay.")
        elif mode == "DỮ LIỆU THẬT - SSI" and not has_ssi_keys():
            st.error("Chưa có khóa SSI.")
        else:
            with st.spinner("Đang quét lịch sử, tính chỉ báo, xếp hạng và cập nhật danh mục. Lần đầu có thể mất một lúc; các lần sau trong ngày dùng cache sẽ nhanh hơn..."):
                refresh_data(mode)
            st.session_state["update_message"] = "Đã cập nhật toàn bộ dữ liệu, chỉ báo, xếp hạng và danh mục."
            st.rerun()
    except Exception as e:
        st.error(f"Cập nhật toàn bộ thất bại: {e}")

message = st.session_state.pop("update_message", None)
if message:
    status_rows.append(("ok", html.escape(message)))

technical_live = _is_technical_live(bundle)
live_quotes = st.session_state.get("live_quotes")
quote_live = isinstance(live_quotes, pd.DataFrame) and not live_quotes.empty
quote_time = st.session_state.get("quote_updated_at")
# Sau CẬP NHẬT TOÀN BỘ, nếu người dùng bấm ĐỒNG BỘ thì bảng giá mới hơn lần quét kỹ thuật.
quotes_newer = (
    technical_live and quote_live and isinstance(quote_time, datetime)
    and isinstance(getattr(bundle, "updated_at", None), datetime) and quote_time > bundle.updated_at
)

age_days = _data_age_days()
if age_days is not None and age_days >= 1:
    last_at = _last_real_data_at()
    err = st.session_state.get("auto_sync_error")
    status_rows.append((
        "bad",
        f'<b>Dữ liệu cũ {age_days} ngày</b> (lần cuối {last_at:%d/%m %H:%M}). Bấm ĐỒNG BỘ để lấy giá mới.'
        + (f' Tự đồng bộ lỗi: {html.escape(err)}' if err else ''),
    ))

if technical_live:
    synced = f" · giá lúc {quote_time:%H:%M:%S}" if quotes_newer else ""
    status_rows.append((
        "ok",
        f'<b>Dữ liệu phân tích thật</b> · {html.escape(bundle.source)}{synced}',
    ))
elif quote_live:
    time_text = quote_time.strftime("%H:%M:%S %d/%m") if quote_time else "vừa cập nhật"
    status_rows.append((
        "info",
        f'<b>Giá nhận lúc {time_text}.</b> Bấm một mã để xem phân tích; CẬP NHẬT TOÀN BỘ để quét cả rổ.',
    ))
else:
    status_rows.append((
        "wait",
        '<b>Chưa có dữ liệu thị trường.</b> Bấm ĐỒNG BỘ để lấy bảng giá thật.',
    ))

# Header + mọi thông báo trạng thái gộp trong MỘT thẻ.
if CLASSIC_UI:
    _status_html = "".join(f'<div class="nm-status nm-{kind}">{body}</div>' for kind, body in status_rows)
    header_ph.markdown(
        f"""<div class="nm-header">
          <div class="nm-head-top">
            <div class="nm-logo">{icon("radar", 26)}</div>
            <div><div class="nm-title">Stock Radar</div>
            <div class="nm-sub">Bản V{VERSION}</div></div>
          </div>
          {_status_html}
        </div>""",
        unsafe_allow_html=True,
    )
else:
    header_ph.markdown(
        f"""<div class="nm-brand"><div class="nm-logo">{icon("radar", 24)}</div>
        <div><div class="nm-title">Stock Radar</div><div class="nm-sub">Bản V{VERSION}</div></div></div>""",
        unsafe_allow_html=True,
    )
    if not has_vnstock_key() and mode == "DỮ LIỆU THẬT - VNSTOCK":
        status_rows.append(("wait", "<b>Chưa có API key Vnstock</b> (giới hạn 20 request/phút nên tải chậm). Thêm key miễn phí ở thanh bên hoặc mục Secrets <code>VNSTOCK_API_KEY</code>."))
    status_ph.markdown(_chips_html(status_rows), unsafe_allow_html=True)

# Dựng MỘT bộ dữ liệu hiển thị dùng chung cho tab Thị trường và tab Danh mục.
# Ở chế độ quote nhanh, mã đã tải lịch sử được giữ đầy đủ chỉ báo thật;
# mã chưa tải lịch sử chỉ có giá/% thay đổi thật.
quick_indexes = st.session_state.get("quick_indexes", {}) or {}
market_breadth = st.session_state.get("market_breadth", {}) or {}
if technical_live:
    view_market = _overlay_live_quotes(market, live_quotes) if quotes_newer else market
    view_top = _overlay_live_quotes(top30, live_quotes) if quotes_newer else top30
    view_bundle = replace(
        bundle,
        indexes={**bundle.indexes, **quick_indexes} if quotes_newer and quick_indexes else bundle.indexes,
        market_breadth=market_breadth or getattr(bundle, "market_breadth", {}) or {},
    )
    view_regime = regime
elif quote_live:
    view_market = _quick_market_from_quotes(
        live_quotes, market, histories, st.session_state.get("symbol_catalog"),
    )
    # Top 20/30 chỉ xếp hạng rổ theo dõi; mã tìm kiếm lẻ/danh mục (vd. chứng quyền
    # đáo hạn -88%) không được chen lên đầu bảng chỉ vì biến động % lớn.
    view_top = view_market[view_market["ticker"].isin(UNIVERSE_TICKERS)].head(30) if not view_market.empty else view_market
    view_bundle = MarketBundle(
        histories={}, metadata=pd.DataFrame(), indexes=quick_indexes,
        source="VNSTOCK / ĐỒNG BỘ NHANH", updated_at=quote_time or _now_vn(),
        market_breadth=market_breadth,
    )
    view_regime = _quick_regime(live_quotes)
else:
    view_market = pd.DataFrame()

held = set(held_symbols())
portfolio_price_live = technical_live or (
    quote_live and not live_quotes[live_quotes["ticker"].astype(str).str.upper().isin(held)].empty
)

hourly = st.session_state.get("hourly_histories", {})
classic_ui = CLASSIC_UI


def _render_classic():
    tab_choice = st.segmented_control(
        "Chế độ xem",
        options=["THỊ TRƯỜNG & MÃ NỔI BẬT", "DANH MỤC ĐẦU TƯ CỦA TÔI"],
        default=st.session_state.get("main_tab_v1612", "THỊ TRƯỜNG & MÃ NỔI BẬT"),
        key="main_tab_v1612",
        label_visibility="collapsed",
        format_func=lambda x: (":material/monitoring: " if x.startswith("THỊ") else ":material/account_balance_wallet: ") + x,
    )
    if tab_choice == "THỊ TRƯỜNG & MÃ NỔI BẬT":
        if technical_live or quote_live:
            render_analysis(
                view_bundle, view_market, view_top, histories, view_regime, True,
                st.session_state.get("symbol_catalog"), load_symbol_on_demand,
                quick_mode=not technical_live, hourly_histories=hourly, load_hourly=load_hourly_on_demand,
            )
        else:
            st.info("Chưa có dữ liệu thật. Bấm ĐỒNG BỘ. Sau khi nhận quote, giao diện Top 20/30 + biểu đồ sẽ xuất hiện; mã được chọn sẽ tự tải lịch sử thật theo yêu cầu.")
    else:
        render_portfolio(
            PORTFOLIO_PATH, view_market, view_market, histories, regime, portfolio_price_live,
            load_symbol=load_symbol_on_demand, hourly_histories=hourly, load_hourly=load_hourly_on_demand,
        )


def _symbol_options():
    cat = st.session_state.get("symbol_catalog")
    if isinstance(cat, pd.DataFrame) and not cat.empty and "ticker" in cat.columns:
        c = cat.copy()
        c["ticker"] = c["ticker"].astype(str).str.upper().str.strip()
        c = c[c["ticker"].ne("")].drop_duplicates("ticker")
        return c["ticker"].tolist(), c.set_index("ticker").to_dict("index")
    base = sorted(UNIVERSE_TICKERS)
    return base, {}


def _render_radar():
    """Giao diện mới: thanh công cụ bằng widget Streamlit + trang HTML tương tác nhận dữ liệu thật."""
    options, names = _symbol_options()

    def _fmt_symbol(tk):
        r = names.get(tk, {})
        name = str(r.get("company_name", "") or "")
        exch = {"HOSE": "HSX", "HSX": "HSX", "HNX": "HNX", "UPCOM": "UPCoM"}.get(str(r.get("exchange", "") or "").upper(), str(r.get("exchange", "") or ""))
        suffix = " · ".join([x for x in [name, exch] if x and x != tk])
        return f"{tk} — {suffix}" if suffix else tk

    with c_search:
        rev = int(st.session_state.get("v2_search_rev", 0) or 0)
        searched = st.selectbox(
            "Tìm mã bất kỳ trên HSX / HNX / UPCoM", options, index=None, format_func=_fmt_symbol,
            placeholder="Tìm mã cổ phiếu (VD: HPG, FPT, SSI)…", key=f"v2_search_{rev}", label_visibility="collapsed",
            help="Không giới hạn Top 10. Chọn mã để tải riêng lịch sử và mở biểu đồ.",
        )
    with c_pf:
        with st.popover("Danh mục", width="stretch", icon=":material/add:", help="Thêm lần mua, điều chỉnh/xóa vị thế, sao lưu CSV"):
            render_portfolio_manager(PORTFOLIO_PATH)
    with c_dark:
        dark = st.toggle("Giao diện tối", key="dark_ui")
    if dark:
        st.markdown(DARK_CSS, unsafe_allow_html=True)

    if searched:
        try:
            with st.spinner(f"Đang tải lịch sử và phân tích {searched}..."):
                load_symbol_on_demand(searched)
        except Exception as e:
            st.error(f"Không tải được {searched}: {e}")
        st.session_state["v2_open"] = searched
        st.session_state["v2_search_rev"] = rev + 1
        st.rerun()

    if not (technical_live or quote_live):
        st.info("Chưa có dữ liệu thật. Bấm ĐỒNG BỘ để lấy bảng giá; mã được mở sẽ tự tải lịch sử thật theo yêu cầu.")
        return

    open_ticker = st.session_state.pop("v2_open", None)
    render_radar_v2(
        view_bundle, view_market, view_top, st.session_state.get("histories", histories), view_regime, True,
        PORTFOLIO_PATH, hourly_histories=hourly, open_ticker=open_ticker, dark=bool(dark),
    )

    # Đã hiển thị xong giao diện → mới tải lịch sử cho Top 10 (một lần mỗi phiên), rồi làm mới.
    if not technical_live and mode == "DỮ LIỆU THẬT - VNSTOCK" and isinstance(view_top, pd.DataFrame) and not view_top.empty:
        tried = st.session_state.setdefault("v2_tried", set())
        have = set(st.session_state.get("histories", {}).keys())
        need = [str(x).upper() for x in view_top["ticker"].head(RADAR_TOP_N) if str(x).upper() not in have and str(x).upper() not in tried]
        if need:
            status_ph.markdown(_chips_html(status_rows + [("wait", f"<b>Đang tải lịch sử cho {len(need)} mã nổi bật…</b> Giao diện vẫn dùng được; sẽ tự làm mới khi xong.")]), unsafe_allow_html=True)
            for tk in need:
                tried.add(tk)
                try:
                    analyze_symbol_and_store(tk, prefer_live_quote=True)
                except Exception:
                    pass
            _save_runtime_cache()
            st.rerun()


if classic_ui:
    _render_classic()
else:
    _render_radar()
