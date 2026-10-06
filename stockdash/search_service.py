from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import re

import pandas as pd

from .config import CATALOG_CSV, DEFAULT_UNIVERSE
from .providers import SSIProvider, VnstockProvider

CATALOG_PATH = Path(CATALOG_CSV)
FULL_HISTORY_FROM = date(2000, 1, 1)
# V1619L_FULL_HISTORY_MARKER


def _fallback_catalog() -> pd.DataFrame:
    rows = [
        {"ticker": s, "exchange": e, "company_name": s, "sector": sector}
        for s, e, sector in DEFAULT_UNIVERSE
    ]
    return pd.DataFrame(rows)


def _normalize_catalog(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return _fallback_catalog()
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(x) for x in c if str(x) not in ("", "None")]).strip("_") for c in df.columns]
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    symbol_col = next((c for c in ("symbol", "ticker", "code") if c in df.columns), None)
    exch_col = next((c for c in ("exchange", "market", "board") if c in df.columns), None)
    name_col = next((c for c in ("organ_name", "org_name", "company_name", "name", "organ_short_name") if c in df.columns), None)
    if symbol_col is None:
        return _fallback_catalog()

    out = pd.DataFrame()
    out["ticker"] = df[symbol_col].astype(str).str.upper().str.strip()
    out["exchange"] = df[exch_col].astype(str).str.upper().str.strip() if exch_col else ""
    out["company_name"] = df[name_col].fillna("").astype(str).str.strip() if name_col else out["ticker"]
    out["sector"] = ""
    out = out[out["ticker"].str.match(r"^[A-Z][A-Z0-9]{1,7}$", na=False)]
    if exch_col:
        # Chuẩn hóa HSX -> HOSE và chỉ giữ cổ phiếu Việt Nam trên ba sàn chính.
        out["exchange"] = out["exchange"].replace({"HSX": "HOSE", "UPCOM": "UPCOM"})
        mask = out["exchange"].isin(["HOSE", "HNX", "UPCOM", ""])
        out = out[mask]
    out = out.drop_duplicates("ticker", keep="first").sort_values("ticker").reset_index(drop=True)
    return out if not out.empty else _fallback_catalog()


def load_symbol_catalog(force: bool = False, max_age_days: float = 7) -> pd.DataFrame:
    """Danh sách toàn bộ cổ phiếu HOSE/HNX/UPCoM để tìm theo mã hoặc tên.

    Ưu tiên cache cục bộ (mặc định 7 ngày) để không tốn request mỗi lần gõ tìm kiếm.
    """
    try:
        if CATALOG_PATH.exists() and not force:
            age = datetime.now() - datetime.fromtimestamp(CATALOG_PATH.stat().st_mtime)
            if age < timedelta(days=max_age_days):
                cached = pd.read_csv(CATALOG_PATH)
                cached = _normalize_catalog(cached)
                if len(cached) >= 300:
                    return cached
    except Exception:
        pass

    attempts = []
    try:
        from vnstock import Reference
        ref = Reference()
        attempts.extend([
            lambda: ref.equity.list_by_exchange(),
            lambda: ref.equity.list(),
        ])
    except Exception:
        pass
    try:
        from vnstock import Listing
        attempts.extend([
            lambda: Listing(source="VCI").symbols_by_exchange(),
            lambda: Listing(source="VCI").all_symbols(),
        ])
    except Exception:
        pass

    for fn in attempts:
        try:
            out = _normalize_catalog(pd.DataFrame(fn()))
            if len(out) >= 300:
                CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
                out.to_csv(CATALOG_PATH, index=False)
                return out
        except Exception:
            continue

    return _fallback_catalog()


def search_catalog(catalog: pd.DataFrame, query: str, limit: int = 60) -> pd.DataFrame:
    if catalog is None or catalog.empty:
        return pd.DataFrame()
    q = str(query or "").strip()
    if not q:
        return catalog.head(limit).copy()
    norm = q.upper()
    names = catalog["company_name"].fillna("").astype(str).str.upper()
    tickers = catalog["ticker"].fillna("").astype(str).str.upper()
    # Mã bắt đầu bằng từ khóa được ưu tiên, sau đó mới đến tên công ty chứa từ khóa.
    rank = pd.Series(9, index=catalog.index, dtype=int)
    rank[tickers == norm] = 0
    rank[tickers.str.startswith(norm, na=False)] = rank[tickers.str.startswith(norm, na=False)].clip(upper=1)
    rank[tickers.str.contains(re.escape(norm), na=False)] = rank[tickers.str.contains(re.escape(norm), na=False)].clip(upper=2)
    rank[names.str.contains(re.escape(norm), na=False)] = rank[names.str.contains(re.escape(norm), na=False)].clip(upper=3)
    out = catalog[rank < 9].copy()
    out["_rank"] = rank[rank < 9]
    return out.sort_values(["_rank", "ticker"]).drop(columns="_rank").head(limit).reset_index(drop=True)


def fetch_single_history(mode: str, ticker: str) -> tuple[pd.DataFrame, dict]:
    ticker = str(ticker).upper().strip()
    if not ticker:
        raise ValueError("Mã cổ phiếu không hợp lệ")

    now = datetime.now(timezone(timedelta(hours=7))).replace(tzinfo=None)
    to_date = now.date()
    from_date = FULL_HISTORY_FROM
    catalog = load_symbol_catalog(force=False)
    meta = {"ticker": ticker, "exchange": "", "sector": "", "company_name": ticker}
    if not catalog.empty and ticker in set(catalog["ticker"].astype(str)):
        r = catalog.set_index("ticker").loc[ticker]
        meta.update({
            "exchange": str(r.get("exchange", "") or ""),
            "company_name": str(r.get("company_name", ticker) or ticker),
        })

    if mode == "DỮ LIỆU THẬT - SSI":
        p = SSIProvider(universe=[(ticker, meta["exchange"], meta["sector"])], extra_symbols=[ticker])
        hist = p.daily_ohlc(ticker, from_date, to_date)
        try:
            snaps = p.current_stock_snapshot(to_date)
            hist = p._overlay_snapshot(hist, snaps.get(ticker))
            if snaps.get(ticker, {}).get("exchange"):
                meta["exchange"] = snaps[ticker]["exchange"]
        except Exception:
            pass
        return hist, meta


    p = VnstockProvider(universe=[(ticker, meta["exchange"], meta["sector"])], extra_symbols=[ticker])
    hist = p._stock_history(ticker, from_date, to_date)
    try:
        snap = p._batch_quotes([ticker]).get(ticker)
        hist = p._overlay_snapshot(hist, snap)
        if snap and snap.get("exchange"):
            meta["exchange"] = str(snap.get("exchange") or meta["exchange"])
    except Exception:
        pass
    return hist, meta


def fetch_single_timeframe(mode: str, ticker: str, interval: str = "1H") -> pd.DataFrame:
    """Lấy dữ liệu theo khung nến cho mã đang mở. 1H tải tối đa khoảng 12 tháng gần nhất.

    Khung tuần/tháng được Dashboard gộp cục bộ từ lịch sử ngày, vì vậy hàm này
    chủ yếu phục vụ nến giờ để tránh tăng request không cần thiết.
    """
    ticker = str(ticker).upper().strip()
    interval = str(interval or "1H").upper().strip()
    if not ticker:
        return pd.DataFrame()
    now = datetime.now(timezone(timedelta(hours=7))).replace(tzinfo=None)
    to_date = now.date()
    if interval == "1H":
        from_date = to_date - timedelta(days=370)
    else:
        from_date = to_date - timedelta(days=1500)
    if mode == "DỮ LIỆU THẬT - SSI":
        # SSI hiện tại của Dashboard chưa duy trì chuỗi IntradayOhlc nhiều ngày ổn định.
        # Trả rỗng để UI tự vô hiệu hóa 1H thay vì dựng dữ liệu giả.
        return pd.DataFrame()
    p = VnstockProvider(universe=[(ticker, "", "")], extra_symbols=[ticker])
    return p.timeframe_history(ticker, from_date, to_date, interval=interval)
