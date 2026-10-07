from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import os
import threading

import numpy as np
import pandas as pd
import requests

from .config import DEFAULT_UNIVERSE, VNSTOCK_CACHE_DIR

# register_user() chỉ cần gọi một lần cho mỗi key trong vòng đời tiến trình.
_REGISTERED_KEYS: set[str] = set()
_REGISTER_LOCK = threading.Lock()


def register_vnstock_key(api_key: str) -> bool:
    """Trả True nếu Vnstock chấp nhận key. Kết quả thành công được nhớ theo tiến trình."""
    if not api_key:
        return False
    with _REGISTER_LOCK:
        if api_key in _REGISTERED_KEYS:
            return True
        from vnstock import register_user
        # API key Cộng đồng miễn phí nâng hạn mức từ 20 lên 60 request/phút.
        ok = register_user(api_key) is not False
        if ok:
            _REGISTERED_KEYS.add(api_key)
        return ok


@dataclass
class MarketBundle:
    histories: dict[str, pd.DataFrame]
    metadata: pd.DataFrame
    indexes: dict[str, pd.DataFrame]
    source: str
    updated_at: datetime
    market_breadth: dict[str, dict] | None = None


def _merge_universe(base, extra_symbols=None):
    rows = list(base)
    known = {str(x[0]).upper() for x in rows}
    for sym in extra_symbols or []:
        s = str(sym or "").upper().strip()
        if s and s != "NAN" and s not in known:
            rows.append((s, "", "Danh mục đang nắm giữ"))
            known.add(s)
    return rows


def _normalize_exchange_code(value: str) -> str:
    x = str(value or "").upper().strip()
    return {"HSX": "HOSE", "HOSE": "HOSE", "HNX": "HNX", "UPCOM": "UPCOM"}.get(x, x)


def _market_breadth_from_snapshots(snapshots: dict[str, dict], catalog: pd.DataFrame | None = None) -> dict[str, dict]:
    cat_map: dict[str, str] = {}
    totals = {"HOSE": 0, "HNX": 0, "UPCOM": 0}
    if isinstance(catalog, pd.DataFrame) and not catalog.empty and "ticker" in catalog.columns:
        for _, r in catalog.iterrows():
            sym = str(r.get("ticker", "") or "").upper().strip()
            exch = _normalize_exchange_code(r.get("exchange", ""))
            if sym:
                cat_map[sym] = exch
                if exch in totals:
                    totals[exch] += 1
    out = {
        exch: {"up": 0, "down": 0, "flat": 0, "volume": 0.0, "quoted": 0, "total": totals.get(exch, 0)}
        for exch in ("HOSE", "HNX", "UPCOM")
    }
    for sym, snap in (snapshots or {}).items():
        symbol = str(sym or "").upper().strip()
        exch = _normalize_exchange_code((snap or {}).get("exchange", "")) or cat_map.get(symbol, "")
        if exch not in out:
            exch = cat_map.get(symbol, "")
        if exch not in out:
            continue
        chg = pd.to_numeric((snap or {}).get("change_pct"), errors="coerce")
        vol = pd.to_numeric((snap or {}).get("volume"), errors="coerce")
        if pd.notna(chg):
            out[exch]["quoted"] += 1
            if float(chg) > 1e-9:
                out[exch]["up"] += 1
            elif float(chg) < -1e-9:
                out[exch]["down"] += 1
            else:
                out[exch]["flat"] += 1
        if pd.notna(vol) and float(vol) > 0:
            out[exch]["volume"] += float(vol)
    return out


# V1619M_MARKET_WIDE_BREADTH
class VnstockProvider:
    """Nguồn dữ liệu thật không cần API key, dùng thư viện Vnstock v4+.

    - Giá hiện tại: lấy bảng giá/quote; với mã đang nắm giữ sẽ thử lấy giao dịch
      gần nhất trong phiên để ưu tiên mức giá mới nhất.
    - Lịch sử: OHLCV thật, lưu cache theo ngày để các lần bấm CẬP NHẬT sau nhanh.
    - Vnstock là công cụ kết nối dữ liệu công khai, không phải sở giao dịch; vì vậy
      dashboard luôn hiển thị thời điểm dữ liệu để người dùng tự kiểm tra độ mới.
    """

    def __init__(self, universe=None, extra_symbols=None):
        try:
            from vnstock import Market
        except Exception as e:
            raise RuntimeError(
                "Chưa cài thư viện Vnstock. Hãy đóng Dashboard và mở lại bằng 01_MO_DASHBOARD.cmd; "
                "V1.6.6 sẽ tự cài thư viện này một lần."
            ) from e
        self.Market = Market
        self.api_key = os.getenv("VNSTOCK_API_KEY", "").strip()
        if self.api_key:
            try:
                register_vnstock_key(self.api_key)
            except Exception as e:
                raise RuntimeError(f"Không xác thực được API key Vnstock: {e}") from e
        self.priority_symbols = [str(x).upper().strip() for x in (extra_symbols or []) if str(x).strip()]
        self.universe = _merge_universe(universe or DEFAULT_UNIVERSE, self.priority_symbols)
        self.cache_dir = Path(VNSTOCK_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _api(self):
        return self.Market()

    @staticmethod
    def _flat(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if isinstance(out.columns, pd.MultiIndex):
            out.columns = ["_".join([str(x) for x in c if str(x) not in ("", "None")]).strip("_") for c in out.columns]
        out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
        return out

    @staticmethod
    def _num(v):
        return pd.to_numeric(v, errors="coerce")

    @staticmethod
    def _pick_field(row, names, default=np.nan):
        """Đọc trường cả khi DataFrame có tên cột prefix_tên_trường sau khi flatten."""
        for name in names:
            if name in row.index:
                v = row.get(name)
                if v is not None and str(v).lower() != "nan":
                    return v
        for name in names:
            suffix = "_" + name
            for col in row.index:
                if str(col).endswith(suffix):
                    v = row.get(col)
                    if v is not None and str(v).lower() != "nan":
                        return v
        return default

    @staticmethod
    def _parse_source_datetime(value):
        """Chuẩn hóa timestamp của nguồn về giờ Việt Nam thay vì hiện số epoch thô."""
        if value is None or str(value).strip() == "" or str(value).lower() == "nan":
            return pd.NaT
        try:
            x = float(value)
            if np.isfinite(x):
                if abs(x) >= 1e12:
                    return pd.to_datetime(x, unit="ms", utc=True, errors="coerce").tz_convert("Asia/Ho_Chi_Minh")
                if abs(x) >= 1e9:
                    return pd.to_datetime(x, unit="s", utc=True, errors="coerce").tz_convert("Asia/Ho_Chi_Minh")
        except Exception:
            pass
        ts = pd.to_datetime(value, errors="coerce")
        if pd.isna(ts):
            return pd.NaT
        try:
            if getattr(ts, "tzinfo", None) is not None:
                return ts.tz_convert("Asia/Ho_Chi_Minh")
        except Exception:
            pass
        return ts

    @classmethod
    def _format_source_time(cls, value):
        ts = cls._parse_source_datetime(value)
        if pd.notna(ts):
            try:
                return pd.Timestamp(ts).strftime("%d/%m/%Y %H:%M:%S")
            except Exception:
                pass
        s = str(value or "").strip()
        return "" if s.lower() == "nan" else s

    @staticmethod
    def _to_vnd(v):
        x = pd.to_numeric(v, errors="coerce")
        if pd.isna(x):
            return np.nan
        x = float(x)
        # Dữ liệu lịch sử/intraday của một số nguồn Vnstock dùng đơn vị nghìn đồng,
        # trong khi bảng giá dùng đồng. Chuẩn hóa toàn bộ cổ phiếu về VND.
        return x * 1000.0 if 0 < abs(x) < 1000 else x

    def _normalize_stock_ohlcv(self, raw: pd.DataFrame) -> pd.DataFrame:
        if raw is None or raw.empty:
            return pd.DataFrame()
        df = self._flat(raw)
        date_col = next((c for c in ("time", "date", "trading_date", "tradingdate") if c in df.columns), None)
        if date_col is None:
            return pd.DataFrame()
        out = pd.DataFrame()
        out["date"] = pd.to_datetime(df[date_col], errors="coerce")
        for c in ("open", "high", "low", "close"):
            if c in df.columns:
                vals = pd.to_numeric(df[c], errors="coerce")
                # Stock OHLCV từ KBS/VCI thường ở dạng xx.xx (nghìn đồng).
                med = vals.dropna().abs().median() if vals.notna().any() else np.nan
                out[c] = vals * 1000.0 if pd.notna(med) and med < 1000 else vals
            else:
                out[c] = np.nan
        vol_col = next((c for c in ("volume", "total_volume", "match_volume") if c in df.columns), None)
        out["volume"] = pd.to_numeric(df[vol_col], errors="coerce").fillna(0) if vol_col else 0
        out["price_time"] = "Dữ liệu Vnstock"
        out = out.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date", keep="last")
        return out.reset_index(drop=True)

    def _normalize_index_ohlcv(self, raw: pd.DataFrame) -> pd.DataFrame:
        if raw is None or raw.empty:
            return pd.DataFrame()
        df = self._flat(raw)
        date_col = next((c for c in ("time", "date", "trading_date", "tradingdate") if c in df.columns), None)
        if date_col is None:
            return pd.DataFrame()
        out = pd.DataFrame()
        # Chuẩn hoá về 00:00 để khớp với ngày của cổ phiếu (một số nguồn trả 07:00 → không ghép được RS20).
        out["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
        for c in ("open", "high", "low", "close"):
            out[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
        vol_col = next((c for c in ("volume", "total_volume", "match_volume") if c in df.columns), None)
        out["volume"] = pd.to_numeric(df[vol_col], errors="coerce").fillna(0) if vol_col else 0
        out["price_time"] = "Dữ liệu Vnstock"
        return out.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)

    def _history_attempts_for_window(self, symbol: str, start: date, end: date):
        """Các nguồn OHLCV cho một cửa sổ ngày, ưu tiên VCI cho chuỗi dài."""
        attempts = []
        # Legacy Quote vẫn được vnstock 4.0.7 duy trì; VCI phù hợp hơn với lịch sử dài.
        try:
            from vnstock import Quote
            attempts.append(("VCI", lambda: Quote(source="VCI", symbol=symbol).history(
                start=start.isoformat(), end=end.isoformat(), interval="1D"
            )))
            attempts.append(("KBS", lambda: Quote(source="KBS", symbol=symbol).history(
                start=start.isoformat(), end=end.isoformat(), interval="1D"
            )))
        except Exception:
            pass

        # Unified UI là fallback cuối cùng.
        try:
            api = self._api()
            equity = getattr(api, "equity", None)
            if equity is not None:
                if hasattr(equity, "ohlcv"):
                    attempts.append(("Unified", lambda: equity.ohlcv(
                        symbol=symbol, start=start.isoformat(), end=end.isoformat(), interval="1D"
                    )))
                    attempts.append(("Unified", lambda: equity.ohlcv(
                        symbol=symbol, start=start.isoformat(), end=end.isoformat()
                    )))
                if hasattr(equity, "history"):
                    attempts.append(("Unified", lambda: equity.history(
                        symbol=symbol, start=start.isoformat(), end=end.isoformat()
                    )))
                if callable(equity):
                    attempts.append(("Unified", lambda: equity(symbol).ohlcv(
                        start=start.isoformat(), end=end.isoformat(), interval="1D"
                    )))
                    attempts.append(("Unified", lambda: equity(symbol).ohlcv(
                        start=start.isoformat(), end=end.isoformat()
                    )))
        except Exception:
            pass
        return attempts

    def _stock_history_window(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        errors = []
        best = pd.DataFrame()
        for source, fn in self._history_attempts_for_window(symbol, start, end):
            try:
                df = self._normalize_stock_ohlcv(fn())
                if df is None or df.empty:
                    continue
                if len(df) > len(best):
                    best = df
                # Cửa sổ <= 3 năm thường dưới 800 phiên; >=120 dòng đã đủ để
                # phân biệt với phản hồi bị cắt chỉ vài chục phiên gần nhất.
                if len(df) >= 120:
                    return df
            except Exception as e:
                errors.append(f"{source}: {e}")
        if best is not None and not best.empty:
            return best
        raise RuntimeError("Không tải được OHLCV cửa sổ: " + " | ".join(errors[-3:]))

    def _stock_history_full(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Tải chuỗi dài theo cửa sổ 3 năm rồi ghép, tránh API cắt dữ liệu dài."""
        # Thử nguyên khoảng trước. vnstock 4.0.7 đã cải thiện phân trang; nếu chuỗi
        # thực sự dài thì chỉ cần 1 request.
        try:
            full = self._stock_history_window(symbol, start, end)
            if full is not None and not full.empty:
                d0 = pd.to_datetime(full["date"], errors="coerce").min()
                d1 = pd.to_datetime(full["date"], errors="coerce").max()
                span = (d1 - d0).days if pd.notna(d0) and pd.notna(d1) else 0
                # Chấp nhận nguyên khoảng nếu đã phủ ít nhất ~7 năm; với mã mới
                # chuỗi ngắn sẽ được xác nhận lại bằng cơ chế cửa sổ phía dưới.
                if span >= 2500:
                    return full.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
        except Exception:
            pass

        frames = []
        cursor_end = end
        consecutive_empty = 0
        found_any = False
        # Đi ngược từ hiện tại về 2000, mỗi cửa sổ 3 năm (~750 phiên).
        while cursor_end >= start:
            cursor_start = max(start, cursor_end - timedelta(days=1095))
            try:
                df = self._stock_history_window(symbol, cursor_start, cursor_end)
            except Exception:
                df = pd.DataFrame()
            if df is not None and not df.empty:
                found_any = True
                consecutive_empty = 0
                frames.append(df)
            else:
                consecutive_empty += 1
                # Sau khi đã gặp dữ liệu niêm yết, 2 cửa sổ liên tiếp rỗng phía trước
                # đủ để coi là đã đi qua ngày niêm yết.
                if found_any and consecutive_empty >= 2:
                    break
            if cursor_start <= start:
                break
            cursor_end = cursor_start - timedelta(days=1)

        if not frames:
            raise RuntimeError(f"Không tải được lịch sử dài của {symbol}")
        out = pd.concat(frames, ignore_index=True)
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date", keep="last")
        return out.reset_index(drop=True)

    def _cache_path(self, symbol: str, full_request: bool = False) -> Path:
        suffix = "full_1d" if full_request else "3y_1d"
        return self.cache_dir / f"{symbol.upper()}_{suffix}.csv"

    def _cache_meta_path(self, symbol: str) -> Path:
        return self.cache_dir / f"{symbol.upper()}_full_1d.meta.json"

    def _stock_history(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        full_request = (end - start).days >= 3650
        path = self._cache_path(symbol, full_request=full_request)
        meta_path = self._cache_meta_path(symbol) if full_request else None
        if path.exists():
            try:
                mdate = datetime.fromtimestamp(path.stat().st_mtime).date()
                cached = pd.read_csv(path, parse_dates=["date"])
                if not full_request and mdate == end and len(cached) >= 60:
                    return cached
                if full_request and meta_path is not None and meta_path.exists():
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    if bool(meta.get("complete")) and len(cached) >= 60:
                        if mdate == end:
                            return cached
                        # Lịch sử cũ không đổi: chỉ tải phần đuôi kể từ phiên cuối trong cache
                        # (≈1 request) thay vì tải lại cả chuỗi nhiều năm mỗi ngày.
                        last = pd.to_datetime(cached["date"], errors="coerce").max()
                        if pd.notna(last):
                            tail_start = (pd.Timestamp(last) - pd.Timedelta(days=10)).date()
                            tail = self._stock_history_window(symbol, tail_start, end)
                            df = (pd.concat([cached, tail], ignore_index=True)
                                  .dropna(subset=["date", "close"]).sort_values("date")
                                  .drop_duplicates("date", keep="last").reset_index(drop=True))
                            self._write_history_cache(path, meta_path, df, start, end)
                            return df
            except Exception:
                pass

        df = self._stock_history_full(symbol, start, end) if full_request else self._stock_history_window(symbol, start, end)
        self._write_history_cache(path, meta_path, df, start, end)
        return df

    @staticmethod
    def _write_history_cache(path: Path, meta_path: Path | None, df: pd.DataFrame, start: date, end: date) -> None:
        try:
            df.to_csv(path, index=False)
            if meta_path is not None:
                d0 = pd.to_datetime(df["date"], errors="coerce").min()
                d1 = pd.to_datetime(df["date"], errors="coerce").max()
                meta_path.write_text(json.dumps({
                    "complete": True,
                    "requested_start": start.isoformat(),
                    "requested_end": end.isoformat(),
                    "first_date": None if pd.isna(d0) else pd.Timestamp(d0).date().isoformat(),
                    "last_date": None if pd.isna(d1) else pd.Timestamp(d1).date().isoformat(),
                    "rows": int(len(df)),
                }, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    # V1619N_FULL_HISTORY_WINDOWED

    def _index_history(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        api = self._api()
        idx = getattr(api, "index", None)
        attempts = []
        # Ưu tiên Quote: trả đủ lịch sử (8 năm) cho chỉ số; hàm index() cũ chỉ trả ~100 phiên gần nhất,
        # làm MA200 của VN-Index trống và điểm "Trạng thái thị trường" sai.
        try:
            from vnstock import Quote
            attempts.append(lambda: Quote(source="VCI", symbol=symbol).history(start=start.isoformat(), end=end.isoformat(), interval="1D"))
            attempts.append(lambda: Quote(source="KBS", symbol=symbol).history(start=start.isoformat(), end=end.isoformat(), interval="1D"))
        except Exception:
            pass
        if idx is not None:
            if hasattr(idx, "ohlcv"):
                attempts.append(lambda: idx.ohlcv(symbol=symbol, start=start.isoformat(), end=end.isoformat()))
                attempts.append(lambda: idx.ohlcv(symbol=symbol, length="1Y"))
            if callable(idx):
                attempts.append(lambda: idx(symbol).ohlcv(start=start.isoformat(), end=end.isoformat()))
                attempts.append(lambda: idx(symbol).ohlcv(length="1Y"))
        for fn in attempts:
            try:
                df = self._normalize_index_ohlcv(fn())
                if len(df) >= 20:
                    return df
            except Exception:
                pass
        raise RuntimeError(f"Không tải được chỉ số {symbol}")

    def _batch_quotes(self, symbols: list[str]) -> dict[str, dict]:
        api = self._api()
        wanted_list = [str(x).upper().strip() for x in symbols if str(x).strip()]
        raw = None
        attempts = []
        if hasattr(api, "quote"):
            attempts.extend([lambda: api.quote(wanted_list), lambda: api.quote(symbols=wanted_list)])
        equity = getattr(api, "equity", None)
        if equity is not None and hasattr(equity, "quote"):
            attempts.append(lambda: equity.quote(symbols=wanted_list))
        for fn in attempts:
            try:
                candidate = fn()
                if candidate is not None and len(candidate):
                    raw = candidate
                    break
            except Exception:
                continue
        if raw is None or len(raw) == 0:
            return {}
        df = self._flat(pd.DataFrame(raw))
        out = {}
        now = datetime.now(timezone(timedelta(hours=7))).replace(tzinfo=None)
        wanted = set(wanted_list)
        for _, r in df.iterrows():
            sym = str(self._pick_field(r,["symbol","ticker","code"],"") or "").upper().strip()
            if not sym or (wanted and sym not in wanted):
                continue
            close = np.nan; price_field = ""
            for field in ["match_price","last_price","close_price","price"]:
                v = pd.to_numeric(self._pick_field(r,[field]), errors="coerce")
                if pd.notna(v) and float(v) > 0:
                    close = self._to_vnd(v); price_field = field; break
            ref = self._to_vnd(self._pick_field(r,["reference_price","ref_price"]))
            direct_change = pd.to_numeric(self._pick_field(r,["percent_change","change_percent","change_pct"]), errors="coerce")
            change = float(direct_change) if pd.notna(direct_change) else ((close/ref-1)*100 if pd.notna(close) and pd.notna(ref) and ref else np.nan)
            source_raw = self._pick_field(r,["match_time","trading_time","time","last_update","update_time","updated_at"],"")
            source_time = self._format_source_time(source_raw)
            source_date = self._pick_field(r,["trading_date","date"],None)
            parsed_date = self._parse_source_datetime(source_date)
            # Nếu trường time chứa cả ngày/giờ còn trường date thiếu, dùng chính time.
            if pd.isna(parsed_date):
                parsed_date = self._parse_source_datetime(source_raw)
            try:
                snap_date = pd.Timestamp(parsed_date).tz_localize(None).normalize() if pd.notna(parsed_date) else pd.Timestamp(now.date())
            except Exception:
                snap_date = pd.Timestamp(now.date())
            if pd.notna(close) and pd.notna(ref) and ref > 0:
                ratio=float(close)/float(ref)
                if ratio < 0.75 or ratio > 1.25:
                    close=np.nan; price_field="rejected_outlier"
            out[sym]={
                "close":close,
                "open":self._to_vnd(self._pick_field(r,["open_price","open"])),
                "high":self._to_vnd(self._pick_field(r,["high_price","high"])),
                "low":self._to_vnd(self._pick_field(r,["low_price","low"])),
                "volume":pd.to_numeric(self._pick_field(r,["total_volume","match_volume","volume"]),errors="coerce"),
                "change_pct":change,
                "exchange":str(self._pick_field(r,["exchange","market","board"],"") or "").upper(),
                "time":source_time,
                "date":snap_date,
                "received_at":now.strftime("%d/%m/%Y %H:%M:%S"),
                "ref_price":ref,
                "price_field":price_field,
            }
        return out

    def _align_history_to_snapshot(self, hist: pd.DataFrame, snap: dict | None) -> pd.DataFrame:
        if hist is None or hist.empty or not snap or pd.isna(snap.get("close")):
            return hist
        out=hist.copy().sort_values("date").reset_index(drop=True)
        last_close=pd.to_numeric(out.iloc[-1].get("close"),errors="coerce")
        if pd.isna(last_close) or float(last_close)<=0:
            return out
        snap_date=pd.to_datetime(snap.get("date"),errors="coerce")
        hist_date=pd.to_datetime(out.iloc[-1].get("date"),errors="coerce")
        target=pd.to_numeric(snap.get("close"),errors="coerce")
        if pd.notna(snap_date) and pd.notna(hist_date) and hist_date.normalize()<snap_date.normalize():
            ref=pd.to_numeric(snap.get("ref_price"),errors="coerce")
            if pd.notna(ref) and float(ref)>0: target=ref
        if pd.isna(target) or float(target)<=0: return out
        factor=float(target)/float(last_close)
        if (factor<0.88 or factor>1.12) and 0.10<=factor<=10.0:
            for c in ["open","high","low","close"]:
                if c in out.columns: out[c]=pd.to_numeric(out[c],errors="coerce")*factor
            out["history_scale_factor"]=factor
        return out

    def quote_snapshot(self) -> pd.DataFrame:
        """Lấy bảng giá hiện tại chỉ bằng một request cho toàn bộ rổ theo dõi.

        Chế độ này được dùng khi người dùng chưa đăng ký API key Vnstock:
        vẫn cập nhật được Giá hiện tại/Lãi-Lỗ của danh mục mà không quét 48 chuỗi OHLCV.
        """
        symbols = [x[0] for x in self.universe]
        snapshots = self._batch_quotes(symbols)
        rows = []
        for sym, snap in snapshots.items():
            rows.append({
                "ticker": sym,
                "exchange": snap.get("exchange", ""),
                "open": snap.get("open"),
                "high": snap.get("high"),
                "low": snap.get("low"),
                "close": snap.get("close"),
                "volume": snap.get("volume"),
                "ref_price": snap.get("ref_price"),
                "change_pct": snap.get("change_pct"),
                "price_time": snap.get("time") or "Cập nhật gần nhất",
                "received_at": snap.get("received_at", ""),
                "price_field": snap.get("price_field", ""),
                "date": snap.get("date"),
            })
        return pd.DataFrame(rows)

    def _normalize_price_board(self, raw, exchange_hint: str = "") -> dict[str, dict]:
        if raw is None:
            return {}
        try:
            df = self._flat(pd.DataFrame(raw))
        except Exception:
            return {}
        if df.empty:
            return {}
        out: dict[str, dict] = {}
        now = datetime.now(timezone(timedelta(hours=7))).replace(tzinfo=None)
        for _, r in df.iterrows():
            sym = str(self._pick_field(r, ["symbol", "ticker", "code"], "") or "").upper().strip()
            if not sym:
                continue
            ref = self._to_vnd(self._pick_field(r, ["reference_price", "ref_price", "reference", "refprice"]))
            close = np.nan
            for field in ["match_price", "last_price", "close_price", "current_price", "price", "close"]:
                v = pd.to_numeric(self._pick_field(r, [field]), errors="coerce")
                if pd.notna(v) and float(v) > 0:
                    close = self._to_vnd(v)
                    break
            pct = pd.to_numeric(self._pick_field(r, ["percent_change", "change_percent", "change_pct", "per_price_change"]), errors="coerce")
            if pd.isna(pct):
                abs_chg = pd.to_numeric(self._pick_field(r, ["price_change", "change", "change_price"]), errors="coerce")
                if pd.notna(abs_chg) and pd.notna(ref) and float(ref) != 0:
                    pct = float(self._to_vnd(abs_chg)) / float(ref) * 100.0
                elif pd.notna(close) and pd.notna(ref) and float(ref) != 0:
                    pct = (float(close) / float(ref) - 1.0) * 100.0
            vol = pd.to_numeric(self._pick_field(r, [
                "total_volume", "total_match_vol", "total_match_volume", "match_volume",
                "volume_accumulated", "accumulated_volume", "volume", "totalvol"
            ]), errors="coerce")
            exch = _normalize_exchange_code(self._pick_field(r, ["exchange", "market", "board"], exchange_hint))
            out[sym] = {
                "close": close, "ref_price": ref, "change_pct": pct,
                "volume": vol, "exchange": exch or _normalize_exchange_code(exchange_hint),
                "time": self._format_source_time(self._pick_field(r, ["match_time", "trading_time", "time", "updated_at"], "")),
                "received_at": now.strftime("%d/%m/%Y %H:%M:%S"),
                "date": pd.Timestamp(now.date()),
            }
        return out

    def _trading_board_exchange(self, symbols: list[str], exchange: str) -> dict[str, dict]:
        if not symbols:
            return {}
        attempts = []
        try:
            from vnstock import Trading
            seed = symbols[0]
            trading = Trading(source="KBS", symbol=seed)
            attempts.extend([
                lambda: trading.price_board(symbols_list=symbols, exchange=exchange, get_all=True),
                lambda: trading.price_board(symbols_list=symbols, exchange=exchange),
            ])
        except Exception:
            pass
        for fn in attempts:
            try:
                parsed = self._normalize_price_board(fn(), exchange_hint=exchange)
                if parsed:
                    return parsed
            except Exception:
                continue
        return {}

    def market_wide_snapshot(self, catalog: pd.DataFrame, chunk_size: int = 100) -> tuple[pd.DataFrame, dict[str, dict]]:
        """Bảng giá toàn thị trường, ưu tiên KBS price_board theo từng sàn.

        Nếu một sàn trả thiếu, chỉ retry các mã thiếu bằng Market.quote theo batch nhỏ.
        Không fallback sang rổ Top/portfolio để tránh hiển thị breadth sai.
        """
        if catalog is None or catalog.empty or "ticker" not in catalog.columns:
            return pd.DataFrame(), {}
        cat = catalog.copy()
        cat["ticker"] = cat["ticker"].astype(str).str.upper().str.strip()
        cat["exchange"] = cat.get("exchange", "").astype(str).map(_normalize_exchange_code)
        cat = cat[cat["ticker"].str.match(r"^[A-Z][A-Z0-9]{1,7}$", na=False)]
        cat = cat[cat["exchange"].isin(["HOSE", "HNX", "UPCOM"])].drop_duplicates("ticker")

        snapshots: dict[str, dict] = {}
        for exch in ("HOSE", "HNX", "UPCOM"):
            symbols = cat.loc[cat["exchange"] == exch, "ticker"].tolist()
            if not symbols:
                continue
            # Cố gắng 1 request toàn sàn bằng Trading.price_board.
            part = self._trading_board_exchange(symbols, exch)
            snapshots.update(part)

            missing = [x for x in symbols if x not in snapshots]
            if missing:
                # Fallback theo batch 100; nếu batch trả rỗng, chia đôi tiếp.
                size = max(40, min(100, int(chunk_size or 100)))
                for i in range(0, len(missing), size):
                    chunk = missing[i:i + size]
                    q = self._batch_quotes(chunk)
                    if not q and len(chunk) > 40:
                        mid = len(chunk) // 2
                        q = {}
                        q.update(self._batch_quotes(chunk[:mid]))
                        q.update(self._batch_quotes(chunk[mid:]))
                    for sym, snap in (q or {}).items():
                        if not snap.get("exchange"):
                            snap["exchange"] = exch
                    snapshots.update(q or {})

        rows = []
        for sym, snap in snapshots.items():
            rows.append({
                "ticker": sym, "exchange": snap.get("exchange", ""),
                "close": snap.get("close"), "volume": snap.get("volume"),
                "ref_price": snap.get("ref_price"), "change_pct": snap.get("change_pct"),
                "price_time": snap.get("time") or "Cập nhật gần nhất",
                "received_at": snap.get("received_at", ""), "date": snap.get("date"),
            })
        breadth = _market_breadth_from_snapshots(snapshots, cat)
        # Ghi coverage để UI biết có thực sự là toàn sàn hay chưa.
        for exch in ("HOSE", "HNX", "UPCOM"):
            total = int((cat["exchange"] == exch).sum())
            quoted = int(breadth.get(exch, {}).get("quoted", 0) or 0)
            breadth.setdefault(exch, {})["total"] = total
            breadth[exch]["coverage"] = (quoted / total) if total else 0.0
        return pd.DataFrame(rows), breadth

    # V1619O_TIMEFRAME_HISTORY
    def timeframe_history(self, symbol: str, start: date, end: date, interval: str = "1H") -> pd.DataFrame:
        """OHLCV theo khung thời gian cho mã đang mở; cache riêng từng interval."""
        symbol = str(symbol).upper().strip()
        interval = str(interval or "1H").upper().strip()
        if interval == "1D":
            return self._stock_history(symbol, start, end)
        safe = interval.lower().replace("/", "_")
        path = self.cache_dir / f"{symbol}_{safe}.csv"
        try:
            if path.exists():
                age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
                cached = pd.read_csv(path, parse_dates=["date"])
                if not cached.empty and age < timedelta(minutes=30):
                    return cached
        except Exception:
            pass

        attempts = []
        try:
            from vnstock import Quote
            attempts.append(lambda: Quote(source="VCI", symbol=symbol).history(
                start=start.isoformat(), end=end.isoformat(), interval=interval
            ))
            attempts.append(lambda: Quote(source="KBS", symbol=symbol).history(
                start=start.isoformat(), end=end.isoformat(), interval=interval
            ))
        except Exception:
            pass
        try:
            api = self._api()
            equity = getattr(api, "equity", None)
            if equity is not None:
                if hasattr(equity, "ohlcv"):
                    attempts.append(lambda: equity.ohlcv(
                        symbol=symbol, start=start.isoformat(), end=end.isoformat(), interval=interval
                    ))
                if callable(equity):
                    attempts.append(lambda: equity(symbol).ohlcv(
                        start=start.isoformat(), end=end.isoformat(), interval=interval
                    ))
        except Exception:
            pass

        best = pd.DataFrame()
        for fn in attempts:
            try:
                df = self._normalize_stock_ohlcv(pd.DataFrame(fn()))
                if df is not None and not df.empty and len(df) > len(best):
                    best = df
            except Exception:
                continue
        if best is None or best.empty:
            return pd.DataFrame()
        best = best.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
        try:
            best.to_csv(path, index=False)
        except Exception:
            pass
        return best

    # V1619N_MARKET_WIDE_PRICE_BOARD

    def quick_indexes(self) -> dict[str, pd.DataFrame]:
        """Lấy 3 chỉ số thật cho dải tổng quan khi bấm ĐỒNG BỘ.

        Nếu nguồn không trả được một chỉ số, bỏ trống chỉ số đó thay vì dựng số giả.
        """
        now = datetime.now(timezone(timedelta(hours=7))).replace(tzinfo=None)
        end = now.date()
        start = end - timedelta(days=45)
        out: dict[str, pd.DataFrame] = {}
        for symbol in ("VNINDEX", "HNXINDEX", "UPCOMINDEX"):
            try:
                df = self._index_history(symbol, start, end)
                if df is not None and not df.empty:
                    out[symbol] = df
            except Exception:
                continue
        return out

    def _overlay_snapshot(self, df: pd.DataFrame, snap: dict | None) -> pd.DataFrame:
        if not snap or pd.isna(snap.get("close")):
            return df
        out = self._align_history_to_snapshot(df, snap).copy()
        snap_date = pd.Timestamp(snap.get("date") if pd.notna(snap.get("date")) else pd.Timestamp.today()).normalize()
        close = snap.get("close")
        row = {
            "date": snap_date,
            "open": snap.get("open") if pd.notna(snap.get("open")) else close,
            "high": snap.get("high") if pd.notna(snap.get("high")) else close,
            "low": snap.get("low") if pd.notna(snap.get("low")) else close,
            "close": close,
            "volume": snap.get("volume") if pd.notna(snap.get("volume")) else 0,
            "price_time": snap.get("time") or "Cập nhật gần nhất",
            "source_change_pct": snap.get("change_pct"),
        }
        same = pd.to_datetime(out["date"]).dt.normalize() == snap_date
        if same.any():
            idx = out.index[same][-1]
            # Giữ low/high lịch sử nếu snapshot chỉ có last price; mở rộng nếu cần.
            old = out.loc[idx]
            if pd.isna(snap.get("open")):
                row["open"] = old.get("open", close)
            if pd.isna(snap.get("high")):
                row["high"] = max(float(old.get("high", close)), float(close))
            if pd.isna(snap.get("low")):
                row["low"] = min(float(old.get("low", close)), float(close))
            if pd.isna(snap.get("volume")):
                row["volume"] = old.get("volume", 0)
            for k, v in row.items():
                out.loc[idx, k] = v
        else:
            out = pd.concat([out, pd.DataFrame([row])], ignore_index=True)
        return out.sort_values("date").reset_index(drop=True)

    def fetch(self) -> MarketBundle:
        now = datetime.now(timezone(timedelta(hours=7))).replace(tzinfo=None)
        to_date = now.date()
        from_date = to_date - timedelta(days=1100)
        symbols = [x[0] for x in self.universe]

        # Một lệnh bảng giá cho toàn bộ danh sách, nhanh hơn gọi từng mã.
        snapshots = self._batch_quotes(symbols)

        # Không gọi intraday riêng cho từng mã danh mục ở đây. Bảng quote một lần
        # đã đủ để cập nhật giá hiện tại và giúp giữ số request trong hạn mức 60/phút.

        histories: dict[str, pd.DataFrame] = {}
        errors = []
        # Lịch sử được cache theo ngày; do đó lần Update thứ hai trong ngày chủ yếu
        # chỉ còn một lần gọi bảng giá + vài mã danh mục, rất nhanh.
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = {ex.submit(self._stock_history, sym, from_date, to_date): (sym, exch, sector) for sym, exch, sector in self.universe}
            for fut in as_completed(futs):
                sym, exch, sector = futs[fut]
                try:
                    histories[sym] = self._overlay_snapshot(fut.result(), snapshots.get(sym))
                except Exception as e:
                    errors.append(f"{sym}: {e}")

        if len(histories) < 8:
            raise RuntimeError("Không tải được đủ dữ liệu thị trường từ Vnstock. " + " | ".join(errors[:4]))

        rows = []
        meta_map = {sym: (exch, sector) for sym, exch, sector in self.universe}
        for sym in histories:
            exch, sector = meta_map.get(sym, ("", ""))
            exch = snapshots.get(sym, {}).get("exchange") or exch
            rows.append({"ticker": sym, "exchange": exch, "sector": sector})

        # V1.5.1 không gọi thêm 3 API chỉ số trong lần quét đầy đủ.
        # Thay vào đó dựng chỉ số đại diện từ chính các cổ phiếu thật trong từng sàn,
        # giúp tổng số request đầu tiên khoảng 49 (< 60 request/phút của tài khoản Cộng đồng).
        def _proxy_index(exchange: str, base: float) -> pd.DataFrame:
            members = []
            for sym, df in histories.items():
                exch = str(meta_map.get(sym, ("", ""))[0]).upper()
                if exchange and exch != exchange:
                    continue
                members.append(df.set_index("date")["close"].rename(sym))
            if not members:
                members = [df.set_index("date")["close"].rename(sym) for sym, df in histories.items()]
            panel = pd.concat(members, axis=1).sort_index().ffill().dropna(how="all")
            norm = panel / panel.iloc[0]
            proxy = norm.mean(axis=1) * base
            return pd.DataFrame({
                "date": proxy.index, "open": proxy.values, "high": proxy.values,
                "low": proxy.values, "close": proxy.values, "volume": 0,
                "price_time": "Chỉ số đại diện từ rổ theo dõi",
            }).reset_index(drop=True)

        indexes: dict[str, pd.DataFrame] = {}
        index_errors = []
        for symbol, exch, base in [
            ("VNINDEX", "HOSE", 1250.0),
            ("HNXINDEX", "HNX", 235.0),
            ("UPCOMINDEX", "UPCOM", 92.0),
        ]:
            try:
                real_idx = self._index_history(symbol, from_date, to_date)
                if real_idx is not None and len(real_idx) >= 20:
                    indexes[symbol] = real_idx
                    continue
            except Exception as e:
                index_errors.append(f"{symbol}: {e}")
            # Chỉ dùng đường đại diện để engine vẫn tính được khi API index tạm lỗi;
            # price_time ghi rõ để giao diện không hiểu nhầm là chỉ số chính thức.
            proxy = _proxy_index(exch, base)
            proxy["price_time"] = "CHỈ SỐ ĐẠI DIỆN - API INDEX KHÔNG SẴN SÀNG"
            indexes[symbol] = proxy

        return MarketBundle(
            histories=histories,
            metadata=pd.DataFrame(rows),
            indexes=indexes,
            source="VNSTOCK / DỮ LIỆU THỊ TRƯỜNG CÔNG KHAI",
            updated_at=now,
        )


class SSIProvider:
    """Nguồn dữ liệu SSI FastConnect Data.

    V1.3 dùng DailyOhlc để tạo lịch sử và gọi DailyStockPrice cho phiên hiện
    tại khi người dùng bấm CẬP NHẬT. DailyStockPrice được dùng như snapshot
    hiện tại cho toàn bộ HOSE/HNX/UPCOM; vì vậy giá trong danh mục không còn
    lấy từ dữ liệu mô phỏng. Với mã không có snapshot trong ngày, hệ thống giữ
    giá phiên gần nhất và hiển thị thời gian dữ liệu để người dùng biết độ mới.
    """

    def __init__(self, universe=None, extra_symbols=None):
        self.consumer_id = os.getenv("SSI_CONSUMER_ID", "").strip()
        self.consumer_secret = os.getenv("SSI_CONSUMER_SECRET", "").strip()
        self.base_url = os.getenv(
            "SSI_BASE_URL", "https://fc-data.ssi.com.vn/api/v2/Market"
        ).rstrip("/")
        self.priority_symbols = [str(x).upper().strip() for x in (extra_symbols or []) if str(x).strip()]
        self.universe = _merge_universe(universe or DEFAULT_UNIVERSE, self.priority_symbols)
        self.session = requests.Session()
        self._token: str | None = None

    def _require_credentials(self):
        if not self.consumer_id or not self.consumer_secret:
            raise RuntimeError(
                "Chưa có khóa SSI FastConnect Data. Hãy nhập mã định danh kết nối và khóa bí mật SSI ở thanh bên trái rồi bấm LƯU KẾT NỐI."
            )

    def _access_token(self) -> str:
        self._require_credentials()
        if self._token:
            return self._token
        r = self.session.post(
            f"{self.base_url}/AccessToken",
            json={"consumerID": self.consumer_id, "consumerSecret": self.consumer_secret},
            timeout=20,
        )
        r.raise_for_status()
        body = r.json()
        data = body.get("data") if isinstance(body.get("data"), dict) else {}
        token = body.get("accessToken") or data.get("accessToken")
        if not token:
            raise RuntimeError(f"SSI không trả về accessToken. Phản hồi: {body}")
        self._token = str(token)
        return self._token

    def _get(self, endpoint: str, params: dict) -> dict:
        token = self._access_token()
        r = self.session.get(
            f"{self.base_url}/{endpoint}",
            params=params,
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        if r.status_code == 401:
            self._token = None
            token = self._access_token()
            r = self.session.get(
                f"{self.base_url}/{endpoint}",
                params=params,
                headers={"Authorization": f"Bearer {token}"},
                timeout=30,
            )
        r.raise_for_status()
        return r.json()

    @staticmethod
    def _records(body: dict) -> list[dict]:
        candidates = [
            body.get("data"), body.get("Data"), body.get("items"), body.get("Items"),
            body.get("records"), body.get("Records"),
        ]
        for x in candidates:
            if isinstance(x, list):
                return x
            if isinstance(x, dict):
                for key in ("data", "items", "records"):
                    if isinstance(x.get(key), list):
                        return x[key]
        return []

    @staticmethod
    def _pick(record: dict, *keys, default=None):
        normalized = {str(k).lower(): v for k, v in record.items()}
        for key in keys:
            if key.lower() in normalized:
                return normalized[key.lower()]
        return default

    @staticmethod
    def _num(value):
        return pd.to_numeric(value, errors="coerce")

    def daily_ohlc(self, symbol: str, from_date: date, to_date: date) -> pd.DataFrame:
        # Lịch sử dài (từ 2000 ≈ 6.400 phiên) vượt pageSize 1000. Không phân trang thì
        # với ascending=True chỉ nhận được các phiên CŨ NHẤT và mất toàn bộ dữ liệu gần đây.
        page_size = 1000
        records: list[dict] = []
        for page in range(1, 11):
            body = self._get("DailyOhlc", {
                "Symbol": symbol,
                "Fromdate": from_date.strftime("%d/%m/%Y"),
                "Todate": to_date.strftime("%d/%m/%Y"),
                "pageIndex": page,
                "pageSize": page_size,
                "ascending": True,
            })
            batch = self._records(body)
            records.extend(batch)
            if len(batch) < page_size:
                break
        rows = []
        for r in records:
            dt = self._pick(r, "TradingDate", "Date")
            rows.append({
                "date": pd.to_datetime(dt, errors="coerce", dayfirst=True),
                "open": self._num(self._pick(r, "Open", "OpenPrice")),
                "high": self._num(self._pick(r, "High", "Highest", "HighPrice")),
                "low": self._num(self._pick(r, "Low", "Lowest", "LowPrice")),
                "close": self._num(self._pick(r, "Close", "ClosePrice", "Price")),
                "volume": self._num(self._pick(r, "Volume", "TotalVolume", "MatchVolume")),
                "price_time": self._pick(r, "Time", default="") or "Kết phiên",
            })
        df = pd.DataFrame(rows)
        if df.empty:
            raise RuntimeError(f"Không nhận được DailyOhlc cho {symbol}")
        df = df.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
        df["volume"] = df["volume"].fillna(0)
        return df

    def current_stock_snapshot(self, on_date: date) -> dict[str, dict]:
        """Lấy snapshot toàn thị trường bằng 3 lệnh DailyStockPrice."""
        out: dict[str, dict] = {}
        for market in ("HOSE", "HNX", "UPCOM"):
            body = self._get("DailyStockPrice", {
                "market": market,
                "Fromdate": on_date.strftime("%d/%m/%Y"),
                "Todate": on_date.strftime("%d/%m/%Y"),
                "pageIndex": 1,
                "pageSize": 1000,
            })
            for r in self._records(body):
                sym = str(self._pick(r, "Symbol", default="") or "").upper().strip()
                if not sym:
                    continue
                out[sym] = {
                    "exchange": str(self._pick(r, "Market", default=market) or market).upper(),
                    "date": pd.to_datetime(self._pick(r, "TradingDate"), errors="coerce", dayfirst=True),
                    "time": str(self._pick(r, "Time", default="") or "").strip(),
                    "open": self._num(self._pick(r, "OpenPrice")),
                    "high": self._num(self._pick(r, "HighestPrice")),
                    "low": self._num(self._pick(r, "LowestPrice")),
                    "close": self._num(self._pick(r, "ClosePrice")),
                    "volume": self._num(self._pick(r, "TotalMatchVol")),
                    "change_pct": self._num(self._pick(r, "PerPriceChange")),
                    "ref_price": self._num(self._pick(r, "RefPrice")),
                }
        return out

    def intraday_latest(self, symbol: str, on_date: date) -> dict | None:
        """Lấy giá khớp gần nhất trong ngày cho mã ưu tiên (đặc biệt là danh mục)."""
        body = self._get("IntradayOhlc", {
            "Symbol": symbol,
            "Fromdate": on_date.strftime("%d/%m/%Y"),
            "Todate": on_date.strftime("%d/%m/%Y"),
            "pageIndex": 1,
            "pageSize": 1000,
            "ascending": False,
            "resollution": 1,
        })
        rows = []
        for r in self._records(body):
            dt = pd.to_datetime(self._pick(r, "TradingDate"), errors="coerce", dayfirst=True)
            tm = str(self._pick(r, "Time", default="") or "").strip()
            close = self._num(self._pick(r, "Close", "Value"))
            if pd.isna(close):
                continue
            stamp = pd.to_datetime(
                f"{dt.strftime('%Y-%m-%d') if not pd.isna(dt) else on_date.isoformat()} {tm or '00:00:00'}",
                errors="coerce",
            )
            rows.append({
                "stamp": stamp,
                "date": dt if not pd.isna(dt) else pd.Timestamp(on_date),
                "time": tm,
                "open": self._num(self._pick(r, "Open")),
                "high": self._num(self._pick(r, "High")),
                "low": self._num(self._pick(r, "Low")),
                "close": close,
                "volume": self._num(self._pick(r, "Volume")),
            })
        if not rows:
            return None
        rows.sort(key=lambda x: pd.Timestamp.min if pd.isna(x["stamp"]) else x["stamp"])
        latest = rows[-1].copy()
        latest.pop("stamp", None)
        # IntradayOhlc là dữ liệu theo phút/tick. Phần trăm thay đổi sẽ được
        # engine tính lại so với giá đóng cửa phiên trước.
        latest["change_pct"] = np.nan
        return latest

    def daily_index_history(self, index_id: str, from_date: date, to_date: date) -> pd.DataFrame:
        body = self._get("DailyIndex", {
            "indexID": index_id,
            "Fromdate": from_date.strftime("%d/%m/%Y"),
            "Todate": to_date.strftime("%d/%m/%Y"),
            "pageIndex": 1,
            "pageSize": 1000,
            "ascending": True,
        })
        rows = []
        for r in self._records(body):
            v = self._num(self._pick(r, "IndexValue"))
            dt = pd.to_datetime(self._pick(r, "TradingDate"), errors="coerce", dayfirst=True)
            if pd.isna(v) or pd.isna(dt):
                continue
            rows.append({
                "date": dt,
                "open": v,
                "high": v,
                "low": v,
                "close": v,
                "volume": self._num(self._pick(r, "TotalMatchVol")),
                "price_time": self._pick(r, "Time", default="") or "Kết phiên",
                "source_change_pct": self._num(self._pick(r, "RatioChange")),
                "advances": self._num(self._pick(r, "Advances")),
                "declines": self._num(self._pick(r, "Declines")),
                "nochanges": self._num(self._pick(r, "NoChanges")),
            })
        df = pd.DataFrame(rows)
        if df.empty:
            raise RuntimeError(f"Không nhận được DailyIndex cho {index_id}")
        return df.sort_values("date").reset_index(drop=True)

    @staticmethod
    def _overlay_snapshot(df: pd.DataFrame, snap: dict | None) -> pd.DataFrame:
        if not snap or pd.isna(snap.get("close")):
            return df
        out = df.copy()
        snap_date = snap.get("date")
        if pd.isna(snap_date):
            snap_date = pd.Timestamp.today().normalize()
        snap_date = pd.Timestamp(snap_date).normalize()

        row = {
            "date": snap_date,
            "open": snap.get("open"),
            "high": snap.get("high"),
            "low": snap.get("low"),
            "close": snap.get("close"),
            "volume": snap.get("volume"),
            "price_time": snap.get("time") or "Cập nhật trong phiên",
            "source_change_pct": snap.get("change_pct"),
        }
        for k in ("open", "high", "low"):
            if pd.isna(row[k]):
                row[k] = row["close"]
        if pd.isna(row["volume"]):
            row["volume"] = 0

        same = out["date"].dt.normalize() == snap_date
        if same.any():
            idx = out.index[same][-1]
            for k, v in row.items():
                out.loc[idx, k] = v
        else:
            out = pd.concat([out, pd.DataFrame([row])], ignore_index=True)
        return out.sort_values("date").reset_index(drop=True)

    def fetch(self) -> MarketBundle:
        now = datetime.now(timezone(timedelta(hours=7))).replace(tzinfo=None)
        to_date = now.date()
        from_date = to_date - timedelta(days=1100)

        # Snapshot hiện tại được lấy một lần cho toàn sàn, giúp bảng danh mục và
        # 30 mã dùng cùng một giá khi người dùng bấm CẬP NHẬT.
        snapshots = self.current_stock_snapshot(to_date)

        # Với các mã người dùng đang nắm giữ, ưu tiên IntradayOhlc để vùng
        # Giá hiện tại / Tổng giá hiện tại / Lãi-Lỗ dùng giá gần nhất khi bấm Update.
        for symbol in self.priority_symbols:
            try:
                latest = self.intraday_latest(symbol, to_date)
                if latest:
                    old = snapshots.get(symbol, {})
                    if old:
                        # Chỉ thay giá khớp và thời điểm bằng intraday; giữ OHLC
                        # cả phiên + tổng khối lượng từ DailyStockPrice.
                        merged = dict(old)
                        merged["close"] = latest.get("close")
                        merged["date"] = latest.get("date") or old.get("date")
                        merged["time"] = latest.get("time") or old.get("time")
                        snapshots[symbol] = merged
                    else:
                        latest["exchange"] = ""
                        snapshots[symbol] = latest
            except Exception:
                # Nếu intraday không có (ngoài phiên/ngày nghỉ), giữ snapshot
                # DailyStockPrice hoặc giá phiên gần nhất.
                pass

        histories: dict[str, pd.DataFrame] = {}
        rows = []
        errors = []
        for symbol, configured_exchange, sector in self.universe:
            try:
                hist = self.daily_ohlc(symbol, from_date, to_date)
                snap = snapshots.get(symbol)
                hist = self._overlay_snapshot(hist, snap)
                histories[symbol] = hist
                exchange = (snap or {}).get("exchange") or configured_exchange
                rows.append({"ticker": symbol, "exchange": exchange, "sector": sector})
            except Exception as e:
                errors.append(f"{symbol}: {e}")

        if len(histories) < 8:
            raise RuntimeError("Không tải được dữ liệu SSI cho phần lớn mã. " + " | ".join(errors[:5]))

        indexes: dict[str, pd.DataFrame] = {}
        for idx in ("VNINDEX", "HNXINDEX", "UPCOMINDEX"):
            try:
                # DailyIndex phù hợp hơn cho chỉ số và có cả tỷ lệ tăng/giảm.
                indexes[idx] = self.daily_index_history(idx, from_date, to_date)
            except Exception:
                try:
                    indexes[idx] = self.daily_ohlc(idx, from_date, to_date)
                except Exception:
                    pass

        if "VNINDEX" not in indexes:
            closes = []
            for sym, df in histories.items():
                closes.append(df.set_index("date")["close"].rename(sym))
            panel = pd.concat(closes, axis=1).sort_index().ffill()
            norm = panel / panel.iloc[0]
            proxy_close = norm.mean(axis=1) * 1250
            indexes["VNINDEX"] = pd.DataFrame({
                "date": proxy_close.index,
                "open": proxy_close.values,
                "high": proxy_close.values,
                "low": proxy_close.values,
                "close": proxy_close.values,
                "volume": 0,
                "price_time": "Chỉ số đại diện",
            }).reset_index(drop=True)
        indexes.setdefault("HNXINDEX", indexes["VNINDEX"].copy())
        indexes.setdefault("UPCOMINDEX", indexes["VNINDEX"].copy())

        return MarketBundle(
            histories=histories,
            metadata=pd.DataFrame(rows),
            indexes=indexes,
            source="SSI FASTCONNECT / DỮ LIỆU THẬT",
            updated_at=now,
            market_breadth=_market_breadth_from_snapshots(snapshots),
        )
