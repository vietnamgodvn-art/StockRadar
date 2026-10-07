from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_portfolio(path: str) -> pd.DataFrame:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    base_cols = ["ticker", "buy_date", "buy_price", "quantity", "fee", "note"]
    if not p.exists():
        df = pd.DataFrame(columns=base_cols)
        df.to_csv(p, index=False)
        return df
    df = pd.read_csv(p)
    for c in base_cols:
        if c not in df.columns:
            df[c] = 0 if c in {"buy_price", "quantity", "fee"} else ""
    return df[base_cols]


def save_portfolio(path: str, df: pd.DataFrame) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    clean = df.copy()
    if "ticker" not in clean.columns:
        clean["ticker"] = ""
    clean["ticker"] = clean["ticker"].astype(str).str.upper().str.strip()
    clean = clean[(clean["ticker"] != "") & (clean["ticker"] != "NAN")]
    for c in ["buy_date", "buy_price", "quantity", "note"]:
        if c not in clean.columns:
            clean[c] = "" if c in {"buy_date", "note"} else 0
    if "fee" not in clean.columns:
        clean["fee"] = 0
    # Chỉ lưu dữ liệu đầu vào; các cột giá hiện tại/P&L được tính lại mỗi lần mở hoặc cập nhật.
    clean = clean[["ticker", "buy_date", "buy_price", "quantity", "fee", "note"]]
    clean.to_csv(p, index=False)


def add_purchase(path: str, ticker: str, buy_price: float, quantity: float) -> None:
    """Thêm một lần mua mới; ngày mua/ghi chú không còn dùng ở giao diện V1.3."""
    df = load_portfolio(path)
    row = {
        "ticker": str(ticker).upper().strip(),
        "buy_date": "",
        "buy_price": float(buy_price),
        "quantity": float(quantity),
        "fee": 0.0,
        "note": "",
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_portfolio(path, df)


def delete_position(path: str, ticker: str) -> None:
    """Xóa toàn bộ các lần mua của một mã khỏi danh mục."""
    df = load_portfolio(path)
    t = str(ticker).upper().strip()
    if df.empty:
        return
    df = df[df["ticker"].astype(str).str.upper().str.strip() != t].copy()
    save_portfolio(path, df)


def replace_position(path: str, ticker: str, avg_cost: float, quantity: float) -> None:
    """Điều chỉnh nhanh một vị thế thành một dòng giá vốn bình quân + số lượng."""
    delete_position(path, ticker)
    if float(quantity) > 0:
        add_purchase(path, ticker, avg_cost, quantity)


_SALE_COLS = ["date", "ticker", "quantity", "sell_price", "avg_cost", "pnl"]


def sales_path(portfolio_path: str) -> Path:
    return Path(portfolio_path).with_name("realized_sales.csv")


def load_sales(portfolio_path: str) -> pd.DataFrame:
    p = sales_path(portfolio_path)
    if not p.exists():
        return pd.DataFrame(columns=_SALE_COLS)
    try:
        df = pd.read_csv(p)
    except Exception:
        return pd.DataFrame(columns=_SALE_COLS)
    for c in _SALE_COLS:
        if c not in df.columns:
            df[c] = "" if c in {"date", "ticker"} else 0
    return df[_SALE_COLS]


def record_sale(portfolio_path: str, ticker: str, quantity: float, sell_price: float, avg_cost: float, date_text: str) -> float:
    """Bán một phần/bán hết theo giá vốn bình quân. Ghi lãi/lỗ đã chốt; trả về lãi/lỗ của lần bán này."""
    q, sp, ac = float(quantity), float(sell_price), float(avg_cost)
    pnl = (sp - ac) * q
    df = load_sales(portfolio_path)
    row = {"date": date_text, "ticker": str(ticker).upper().strip(), "quantity": q, "sell_price": sp, "avg_cost": ac, "pnl": pnl}
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    p = sales_path(portfolio_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return pnl


def realized_total(portfolio_path: str) -> float:
    df = load_sales(portfolio_path)
    return float(pd.to_numeric(df["pnl"], errors="coerce").fillna(0).sum()) if not df.empty else 0.0
