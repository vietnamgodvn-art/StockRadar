from __future__ import annotations

import pandas as pd

from .indicators import add_indicators
from .scoring import market_regime, score_stock


def build_market_tables(bundle) -> tuple[pd.DataFrame, dict, dict[str, pd.DataFrame]]:
    index_calc: dict[str, pd.DataFrame] = {k: add_indicators(v) for k, v in bundle.indexes.items()}
    benchmark = index_calc.get("VNINDEX")

    calc_histories: dict[str, pd.DataFrame] = {}
    rows = []
    meta = bundle.metadata.set_index("ticker") if not bundle.metadata.empty else pd.DataFrame()
    for ticker, hist in bundle.histories.items():
        calc = add_indicators(hist, benchmark=benchmark)
        calc_histories[ticker] = calc
        r = calc.iloc[-1].to_dict()
        r["ticker"] = ticker
        if ticker in meta.index:
            r["exchange"] = meta.loc[ticker].get("exchange", "")
            r["sector"] = meta.loc[ticker].get("sector", "")
        else:
            r["exchange"] = ""
            r["sector"] = ""
        src_chg = r.get("source_change_pct")
        r["change_pct"] = float(src_chg) if pd.notna(src_chg) else float(r.get("ret_1d", 0) or 0)
        rows.append(r)

    latest = pd.DataFrame(rows)
    regime = market_regime(benchmark, latest)

    scored = []
    for _, row in latest.iterrows():
        s = score_stock(row, regime["score"])
        merged = row.to_dict()
        merged.update({k: v for k, v in s.items() if k != "components"})
        merged["components"] = s["components"]
        merged["rank_score"] = round(s["score"] * 0.72 + s["attention"] * 2.8, 1)
        scored.append(merged)
    market = pd.DataFrame(scored).sort_values(["rank_score", "score"], ascending=False).reset_index(drop=True)
    return market, regime, calc_histories


