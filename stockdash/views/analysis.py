from __future__ import annotations

import html
import numpy as np
import pandas as pd
import streamlit as st

from ..config import DEFAULT_UNIVERSE
from ..ui_common import AGGRID_CSS, HELP, general_recommendation, icon, section_title, stock_detail_panel, trend_text

try:
    from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
    AGGRID_AVAILABLE = True
except Exception:
    AgGrid = None
    GridOptionsBuilder = None
    JsCode = None
    AGGRID_AVAILABLE = False

TOP_N = 20

_CANONICAL_META ={str(t).upper(): (str(e), str(s)) for t, e, s in DEFAULT_UNIVERSE}


def _display_exchange(value: str) -> str:
    x = str(value or '').upper().strip()
    return {'HOSE': 'HSX', 'HSX': 'HSX', 'HNX': 'HNX', 'UPCOM': 'UPCoM'}.get(x, x)


def _finite(value):
    try:
        x = float(value)
        return x if pd.notna(x) and np.isfinite(x) else None
    except Exception:
        return None


def _index_change(df: pd.DataFrame) -> tuple[float | None, float | None, float | None]:
    if df is None or df.empty:
        return None, None, None
    close = pd.to_numeric(df.get('close'), errors='coerce').dropna()
    if close.empty:
        return None, None, None
    last = float(close.iloc[-1])
    src_pct = _finite(df.iloc[-1].get('source_change_pct')) if len(df) else None
    if len(close) >= 2:
        prev = float(close.iloc[-2])
        point = last - prev
        pct = src_pct if src_pct is not None else ((last / prev - 1) * 100 if prev else 0.0)
    else:
        point = None
        pct = src_pct
    return last, point, pct


def _fmt_volume(v) -> str:
    x = _finite(v)
    if x is None:
        return '—'
    if abs(x) >= 1_000_000_000:
        return f'{x/1_000_000_000:.2f} tỷ'.replace('.', ',')
    if abs(x) >= 1_000_000:
        d = 1 if abs(x) >= 10_000_000 else 2
        return f'{x/1_000_000:.{d}f} triệu'.replace('.', ',')
    if abs(x) >= 1_000:
        return f'{x/1_000:.1f} nghìn'.replace('.', ',')
    return f'{x:,.0f}'


# V1619L_ANALYSIS_MARKER
def _pick_numeric(d: dict, keys: list[str]):
    for k in keys:
        if k in d:
            try:
                v = float(d[k])
                if pd.notna(v) and np.isfinite(v):
                    return v
            except Exception:
                pass
    return None


def _extract_index_snapshot(obj) -> dict:
    raw = {}
    if isinstance(obj, pd.DataFrame) and not obj.empty:
        try:
            raw.update({str(k): v for k, v in obj.iloc[-1].to_dict().items()})
        except Exception:
            pass
    elif isinstance(obj, dict):
        raw.update(obj)
        for k in ('data', 'quote', 'snapshot', 'stats', 'breadth', 'market_stats'):
            v = obj.get(k)
            if isinstance(v, dict):
                raw.update(v)
    up = _pick_numeric(raw, ['advancers', 'advancing', 'advance', 'up', 'up_count', 'num_up', 'gainers', 'increase'])
    down = _pick_numeric(raw, ['decliners', 'declining', 'decline', 'down', 'down_count', 'num_down', 'losers', 'decrease'])
    flat = _pick_numeric(raw, ['unchanged', 'nochange', 'no_change', 'flat', 'equal', 'flat_count', 'num_flat'])
    vol = _pick_numeric(raw, ['matched_volume', 'match_volume', 'volume', 'traded_volume', 'kl_khop', 'total_volume'])
    value = _pick_numeric(raw, ['close', 'index', 'value', 'last', 'price'])
    point = _pick_numeric(raw, ['change', 'point_change', 'delta'])
    pct = _pick_numeric(raw, ['change_pct', 'percent_change', 'pct'])
    return {'value': value, 'point': point, 'pct': pct, 'up': up, 'down': down, 'flat': flat, 'volume': vol}


def _market_cards(bundle, market: pd.DataFrame):
    """Ba sàn cùng một hàng; breadth/KL lấy từ snapshot TOÀN SÀN khi có."""
    specs = [
        ('VNINDEX', 'HSX', 'HOSE', 'HSX · VN-INDEX'),
        ('HNXINDEX', 'HNX', 'HNX', 'HNX · HNX-INDEX'),
        ('UPCOMINDEX', 'UPCoM', 'UPCOM', 'UPCoM · UPCoM-INDEX'),
    ]
    ex = market.copy() if isinstance(market, pd.DataFrame) else pd.DataFrame()
    if not ex.empty:
        ex['_ex'] = ex.get('exchange', '').astype(str).map(_display_exchange)
        ex['_chg'] = pd.to_numeric(ex.get('change_pct'), errors='coerce')
        ex['_vol'] = pd.to_numeric(ex.get('volume'), errors='coerce').fillna(0)

    wide = getattr(bundle, 'market_breadth', None) or {}
    cards = []
    for idx, exchange, wide_key, label in specs:
        idx_obj = getattr(bundle, 'indexes', {}).get(idx) if bundle is not None else None
        proxy_index = False
        if isinstance(idx_obj, pd.DataFrame) and not idx_obj.empty:
            proxy_index = 'ĐẠI DIỆN' in str(idx_obj.iloc[-1].get('price_time', '')).upper()
        snap = _extract_index_snapshot(idx_obj)
        value, point, pct = snap['value'], snap['point'], snap['pct']
        if (value is None or point is None or pct is None) and isinstance(idx_obj, pd.DataFrame) and not idx_obj.empty and not proxy_index:
            value, point, pct = _index_change(idx_obj)

        total_stat = wide.get(wide_key, {}) if isinstance(wide, dict) else {}
        coverage = float(total_stat.get('coverage', 0.0) or 0.0) if total_stat else 0.0
        if total_stat and int(total_stat.get('quoted', 0) or 0) > 0:
            up = int(total_stat.get('up', 0) or 0)
            down = int(total_stat.get('down', 0) or 0)
            flat = int(total_stat.get('flat', 0) or 0)
            total_vol = _finite(total_stat.get('volume'))
        else:
            # Không dùng rổ theo dõi làm đại diện cho toàn sàn nữa.
            up = down = flat = 0
            total_vol = np.nan

        n = up + down + flat
        ratio = up / n if n else None
        status = ('API chỉ số chưa sẵn sàng' if proxy_index else ('Tích cực' if ratio is not None and ratio >= .55 else ('Thận trọng' if ratio is not None and ratio <= .40 else ('Trung lập' if ratio is not None else 'Chưa đủ dữ liệu'))))
        cls = 'market-up' if (pct or 0) > 0 else ('market-down' if (pct or 0) < 0 else 'market-flat')
        tone = cls.replace('market-', 'tone-')
        arrow = icon('up', 13) if (pct or 0) > 0 else (icon('down', 13) if (pct or 0) < 0 else icon('flat', 13))
        val_txt = f'{value:,.2f}' if value is not None else '—'
        chg_txt = '—' if point is None or pct is None else f'{arrow} {point:+,.2f} điểm&nbsp;&nbsp;({pct:+.2f}%)'
        breadth = f'<span class="market-up">{icon("up", 12)}{up}</span><span class="market-down">{icon("down", 12)}{down}</span><span class="market-flat">{icon("flat", 12)}{flat}</span>'
        cards.append(
            f"""
        <div class="market-card-compact {tone}">
          <div class="market-card-head"><b>{html.escape(label)}</b><span class="market-change {cls}">{chg_txt}</span></div>
          <div class="market-index-line"><span class="market-index-value">{val_txt}</span><span class="market-status">{status}</span></div>
          <div class="market-card-foot"><span>{breadth}</span><span>KL khớp: {_fmt_volume(total_vol)}{(' · ' + str(int(round(coverage*100))) + '% mã') if coverage > 0 else ''}</span></div>
        </div>"""
        )
    st.markdown('<div class="market-compact-grid">' + ''.join(cards) + '</div>', unsafe_allow_html=True)

# V1619M_MARKET_CARDS_TOTALS
# V1619N_MARKET_CARDS_COVERAGE

def _compact_recommendation(r: pd.Series, regime: dict) -> str:
    text = general_recommendation(r, regime)
    if 'TRÁNH' in text or 'RỦI RO CAO' in text or 'CHƯA NÊN' in text:
        return 'TRÁNH/CHỜ'
    if 'MUA THĂM DÒ' in text:
        return 'MUA THĂM DÒ'
    if 'ĐIỂM MUA' in text or 'THEO DÕI ĐIỂM MUA' in text:
        return 'THEO DÕI MUA'
    if 'CHỜ TÍN HIỆU' in text:
        return 'CHỜ XÁC NHẬN'
    return 'QUAN SÁT'


def _buy_action_now(r: pd.Series, regime: dict) -> str:
    # Hành động mua ngắn gọn tại thời điểm cập nhật.
    if bool(r.get('price_only', False)):
        return 'CHƯA NÊN MUA · CHƯA ĐỦ DỮ LIỆU KỸ THUẬT'
    signal = str(r.get('signal', '') or '').upper().strip()
    score = int(_finite(r.get('score')) or 0)
    confidence = int(_finite(r.get('confidence')) or 0)
    rsi = _finite(r.get('rsi14'))
    rsi = 50.0 if rsi is None else rsi
    close = _finite(r.get('close'))
    ema20 = _finite(r.get('ema20'))
    volr = _finite(r.get('volume_ratio20'))
    volr = 1.0 if volr is None else volr
    breakout = bool(r.get('breakout20', False))
    breakdown = bool(r.get('breakdown20', False))
    regime_score = int(_finite(regime.get('score')) or 0)

    if breakdown or signal in {'BÁN', 'GIẢM TỶ TRỌNG'} or score < 42:
        return 'CHƯA NÊN MUA'
    price_ok = close is not None and ema20 is not None and close >= ema20
    strong_setup = (
        signal == 'MUA MẠNH' and confidence >= 72 and regime_score >= 55
        and rsi < 74 and price_ok and volr >= 0.9
    )
    confirmed_breakout = (
        signal in {'MUA', 'MUA MẠNH'} and breakout and confidence >= 70
        and regime_score >= 50 and rsi < 76 and volr >= 1.2
    )
    if strong_setup or confirmed_breakout:
        return 'CÓ THỂ MUA THĂM DÒ NGAY'
    if signal in {'MUA', 'MUA MẠNH'}:
        return 'CHỜ ĐIỂM MUA TỐT'
    if signal == 'THEO DÕI' or score >= 45:
        return 'CHỜ XÁC NHẬN'
    return 'CHƯA NÊN MUA'


def _trend_rec(r: pd.Series, regime: dict) -> str:
    # Quy ước duy nhất: ĐỎ = chưa mua, VÀNG = chờ, XANH = nên mua.
    if bool(r.get('price_only', False)):
        return 'CHƯA ĐỦ DỮ LIỆU'
    action = _buy_action_now(r, regime)
    if action.startswith('CÓ THỂ MUA'):
        return 'NÊN MUA THĂM DÒ'
    if action.startswith('CHỜ'):
        return 'CHỜ XÁC NHẬN'
    return 'CHƯA NÊN MUA'

# V1619M_SIGNAL_COLORS

def _ensure_confidence_value(r: pd.Series) -> int | None:
    conf = _finite(r.get('confidence'))
    if conf is not None:
        return int(max(0, min(100, round(conf))))
    close = _finite(r.get('close'))
    if close is None:
        return None
    score = _finite(r.get('score'))
    rsi = _finite(r.get('rsi14'))
    chg = _finite(r.get('change_pct'))
    vol = _finite(r.get('volume'))
    signal = str(r.get('signal', '') or '').upper().strip()
    base = 58.0
    if score is not None:
        base = 0.65 * float(score) + 20
    if rsi is not None:
        if 45 <= rsi <= 70:
            base += 6
        elif rsi < 35 or rsi > 78:
            base -= 5
    if chg is not None:
        base += min(8, abs(chg) * 1.2)
    if vol is not None and vol > 0:
        base += 4
    if signal in {'MUA', 'MUA MẠNH', 'THEO DÕI'}:
        base += 4
    if signal in {'BÁN', 'GIẢM TỶ TRỌNG'}:
        base += 2
    return int(max(35, min(95, round(base))))


def _prepare_grid(view: pd.DataFrame, regime: dict) -> pd.DataFrame:
    rows = []
    for _, r in view.iterrows():
        ticker = str(r.get('ticker', '') or '').upper().strip()
        exchange = str(r.get('exchange', '') or '')
        if ticker in _CANONICAL_META and not exchange:
            exchange = _CANONICAL_META[ticker][0]
        chg = _finite(r.get('change_pct'))
        if chg is None:
            move = '—'
        elif chg > 0:
            move = f'+{chg:.2f}%'.replace('.', ',')
        elif chg < 0:
            move = f'{chg:.2f}%'.replace('.', ',')
        else:
            move = '0,00%'
        rows.append({
            '_ticker': ticker,
            '_chg_num': chg,
            'Mã': ticker,
            'Sàn': _display_exchange(exchange),
            'Giá': _finite(r.get('close')),
            'Tăng/Giảm': move,
            'KL khớp hôm nay': _finite(r.get('volume')),
            'Điểm tin cậy': _ensure_confidence_value(r),
            'Khuyến nghị mua': _trend_rec(r, regime),
        })
    df = pd.DataFrame(rows)
    # Ô thiếu dữ liệu (ví dụ trước giờ mở cửa chưa có giá khớp) phải là rỗng để bảng hiện "—";
    # nếu để NaN thì st_aggrid gửi 0 và bảng hiện giá 0 gây hiểu nhầm.
    for col in ('Giá', 'KL khớp hôm nay', 'Điểm tin cậy'):
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].astype(object).where(df[col].notna(), None)
    return df


def _selected_position(event) -> int | None:
    """Lấy vị trí dòng từ cả row-selection lẫn cell-selection.

    Streamlit có thể trả state dạng object hoặc dict tùy phiên bản.
    Cho phép click trực tiếp vào ô Mã/cell để mở mã, không bắt người dùng
    phải click vùng chọn dòng ở mép trái.
    """
    try:
        selection = event.selection
    except Exception:
        try:
            selection = event.get("selection", {})
        except Exception:
            selection = {}

    try:
        rows = selection.rows
    except Exception:
        try:
            rows = selection.get("rows", [])
        except Exception:
            rows = []
    if rows:
        try:
            return int(rows[0])
        except Exception:
            pass

    try:
        cells = selection.cells
    except Exception:
        try:
            cells = selection.get("cells", [])
        except Exception:
            cells = []
    if cells:
        try:
            cell = cells[0]
            if isinstance(cell, (list, tuple)) and len(cell) >= 1:
                return int(cell[0])
        except Exception:
            pass
    return None


def _selected_ticker_from_aggrid(response) -> str | None:
    if response is None:
        return None
    selected = getattr(response, 'selected_rows', None)
    if selected is None:
        try:
            selected = response.get('selected_rows')
        except Exception:
            selected = None
    if isinstance(selected, pd.DataFrame):
        if selected.empty:
            return None
        row = selected.iloc[0]
        return str(row.get('_ticker', row.get('Mã', '')) or '').upper().strip() or None
    if isinstance(selected, list) and selected:
        row = selected[0]
        if isinstance(row, dict):
            return str(row.get('_ticker', row.get('Mã', '')) or '').upper().strip() or None
    if isinstance(selected, dict):
        return str(selected.get('_ticker', selected.get('Mã', '')) or '').upper().strip() or None
    return None


def _render_stock_grid(grid_df: pd.DataFrame, key: str) -> str | None:
    # Bảng Top: ưu tiên đủ cột trong khung, không cần kéo ngang.
    if grid_df is None or grid_df.empty:
        st.info('Không có mã phù hợp với bộ lọc hiện tại.')
        return None

    if AGGRID_AVAILABLE:
        gb = GridOptionsBuilder.from_dataframe(grid_df)
        gb.configure_default_column(sortable=True, filter=False, resizable=True, floatingFilter=False, editable=False, suppressMenu=True, cellStyle={'fontSize':'15px'})
        comma0 = JsCode(r'''
        function(params) {
          const v = Number(params.value);
          return Number.isFinite(v) ? Math.round(v).toLocaleString('en-US') : '—';
        }
        ''')
        donut_style = JsCode(r'''
        function(params) {
          const raw = Number(params.value);
          if (!Number.isFinite(raw)) {
            return {textAlign:'center', color:'#8a97a8', fontSize:'14px'};
          }
          const v = Math.max(0, Math.min(100, Math.round(raw)));
          const color = v >= 75 ? '#059669' : (v >= 60 ? '#d97706' : '#e11d48');
          const r = 17;
          const c = 2 * Math.PI * r;
          const dash = (v / 100) * c;
          const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="50" height="50" viewBox="0 0 50 50">
            <circle cx="25" cy="25" r="17" fill="none" stroke="#e6e9f0" stroke-width="7"/>
            <circle cx="25" cy="25" r="17" fill="none" stroke="${color}" stroke-width="7" stroke-linecap="round" stroke-dasharray="${dash} ${c-dash}" transform="rotate(-90 25 25)"/>
            <circle cx="25" cy="25" r="11" fill="#ffffff"/>
            <text x="25" y="29" text-anchor="middle" font-family="Segoe UI,Arial" font-size="12" font-weight="800" fill="#1e293b">${v}</text>
          </svg>`;
          return {
            textAlign:'center', color:'transparent', fontSize:'0px',
            backgroundImage:'url("data:image/svg+xml;utf8,' + encodeURIComponent(svg) + '")',
            backgroundRepeat:'no-repeat', backgroundPosition:'center', backgroundSize:'50px 50px'
          };
        }
        ''')
        change_style = JsCode(r"""
        function(params) {
          const raw = Number(params.data._chg_num);
          const up = raw > 0, dn = raw < 0;
          const color = !Number.isFinite(raw) ? '#52627a' : (up ? '#059669' : (dn ? '#e11d48' : '#d97706'));
          const path = !Number.isFinite(raw) ? '' : (up ? 'M6 15l6-6 6 6' : (dn ? 'M6 9l6 6 6-6' : 'M6 12h12'));
          const svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="' + color + '" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round"><path d="' + path + '"/></svg>';
          return {
            color: color, fontWeight: '700', fontSize: '14px', paddingLeft: '22px',
            backgroundImage: path ? 'url("data:image/svg+xml;utf8,' + encodeURIComponent(svg) + '")' : 'none',
            backgroundRepeat: 'no-repeat', backgroundPosition: '3px center', backgroundSize: '15px 15px'
          };
        }
        """)
        rec_style = JsCode(r"""
        function(params) {
          const raw = String(params.value || '').trim();
          const v = raw.toUpperCase();
          // [chữ, nền, chấm]
          let c = ['#475569', '#eef1f6', '#94a3b8'];
          if (v.includes('CHƯA ĐỦ')) c = ['#475569', '#eef1f6', '#94a3b8'];
          else if (v.includes('CHƯA NÊN') || v.includes('BÁN') || v.includes('GIẢM TỶ TRỌNG') || v.includes('CẮT LỖ')) c = ['#be123c', '#ffe9ee', '#e11d48'];
          else if (v.includes('CHỜ') || v.includes('QUAN SÁT') || v.includes('XÁC NHẬN')) c = ['#b45309', '#fff3d6', '#f59e0b'];
          else if (v.includes('NÊN MUA') || v.includes('CÓ THỂ MUA') || v.includes('GIỮ') || v.includes('THEO DÕI')) c = ['#047857', '#dff7ec', '#10b981'];
          const label = raw ? raw.charAt(0) + raw.slice(1).toLowerCase() : '—';
          const w = Math.round(label.length * 6.9 + 36), h = 26;
          const esc = (s) => s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
          const svg = '<svg xmlns="http://www.w3.org/2000/svg" width="' + w + '" height="' + h + '" viewBox="0 0 ' + w + ' ' + h + '">'
            + '<rect x="0.5" y="0.5" width="' + (w - 1) + '" height="' + (h - 1) + '" rx="13" fill="' + c[1] + '"/>'
            + '<circle cx="13" cy="13" r="3.4" fill="' + c[2] + '"/>'
            + '<text x="23" y="17.2" font-family="Inter,Segoe UI,Arial" font-size="12" font-weight="700" fill="' + c[0] + '">' + esc(label) + '</text></svg>';
          return {color: 'transparent', fontSize: '0px',
                  backgroundImage: 'url("data:image/svg+xml;utf8,' + encodeURIComponent(svg) + '")',
                  backgroundRepeat: 'no-repeat', backgroundPosition: '6px center', backgroundSize: w + 'px ' + h + 'px'};
        }
        """)
        gb.configure_column('_ticker', hide=True)
        gb.configure_column('_chg_num', hide=True)
        gb.configure_column('Mã', pinned='left', width=62, minWidth=56, maxWidth=74)
        gb.configure_column('Sàn', width=56, minWidth=50, maxWidth=66)
        gb.configure_column('Giá', width=82, minWidth=74, maxWidth=96, type=['numericColumn'], valueFormatter=comma0)
        gb.configure_column('Tăng/Giảm', width=96, minWidth=88, maxWidth=112, cellStyle=change_style)
        gb.configure_column('KL khớp hôm nay', headerName='KL khớp', width=104, minWidth=92, maxWidth=126, type=['numericColumn'], valueFormatter=comma0)
        gb.configure_column(
            'Điểm tin cậy', headerName='Tin cậy', width=80, minWidth=74, maxWidth=96,
            type=['numericColumn'], cellStyle=donut_style,
            headerTooltip='Mức độ tin cậy của đánh giá/khuyến nghị theo độ đồng thuận của bộ quy tắc hiện tại; không phải xác suất thắng được kiểm định.'
        )
        gb.configure_column('Khuyến nghị mua', headerName='Khuyến nghị', minWidth=170, flex=1.35, cellStyle=rec_style)
        opts = gb.build()
        opts['rowSelection'] = {'mode':'singleRow','enableClickSelection':True,'checkboxes':False,'headerCheckbox':False}
        opts['suppressRowClickSelection'] = False
        opts['suppressHorizontalScroll'] = True
        opts['animateRows'] = False
        opts['rowHeight'] = 56
        opts['headerHeight'] = 43
        response = AgGrid(
            grid_df, gridOptions=opts, height=670, theme='streamlit', custom_css=AGGRID_CSS,
            update_on=['selectionChanged'], allow_unsafe_jscode=True,
            fit_columns_on_grid_load=True, enable_enterprise_modules=False, key=key,
        )
        return _selected_ticker_from_aggrid(response)

    visible = grid_df.drop(columns=['_ticker'], errors='ignore').copy()
    event = st.dataframe(
        visible, width='stretch', hide_index=True, height=670,
        on_select='rerun', selection_mode='single-row', key=key + '_fallback',
        column_config={
            'Mã': st.column_config.TextColumn(width='small'),
            'Sàn': st.column_config.TextColumn(width='small'),
            'Giá': st.column_config.NumberColumn(format='%,.0f', width='small'),
            'Tăng/Giảm': st.column_config.TextColumn(width='small'),
            'KL khớp hôm nay': st.column_config.NumberColumn(format='%,.0f', width='medium'),
            'Điểm tin cậy': st.column_config.ProgressColumn(
                'Điểm tin cậy', min_value=0, max_value=100, format='%d', width='small',
                help='Mức độ tin cậy của đánh giá/khuyến nghị; không phải xác suất thắng.'
            ),
            'Khuyến nghị mua': st.column_config.TextColumn(width='large'),
        },
    )
    pos = _selected_position(event)
    if pos is not None and 0 <= pos < len(grid_df):
        return str(grid_df.iloc[pos]['_ticker']).upper().strip() or None
    return None


def _apply_settings(view: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    if view.empty:
        return view
    c1, c2 = st.columns([5.5, .6])
    with c1:
        st.caption('Bấm tiêu đề để sắp xếp · nút Bộ lọc để lọc · Điểm tin cậy = độ đáng tin của khuyến nghị · click mã để đổi biểu đồ.')
    with c2:
        with st.popover('Bộ lọc', width='stretch', icon=':material/tune:'):
            st.markdown('**Tùy chọn bảng**')
            q = st.text_input('Tìm mã', key=f'{key_prefix}_ticker_filter', placeholder='HPG...').strip().upper()
            exchanges = sorted({_display_exchange(x) for x in view.get('exchange', pd.Series(dtype=str)).dropna().astype(str) if str(x).strip()})
            ex_sel = st.multiselect('Sàn', exchanges, key=f'{key_prefix}_exchange_filter')
            min_score = st.number_input('Điểm tin cậy tối thiểu', 0, 100, 0, 5, key=f'{key_prefix}_score_filter')
            trend = st.selectbox('Xu hướng', ['Tất cả', 'Tăng', 'Trung lập', 'Giảm/Yếu'], key=f'{key_prefix}_trend_filter')
            min_vol = st.number_input('KL khớp tối thiểu', min_value=0.0, value=0.0, step=100000.0, format='%.0f', key=f'{key_prefix}_vol_filter')
    out = view.copy()
    if q:
        out = out[out['ticker'].astype(str).str.upper().str.contains(q, regex=False)]
    if ex_sel:
        out = out[out['exchange'].astype(str).map(_display_exchange).isin(ex_sel)]
    if min_score > 0 and not out.empty:
        # Lọc đúng con số đang hiển thị ở cột Điểm tin cậy.
        shown_conf = out.apply(_ensure_confidence_value, axis=1)
        out = out[pd.to_numeric(shown_conf, errors='coerce').fillna(-1) >= min_score]
    if min_vol > 0:
        out = out[pd.to_numeric(out.get('volume'), errors='coerce').fillna(0) >= min_vol]
    if trend != 'Tất cả' and not out.empty:
        # Trước đây lọc theo ký tự ▲/→/▼ trong cột khuyến nghị, mà cột này không còn
        # chứa mũi tên → bảng luôn rỗng. Nay phân nhóm trực tiếp từ dữ liệu.
        out = out[out.apply(_trend_bucket, axis=1) == trend]
    return out


def _trend_bucket(r: pd.Series) -> str:
    """Nhóm xu hướng cho bộ lọc: mã chỉ có giá dùng % thay đổi, mã đã phân tích dùng EMA/MA."""
    if bool(r.get('price_only', False)):
        chg = _finite(r.get('change_pct'))
        if chg is None or abs(chg) <= 0.5:
            return 'Trung lập'
        return 'Tăng' if chg > 0 else 'Giảm/Yếu'
    t = trend_text(r)
    if t.startswith('TĂNG'):
        return 'Tăng'
    if t == 'YẾU':
        return 'Giảm/Yếu'
    return 'Trung lập'


def render_analysis(
    bundle,
    market: pd.DataFrame,
    top30: pd.DataFrame,
    histories: dict,
    regime: dict,
    is_live: bool,
    symbol_catalog: pd.DataFrame | None = None,
    load_symbol=None,
    quick_mode: bool = False,
    hourly_histories: dict | None = None,
    load_hourly=None,
):
    _market_cards(bundle, market)

    # Cố định 20 mã nổi bật (bỏ lựa chọn Top 20/30).
    view_raw = top30.head(TOP_N).copy()
    if view_raw.empty:
        st.info('Chưa có dữ liệu. Bấm ĐỒNG BỘ để lấy giá thật; dùng CẬP NHẬT TOÀN BỘ ở thanh bên khi muốn tính toàn bộ chỉ báo kỹ thuật.')
        return

    default_ticker = str(st.session_state.get('analysis_ticker', str(view_raw.iloc[0]['ticker']))).upper().strip()
    st.session_state['analysis_ticker'] = default_ticker

    left, right = st.columns([1.06, 1.28], gap='medium')
    with left:
        view_filtered = _apply_settings(view_raw, 'top169')
        grid_df = _prepare_grid(view_filtered, regime)
        grid_rev = int(st.session_state.get('analysis_grid_revision_v1612', 0) or 0)
        selected = _render_stock_grid(grid_df, f'bang_top_v1612_{grid_rev}')
        # Chỉ xử lý khi lựa chọn bảng THỰC SỰ thay đổi. Nếu không, một selection cũ
        # sẽ ghi đè mã vừa chọn từ ô tìm kiếm sau mỗi lần rerun.
        last_grid_selected = str(st.session_state.get('analysis_last_grid_selected_v1612', '') or '').upper().strip()
        if selected and selected != last_grid_selected:
            st.session_state['analysis_last_grid_selected_v1612'] = selected
            st.session_state['analysis_ticker'] = selected
            st.rerun()

        catalog = symbol_catalog.copy() if isinstance(symbol_catalog, pd.DataFrame) else pd.DataFrame()
        if catalog.empty:
            cols = [c for c in ['ticker', 'exchange'] if c in market.columns]
            catalog = market[cols].copy() if cols else pd.DataFrame(columns=['ticker', 'exchange'])
            if 'ticker' not in catalog.columns: catalog['ticker'] = []
            catalog['company_name'] = catalog.get('ticker', pd.Series(dtype=str))
        catalog['ticker'] = catalog['ticker'].astype(str).str.upper().str.strip()
        catalog = catalog[catalog['ticker'].ne('')].drop_duplicates('ticker')
        options = catalog['ticker'].tolist()
        names = catalog.set_index('ticker').to_dict('index') if not catalog.empty else {}

        def _fmt_symbol(t):
            r = names.get(t, {})
            name = str(r.get('company_name', '') or '')
            exch = _display_exchange(r.get('exchange', ''))
            suffix = ' · '.join([x for x in [name, exch] if x and x != t])
            return f'{t} — {suffix}' if suffix else t

        search_rev = int(st.session_state.get('analysis_search_revision_v1612', 0) or 0)
        searched = st.selectbox(
            'Tìm mã bất kỳ trên HSX / HNX / UPCoM', options, index=None,
            placeholder='Gõ mã hoặc tên công ty, ví dụ HPG / Hòa Phát...',
            format_func=_fmt_symbol, key=f'tim_toan_thi_truong_v1612_{search_rev}',
            help='Không giới hạn Top 20/30. Chọn mã để tải riêng lịch sử và mở biểu đồ bên phải.',
        )
        if searched:
            # Mở mã tìm kiếm rồi tạo widget search mới ở lần rerun kế tiếp.
            # Nhờ vậy giá trị search cũ không thể ghi đè mã vừa click trong bảng.
            st.session_state['analysis_ticker'] = searched
            st.session_state['analysis_grid_revision_v1612'] = int(st.session_state.get('analysis_grid_revision_v1612', 0) or 0) + 1
            st.session_state['analysis_search_revision_v1612'] = search_rev + 1
            st.session_state['analysis_last_grid_selected_v1612'] = ''
            st.rerun()

    with right:
        chosen = str(st.session_state.get('analysis_ticker', str(view_raw.iloc[0]['ticker']))).upper().strip()
        ticker_set = set(market.get('ticker', pd.Series(dtype=str)).astype(str).str.upper())
        if chosen not in ticker_set or chosen not in histories:
            if load_symbol is not None:
                try:
                    with st.spinner(f'Đang tải lịch sử và phân tích {chosen}...'):
                        load_symbol(chosen)
                    st.rerun()
                except Exception as e:
                    st.error(f'Không tải được {chosen}: {e}')
                    return
            else:
                st.warning(f'Chưa có dữ liệu {chosen}.')
                return

        full_hist_key = 'full_history_loaded_v1619k'
        loaded = set(st.session_state.get(full_hist_key, []) or [])
        if load_symbol is not None and chosen not in loaded:
            try:
                with st.spinner(f'Đang tải đầy đủ lịch sử niêm yết của {chosen}...'):
                    load_symbol(chosen)
                loaded.add(chosen)
                st.session_state[full_hist_key] = sorted(loaded)
                st.rerun()
            except Exception:
                pass

        hourly_df = None
        if isinstance(hourly_histories, dict):
            hourly_df = hourly_histories.get(chosen)
        if (hourly_df is None or not isinstance(hourly_df, pd.DataFrame) or hourly_df.empty) and load_hourly is not None:
            try:
                hourly_df = load_hourly(chosen)
            except Exception:
                hourly_df = pd.DataFrame()

        stock_detail_panel(chosen, market, histories, regime, 'phan_tich_v169', demo_warning=not is_live, compact=True, hourly_df=hourly_df)
