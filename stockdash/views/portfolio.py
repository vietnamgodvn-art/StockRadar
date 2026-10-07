from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from ..portfolio import aggregate_portfolio
from ..storage import add_purchase, delete_position, load_portfolio, replace_position, save_portfolio, load_sales, record_sale
from ..ui_common import AGGRID_CSS, HELP, THEME, icon, section_title, stock_detail_panel

try:
    from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
    AGGRID_AVAILABLE = True
except Exception:
    AgGrid = None
    GridOptionsBuilder = None
    JsCode = None
    AGGRID_AVAILABLE = False


def _selected_position(event) -> int | None:
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


def _ensure_confidence_value(r: pd.Series) -> int | None:
    try:
        conf = float(r.get('confidence'))
        if np.isfinite(conf):
            return int(max(0, min(100, round(conf))))
    except Exception:
        pass
    try:
        close = float(r.get('close'))
        if not np.isfinite(close):
            return None
    except Exception:
        return None
    base = 58.0
    try:
        score = float(r.get('score'))
        if np.isfinite(score):
            base = 0.65 * score + 20
    except Exception:
        pass
    try:
        pnl_pct = abs(float(r.get('pnl_pct')))
        if np.isfinite(pnl_pct):
            base += min(8, pnl_pct / 3.0)
    except Exception:
        pass
    return int(max(35, min(95, round(base))))



# V1619L_PORTFOLIO_DAY_CHANGE_COLUMN
def _portfolio_grid(portfolio: pd.DataFrame, key: str) -> str | None:
    avg_cost = pd.to_numeric(portfolio['avg_cost'], errors='coerce')
    close_now = pd.to_numeric(portfolio['close'], errors='coerce')
    pnl_abs = pd.to_numeric(portfolio['pnl'], errors='coerce')
    pnl_pct = pd.to_numeric(portfolio['pnl_pct'], errors='coerce')
    day_change = pd.to_numeric(portfolio.get('change_pct', pd.Series([np.nan] * len(portfolio), index=portfolio.index)), errors='coerce')

    def fmt0(v):
        return f'{float(v):,.0f}' if pd.notna(v) and np.isfinite(float(v)) else '—'

    def day_label(v):
        if pd.isna(v) or not np.isfinite(float(v)):
            return '—'
        x = float(v)
        if x > 0:
            return f'+{x:.2f}%'.replace('.', ',')
        if x < 0:
            return f'{x:.2f}%'.replace('.', ',')
        return '0,00%'

    def pnl_label(a, b):
        if pd.isna(a) or not np.isfinite(float(a)):
            return '—'
        sign = '+' if float(a) > 0 else ('-' if float(a) < 0 else '')
        pct_txt = '—' if pd.isna(b) or not np.isfinite(float(b)) else f'{float(b):+.2f}%'.replace('.', ',')
        return f'{sign}{abs(float(a)):,.0f}\n({pct_txt})'

    view = pd.DataFrame({
        '_ticker': portfolio['ticker'].astype(str),
        'Mã': portfolio['ticker'].astype(str),
        'SL': pd.to_numeric(portfolio['quantity'], errors='coerce'),
        '_avg_cost': avg_cost,
        '_close_now': close_now,
        '_day_change': day_change,
        '_pnl': pnl_abs,
        'Giá vốn / Hiện tại': [f'Vốn: {fmt0(a)}\nHiện tại: {fmt0(b)}' for a, b in zip(avg_cost, close_now)],
        '% phiên trước': [day_label(c) for c in day_change],
        'Lãi/Lỗ': [pnl_label(a, b) for a, b in zip(pnl_abs, pnl_pct)],
        'Độ rõ tín hiệu': portfolio.apply(_ensure_confidence_value, axis=1),
        'Khuyến nghị': portfolio.get('action_now', pd.Series([''] * len(portfolio))).astype(str),
    })

    if AGGRID_AVAILABLE:
        gb = GridOptionsBuilder.from_dataframe(view)
        gb.configure_default_column(
            sortable=True, filter=False, resizable=True, floatingFilter=False,
            editable=False, suppressMenu=True, cellStyle={'fontSize': '15px', 'fontFamily': 'Segoe UI, Arial, sans-serif', 'fontWeight': '400'}
        )

        comma0 = JsCode("""
        function(params) {
          const v = Number(params.value);
          return Number.isFinite(v) ? Math.round(v).toLocaleString('en-US') : '—';
        }
        """)
        pnl_style = JsCode("""
        function(params) {
          function arrowStyle(val, bold) {
            const n = Number(val);
            const up = n > 0, dn = n < 0;
            const color = !Number.isFinite(n) ? '#52627a' : (up ? '#059669' : (dn ? '#e11d48' : '#d97706'));
            const path = !Number.isFinite(n) ? '' : (up ? 'M6 15l6-6 6 6' : (dn ? 'M6 9l6 6 6-6' : 'M6 12h12'));
            const svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="' + color + '" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round"><path d="' + path + '"/></svg>';
            return {color: color, fontWeight: bold, fontSize: '14px', paddingLeft: '22px', whiteSpace: 'pre-line', lineHeight: '1.25',
                    backgroundImage: path ? 'url("data:image/svg+xml;utf8,' + encodeURIComponent(svg) + '")' : 'none',
                    backgroundRepeat: 'no-repeat', backgroundPosition: '3px 12px', backgroundSize: '15px 15px'};
          }
          return arrowStyle(params.data._pnl, '700');
        }
        """)
        cost_now_style = JsCode("""
        function(params) {
          const fmt0 = (v) => Number.isFinite(Number(v)) ? Math.round(Number(v)).toLocaleString('en-US') : '—';
          const avg = fmt0(params.data._avg_cost);
          const now = fmt0(params.data._close_now);
          const esc = (s) => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
          const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="190" height="52" viewBox="0 0 190 52">
            <text x="4" y="18" font-family="Segoe UI,Arial" font-size="14" font-weight="400" fill="#1e293b">Vốn: ${esc(avg)}</text>
            <text x="4" y="38" font-family="Segoe UI,Arial" font-size="14" font-weight="400" fill="#1e293b">Hiện tại: ${esc(now)}</text>
          </svg>`;
          return {
            color:'transparent', fontSize:'0px',
            backgroundImage:'url("data:image/svg+xml;utf8,' + encodeURIComponent(svg) + '")',
            backgroundRepeat:'no-repeat', backgroundPosition:'left center', backgroundSize:'190px 52px',
            whiteSpace:'normal', lineHeight:'1.2', fontFamily:'Segoe UI, Arial, sans-serif'
          };
        }
        """)
        day_style = JsCode("""
        function(params) {
          function arrowStyle(val, bold) {
            const n = Number(val);
            const up = n > 0, dn = n < 0;
            const color = !Number.isFinite(n) ? '#52627a' : (up ? '#059669' : (dn ? '#e11d48' : '#d97706'));
            const path = !Number.isFinite(n) ? '' : (up ? 'M6 15l6-6 6 6' : (dn ? 'M6 9l6 6 6-6' : 'M6 12h12'));
            const svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="' + color + '" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round"><path d="' + path + '"/></svg>';
            return {color: color, fontWeight: bold, fontSize: '14px', paddingLeft: '22px', whiteSpace: 'pre-line', lineHeight: '1.25',
                    backgroundImage: path ? 'url("data:image/svg+xml;utf8,' + encodeURIComponent(svg) + '")' : 'none',
                    backgroundRepeat: 'no-repeat', backgroundPosition: '3px 12px', backgroundSize: '15px 15px'};
          }
          return arrowStyle(params.data._day_change, '700');
        }
        """)
        pnl_sort = JsCode("""
        function(valueA, valueB, nodeA, nodeB) {
          const a = Number(nodeA && nodeA.data ? nodeA.data._pnl : NaN);
          const b = Number(nodeB && nodeB.data ? nodeB.data._pnl : NaN);
          if (!Number.isFinite(a) && !Number.isFinite(b)) return 0;
          if (!Number.isFinite(a)) return -1;
          if (!Number.isFinite(b)) return 1;
          return a - b;
        }
        """)
        donut_style = JsCode("""
        function(params) {
          const raw = Number(params.value);
          if (!Number.isFinite(raw)) return {textAlign:'center', color:'#8a97a8', fontSize:'14px'};
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
        gb.configure_column('_avg_cost', hide=True)
        gb.configure_column('_close_now', hide=True)
        gb.configure_column('_day_change', hide=True)
        gb.configure_column('_pnl', hide=True)
        gb.configure_column('Mã', pinned='left', width=70, minWidth=64, maxWidth=82)
        gb.configure_column('SL', width=80, minWidth=72, maxWidth=90, type=['numericColumn'], valueFormatter=comma0)
        gb.configure_column('Giá vốn / Hiện tại', width=166, minWidth=154, maxWidth=180, cellStyle=cost_now_style)
        gb.configure_column('% phiên trước', width=104, minWidth=96, maxWidth=114, cellStyle=day_style)
        gb.configure_column('Lãi/Lỗ', width=132, minWidth=122, maxWidth=144, cellStyle=pnl_style, comparator=pnl_sort)
        gb.configure_column('Độ rõ tín hiệu', width=112, minWidth=104, maxWidth=118, type=['numericColumn'], cellStyle=donut_style)
        gb.configure_column('Khuyến nghị', minWidth=178, flex=0.9, cellStyle=rec_style)

        opts = gb.build()
        opts['rowSelection'] = {'mode': 'singleRow', 'enableClickSelection': True, 'checkboxes': False, 'headerCheckbox': False}
        opts['suppressRowClickSelection'] = False
        opts['suppressHorizontalScroll'] = True
        opts['animateRows'] = False
        opts['rowHeight'] = 78
        opts['headerHeight'] = 42
        response = AgGrid(
            view[['Mã', 'SL', 'Giá vốn / Hiện tại', '% phiên trước', 'Lãi/Lỗ', 'Độ rõ tín hiệu', 'Khuyến nghị', '_ticker', '_avg_cost', '_close_now', '_day_change', '_pnl']],
            gridOptions=opts,
            height=610,
            theme='streamlit', custom_css=AGGRID_CSS,
            update_on=['selectionChanged'],
            allow_unsafe_jscode=True,
            fit_columns_on_grid_load=True,
            enable_enterprise_modules=False,
            key=key,
        )
        return _selected_ticker_from_aggrid(response)

    fallback = view[['_ticker', 'Mã', 'SL', 'Giá vốn / Hiện tại', '% phiên trước', 'Lãi/Lỗ', 'Độ rõ tín hiệu', 'Khuyến nghị']].copy()
    event = st.dataframe(
        fallback.drop(columns=['_ticker']), width='stretch', hide_index=True, height=610,
        on_select='rerun', selection_mode='single-row', key=key + '_fallback',
        column_config={
            'Mã': st.column_config.TextColumn(width='small'),
            'SL': st.column_config.NumberColumn(format='%,.0f', width='small'),
            'Giá vốn / Hiện tại': st.column_config.TextColumn(width='medium'),
            '% phiên trước': st.column_config.TextColumn(width='small'),
            'Lãi/Lỗ': st.column_config.TextColumn(width='medium'),
            'Độ rõ tín hiệu': st.column_config.ProgressColumn('Độ rõ tín hiệu', min_value=0, max_value=100, format='%d', width='small'),
            'Khuyến nghị': st.column_config.TextColumn(width='medium'),
        },
    )
    pos = _selected_position(event)
    if pos is not None and 0 <= pos < len(fallback):
        return str(fallback.iloc[pos]['_ticker']).upper().strip() or None
    return None



def _lerp_hex(c1: str, c2: str, k: float) -> str:
    k = max(0.0, min(1.0, float(k)))
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * k):02x}" for x, y in zip(a, b))


def _treemap_color(k: float) -> str:
    """Thang màu chàm: ô nhỏ nhạt, ô lớn đậm hơn (pastel, cùng tông chàm của giao diện)."""
    stops = ["#dfe5ff", "#b9c4ff", "#8e9bf7", "#6a73ee"]
    k = max(0.0, min(1.0, float(k))) * (len(stops) - 1)
    i = min(int(k), len(stops) - 2)
    return _lerp_hex(stops[i], stops[i + 1], k - i)


def _portfolio_treemap_figure(df: pd.DataFrame, value_col: str, title: str, note: str = ""):
    data = df.copy()
    data = data[pd.to_numeric(data[value_col], errors='coerce').fillna(0) > 0].copy()
    if data.empty:
        return None
    vals = pd.to_numeric(data[value_col], errors='coerce').fillna(0).astype(float).tolist()
    total = float(sum(vals))
    labels = data['ticker'].astype(str).tolist()
    top = max(vals) or 1.0
    text = [
        f"<b>{t}</b><br>{v:,.0f}<br>{(v / total * 100 if total else 0):.1f}%".replace('.', ',')
        for t, v in zip(labels, vals)
    ]
    root = "Danh mục"
    shades = [_treemap_color((v / top) ** 0.6) for v in vals]
    txt_colors = [THEME['ink'] if (v / top) ** 0.6 < 0.62 else '#ffffff' for v in vals]
    # Có ô gốc tường minh (màu trắng) để Plotly không tự vẽ ô gốc xám #444 phía sau.
    fig = go.Figure(go.Treemap(
        labels=[root] + labels,
        parents=[""] + [root] * len(labels),
        values=[total] + vals,
        branchvalues='total',
        text=[""] + text,
        textinfo='text',
        textfont=dict(size=14, color=['#ffffff'] + txt_colors, family="Inter, Segoe UI, Arial"),
        marker=dict(
            colors=["#ffffff"] + shades,
            line=dict(color='#ffffff', width=4),
            cornerradius=12,
        ),
        hovertemplate='<b>%{label}</b><br>Giá trị: %{value:,.0f} VND<br>Tỷ trọng: %{percentRoot:.1%}<extra></extra>',
        tiling=dict(pad=0),
        pathbar=dict(visible=False),
        root=dict(color='#ffffff'),
        sort=True,
    ))
    fig.update_layout(
        margin=dict(l=12, r=12, t=12, b=12),
        height=330,
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        font=dict(size=14, color=THEME['ink'], family="Inter, Segoe UI, Arial"),
        hoverlabel=dict(bgcolor='#0f172a', font=dict(color='#ffffff', family="Inter, Segoe UI, Arial", size=13), bordercolor='#0f172a'),
    )
    return fig


def _render_portfolio_treemaps(portfolio: pd.DataFrame, is_live: bool):
    t1, t2 = st.columns(2, gap='small')
    with t1:
        st.markdown("<div class='treemap-card-head'>TỶ TRỌNG VỐN BAN ĐẦU</div>", unsafe_allow_html=True)
        st.markdown("<div class='treemap-card-wrap'>", unsafe_allow_html=True)
        fig_cost = _portfolio_treemap_figure(portfolio, 'total_cost', 'TỶ TRỌNG VỐN BAN ĐẦU')
        if fig_cost is not None:
            st.plotly_chart(fig_cost, width='stretch', config={'displayModeBar': False})
        else:
            st.info('Chưa có dữ liệu vốn ban đầu để vẽ sơ đồ.')
        st.markdown("</div>", unsafe_allow_html=True)
    with t2:
        st.markdown("<div class='treemap-card-head'>TỶ TRỌNG TÀI SẢN HIỆN TẠI</div>", unsafe_allow_html=True)
        st.markdown("<div class='treemap-card-wrap'>", unsafe_allow_html=True)
        if is_live and 'market_value' in portfolio.columns:
            current_df = portfolio[pd.to_numeric(portfolio['market_value'], errors='coerce').fillna(0) > 0].copy()
            fig_val = _portfolio_treemap_figure(current_df, 'market_value', 'TỶ TRỌNG TÀI SẢN HIỆN TẠI')
            if fig_val is not None:
                st.plotly_chart(fig_val, width='stretch', config={'displayModeBar': False})
            else:
                st.info('Chưa có đủ giá hiện tại để vẽ tỷ trọng tài sản.')
        else:
            st.info('Bấm CẬP NHẬT để lấy giá hiện tại rồi mới vẽ tỷ trọng tài sản.')
        st.markdown("</div>", unsafe_allow_html=True)



def _portfolio_overview_html(portfolio: pd.DataFrame, pnl_pct: float, priced_count: int, total_count: int) -> str:
    """Tạo dải tổng quan danh mục từ chính dữ liệu hiện tại, không dùng số giả."""
    if portfolio is None or portfolio.empty:
        return ''

    x = portfolio.copy()
    x['_action'] = x.get('action_now', pd.Series([''] * len(x), index=x.index)).astype(str).str.upper().str.strip()
    x['_pnl_pct'] = pd.to_numeric(x.get('pnl_pct'), errors='coerce')
    x['_score'] = pd.to_numeric(x.get('score'), errors='coerce')
    x['_confidence'] = x.apply(_ensure_confidence_value, axis=1)

    sell_mask = x['_action'].str.contains('BÁN|GIẢM', regex=True, na=False)
    hold_mask = x['_action'].str.contains('GIỮ', regex=True, na=False) & ~sell_mask
    wait_mask = ~(sell_mask | hold_mask)

    sell_count = int(sell_mask.sum())
    hold_count = int(hold_mask.sum())
    wait_count = int(wait_mask.sum())
    denom = max(len(x), 1)
    sell_ratio = sell_count / denom
    hold_ratio = hold_count / denom

    priced_pnl = x['_pnl_pct'].dropna()
    win_count = int((priced_pnl > 0).sum())
    loss_count = int((priced_pnl < 0).sum())
    conf = pd.to_numeric(x['_confidence'], errors='coerce').dropna()
    avg_conf = int(round(float(conf.mean()))) if not conf.empty else 0

    pnl_known = bool(np.isfinite(pnl_pct))
    if (pnl_known and pnl_pct <= -15) or sell_ratio >= 0.50:
        state = 'PHÒNG THỦ · ƯU TIÊN HẠ RỦI RO'
        state_color = '#e11d48'
        state_desc = (
            f"Danh mục hiện {'âm' if pnl_pct < 0 else 'tăng'} khoảng <b>{abs(pnl_pct):.1f}%</b> so với giá vốn. "
            f"Có <b>{sell_count}/{len(x)}</b> mã đang ở trạng thái Bán/Giảm tỷ trọng. "
            "Ưu tiên bảo toàn vốn và xử lý mã yếu trước, chưa bình quân giá xuống toàn danh mục."
        ) if pnl_known else (
            f"Có <b>{sell_count}/{len(x)}</b> mã đang ở trạng thái Bán/Giảm tỷ trọng. "
            "Ưu tiên xử lý mã yếu trước và hạn chế tăng thêm rủi ro."
        )
    elif (pnl_known and pnl_pct < -5) or sell_ratio >= 0.35:
        state = 'THẬN TRỌNG · ƯU TIÊN TÁI CƠ CẤU'
        state_color = '#d97706'
        state_desc = (
            f"Danh mục đang {'âm' if pnl_pct < 0 else 'tăng'} khoảng <b>{abs(pnl_pct):.1f}%</b>; "
            f"{sell_count} mã cần giảm rủi ro và {hold_count} mã có thể tiếp tục giữ. "
            "Nên tái cơ cấu theo chất lượng tín hiệu thay vì xử lý đồng loạt."
        ) if pnl_known else (
            f"{sell_count} mã cần giảm rủi ro và {hold_count} mã có thể tiếp tục giữ. "
            "Ưu tiên tái cơ cấu theo chất lượng tín hiệu."
        )
    elif pnl_known and pnl_pct >= 0 and hold_ratio >= 0.60:
        state = 'TÍCH CỰC · ƯU TIÊN GIỮ MÃ KHỎE'
        state_color = '#059669'
        state_desc = (
            f"Danh mục đang tăng khoảng <b>{abs(pnl_pct):.1f}%</b>; phần lớn mã vẫn ở trạng thái giữ. "
            "Ưu tiên giữ mã khỏe, dời mốc bảo vệ lợi nhuận và chỉ tăng tỷ trọng khi giá/khối lượng xác nhận."
        )
    else:
        state = 'CÂN BẰNG · ƯU TIÊN CHỌN LỌC'
        state_color = '#4f46e5'
        pnl_text = f"P/L toàn danh mục khoảng <b>{pnl_pct:+.1f}%</b>. " if pnl_known else ''
        state_desc = (
            pnl_text + f"Hiện có {sell_count} mã Bán/Giảm, {hold_count} mã Giữ và {wait_count} mã Chờ. "
            "Nên hành động theo từng mã, không tăng tỷ trọng đồng loạt."
        )

    def names(mask, kind):
        z = x.loc[mask, ['ticker', '_pnl_pct', '_score', '_confidence']].copy()
        if z.empty:
            return '—'
        if kind == 'sell':
            z = z.sort_values(['_pnl_pct', '_confidence'], ascending=[True, False], na_position='last')
        elif kind == 'hold':
            z = z.sort_values(['_score', '_confidence'], ascending=[False, False], na_position='last')
        else:
            z = z.sort_values(['_confidence', '_score'], ascending=[False, False], na_position='last')
        vals = z['ticker'].astype(str).str.upper().tolist()[:5]
        return ' · '.join(vals) if vals else '—'

    sell_names = names(sell_mask, 'sell')
    hold_names = names(hold_mask, 'hold')
    wait_names = names(wait_mask, 'wait')

    coverage = f"{priced_count}/{total_count} mã đã có giá" if total_count else '—'
    return f"""
    <div id="portfolio-overview-1619j" class="pf-ov">
      <div class="pf-card">
        <div class="pf-k">TỔNG QUAN VỊ THẾ DANH MỤC</div>
        <div class="pf-state" style="color:{state_color};">{state}</div>
        <div class="pf-desc">{state_desc}</div>
        <div class="pf-foot">{coverage} · Nội dung tự đổi theo lần Đồng bộ gần nhất.</div>
      </div>
      <div class="pf-card">
        <div class="pf-k">CƠ CẤU HÀNH ĐỘNG HIỆN TẠI</div>
        <div class="pf-stats">
          <div class="pf-stat"><div class="n" style="color:var(--down);">{sell_count}</div><div class="t">Bán / Giảm</div></div>
          <div class="pf-stat"><div class="n" style="color:var(--up);">{hold_count}</div><div class="t">Giữ tiếp</div></div>
          <div class="pf-stat"><div class="n" style="color:var(--wait);">{wait_count}</div><div class="t">Chờ / Chưa rõ</div></div>
        </div>
        <div class="pf-foot">Độ rõ TB: <b>{avg_conf}/100</b> · Mã đang lãi: <b style="color:var(--up);">{win_count}</b> · Mã đang lỗ: <b style="color:var(--down);">{loss_count}</b></div>
      </div>
      <div class="pf-card">
        <div class="pf-k">ĐỊNH HƯỚNG HÀNH ĐỘNG CHO LIST MÃ HIỆN TẠI</div>
        <div class="pf-desc">
          <div class="pf-line"><span class="pf-tag down">ƯU TIÊN XỬ LÝ</span>{sell_names} — theo dõi hỗ trợ/cắt giảm trước, không bình quân giá xuống khi xu hướng chưa cải thiện.</div>
          <div class="pf-line"><span class="pf-tag up">TIẾP TỤC GIỮ</span>{hold_names} — giữ khi còn trên mốc bảo vệ và chưa xuất hiện tín hiệu bán mới.</div>
          <div class="pf-line"><span class="pf-tag wait">CHỜ XÁC NHẬN</span>{wait_names} — chưa mua thêm chỉ vì giá giảm hoặc một phiên hồi.</div>
        </div>
        <div class="pf-foot"><b>Nguyên tắc:</b> xử lý mã yếu trước → giữ mã khỏe → chỉ tăng tỷ trọng khi giá và khối lượng cùng xác nhận.</div>
      </div>
    </div>
    """


def _render_portfolio_overview(portfolio: pd.DataFrame, pnl_pct: float, priced_count: int, total_count: int):
    html = _portfolio_overview_html(portfolio, pnl_pct, priced_count, total_count)
    if html:
        st.markdown(html, unsafe_allow_html=True)

_BACKUP_COLUMNS = ["ticker", "buy_date", "buy_price", "quantity", "fee", "note"]


def _render_portfolio_backup(portfolio_path: str, lots: pd.DataFrame) -> None:
    """Sao lưu/khôi phục danh mục bằng file CSV.

    Khi chạy trên dịch vụ đám mây, ổ đĩa bị xóa mỗi lần khởi động lại; xuất CSV để giữ danh mục
    và nhập lại khi cần.
    """
    st.markdown("**Sao lưu danh mục**")
    b1, b2 = st.columns(2)
    with b1:
        data = lots.to_csv(index=False).encode("utf-8-sig") if lots is not None and not lots.empty else b""
        st.download_button(
            "TẢI DANH MỤC (CSV)", data=data, file_name="portfolio.csv", mime="text/csv",
            width="stretch", disabled=not data, key="portfolio_export_csv",
        )
    with b2:
        up = st.file_uploader("Nhập danh mục từ CSV (thay thế danh mục hiện tại)", type=["csv"], key="portfolio_import_csv")
    if up is None:
        return
    token = f"{up.name}:{up.size}"
    if st.session_state.get("portfolio_import_done") == token:
        return
    try:
        if up.size > 200_000:
            raise ValueError("File quá lớn (tối đa 200 KB).")
        df = pd.read_csv(up)
        df.columns = [str(c).strip().lower() for c in df.columns]
        missing = {"ticker", "buy_price", "quantity"} - set(df.columns)
        if missing:
            raise ValueError("Thiếu cột: " + ", ".join(sorted(missing)))
        if len(df) > 500:
            raise ValueError("Tối đa 500 dòng.")
        for c in _BACKUP_COLUMNS:
            if c not in df.columns:
                df[c] = "" if c in {"buy_date", "note"} else 0
        for c in ("buy_price", "quantity", "fee"):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        df = df[(df["buy_price"] >= 0) & (df["quantity"] > 0)]
        if df.empty:
            raise ValueError("Không có dòng hợp lệ (cần mã, giá mua ≥ 0, số lượng > 0).")
        save_portfolio(portfolio_path, df[_BACKUP_COLUMNS])
        st.session_state["portfolio_import_done"] = token
        st.success(f"Đã nhập {len(df)} dòng.")
        st.rerun()
    except Exception as e:
        st.error(f"Không nhập được file: {e}")


def _render_partial_sell(portfolio_path: str, portfolio: pd.DataFrame, prices: dict) -> None:
    """Bán một phần hoặc bán hết (giá vốn bình quân giữ nguyên). Hiện trước số còn lại và lãi/lỗ đã chốt."""
    from datetime import datetime as _dt
    st.markdown("**Bán một phần / bán hết**")
    flash = st.session_state.pop("sell_flash", None)
    if flash:
        st.success(flash)
    held = portfolio["ticker"].astype(str).tolist()
    tk = st.selectbox("Mã đang giữ", held, key="sell_tk_v19")
    raw = portfolio.set_index("ticker").loc[tk]
    have = int(round(float(raw["quantity"])))
    avg = float(raw["avg_cost"])
    cur = prices.get(str(tk).upper())
    c1, c2 = st.columns(2)
    with c1:
        qty = st.number_input(f"Số lượng bán (đang có {have:,})", min_value=0, max_value=have, value=0, step=100, key=f"sell_qty_{tk}")
    with c2:
        price = st.number_input("Giá bán", min_value=0.0, value=float(cur if cur and cur > 0 else avg), step=100.0, format="%.0f", key=f"sell_px_{tk}")
    if qty > 0:
        remain = have - int(qty)
        pnl = (float(price) - avg) * int(qty)
        pnl_txt = f":green[+{pnl:,.0f}]" if pnl >= 0 else f":red[{pnl:,.0f}]"
        lines = [
            f"- Bán **{int(qty):,}** cổ phiếu {tk} giá **{float(price):,.0f}** → thu về **{float(price) * int(qty):,.0f}**",
            f"- **Lãi/lỗ đã chốt:** {pnl_txt} ({(float(price) / avg - 1) * 100:+.2f}% so với giá vốn bình quân {avg:,.0f})" if avg > 0 else f"- Thu về {float(price) * int(qty):,.0f}",
        ]
        if remain > 0:
            lines.append(f"- **Còn lại: {remain:,} cổ phiếu** · giá vốn bình quân giữ nguyên **{avg:,.0f}** · vốn còn lại **{remain * avg:,.0f}**")
            if cur and cur > 0:
                unreal = (float(cur) - avg) * remain
                u_txt = f":green[+{unreal:,.0f}]" if unreal >= 0 else f":red[{unreal:,.0f}]"
                lines.append(f"- Giá trị còn lại theo giá hiện tại {float(cur):,.0f}: **{remain * float(cur):,.0f}** · lãi/lỗ tạm tính {u_txt}")
        else:
            lines.append("- **Bán hết:** mã sẽ được xóa khỏi danh mục.")
        st.markdown("\n".join(lines))
        if st.button("XÁC NHẬN BÁN", type="primary", width="stretch", key=f"sell_ok_{tk}"):
            if float(price) <= 0:
                st.error("Cần nhập giá bán lớn hơn 0.")
            else:
                got = record_sale(portfolio_path, tk, int(qty), float(price), avg, _dt.now().strftime("%Y-%m-%d"))
                if remain > 0:
                    replace_position(portfolio_path, tk, avg, remain)
                else:
                    delete_position(portfolio_path, tk)
                st.session_state["sell_flash"] = f"Đã bán {int(qty):,} {tk}. Lãi/lỗ đã chốt {got:+,.0f}. " + (f"Còn lại {remain:,} cổ phiếu." if remain > 0 else "Đã bán hết, xóa khỏi danh mục.")
                st.rerun()
    sales = load_sales(portfolio_path)
    if not sales.empty:
        total = float(pd.to_numeric(sales["pnl"], errors="coerce").fillna(0).sum())
        st.caption(f"Tổng lãi/lỗ đã chốt: {total:+,.0f} VND ({len(sales)} lần bán). Lịch sử bán mất khi máy chủ web khởi động lại; hãy tải về để giữ.")
        st.download_button("TẢI LỊCH SỬ BÁN (CSV)", data=sales.to_csv(index=False).encode("utf-8-sig"), file_name="realized_sales.csv", mime="text/csv", width="stretch", key="sales_export_csv")


def render_portfolio_manager(portfolio_path: str, prices: dict | None = None) -> None:
    """Thêm lần mua, điều chỉnh/xóa vị thế, sao lưu CSV. Dùng trong expander hoặc popover."""
    lots = load_portfolio(portfolio_path)
    portfolio = aggregate_portfolio(lots, pd.DataFrame())
    with st.form("them_lan_mua_v1612", clear_on_submit=True):
        f1, f2, f3, f4 = st.columns([1.0, 1.0, 1.0, 0.9])
        with f1:
            new_ticker = st.text_input("Mã cổ phiếu", placeholder="Ví dụ: HPG")
        with f2:
            new_price = st.number_input("Giá mua", min_value=0.0, step=100.0, format="%.0f")
        with f3:
            new_qty = st.number_input("Số lượng", min_value=0.0, step=100.0, format="%.0f")
        with f4:
            st.write("")
            st.write("")
            add_clicked = st.form_submit_button("+ THÊM LẦN MUA", type="primary", width='stretch')
        if add_clicked:
            if not new_ticker.strip() or new_price <= 0 or new_qty <= 0:
                st.error("Cần nhập đủ Mã, Giá mua và Số lượng lớn hơn 0.")
            else:
                add_purchase(portfolio_path, new_ticker, new_price, new_qty)
                st.success("Đã thêm vị thế. Bấm CẬP NHẬT GIÁ để lấy giá hiện tại của mã mới.")
                st.rerun()

    if not portfolio.empty:
        held = portfolio["ticker"].astype(str).tolist()
        edit_ticker = st.selectbox("Mã cần điều chỉnh/xóa", held, key="edit_position_ticker_v1612")
        raw = portfolio.set_index("ticker").loc[edit_ticker]
        e1, e2, e3 = st.columns([1, 1, 0.8])
        with e1:
            edit_avg = st.number_input("Giá vốn bình quân mới", min_value=0.0, value=float(raw["avg_cost"]), step=100.0, format="%.0f", key="edit_avg_v1612")
        with e2:
            edit_qty = st.number_input("Số lượng mới", min_value=0.0, value=float(raw["quantity"]), step=100.0, format="%.0f", key="edit_qty_v1612")
        with e3:
            if st.button("LƯU ĐIỀU CHỈNH", width='stretch', key="save_edit_v1612"):
                replace_position(portfolio_path, edit_ticker, edit_avg, edit_qty)
                st.rerun()
            if st.button("XÓA VỊ THẾ", width='stretch', key="delete_edit_v1612"):
                delete_position(portfolio_path, edit_ticker)
                st.rerun()

    if not portfolio.empty:
        st.divider()
        _render_partial_sell(portfolio_path, portfolio, prices or {})

    st.divider()
    _render_portfolio_backup(portfolio_path, lots)


def render_portfolio(
    portfolio_path: str,
    portfolio_market: pd.DataFrame,
    market: pd.DataFrame,
    histories: dict,
    regime: dict,
    is_live: bool,
    load_symbol=None,
    hourly_histories: dict | None = None,
    load_hourly=None,
):
    lots = load_portfolio(portfolio_path)
    portfolio = aggregate_portfolio(lots, portfolio_market)

    with st.expander("+ THÊM / ĐIỀU CHỈNH DANH MỤC", expanded=portfolio.empty):
        render_portfolio_manager(portfolio_path)

    if portfolio.empty:
        st.info("Danh mục đang trống. Mở mục THÊM / ĐIỀU CHỈNH DANH MỤC để thêm vị thế đầu tiên.")
        return

    priced_mask = portfolio["close"].notna() if is_live else pd.Series(False, index=portfolio.index)
    priced_count = int(priced_mask.sum())
    total_count = len(portfolio)
    total_cost = float(portfolio["total_cost"].sum())
    priced_cost = float(portfolio.loc[priced_mask, "total_cost"].sum()) if priced_count else np.nan
    total_value = float(portfolio.loc[priced_mask, "market_value"].sum()) if priced_count else np.nan
    pnl = total_value - priced_cost if priced_count else np.nan
    pnl_pct = pnl / priced_cost * 100 if priced_count and priced_cost else np.nan

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("TỔNG GIÁ VỐN", f"{total_cost:,.0f} VND", help=HELP["total_cost"])
    m2.metric("GIÁ TRỊ HIỆN TẠI", f"{total_value:,.0f} VND" if priced_count else "—", help=HELP["market_value"])
    m3.metric("LÃI/LỖ TẠM TÍNH", f"{pnl:,.0f} VND" if priced_count else "—", f"{pnl_pct:+.2f}%" if priced_count else None, help=HELP["pnl"])
    m4.metric("SỐ MÃ", str(total_count), f"{priced_count}/{total_count} đã có giá")

    _render_portfolio_overview(portfolio, pnl_pct, priced_count, total_count)

    left, right = st.columns([0.94, 1.34], gap="medium")
    with left:
        selected = _portfolio_grid(portfolio, "bang_danh_muc_v1612")
        last_selected = str(st.session_state.get("portfolio_last_grid_selected_v1612", "") or "").upper().strip()
        if selected and selected != last_selected:
            st.session_state["portfolio_last_grid_selected_v1612"] = selected
            st.session_state["portfolio_ticker"] = selected
            st.rerun()
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        _render_portfolio_treemaps(portfolio, is_live)

    with right:
        held = portfolio["ticker"].astype(str).tolist()
        chosen = str(st.session_state.get("portfolio_ticker", held[0])).upper().strip()
        if chosen not in held:
            chosen = held[0]
            st.session_state["portfolio_ticker"] = chosen

        if chosen not in set(market.get("ticker", pd.Series(dtype=str)).astype(str)) or chosen not in histories:
            if load_symbol is not None:
                try:
                    with st.spinner(f"Đang tải lịch sử thật và phân tích vị thế {chosen}..."):
                        load_symbol(chosen)
                    st.rerun()
                except Exception as e:
                    st.error(f"Không tải được lịch sử {chosen}: {e}")
                    return
            else:
                st.info(f"{chosen}: đã có giá hiện tại nhưng chưa có lịch sử kỹ thuật. Bấm CẬP NHẬT TOÀN BỘ để mở biểu đồ.")
                return

        full_hist_key = 'full_history_loaded_v1619k'
        loaded = set(st.session_state.get(full_hist_key, []) or [])
        if load_symbol is not None and chosen not in loaded:
            try:
                with st.spinner(f"Đang tải đầy đủ lịch sử niêm yết của {chosen}..."):
                    load_symbol(chosen)
                loaded.add(chosen)
                st.session_state[full_hist_key] = sorted(loaded)
                st.rerun()
            except Exception:
                pass

        prow = portfolio.set_index("ticker").loc[chosen]
        hourly_df = None
        if isinstance(hourly_histories, dict):
            hourly_df = hourly_histories.get(chosen)
        if (hourly_df is None or not isinstance(hourly_df, pd.DataFrame) or hourly_df.empty) and load_hourly is not None:
            try:
                hourly_df = load_hourly(chosen)
            except Exception:
                hourly_df = pd.DataFrame()

        stock_detail_panel(
            chosen, market, histories, regime, "danh_muc_v1612",
            demo_warning=not is_live, portfolio_row=prow, compact=True, hourly_df=hourly_df,
        )
