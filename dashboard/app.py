"""StockSense - NovaMart store manager dashboard (Streamlit).

Run from the project root:
    streamlit run dashboard/app.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import HIGH_RISK, MEDIUM_RISK  # noqa: E402

PROC, REP, MOD = ROOT / "data" / "processed", ROOT / "reports", ROOT / "models"

PAPER, INK, MUTED, RULE, TAG = "#F4F6F3", "#1B2A2F", "#66777C", "#D5DBD8", "#FFD23F"
TEAL, SAGE = "#1F5F6B", "#8DB3A8"
RISK = {"HIGH": "#C8332F", "MEDIUM": "#D9822B", "LOW": "#2E7D5B"}
SERIES = [TEAL, "#C99A1E", SAGE, "#4B6F8A", "#A45A52", "#6E8B3D", "#8A6FA0"]

st.set_page_config(page_title="StockSense | NovaMart", page_icon="🏷️", layout="wide")

# --------------------------------------------------------------------------- #
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,400..800&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, p, li, label, input, button, select, textarea {{
  font-family: 'Archivo', 'Helvetica Neue', Arial, sans-serif !important;
}}
.block-container {{ padding-top: 1.6rem; max-width: 1380px; }}
[data-testid="stSidebar"] {{ background: {INK}; }}
[data-testid="stSidebar"] * {{ color: #E6ECEA !important; }}
[data-testid="stSidebar"] [data-baseweb="select"] > div {{ background: #26393F; border-color: #3A4F55; }}
[data-testid="stSidebar"] [data-baseweb="tag"] {{ background: {TAG} !important; }}
[data-testid="stSidebar"] [data-baseweb="tag"] * {{ color: {INK} !important; }}
.brand {{ font-stretch: 125%; font-weight: 800; font-size: 1.55rem; letter-spacing: -.01em; line-height: 1; }}
.brand small {{ display:block; font-stretch:100%; font-weight:400; font-size:.8rem; opacity:.7; margin-top:.35rem; }}
h1, h2, h3 {{ font-family:'Archivo',sans-serif !important; font-stretch:112%; letter-spacing:-.015em; color:{INK}; }}
.lede {{ color:{MUTED}; max-width: 70ch; font-size: 1rem; margin-top:-.4rem; }}

/* hero: the day's order count */
.hero {{ display:grid; grid-template-columns: minmax(260px, 1fr) 2fr; gap: 2.2rem; align-items:end;
         border-bottom: 2px solid {INK}; padding-bottom: 1.2rem; margin-bottom: 1.4rem; }}
.hero .count {{ font-stretch:125%; font-weight:800; font-size: clamp(4rem, 9vw, 7.5rem); line-height:.85; color:{INK}; }}
.hero .count-label {{ font-size:1.05rem; margin-top:.5rem; }}
.hero .count-label b {{ background:{TAG}; padding:0 .3rem; }}
.kpis {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(150px,1fr)); gap: 1.2rem 1.6rem; }}
.kpi .v {{ font-stretch:115%; font-weight:700; font-size:1.7rem; line-height:1.05; font-variant-numeric: tabular-nums; }}
.kpi .l {{ color:{MUTED}; font-size:.85rem; }}
.kpi .d {{ font-size:.82rem; font-weight:600; }}
.up {{ color:{RISK['LOW']}; }} .down {{ color:{RISK['HIGH']}; }}

/* shelf-edge tag */
.tags {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(330px, 1fr)); gap: 1rem; }}
.tag {{ position:relative; background:{TAG}; color:{INK}; padding: .95rem 1.1rem 1rem 1.35rem;
        border-radius: 3px 3px 3px 3px; box-shadow: 0 1px 0 rgba(27,42,47,.25); }}
.tag::before {{ content:""; position:absolute; left:0; top:0; bottom:0; width:7px; background: var(--risk); }}
.tag::after {{ content:""; position:absolute; right:14px; top:14px; width:10px; height:10px; border-radius:50%;
               background:{PAPER}; box-shadow: inset 0 1px 2px rgba(0,0,0,.25); }}
.tag .who {{ font-size:.82rem; }}
.tag .name {{ font-stretch:112%; font-weight:700; font-size:1.15rem; margin:.1rem 0 .55rem; padding-right:1.4rem; }}
.tag .row {{ display:flex; align-items:flex-end; justify-content:space-between; gap:1rem; }}
.tag .qty {{ font-stretch:125%; font-weight:800; font-size:2.6rem; line-height:.9; font-variant-numeric: tabular-nums; }}
.tag .qty small {{ display:block; font-stretch:100%; font-size:.75rem; font-weight:500; margin-top:.2rem; }}
.tag .facts {{ text-align:right; font-size:.8rem; line-height:1.45; font-variant-numeric: tabular-nums; }}
.tag .facts b {{ font-weight:700; }}
.tag .why {{ border-top: 1px dashed rgba(27,42,47,.45); margin-top:.7rem; padding-top:.55rem; font-size:.84rem; }}
.tag .why span {{ display:inline-block; margin: 0 .35rem .25rem 0; padding: .05rem .4rem; background: rgba(255,255,255,.55); border-radius:2px; }}
.tag .act {{ font-weight:700; margin-top:.35rem; font-size:.9rem; }}
.pill {{ display:inline-block; padding:.05rem .45rem; border-radius: 2px; color:white; font-size:.72rem; font-weight:700; }}

.note {{ border-left: 4px solid {TEAL}; padding: .35rem .9rem; color:{INK}; background: #E8ECE9; margin:.6rem 0 1rem; max-width: 90ch; }}
div[data-testid="stDataFrame"] {{ border: 1px solid {RULE}; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
@st.cache_data
def load():
    rec = pd.read_csv(PROC / "recommendations.csv")
    eda = json.loads((REP / "eda_summary.json").read_text())
    meta = json.loads((MOD / "model_metadata.json").read_text())
    stats = json.loads((REP / "statistical_tests.json").read_text())
    testp = pd.read_csv(PROC / "test_predictions.csv", parse_dates=["date"])
    dq = pd.read_csv(REP / "data_quality_log.csv")
    return rec, eda, meta, stats, testp, dq


@st.cache_resource
def models():
    from src.recommend import load_models
    return load_models()


@st.cache_data
def features():
    return pd.read_csv(PROC / "features.csv", parse_dates=["date"])


def fig_base(fig, h=360):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor=PAPER,
                      plot_bgcolor=PAPER, font=dict(family="Archivo, sans-serif", color=INK, size=13),
                      legend=dict(orientation="h", y=1.1, x=0, bgcolor="rgba(0,0,0,0)"),
                      title_font=dict(size=15), hoverlabel=dict(font_family="Archivo"))
    fig.update_xaxes(gridcolor=RULE, zeroline=False, linecolor=RULE)
    fig.update_yaxes(gridcolor=RULE, zeroline=False, linecolor=RULE)
    return fig


def rupees(v):
    v = float(v)
    if abs(v) >= 1e7:
        return f"Rs {v/1e7:.2f} Cr"
    if abs(v) >= 1e5:
        return f"Rs {v/1e5:.1f} L"
    if abs(v) >= 1e3:
        return f"Rs {v/1e3:.1f} K"
    return f"Rs {v:,.0f}"


def tag_html(r):
    reasons = json.loads(r.reasons) if isinstance(r.reasons, str) else []
    why = "".join(f"<span><b>{n}</b></span>" if v == "rule" else f"<span>{n} · {v}</span>"
                  for n, v in reasons) or "<span>Stock covers forecast</span>"
    new = " · new product, look-alike forecast" if r.sparse_history_flag else ""
    return f"""
<div class="tag" style="--risk:{RISK[r.risk_level]}">
  <div class="who">{r.store_id} {r.city} · {r.category}{new}</div>
  <div class="name">{r.product_name}</div>
  <div class="row">
    <div class="qty">{int(r.reorder_qty):,}<small>units to order</small></div>
    <div class="facts">
      <span class="pill" style="background:{RISK[r.risk_level]}">{r.risk_level} {r.stockout_probability:.0%}</span><br>
      <span style="opacity:.75">model {r.model_probability:.0%}{" + rule" if r.risk_basis != "Model" else ""}</span><br>
      On shelf <b>{int(r.current_stock):,}</b> · Next 7 days <b>{r.forecast_7d:,.0f}</b><br>
      Cover <b>{r.days_of_cover:.1f} days</b> · Lead time <b>{int(r.lead_time_days)} d</b>
    </div>
  </div>
  <div class="why">{why}</div>
  <div class="act">{r.manager_action}</div>
</div>"""


rec, eda, meta, stats, testp, dq = load()
kpi = eda["kpi"]

# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown('<div class="brand">StockSense<small>NovaMart replenishment desk</small></div>',
                unsafe_allow_html=True)
    st.write("")
    page = st.radio("Go to", ["Today's orders", "Executive summary", "Demand", "Inventory risk",
                              "Model performance", "Why the model says so", "What-if planner",
                              "Data quality"], label_visibility="collapsed")
    st.write("")
    stores = st.multiselect("Stores", sorted(rec.store_id.unique()), default=sorted(rec.store_id.unique()))
    cats = st.multiselect("Categories", sorted(rec.category.unique()), default=sorted(rec.category.unique()))
    st.caption(f"Data to {kpi['date_max']}. Forecast window: next 7 days.")

view = rec[rec.store_id.isin(stores) & rec.category.isin(cats)]

# --------------------------------------------------------------------------- #
if page == "Today's orders":
    high = view[view.risk_level == "HIGH"]
    med = view[view.risk_level == "MEDIUM"]
    to_order = view[view.reorder_qty > 0]
    st.markdown(f"""
<div class="hero">
  <div><div class="count">{len(high)}</div>
       <div class="count-label">items need an order <b>today</b></div></div>
  <div class="kpis">
    <div class="kpi"><div class="v">{len(med)}</div><div class="l">to order within 2 days</div></div>
    <div class="kpi"><div class="v">{int(to_order.reorder_qty.sum()):,}</div><div class="l">units recommended, {len(to_order)} lines</div></div>
    <div class="kpi"><div class="v">{rupees(view.revenue_at_risk.sum())}</div><div class="l">sales at risk this week</div></div>
    <div class="kpi"><div class="v">{rupees(view.excess_value.sum())}</div><div class="l">cash tied up in overstock</div></div>
  </div>
</div>""", unsafe_allow_html=True)

    c1, c2 = st.columns([3, 1])
    with c2:
        show = st.selectbox("Show", ["High risk", "High and medium", "All with an order"])
    pick = {"High risk": high, "High and medium": pd.concat([high, med]),
            "All with an order": to_order}[show]
    with c1:
        st.markdown(f"### Order cards ({len(pick)})")
        st.markdown('<p class="lede">Sorted by sales at risk. Each card shows what to order, '
                    'why the model flagged it and what to do next.</p>', unsafe_allow_html=True)
    if pick.empty:
        st.info("Nothing to order for this selection. Widen the store or category filter.")
    else:
        st.markdown('<div class="tags">' + "".join(tag_html(r) for r in pick.head(24).itertuples())
                    + "</div>", unsafe_allow_html=True)
        if len(pick) > 24:
            st.caption(f"Showing the top 24 of {len(pick)}. The full list is in the table below.")
    st.markdown("### Full action list")
    tbl = pick[["store_id", "product_name", "current_stock", "forecast_7d", "stockout_probability",
                "risk_level", "reorder_qty", "manager_action"]].rename(columns={
        "store_id": "Store", "product_name": "Product", "current_stock": "Current stock",
        "forecast_7d": "7-day forecast", "stockout_probability": "Stock-out prob.",
        "risk_level": "Risk", "reorder_qty": "Recommended order", "manager_action": "Action"})
    st.dataframe(tbl, hide_index=True, width="stretch",
                 column_config={"Stock-out prob.": st.column_config.ProgressColumn(
                     format="%.2f", min_value=0, max_value=1)})
    st.download_button("Download order list (CSV)", pick.to_csv(index=False),
                       "stocksense_orders.csv", "text/csv")

    overs = view[view.excess_units > 0].sort_values("excess_value", ascending=False).head(8)
    if not overs.empty:
        st.markdown("### Overstocked - hold orders")
        st.dataframe(overs[["store_id", "product_name", "current_stock", "forecast_7d",
                            "days_of_cover", "excess_value", "manager_action"]].rename(columns={
            "store_id": "Store", "product_name": "Product", "current_stock": "Current stock",
            "forecast_7d": "7-day forecast", "days_of_cover": "Days of cover",
            "excess_value": "Excess value (Rs)", "manager_action": "Action"}),
            hide_index=True, width="stretch")

# --------------------------------------------------------------------------- #
elif page == "Executive summary":
    st.markdown("## Executive summary")
    g = kpi["revenue_growth_28d"]
    st.markdown(f"""
<div class="kpis" style="margin:1rem 0 1.6rem">
 <div class="kpi"><div class="v">{rupees(kpi['revenue_total'])}</div><div class="l">revenue, {kpi['date_min']} to {kpi['date_max']}</div></div>
 <div class="kpi"><div class="v">{rupees(kpi['revenue_last_28d'])}</div><div class="l">last 28 days</div><div class="d {'up' if g>=0 else 'down'}">{g:+.1%} vs prior 28</div></div>
 <div class="kpi"><div class="v">{kpi['stockout_rate']:.1%}</div><div class="l">stock-out rate (store-product days)</div></div>
 <div class="kpi"><div class="v">{rupees(kpi['inventory_value_latest'])}</div><div class="l">inventory value at cost</div></div>
 <div class="kpi"><div class="v">{rupees(kpi['estimated_lost_sales'])}</div><div class="l">estimated lost sales</div></div>
 <div class="kpi"><div class="v">{(rec.risk_level=='HIGH').sum()}</div><div class="l">products at high risk</div></div>
</div>""", unsafe_allow_html=True)
    d = eda["daily"]
    fig = go.Figure()
    fig.add_bar(x=d["dates"], y=d["revenue"], marker_color=SAGE, name="Daily revenue")
    fig.add_scatter(x=d["dates"], y=pd.Series(d["revenue"]).rolling(7).mean(), line=dict(color=INK, width=2.5),
                    name="7-day average")
    st.plotly_chart(fig_base(fig.update_layout(title="Revenue per day, all stores"), 340), width="stretch")
    c1, c2 = st.columns(2)
    gr = pd.Series(eda["store_growth_first4_vs_last4_weeks"]).sort_values()
    so = pd.Series(eda["stockout_rate_by_store"]).reindex(gr.index)
    fig = go.Figure(go.Bar(x=gr.values * 100, y=gr.index, orientation="h",
                           marker_color=[RISK["HIGH"] if v < 0 else TEAL for v in gr.values],
                           text=[f"{v:+.0%}" for v in gr.values], textposition="outside"))
    c1.plotly_chart(fig_base(fig.update_layout(title="Store growth: first 4 vs last 4 weeks (%)"), 320),
                    width="stretch")
    fig = go.Figure(go.Bar(x=so.values * 100, y=so.index, orientation="h", marker_color=RISK["MEDIUM"],
                           text=[f"{v:.1%}" for v in so.values], textposition="outside"))
    c2.plotly_chart(fig_base(fig.update_layout(title="Stock-out rate by store (%)"), 320), width="stretch")
    top = gr.idxmax()
    st.markdown(f'<div class="note">{top} grew fastest ({gr.max():+.0%}) and also has one of the highest '
                f'stock-out rates ({so[top]:.1%}): its reorder levels were set for a smaller store. '
                f'Promotions lift units by {eda["promotion_lift_overall"]:.0%} and weekends by '
                f'{eda["weekend_uplift"]:.0%}, and both now feed the forecast.</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
elif page == "Demand":
    st.markdown("## Demand intelligence")
    st.markdown('<p class="lede">How well the forecast tracked real sales on the held-out test weeks, '
                'and where demand is heading.</p>', unsafe_allow_html=True)
    t = testp[testp.store_id.isin(stores) & testp.category.isin(cats)]
    agg = t.groupby("date")[["next_7_day_demand", "forecast", "baseline_forecast"]].sum().reset_index()
    fig = go.Figure()
    fig.add_scatter(x=agg.date, y=agg.next_7_day_demand, name="Actual", line=dict(color=INK, width=3))
    fig.add_scatter(x=agg.date, y=agg.forecast, name="StockSense forecast", line=dict(color=TEAL, width=2.5))
    fig.add_scatter(x=agg.date, y=agg.baseline_forecast, name="Old rule (last 7-day mean)",
                    line=dict(color=MUTED, width=1.5, dash="dash"))
    st.plotly_chart(fig_base(fig.update_layout(title="Next-7-day units: actual vs forecast (test window)"), 380),
                    width="stretch")
    c1, c2 = st.columns(2)
    wc = eda["weekly_category_units"]
    fig = go.Figure()
    for i, (c, v) in enumerate(wc.items()):
        if c in cats:
            fig.add_scatter(x=v["dates"], y=v["units"], name=c, line=dict(color=SERIES[i % 7], width=2))
    c1.plotly_chart(fig_base(fig.update_layout(title="Weekly units by category"), 380), width="stretch")
    err = t.assign(ae=(t.next_7_day_demand - t.forecast).abs()).groupby("category").apply(
        lambda d: d.ae.sum() / d.next_7_day_demand.sum() * 100, include_groups=False).sort_values()
    fig = go.Figure(go.Bar(x=err.values, y=err.index, orientation="h", marker_color=TEAL,
                           text=[f"{v:.1f}%" for v in err.values], textposition="outside"))
    c2.plotly_chart(fig_base(fig.update_layout(title="Forecast error by category (WAPE %)"), 380),
                    width="stretch")
    ws = eda["weekly_store_revenue"]
    fig = go.Figure()
    for i, (s, v) in enumerate(ws.items()):
        if s in stores:
            fig.add_scatter(x=v["dates"], y=v["revenue"], name=s, line=dict(color=SERIES[i % 7], width=2))
    st.plotly_chart(fig_base(fig.update_layout(title="Weekly revenue by store"), 340), width="stretch")

# --------------------------------------------------------------------------- #
elif page == "Inventory risk":
    st.markdown("## Inventory risk")
    c = view.risk_level.value_counts()
    st.markdown(f"""<div class="kpis" style="margin:.6rem 0 1.2rem">
<div class="kpi"><div class="v" style="color:{RISK['HIGH']}">{c.get('HIGH',0)}</div><div class="l">high risk (p ≥ {HIGH_RISK})</div></div>
<div class="kpi"><div class="v" style="color:{RISK['MEDIUM']}">{c.get('MEDIUM',0)}</div><div class="l">medium ({MEDIUM_RISK}-{HIGH_RISK})</div></div>
<div class="kpi"><div class="v" style="color:{RISK['LOW']}">{c.get('LOW',0)}</div><div class="l">low (p &lt; {MEDIUM_RISK})</div></div></div>""",
                unsafe_allow_html=True)
    hm = view.pivot_table(index="store_id", columns="category", values="stockout_probability", aggfunc="mean")
    fig = go.Figure(go.Heatmap(z=hm.values, x=hm.columns, y=hm.index, zmin=0, zmax=1,
                               colorscale=[[0, "#E7F0EA"], [0.4, "#F3D9A4"], [0.7, "#E39A5B"], [1, "#B8322D"]],
                               text=np.round(hm.values * 100).astype(int), texttemplate="%{text}%",
                               colorbar=dict(title="Avg prob.")))
    st.plotly_chart(fig_base(fig.update_layout(title="Average stock-out probability, next 7 days"), 380),
                    width="stretch")
    fig = go.Figure()
    for lvl in ["LOW", "MEDIUM", "HIGH"]:
        d = view[view.risk_level == lvl]
        fig.add_scatter(x=d.days_of_cover, y=d.stockout_probability, mode="markers", name=lvl,
                        marker=dict(color=RISK[lvl], size=8, opacity=.75),
                        text=d.store_id + " " + d.product_name,
                        hovertemplate="%{text}<br>cover %{x:.1f} d<br>prob %{y:.2f}<extra></extra>")
    fig.add_hline(y=HIGH_RISK, line_dash="dot", line_color=RISK["HIGH"])
    fig.add_hline(y=MEDIUM_RISK, line_dash="dot", line_color=RISK["MEDIUM"])
    fig.update_xaxes(title="Days of stock cover (current stock / forecast daily demand)", range=[0, 30])
    fig.update_yaxes(title="Stock-out probability")
    st.plotly_chart(fig_base(fig.update_layout(title="Less cover, more risk"), 420), width="stretch")
    st.dataframe(view[["store_id", "product_name", "category", "current_stock", "forecast_7d",
                       "days_of_cover", "stockout_probability", "risk_level"]]
                 .sort_values("stockout_probability", ascending=False),
                 hide_index=True, width="stretch")

# --------------------------------------------------------------------------- #
elif page == "Model performance":
    st.markdown("## Model performance")
    r, c = meta["regression"], meta["classification"]
    rt = {x["model"]: x for x in r["test"]}
    sel, base = rt[r["selected"]], rt["Baseline (7-day mean x 7)"]
    ct = c["test"][0]
    st.markdown(f"""<div class="kpis" style="margin:.6rem 0 1.2rem">
<div class="kpi"><div class="v">{sel['WAPE']:.1f}%</div><div class="l">forecast WAPE ({r['selected']})</div><div class="d up">vs {base['WAPE']:.1f}% old rule</div></div>
<div class="kpi"><div class="v">{sel['MAE']:.1f}</div><div class="l">mean abs. error, units / 7 days</div></div>
<div class="kpi"><div class="v">{sel['R2']:.3f}</div><div class="l">R² on test weeks</div></div>
<div class="kpi"><div class="v">{ct['ROC_AUC']:.3f}</div><div class="l">stock-out ROC-AUC ({c['selected']})</div></div>
<div class="kpi"><div class="v">{ct['Recall']:.0%}</div><div class="l">stock-outs caught (recall)</div></div>
<div class="kpi"><div class="v">{ct['Precision']:.0%}</div><div class="l">alerts that were real (precision)</div></div></div>""",
                unsafe_allow_html=True)
    st.markdown("#### Demand models, validation window")
    rv = pd.DataFrame(r["validation"]).drop(columns=["split"]).round(3)
    st.dataframe(rv, hide_index=True, width="stretch")
    st.markdown("#### Stock-out models, validation window")
    cv = pd.DataFrame(c["validation"]).drop(columns=["split"]).round(3)
    st.dataframe(cv, hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    cm = np.array(c["confusion_matrix"])
    fig = go.Figure(go.Heatmap(z=cm, x=["Predicted: no", "Predicted: stock-out"],
                               y=["Actual: no", "Actual: stock-out"], colorscale=[[0, "#E8ECE9"], [1, TEAL]],
                               text=cm, texttemplate="%{text:,}", showscale=False, textfont=dict(size=18)))
    c1.plotly_chart(fig_base(fig.update_layout(title=f"Confusion matrix, test (threshold {c['threshold']:.2f})"), 360),
                    width="stretch")
    t = testp[testp.store_id.isin(stores)]
    res = t.next_7_day_demand - t.forecast
    fig = go.Figure(go.Histogram(x=res, nbinsx=60, marker_color=SAGE))
    fig.add_vline(x=0, line_color=INK)
    c2.plotly_chart(fig_base(fig.update_layout(title=f"Residuals (actual - forecast), mean {res.mean():+.1f}"), 360),
                    width="stretch")
    st.markdown(f'<div class="note">Why trust it: splits are in time order and purged by 7 days, '
                f'model choice used the validation weeks only and the test weeks were scored once. '
                f'Probabilities are calibrated, so a 70% risk means about 7 in 10 such items really ran out.</div>',
                unsafe_allow_html=True)
    st.image(str(REP / "figures" / "11_calibration.png"), width=460)

# --------------------------------------------------------------------------- #
elif page == "Why the model says so":
    st.markdown("## Why the model says so")
    imp = meta["importance"]
    c1, c2 = st.columns(2)
    for col, key, title in ((c1, "demand", "What drives the demand forecast"),
                            (c2, "stockout", "What drives stock-out risk")):
        s = pd.Series(imp[key]).head(12)[::-1]
        fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=TEAL))
        col.plotly_chart(fig_base(fig.update_layout(title=title), 420), width="stretch")
    st.caption("Permutation importance on the test weeks: how much accuracy drops when a feature is shuffled.")
    st.markdown("### Explain one item")
    opts = view.store_id + " · " + view.product_name
    choice = st.selectbox("Store and product", opts.tolist())
    r = view[opts == choice].iloc[0]
    rd = pd.Series(json.loads(r.risk_drivers)).sort_values()
    dd = pd.Series(json.loads(r.demand_drivers)).sort_values()
    c1, c2 = st.columns([1, 2])
    c1.markdown('<div class="tags">' + tag_html(r) + "</div>", unsafe_allow_html=True)
    fig = go.Figure(go.Bar(x=rd.values, y=rd.index, orientation="h",
                           marker_color=[RISK["HIGH"] if v > 0 else RISK["LOW"] for v in rd.values],
                           text=[f"{v:+.0f} pts" for v in rd.values], textposition="outside"))
    c2.plotly_chart(fig_base(fig.update_layout(title="Effect on stock-out probability (points)"), 380),
                    width="stretch")
    fig = go.Figure(go.Bar(x=dd.values, y=dd.index, orientation="h", marker_color=TEAL,
                           text=[f"{v:+.0f}%" for v in dd.values], textposition="outside"))
    st.plotly_chart(fig_base(fig.update_layout(title="Effect on 7-day demand (%)"), 360), width="stretch")
    st.markdown('<div class="note">Each bar answers: if this driver were at a normal level (no promotion, '
                'no festival, typical stock cover...), how different would the prediction be?</div>',
                unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
elif page == "What-if planner":
    st.markdown("## What-if planner")
    st.markdown('<p class="lede">Re-run both models live under a different plan and see how risk and '
                'orders change.</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    disc = c1.slider("Run a promotion at discount (%)", 0, 30, 0, 5)
    delay = c2.slider("Supplier delay (extra days)", 0, 5, 0)
    fest = c3.toggle("Add a local festival this week")
    shock = c4.slider("Extra demand shock (%)", -30, 50, 0, 5)
    from src.recommend import score_latest
    reg, clf = models()
    scen = score_latest(features(), reg, clf, explain_rows=False,
                        scenario=dict(discount_pct=disc, supplier_delay_days=delay, festival=int(fest),
                                      demand_shock_pct=shock))
    scen = scen[scen.store_id.isin(stores) & scen.category.isin(cats)]
    b = view.set_index(["store_id", "product_id"])
    s = scen.set_index(["store_id", "product_id"])
    delta_high = (s.risk_level == "HIGH").sum() - (b.risk_level == "HIGH").sum()
    delta_units = s.reorder_qty.sum() - b.reorder_qty.sum()
    st.markdown(f"""<div class="kpis" style="margin:1rem 0 1.2rem">
<div class="kpi"><div class="v">{(s.risk_level=='HIGH').sum()}</div><div class="l">high-risk items</div><div class="d {'down' if delta_high>0 else 'up'}">{delta_high:+d} vs today's plan</div></div>
<div class="kpi"><div class="v">{int(s.reorder_qty.sum()):,}</div><div class="l">units to order</div><div class="d">{int(delta_units):+,} units</div></div>
<div class="kpi"><div class="v">{s.forecast_7d.sum():,.0f}</div><div class="l">7-day demand</div><div class="d">{(s.forecast_7d.sum()/b.forecast_7d.sum()-1):+.1%}</div></div>
<div class="kpi"><div class="v">{rupees(s.revenue_at_risk.sum())}</div><div class="l">sales at risk</div></div></div>""",
                unsafe_allow_html=True)
    j = b[["product_name", "stockout_probability", "risk_level", "reorder_qty"]].join(
        s[["stockout_probability", "risk_level", "reorder_qty"]], rsuffix="_new").reset_index()
    j["change"] = j.stockout_probability_new - j.stockout_probability
    j = j.sort_values("change", ascending=False)
    st.dataframe(j.rename(columns={"store_id": "Store", "product_name": "Product",
                                   "stockout_probability": "Prob. now", "stockout_probability_new": "Prob. in scenario",
                                   "risk_level": "Risk now", "risk_level_new": "Risk in scenario",
                                   "reorder_qty": "Order now", "reorder_qty_new": "Order in scenario",
                                   "change": "Change"}).drop(columns=["product_id"]).round(3),
                 hide_index=True, width="stretch")

# --------------------------------------------------------------------------- #
elif page == "Data quality":
    st.markdown("## Data quality")
    st.markdown('<p class="lede">Every issue found in the five raw files and what was done about it.</p>',
                unsafe_allow_html=True)
    st.dataframe(dq, hide_index=True, width="stretch")
    st.markdown("### Statistical checks")
    for t in stats:
        st.markdown(f"**{t['question']}**  \n{t['test']} · p = {t['p_value']:.2g} · {t['effect']}  \n"
                    f"{t['interpretation']}")
