"""Build dashboard/stocksense_dashboard.html - a single self-contained file that
opens in any browser (no Python needed). All numbers come from the pipeline.

Run:  python -m src.build_html_dashboard
"""
import json

import numpy as np
import pandas as pd

from src.config import DASHBOARD, MODELS, PROCESSED, REPORTS

FRIENDLY = {
    "rolling_mean_7": "Avg daily sales, last 7 days", "rolling_mean_14": "Avg daily sales, last 14 days",
    "rolling_mean_28": "Avg daily sales, last 28 days", "lag_1": "Sales yesterday",
    "lag_7": "Sales 7 days ago", "lag_14": "Sales 14 days ago", "rolling_std_7": "Sales swings, 7 days",
    "rolling_std_28": "Sales swings, 28 days", "cv_28": "Demand volatility",
    "sales_growth_7v28": "Recent sales growth", "current_stock": "Stock on shelf",
    "days_of_inventory": "Days of stock cover", "inventory_to_demand_ratio": "Stock vs weekly demand",
    "reorder_gap": "Stock above reorder level", "below_reorder_flag": "Below reorder level",
    "received_last_7": "Delivered in last 7 days", "stockout_days_last_14": "Stock-outs, last 14 days",
    "stockout_days_last_28": "Stock-outs, last 28 days", "lead_time_days": "Supplier lead time",
    "days_since_last_receipt": "Days since last delivery", "last_receipt_cover": "Size of last delivery",
    "cover_minus_lead": "Cover minus lead time", "projected_stock_at_lead": "Stock left when order lands",
    "projected_stock_7d": "Stock left in 7 days", "promotion_flag": "Promotion running",
    "discount_pct": "Discount %", "promo_days_last_7": "Promo days, last 7", "price_change": "Price change",
    "festival_next_7": "Festivals next 7 days", "holiday_next_7": "Holidays next 7 days",
    "temp_c": "Temperature", "temp_mean_7": "Avg temperature, 7 days", "rain_mm": "Rainfall",
    "avg_daily_customers": "Store footfall", "floor_area_sqft": "Store size", "mrp": "MRP",
    "shelf_life_days": "Shelf life", "margin_pct": "Margin %", "history_days": "Days of history",
    "category": "Category", "store_type": "Store format", "region": "Region",
    "day_of_week": "Day of week", "week_no": "Week number", "month": "Month",
}


def build_data():
    rec = pd.read_csv(PROCESSED / "recommendations.csv")
    wi = pd.read_csv(PROCESSED / "whatif_grid.csv")
    eda = json.loads((REPORTS / "eda_summary.json").read_text())
    meta = json.loads((MODELS / "model_metadata.json").read_text())
    stats = json.loads((REPORTS / "statistical_tests.json").read_text())
    dq = pd.read_csv(REPORTS / "data_quality_log.csv")
    tp = pd.read_csv(PROCESSED / "test_predictions.csv", parse_dates=["date"])

    items = []
    for r in rec.itertuples():
        items.append(dict(
            s=r.store_id, city=r.city, pid=r.product_id, p=r.product_name, cat=r.category,
            sup=r.supplier_id, stock=int(r.current_stock), fc=round(float(r.forecast_7d), 1),
            prob=round(float(r.stockout_probability), 3), mprob=round(float(r.model_probability), 3),
            risk=r.risk_level, basis=r.risk_basis, ss=int(r.safety_stock), rs=int(r.recommended_stock),
            q=int(r.reorder_qty), cover=float(r.days_of_cover), lead=int(r.lead_time_days),
            shelf=int(r.shelf_life_days), rar=float(r.revenue_at_risk), exu=float(r.excess_units),
            exv=float(r.excess_value), sparse=int(r.sparse_history_flag), act=r.manager_action,
            why=json.loads(r.reasons), rd=json.loads(r.risk_drivers), dd=json.loads(r.demand_drivers)))

    key = rec.store_id + "|" + rec.product_id
    whatif = {}
    for (d, dl, f), g in wi.groupby(["discount_pct_s", "supplier_delay_s", "festival_s"]):
        g = g.set_index(g.store_id + "|" + g.product_id).reindex(key)
        whatif[f"{d}_{dl}_{f}"] = dict(fc=g.forecast_7d.round(1).tolist(),
                                       prob=g.stockout_probability.round(3).tolist(),
                                       risk=g.risk_level.tolist(), q=g.reorder_qty.astype(int).tolist())

    agg = tp.groupby("date")[["next_7_day_demand", "forecast", "baseline_forecast"]].sum()
    wape = tp.assign(ae=(tp.next_7_day_demand - tp.forecast).abs()).groupby("category").apply(
        lambda d: d.ae.sum() / d.next_7_day_demand.sum() * 100, include_groups=False).sort_values()
    res = tp.next_7_day_demand - tp.forecast
    hist, edges = np.histogram(res.clip(-150, 150), bins=40)
    cal_bins = pd.cut(tp.stockout_probability, np.linspace(0, 1, 11), include_lowest=True)
    cal = tp.groupby(cal_bins, observed=True).agg(p=("stockout_probability", "mean"),
                                                  y=("stockout_next_7d", "mean"),
                                                  n=("stockout_next_7d", "size"))
    cal = cal[cal.n >= 20]

    def imp(d):
        return [[FRIENDLY.get(k, k.replace("_", " ")), round(float(v), 4)] for k, v in
                list(d.items())[:12]]

    return dict(
        asof=rec.as_of_date.iloc[0], items=items, whatif=whatif, kpi=eda["kpi"],
        daily=eda["daily"], growth=eda["store_growth_first4_vs_last4_weeks"],
        so_store=eda["stockout_rate_by_store"], heat=eda["stockout_heatmap"],
        wk_store=eda["weekly_store_revenue"], wk_cat=eda["weekly_category_units"],
        promo_lift=eda["promotion_lift_overall"], weekend=eda["weekend_uplift"],
        cat_rev=eda["category_revenue"], pareto=eda["pareto_products_for_80pct"],
        test=dict(dates=agg.index.strftime("%d %b").tolist(),
                  actual=agg.next_7_day_demand.round(0).tolist(),
                  fc=agg.forecast.round(0).tolist(), base=agg.baseline_forecast.round(0).tolist()),
        wape_cat=[[k, round(float(v), 2)] for k, v in wape.items()],
        resid=dict(counts=hist.tolist(), edges=np.round(edges, 1).tolist(), mean=round(float(res.mean()), 2)),
        calib=dict(p=cal.p.round(3).tolist(), y=cal.y.round(3).tolist()),
        reg=meta["regression"], clf=meta["classification"],
        imp_d=imp(meta["importance"]["demand"]), imp_s=imp(meta["importance"]["stockout"]),
        stats=stats, dq=dq.fillna("").to_dict("records"), splits=meta["splits"],
    )


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>StockSense | NovaMart replenishment desk</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,400..800&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.js"></script>
<style>
:root{
  --paper:#F4F6F3; --panel:#FFFFFF; --ink:#1B2A2F; --muted:#5E6F74; --rule:#D5DBD8; --wash:#E8ECE9;
  --rail:#1B2A2F; --rail-ink:#E6ECEA; --rail-muted:#9DB0B3;
  --tag:#FFD23F; --tag-ink:#1B2A2F; --teal:#1F5F6B; --sage:#8DB3A8; --gold:#C99A1E;
  --high:#C8332F; --medium:#D9822B; --low:#2E7D5B;
  --font:'Archivo','Helvetica Neue',Arial,sans-serif;
  color-scheme: light;
  box-sizing:border-box;
  padding-top:env(safe-area-inset-top,0px); padding-bottom:env(safe-area-inset-bottom,0px);
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --paper:#162124; --panel:#1D2B2F; --ink:#E4EBE8; --muted:#9AAEB0; --rule:#33474C; --wash:#22343A;
  --rail:#0F181A; --rail-ink:#E4EBE8; --rail-muted:#7F9699;
  --teal:#5FA8B4; --sage:#6F988D; --gold:#E0B53C; --high:#EF5B53; --medium:#EE9A45; --low:#4FB185;
  color-scheme: dark; } }
:root[data-theme="dark"]{
  --paper:#162124; --panel:#1D2B2F; --ink:#E4EBE8; --muted:#9AAEB0; --rule:#33474C; --wash:#22343A;
  --rail:#0F181A; --rail-ink:#E4EBE8; --rail-muted:#7F9699;
  --teal:#5FA8B4; --sage:#6F988D; --gold:#E0B53C; --high:#EF5B53; --medium:#EE9A45; --low:#4FB185;
  color-scheme: dark; }
html{ scroll-padding-top:env(safe-area-inset-top,0px); height:100%; }
*,*::before,*::after{ box-sizing:border-box; }
body{ margin:0; min-height:100%; background:var(--paper); color:var(--ink); font-family:var(--font);
  font-size:15px; line-height:1.5; -webkit-font-smoothing:antialiased; }
button,select,input{ font:inherit; color:inherit; }
:focus-visible{ outline:3px solid var(--tag); outline-offset:2px; }
.num{ font-variant-numeric: tabular-nums; }

.app{ display:grid; grid-template-columns: 232px minmax(0,1fr); min-height:100vh; }
.rail{ background:var(--rail); color:var(--rail-ink); padding:1.6rem 1.1rem; position:sticky; top:0; height:100vh; overflow-y:auto; }
.brand{ font-stretch:125%; font-weight:800; font-size:1.6rem; line-height:1; letter-spacing:-.01em; }
.brand-sub{ color:var(--rail-muted); font-size:.82rem; margin:.4rem 0 1.8rem; }
.nav{ display:flex; flex-direction:column; gap:2px; }
.nav button{ text-align:left; background:none; border:0; padding:.55rem .7rem; border-radius:3px; cursor:pointer; color:var(--rail-ink); opacity:.78; }
.nav button:hover{ opacity:1; background:rgba(255,255,255,.06); }
.nav button[aria-current="page"]{ opacity:1; background:var(--tag); color:var(--tag-ink); font-weight:700; }
.rail-foot{ margin-top:2rem; color:var(--rail-muted); font-size:.78rem; line-height:1.45; }
.theme-btn{ margin-top:.8rem; background:none; border:1px solid var(--rail-muted); color:var(--rail-ink); border-radius:3px; padding:.3rem .6rem; cursor:pointer; font-size:.8rem; }

main{ padding:1.6rem clamp(1rem,3vw,2.6rem) 3rem; min-width:0; }
.topbar{ display:flex; flex-wrap:wrap; gap:.5rem 1rem; align-items:center; justify-content:space-between; margin-bottom:1.4rem; }
.chips{ display:flex; flex-wrap:wrap; gap:.35rem; }
.chip{ border:1px solid var(--rule); background:var(--panel); padding:.25rem .65rem; border-radius:999px; cursor:pointer; font-size:.85rem; }
.chip[aria-pressed="true"]{ background:var(--ink); color:var(--paper); border-color:var(--ink); }
.asof{ color:var(--muted); font-size:.85rem; }

section[hidden]{ display:none; }
h1{ font-stretch:115%; font-weight:750; font-size:clamp(1.6rem,2.6vw,2.1rem); letter-spacing:-.02em; margin:0 0 .3rem; line-height:1.1; }
h2{ font-stretch:110%; font-weight:700; font-size:1.2rem; margin:2rem 0 .3rem; letter-spacing:-.01em; }
.lede{ color:var(--muted); max-width:68ch; margin:0 0 1.2rem; }

.hero{ display:grid; grid-template-columns:minmax(220px,auto) 1fr; gap:1rem 3rem; align-items:end; padding-bottom:1.3rem; border-bottom:2px solid var(--ink); margin-bottom:1.6rem; }
.hero-count{ font-stretch:125%; font-weight:800; font-size:clamp(4.5rem,10vw,8.5rem); line-height:.82; letter-spacing:-.03em; }
.hero-label{ font-size:1.1rem; margin-top:.6rem; }
.hero-label mark{ background:var(--tag); color:var(--tag-ink); padding:0 .3rem; font-weight:700; }
.kpis{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:1.1rem 1.8rem; }
.kpi .v{ font-stretch:112%; font-weight:700; font-size:1.65rem; line-height:1.05; font-variant-numeric:tabular-nums; }
.kpi .l{ color:var(--muted); font-size:.84rem; margin-top:.15rem; }
.kpi .d{ font-size:.82rem; font-weight:650; margin-top:.1rem; }
.up{ color:var(--low); } .down{ color:var(--high); }

.seg{ display:inline-flex; border:1px solid var(--rule); border-radius:3px; overflow:hidden; background:var(--panel); }
.seg button{ border:0; background:none; padding:.35rem .8rem; cursor:pointer; font-size:.88rem; }
.seg button+button{ border-left:1px solid var(--rule); }
.seg button[aria-pressed="true"]{ background:var(--ink); color:var(--paper); }
.row-between{ display:flex; flex-wrap:wrap; gap:.6rem 1rem; align-items:end; justify-content:space-between; }

/* the shelf-edge tag */
.tags{ display:grid; grid-template-columns:repeat(auto-fill,minmax(310px,1fr)); gap:1rem; margin-top:1rem; }
.tag{ position:relative; background:var(--tag); color:var(--tag-ink); padding:.95rem 1.1rem 1rem 1.4rem; border-radius:3px; box-shadow:0 1px 0 rgba(0,0,0,.28); }
.tag::before{ content:""; position:absolute; inset:0 auto 0 0; width:7px; background:var(--risk); border-radius:3px 0 0 3px; }
.tag::after{ content:""; position:absolute; right:14px; top:14px; width:10px; height:10px; border-radius:50%; background:var(--paper); box-shadow:inset 0 1px 2px rgba(0,0,0,.3); }
.tag .who{ font-size:.8rem; opacity:.85; padding-right:1.4rem; }
.tag .name{ font-stretch:112%; font-weight:720; font-size:1.14rem; line-height:1.2; margin:.15rem 0 .6rem; padding-right:1.4rem; }
.tag .body{ display:flex; justify-content:space-between; align-items:flex-end; gap:.8rem; }
.tag .qty{ font-stretch:125%; font-weight:800; font-size:2.7rem; line-height:.85; font-variant-numeric:tabular-nums; }
.tag .qty small{ display:block; font-stretch:100%; font-size:.74rem; font-weight:500; margin-top:.3rem; }
.tag .facts{ text-align:right; font-size:.79rem; line-height:1.5; font-variant-numeric:tabular-nums; }
.pill{ display:inline-block; padding:.05rem .45rem; border-radius:2px; color:#fff; font-size:.72rem; font-weight:700; background:var(--risk); }
.tag .why{ border-top:1px dashed rgba(27,42,47,.45); margin-top:.75rem; padding-top:.55rem; display:flex; flex-wrap:wrap; gap:.3rem; font-size:.8rem; }
.tag .why span{ background:rgba(255,255,255,.55); padding:.05rem .4rem; border-radius:2px; }
.tag .act{ font-weight:700; font-size:.88rem; margin-top:.45rem; }
.more{ margin-top:1rem; }
.btn{ background:var(--ink); color:var(--paper); border:0; padding:.5rem .9rem; border-radius:3px; cursor:pointer; font-weight:600; }
.btn.ghost{ background:none; color:var(--ink); border:1px solid var(--rule); }

.panel{ background:var(--panel); border:1px solid var(--rule); border-radius:4px; padding:1rem 1.1rem; min-width:0; }
.panel h3{ margin:0 0 .6rem; font-size:.98rem; font-weight:650; }
.grid2{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:1rem; }
.grid3{ display:grid; grid-template-columns:2fr 1fr; gap:1rem; }
.chart{ position:relative; height:300px; }
.chart.tall{ height:360px; }
.note{ border-left:4px solid var(--teal); background:var(--wash); padding:.6rem 1rem; max-width:90ch; margin:1.2rem 0; }

.table-wrap{ overflow-x:auto; border:1px solid var(--rule); border-radius:4px; background:var(--panel); max-height:520px; overflow-y:auto; }
table{ border-collapse:collapse; width:100%; font-size:.86rem; }
th,td{ padding:.5rem .7rem; text-align:left; border-bottom:1px solid var(--rule); white-space:nowrap; }
th{ position:sticky; top:0; background:var(--wash); font-weight:650; z-index:1; }
td.n,th.n{ text-align:right; font-variant-numeric:tabular-nums; }
td.wrap{ white-space:normal; min-width:260px; }
tr.sel td{ background:color-mix(in srgb, var(--tag) 30%, transparent); }
.dot{ display:inline-block; width:.6rem; height:.6rem; border-radius:50%; background:var(--risk); margin-right:.35rem; }
.bar{ display:inline-block; height:.5rem; background:var(--risk); border-radius:1px; vertical-align:middle; margin-right:.4rem; }

.heat{ display:grid; gap:3px; font-size:.82rem; }
.heat div{ padding:.55rem .3rem; text-align:center; border-radius:2px; font-variant-numeric:tabular-nums; }
.heat .h{ background:none; color:var(--muted); font-weight:600; }
.cm{ display:grid; grid-template-columns:auto 1fr 1fr; gap:4px; font-variant-numeric:tabular-nums; }
.cm div{ padding:1rem .6rem; text-align:center; border-radius:3px; }
.cm .h{ color:var(--muted); font-size:.82rem; padding:.3rem; }
.cm .v{ font-stretch:115%; font-size:1.5rem; font-weight:700; }
.cm .v small{ display:block; font-size:.72rem; font-weight:500; font-stretch:100%; }

.controls{ display:flex; flex-wrap:wrap; gap:1.2rem 2rem; align-items:end; margin:1rem 0 1.4rem; }
.control label{ display:block; font-size:.84rem; color:var(--muted); margin-bottom:.3rem; }
select{ background:var(--panel); border:1px solid var(--rule); border-radius:3px; padding:.4rem .6rem; max-width:100%; }
.stat{ border-top:2px solid var(--ink); padding-top:.8rem; }
.stat h3{ margin:0 0 .4rem; font-size:1rem; }
.stat p{ margin:.25rem 0; font-size:.9rem; max-width:75ch; }
.stat .meta{ color:var(--muted); font-size:.84rem; }
.stats{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:1.6rem 2rem; }

@media (max-width: 900px){
  .app{ grid-template-columns:1fr; }
  .rail{ position:sticky; top:0; height:auto; z-index:5; padding:.8rem 1rem calc(.6rem); padding-top:calc(.8rem + env(safe-area-inset-top,0px)); margin-top:calc(-1 * env(safe-area-inset-top,0px)); }
  .brand{ font-size:1.25rem; } .brand-sub,.rail-foot{ display:none; }
  .nav{ flex-direction:row; overflow-x:auto; gap:.3rem; margin-top:.6rem; scrollbar-width:none; }
  .nav button{ white-space:nowrap; padding:.35rem .6rem; font-size:.85rem; }
  .hero{ grid-template-columns:1fr; }
  .grid2,.grid3{ grid-template-columns:1fr; }
  .tags{ grid-template-columns:1fr; }
}
@media (prefers-reduced-motion: reduce){ *{ transition:none !important; animation:none !important; } }
</style>
</head>
<body>
<div class="app">
  <aside class="rail">
    <div class="brand">StockSense</div>
    <div class="brand-sub">NovaMart replenishment desk</div>
    <nav class="nav" id="nav" aria-label="Sections"></nav>
    <div class="rail-foot">
      <div id="railAsof"></div>
      <button class="theme-btn" id="themeBtn" type="button">Switch theme</button>
    </div>
  </aside>
  <main>
    <div class="topbar">
      <div class="chips" id="storeChips" role="group" aria-label="Filter stores"></div>
      <div class="asof" id="asof"></div>
    </div>

    <section id="orders">
      <div class="hero">
        <div><div class="hero-count num" id="heroCount">0</div>
             <div class="hero-label">items need an order <mark>today</mark></div></div>
        <div class="kpis" id="heroKpis"></div>
      </div>
      <div class="row-between">
        <div><h1>Order cards</h1><p class="lede" style="margin:0">Sorted by sales at risk. Each card shows what to order, why it was flagged and what to do next.</p></div>
        <div class="seg" id="orderSeg" role="group" aria-label="Which items">
          <button data-v="HIGH" aria-pressed="true">High risk</button>
          <button data-v="HM" aria-pressed="false">High + medium</button>
          <button data-v="ALL" aria-pressed="false">Every order</button>
        </div>
      </div>
      <div class="tags" id="tags"></div>
      <div class="more"><button class="btn ghost" id="moreBtn" type="button">Show more cards</button></div>
      <h2>Full action list</h2>
      <p class="lede">The same items as a table, ready to hand to the purchasing team.</p>
      <div class="row-between" style="margin-bottom:.6rem"><span></span><button class="btn" id="csvBtn" type="button">Copy as CSV</button></div>
      <div class="table-wrap"><table id="actionTable"></table></div>
      <h2>Overstocked: hold orders</h2>
      <p class="lede">More than two weeks of stock (or more than the shelf life). Pausing these orders frees working capital.</p>
      <div class="table-wrap"><table id="overTable"></table></div>
    </section>

    <section id="summary" hidden>
      <h1>Executive summary</h1>
      <p class="lede" id="sumLede"></p>
      <div class="kpis" id="sumKpis" style="margin-bottom:1.6rem"></div>
      <div class="panel"><h3>Revenue per day, all stores</h3><div class="chart"><canvas id="cRevenue"></canvas></div></div>
      <div class="grid2" style="margin-top:1rem">
        <div class="panel"><h3>Store growth, first 4 vs last 4 weeks</h3><div class="chart"><canvas id="cGrowth"></canvas></div></div>
        <div class="panel"><h3>Stock-out rate by store</h3><div class="chart"><canvas id="cSoStore"></canvas></div></div>
      </div>
      <div class="note" id="sumNote"></div>
      <div class="panel"><h3>Revenue by category</h3><div class="chart"><canvas id="cCatRev"></canvas></div></div>
    </section>

    <section id="demand" hidden>
      <h1>Demand intelligence</h1>
      <p class="lede">How well the forecast tracked real sales over the held-out test weeks, and where demand is heading.</p>
      <div class="panel"><h3>Next-7-day units: actual vs forecast (test weeks, all stores)</h3><div class="chart tall"><canvas id="cAvF"></canvas></div></div>
      <div class="grid2" style="margin-top:1rem">
        <div class="panel"><h3>Weekly units by category</h3><div class="chart"><canvas id="cCatWk"></canvas></div></div>
        <div class="panel"><h3>Forecast error by category (WAPE)</h3><div class="chart"><canvas id="cWape"></canvas></div></div>
      </div>
      <div class="panel" style="margin-top:1rem"><h3>Weekly revenue by store</h3><div class="chart"><canvas id="cStoreWk"></canvas></div></div>
    </section>

    <section id="risk" hidden>
      <h1>Inventory risk</h1>
      <p class="lede">Probability that each store and product runs out in the next 7 days. High is 70% or more, medium 40 to 70%.</p>
      <div class="kpis" id="riskKpis" style="margin-bottom:1.4rem"></div>
      <div class="grid3">
        <div class="panel"><h3>Average stock-out probability, store x category</h3><div class="heat" id="heat"></div></div>
        <div class="panel"><h3>Less cover, more risk</h3><div class="chart"><canvas id="cScatter"></canvas></div></div>
      </div>
      <h2>Risk register</h2>
      <div class="table-wrap"><table id="riskTable"></table></div>
    </section>

    <section id="models" hidden>
      <h1>Model performance</h1>
      <p class="lede" id="modelLede"></p>
      <div class="kpis" id="modelKpis" style="margin-bottom:1.6rem"></div>
      <div class="grid2">
        <div class="panel"><h3>Demand models, validation weeks</h3><div class="table-wrap" style="border:0"><table id="regTable"></table></div></div>
        <div class="panel"><h3>Stock-out models, validation weeks</h3><div class="table-wrap" style="border:0"><table id="clfTable"></table></div></div>
      </div>
      <div class="grid2" style="margin-top:1rem">
        <div class="panel"><h3 id="cmTitle">Confusion matrix, test weeks</h3><div class="cm" id="cm"></div></div>
        <div class="panel"><h3 id="residTitle">Residuals</h3><div class="chart"><canvas id="cResid"></canvas></div></div>
      </div>
      <div class="grid2" style="margin-top:1rem">
        <div class="panel"><h3>Calibration: predicted vs observed stock-out rate</h3><div class="chart"><canvas id="cCalib"></canvas></div></div>
        <div class="note" style="margin:0" id="trustNote"></div>
      </div>
    </section>

    <section id="why" hidden>
      <h1>Why the model says so</h1>
      <p class="lede">Permutation importance: how much accuracy drops when a feature is shuffled. Longer bar, more influence.</p>
      <div class="grid2">
        <div class="panel"><h3>What drives the demand forecast</h3><div class="chart tall"><canvas id="cImpD"></canvas></div></div>
        <div class="panel"><h3>What drives stock-out risk</h3><div class="chart tall"><canvas id="cImpS"></canvas></div></div>
      </div>
      <h2>Explain one item</h2>
      <p class="lede">Each bar answers: if this driver were at a normal level (no promotion, no festival, typical stock cover), how different would the prediction be?</p>
      <div class="control" style="margin-bottom:1rem"><label for="itemSel">Store and product</label><select id="itemSel"></select></div>
      <div class="grid3">
        <div class="panel"><h3>Effect on stock-out probability (points)</h3><div class="chart"><canvas id="cRd"></canvas></div></div>
        <div id="itemTag"></div>
      </div>
      <div class="panel" style="margin-top:1rem"><h3>Effect on 7-day demand (%)</h3><div class="chart" style="height:240px"><canvas id="cDd"></canvas></div></div>
    </section>

    <section id="whatif" hidden>
      <h1>What-if planner</h1>
      <p class="lede">Both models were re-run under each plan. Pick a scenario to see how risk and orders move.</p>
      <div class="controls">
        <div class="control"><label>Promotion discount</label>
          <div class="seg" id="wDisc"><button data-v="0" aria-pressed="true">None</button><button data-v="10" aria-pressed="false">10%</button><button data-v="20" aria-pressed="false">20%</button></div></div>
        <div class="control"><label>Supplier delay</label>
          <div class="seg" id="wDelay"><button data-v="0" aria-pressed="true">On time</button><button data-v="2" aria-pressed="false">+2 days</button></div></div>
        <div class="control"><label>Local festival this week</label>
          <div class="seg" id="wFest"><button data-v="0" aria-pressed="true">No</button><button data-v="1" aria-pressed="false">Yes</button></div></div>
      </div>
      <div class="kpis" id="wKpis" style="margin-bottom:1.4rem"></div>
      <h2>Biggest changes</h2>
      <div class="table-wrap"><table id="wTable"></table></div>
    </section>

    <section id="data" hidden>
      <h1>Data quality and statistics</h1>
      <p class="lede">Every issue found in the five raw files, what was done and why. Then the four hypothesis tests behind the design choices.</p>
      <div class="table-wrap"><table id="dqTable"></table></div>
      <h2>Statistical checks</h2>
      <div class="stats" id="stats" style="margin-top:1rem"></div>
    </section>
  </main>
</div>

<script>
const D = __DATA__;
const SECTIONS = [["orders","Today's orders"],["summary","Executive summary"],["demand","Demand"],
  ["risk","Inventory risk"],["models","Model performance"],["why","Why the model says so"],
  ["whatif","What-if planner"],["data","Data quality"]];
const STORES = [...new Set(D.items.map(i=>i.s))].sort();
const state = { stores:new Set(STORES), orderMode:"HIGH", shown:12, section:"orders",
                w:{disc:"0",delay:"0",fest:"0"} };
const charts = {};
const $ = s => document.querySelector(s);
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const esc = s => String(s).replace(/[&<>"]/g, c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const fmt = n => Number(n).toLocaleString("en-IN",{maximumFractionDigits:0});
const pct = (n,d=0) => (n*100).toFixed(d)+"%";
function rs(v){ v=+v; const a=Math.abs(v);
  if(a>=1e7) return "Rs "+(v/1e7).toFixed(2)+" Cr"; if(a>=1e5) return "Rs "+(v/1e5).toFixed(1)+" L";
  if(a>=1e3) return "Rs "+(v/1e3).toFixed(1)+" K"; return "Rs "+fmt(v); }
const riskVar = r => r==="HIGH"?"var(--high)":r==="MEDIUM"?"var(--medium)":"var(--low)";
const riskCol = r => css(r==="HIGH"?"--high":r==="MEDIUM"?"--medium":"--low");
const view = () => D.items.map((it,idx)=>({...it,idx})).filter(i=>state.stores.has(i.s));
const kpi = (v,l,d="",cls="") => `<div class="kpi"><div class="v">${v}</div><div class="l">${l}</div>${d?`<div class="d ${cls}">${d}</div>`:""}</div>`;

// ---------- theme ----------
function applyTheme(t){ if(t) document.documentElement.setAttribute("data-theme",t);
  else document.documentElement.removeAttribute("data-theme"); try{ t?localStorage.setItem("ss-theme",t):localStorage.removeItem("ss-theme"); }catch(e){} renderAll(); }
try{ const t=localStorage.getItem("ss-theme"); if(t) document.documentElement.setAttribute("data-theme",t); }catch(e){}
$("#themeBtn").onclick=()=>{ const dark = document.documentElement.getAttribute("data-theme")==="dark" ||
  (!document.documentElement.getAttribute("data-theme") && matchMedia("(prefers-color-scheme: dark)").matches);
  applyTheme(dark?"light":"dark"); };
matchMedia("(prefers-color-scheme: dark)").addEventListener("change",()=>renderAll());

// ---------- chart helpers ----------
function chart(id, cfg){
  if(charts[id]) charts[id].destroy();
  const ink=css("--ink"), muted=css("--muted"), rule=css("--rule");
  Chart.defaults.font.family="Archivo, Helvetica Neue, Arial, sans-serif";
  Chart.defaults.color=muted;
  cfg.options = Object.assign({responsive:true, maintainAspectRatio:false, animation:false,
    plugins:{legend:{labels:{color:ink,boxWidth:12,usePointStyle:true}}, tooltip:{backgroundColor:ink,titleColor:css("--paper"),bodyColor:css("--paper")}},
    scales:{x:{grid:{color:rule},ticks:{color:muted,maxRotation:0,autoSkipPadding:14}},y:{grid:{color:rule},ticks:{color:muted}}}}, cfg.options||{});
  charts[id]=new Chart(document.getElementById(id), cfg);
}
const hbar = (id, labels, vals, colors, fmtFn=v=>v, extra={}) => chart(id,{type:"bar",
  data:{labels, datasets:[{data:vals, backgroundColor:colors, borderRadius:2, barThickness:"flex", maxBarThickness:22}]},
  options:Object.assign({indexAxis:"y", plugins:{legend:{display:false}, tooltip:{callbacks:{label:c=>fmtFn(c.raw)}}},
    scales:{x:{grid:{color:css("--rule")},ticks:{color:css("--muted"),callback:v=>fmtFn(v)}},y:{grid:{display:false},ticks:{color:css("--ink")}}}},extra)});

// ---------- nav + filters ----------
function buildNav(){
  $("#nav").innerHTML = SECTIONS.map(([id,l])=>`<button type="button" data-s="${id}" ${id===state.section?'aria-current="page"':''}>${l}</button>`).join("");
  $("#nav").onclick=e=>{ const b=e.target.closest("button"); if(!b) return; state.section=b.dataset.s;
    SECTIONS.forEach(([id])=>{ document.getElementById(id).hidden = id!==state.section; });
    buildNav(); renderSection(); window.scrollTo(0,0); };
  $("#storeChips").innerHTML = `<button class="chip" data-s="ALL" aria-pressed="${state.stores.size===STORES.length}">All stores</button>` +
    STORES.map(s=>{ const c=D.items.find(i=>i.s===s).city; return `<button class="chip" data-s="${s}" aria-pressed="${state.stores.has(s)&&state.stores.size!==STORES.length}">${s} ${c}</button>`;}).join("");
}
$("#storeChips").onclick=e=>{ const b=e.target.closest("button"); if(!b) return; const s=b.dataset.s;
  if(s==="ALL") state.stores=new Set(STORES);
  else if(state.stores.size===STORES.length) state.stores=new Set([s]);
  else { state.stores.has(s)?state.stores.delete(s):state.stores.add(s); if(!state.stores.size) state.stores=new Set(STORES); }
  state.shown=12; buildNav(); renderSection(); };
function segInit(sel, cb){ $(sel).onclick=e=>{ const b=e.target.closest("button"); if(!b) return;
  $(sel).querySelectorAll("button").forEach(x=>x.setAttribute("aria-pressed", x===b)); cb(b.dataset.v); }; }

// ---------- shelf tag ----------
function tagHTML(i){
  const why = (i.why||[]).map(([n,v])=> v==="rule"?`<span><b>${esc(n)}</b></span>`:`<span>${esc(n)} · ${esc(v)}</span>`).join("") || "<span>Stock covers the forecast</span>";
  return `<article class="tag" style="--risk:${riskVar(i.risk)}">
    <div class="who">${i.s} ${esc(i.city)} · ${esc(i.cat)}${i.sparse?" · new item, look-alike forecast":""}</div>
    <div class="name">${esc(i.p)}</div>
    <div class="body"><div class="qty">${fmt(i.q)}<small>units to order</small></div>
      <div class="facts"><span class="pill">${i.risk} ${pct(i.prob)}</span><br>
        <span style="opacity:.75">model ${pct(i.mprob)}${i.basis!=="Model"?" + rule":""}</span><br>
        On shelf <b>${fmt(i.stock)}</b> · Next 7 days <b>${fmt(i.fc)}</b><br>
        Cover <b>${i.cover.toFixed(1)} d</b> · Lead time <b>${i.lead} d</b></div></div>
    <div class="why">${why}</div><div class="act">${esc(i.act)}</div></article>`;
}

// ---------- sections ----------
function renderOrders(){
  const v=view(), high=v.filter(i=>i.risk==="HIGH"), med=v.filter(i=>i.risk==="MEDIUM"), ord=v.filter(i=>i.q>0);
  $("#heroCount").textContent=high.length;
  $("#heroKpis").innerHTML = kpi(med.length,"to order within 2 days") + kpi(fmt(ord.reduce((a,i)=>a+i.q,0)),`units recommended, ${ord.length} lines`)
    + kpi(rs(v.reduce((a,i)=>a+i.rar,0)),"sales at risk this week") + kpi(rs(v.reduce((a,i)=>a+i.exv,0)),"cash tied up in overstock");
  const pick = state.orderMode==="HIGH"?high:state.orderMode==="HM"?high.concat(med):ord;
  $("#tags").innerHTML = pick.length? pick.slice(0,state.shown).map(tagHTML).join("") :
    `<p class="lede">Nothing to order for these stores. Choose "Every order" or add more stores.</p>`;
  $("#moreBtn").hidden = pick.length<=state.shown;
  $("#moreBtn").textContent = `Show more cards (${pick.length-state.shown} left)`;
  $("#actionTable").innerHTML = `<thead><tr><th>Store</th><th>Product</th><th class="n">Current stock</th><th class="n">7-day forecast</th><th>Stock-out prob.</th><th>Risk</th><th class="n">Recommended order</th><th>Action</th></tr></thead><tbody>`+
    pick.map(i=>`<tr><td>${i.s}</td><td>${esc(i.p)}</td><td class="n">${fmt(i.stock)}</td><td class="n">${fmt(i.fc)}</td>
      <td style="--risk:${riskVar(i.risk)}"><span class="bar" style="width:${Math.round(i.prob*60)}px"></span>${i.prob.toFixed(2)}</td>
      <td style="--risk:${riskVar(i.risk)}"><span class="dot"></span>${i.risk}</td><td class="n"><b>${fmt(i.q)}</b></td><td class="wrap">${esc(i.act)}</td></tr>`).join("")+"</tbody>";
  const over=v.filter(i=>i.exu>0).sort((a,b)=>b.exv-a.exv);
  $("#overTable").innerHTML = over.length? `<thead><tr><th>Store</th><th>Product</th><th class="n">Current stock</th><th class="n">7-day forecast</th><th class="n">Days of cover</th><th class="n">Excess value</th><th>Action</th></tr></thead><tbody>`+
    over.map(i=>`<tr><td>${i.s}</td><td>${esc(i.p)}</td><td class="n">${fmt(i.stock)}</td><td class="n">${fmt(i.fc)}</td><td class="n">${i.cover.toFixed(1)}</td><td class="n">${rs(i.exv)}</td><td class="wrap">${esc(i.act)}</td></tr>`).join("")+"</tbody>"
    : `<tbody><tr><td>No overstocked items for these stores.</td></tr></tbody>`;
  $("#csvBtn").onclick=()=>{ const rows=[["store","product","current_stock","forecast_7d","stockout_prob","risk","reorder_qty","action"]]
    .concat(pick.map(i=>[i.s,i.p,i.stock,i.fc,i.prob,i.risk,i.q,i.act])); 
    const txt=rows.map(r=>r.map(x=>`"${String(x).replace(/"/g,'""')}"`).join(",")).join("\n");
    navigator.clipboard.writeText(txt).then(()=>{ $("#csvBtn").textContent="Copied"; setTimeout(()=>$("#csvBtn").textContent="Copy as CSV",1500); }); };
}

function renderSummary(){
  const k=D.kpi, g=k.revenue_growth_28d;
  $("#sumLede").textContent=`All six stores, ${k.date_min} to ${k.date_max}. Figures come from the cleaned master dataset.`;
  $("#sumKpis").innerHTML = kpi(rs(k.revenue_total),"revenue, full period") + kpi(rs(k.revenue_last_28d),"last 28 days",(g>=0?"+":"")+pct(g,1)+" vs prior 28",g>=0?"up":"down")
    + kpi(pct(k.stockout_rate,1),"stock-out rate (store-product days)") + kpi(rs(k.inventory_value_latest),"inventory value at cost")
    + kpi(rs(k.estimated_lost_sales),"estimated lost sales from empty shelves") + kpi(D.items.filter(i=>i.risk==="HIGH").length,"products at high risk now");
  const d=D.daily, ma=d.revenue.map((_,i)=> i<6?null: d.revenue.slice(i-6,i+1).reduce((a,b)=>a+b,0)/7);
  chart("cRevenue",{type:"bar",data:{labels:d.dates.map(x=>x.slice(5)),datasets:[
    {type:"line",label:"7-day average",data:ma,borderColor:css("--ink"),borderWidth:2.5,pointRadius:0,tension:.3},
    {label:"Daily revenue",data:d.revenue,backgroundColor:css("--sage"),borderRadius:1}]},
    options:{scales:{y:{ticks:{callback:v=>rs(v)},grid:{color:css("--rule")}},x:{grid:{display:false},ticks:{color:css("--muted"),maxRotation:0,autoSkipPadding:18}}}}});
  const gr=Object.entries(D.growth).sort((a,b)=>b[1]-a[1]);
  hbar("cGrowth",gr.map(x=>x[0]),gr.map(x=>+(x[1]*100).toFixed(1)),gr.map(x=>x[1]<0?css("--high"):css("--teal")),v=>v+"%");
  const so=gr.map(x=>[x[0],D.so_store[x[0]]]);
  hbar("cSoStore",so.map(x=>x[0]),so.map(x=>+(x[1]*100).toFixed(1)),css("--medium"),v=>v+"%");
  const top=gr[0];
  $("#sumNote").innerHTML=`<b>${top[0]} grew fastest (${pct(top[1])}) and runs out more often than any other store (${pct(D.so_store[top[0]],1)} of days).</b> Its reorder levels were set for a smaller store. Promotions lift units by ${pct(D.promo_lift)} and weekends by ${pct(D.weekend)}; both now feed the forecast. ${D.pareto} products make up 80% of revenue.`;
  const cr=Object.entries(D.cat_rev);
  hbar("cCatRev",cr.map(x=>x[0]),cr.map(x=>x[1]),css("--teal"),v=>rs(v));
}

function renderDemand(){
  const t=D.test;
  chart("cAvF",{type:"line",data:{labels:t.dates,datasets:[
    {label:"Actual",data:t.actual,borderColor:css("--ink"),borderWidth:3,pointRadius:0,tension:.25},
    {label:"StockSense forecast",data:t.fc,borderColor:css("--teal"),borderWidth:2.5,pointRadius:0,tension:.25},
    {label:"Old rule (last 7-day mean)",data:t.base,borderColor:css("--muted"),borderWidth:1.5,borderDash:[5,4],pointRadius:0,tension:.25}]},
    options:{interaction:{mode:"index",intersect:false},scales:{y:{ticks:{callback:v=>fmt(v)},grid:{color:css("--rule")}}}}});
  const pal=[css("--teal"),css("--gold"),css("--sage"),"#4B6F8A","#A45A52","#6E8B3D","#8A6FA0"];
  const cats=Object.entries(D.wk_cat);
  chart("cCatWk",{type:"line",data:{labels:cats[0][1].dates.map(x=>x.slice(5)),datasets:cats.map(([c,v],i)=>({label:c,data:v.units,borderColor:pal[i%7],borderWidth:2,pointRadius:0,tension:.3}))},
    options:{plugins:{legend:{position:"bottom",labels:{color:css("--ink"),boxWidth:10,usePointStyle:true}}}}});
  hbar("cWape",D.wape_cat.map(x=>x[0]),D.wape_cat.map(x=>x[1]),css("--teal"),v=>(+v).toFixed(1)+"%");
  const st=Object.entries(D.wk_store).filter(([s])=>state.stores.has(s));
  chart("cStoreWk",{type:"line",data:{labels:st.length?st[0][1].dates.map(x=>x.slice(5)):[],datasets:st.map(([s,v],i)=>({label:s,data:v.revenue,borderColor:pal[STORES.indexOf(s)%7],borderWidth:2,pointRadius:0,tension:.3}))},
    options:{scales:{y:{ticks:{callback:v=>rs(v)},grid:{color:css("--rule")}}}}});
}

function heatColor(p){ const stops=[[0,[231,240,234]],[.4,[243,217,164]],[.7,[227,154,91]],[1,[184,50,45]]];
  for(let i=1;i<stops.length;i++){ if(p<=stops[i][0]){ const [a,ca]=stops[i-1],[b,cb]=stops[i],t=(p-a)/(b-a);
    return `rgb(${ca.map((c,j)=>Math.round(c+(cb[j]-c)*t)).join(",")})`; } } return "rgb(184,50,45)"; }
function renderRisk(){
  const v=view(), c={HIGH:0,MEDIUM:0,LOW:0}; v.forEach(i=>c[i.risk]++);
  $("#riskKpis").innerHTML = kpi(`<span style="color:var(--high)">${c.HIGH}</span>`,"high risk (70% or more)")+kpi(`<span style="color:var(--medium)">${c.MEDIUM}</span>`,"medium (40 to 70%)")
    +kpi(`<span style="color:var(--low)">${c.LOW}</span>`,"low (under 40%)")+kpi(v.filter(i=>i.basis!=="Model").length,"escalated by a stock rule");
  const cats=[...new Set(D.items.map(i=>i.cat))].sort(), stores=STORES.filter(s=>state.stores.has(s));
  $("#heat").style.gridTemplateColumns=`auto repeat(${cats.length},minmax(52px,1fr))`;
  let h=`<div class="h"></div>`+cats.map(x=>`<div class="h">${x}</div>`).join("");
  stores.forEach(s=>{ h+=`<div class="h">${s}</div>`; cats.forEach(ct=>{ const g=D.items.filter(i=>i.s===s&&i.cat===ct); const p=g.length?g.reduce((a,i)=>a+i.prob,0)/g.length:0;
    h+=`<div style="background:${heatColor(p)};color:${p>.62?"#fff":"#1B2A2F"}" title="${s} ${ct}: ${pct(p)}">${pct(p)}</div>`; }); });
  $("#heat").innerHTML=h;
  chart("cScatter",{type:"scatter",data:{datasets:["LOW","MEDIUM","HIGH"].map(r=>({label:r,data:v.filter(i=>i.risk===r).map(i=>({x:Math.min(i.cover,25),y:i.prob,n:i.s+" "+i.p})),backgroundColor:riskCol(r),pointRadius:4}))},
    options:{plugins:{tooltip:{callbacks:{label:c=>`${c.raw.n}: cover ${c.raw.x.toFixed(1)} d, prob ${c.raw.y.toFixed(2)}`}}},
      scales:{x:{title:{display:true,text:"Days of stock cover",color:css("--muted")},grid:{color:css("--rule")}},y:{min:0,max:1,title:{display:true,text:"Stock-out probability",color:css("--muted")},grid:{color:css("--rule")}}}}});
  const sorted=[...v].sort((a,b)=>b.prob-a.prob);
  $("#riskTable").innerHTML=`<thead><tr><th>Store</th><th>Product</th><th>Category</th><th class="n">Stock</th><th class="n">7-day forecast</th><th class="n">Days of cover</th><th class="n">Model prob.</th><th class="n">Final prob.</th><th>Risk</th><th>Basis</th></tr></thead><tbody>`+
    sorted.map(i=>`<tr><td>${i.s}</td><td>${esc(i.p)}</td><td>${i.cat}</td><td class="n">${fmt(i.stock)}</td><td class="n">${fmt(i.fc)}</td><td class="n">${i.cover.toFixed(1)}</td><td class="n">${i.mprob.toFixed(2)}</td><td class="n"><b>${i.prob.toFixed(2)}</b></td><td style="--risk:${riskVar(i.risk)}"><span class="dot"></span>${i.risk}</td><td>${esc(i.basis)}</td></tr>`).join("")+"</tbody>";
}

function renderModels(){
  const r=D.reg, c=D.clf, rt=Object.fromEntries(r.test.map(x=>[x.model,x])), sel=rt[r.selected], base=rt["Baseline (7-day mean x 7)"], ct=c.test[0];
  $("#modelLede").textContent=`Trained on ${D.splits.train[0]} to ${D.splits.train[1]}, chosen on ${D.splits.valid[0]} to ${D.splits.valid[1]}, tested once on ${D.splits.test[0]} to ${D.splits.test[1]}. Seven-day gaps between splits stop future sales leaking into training.`;
  $("#modelKpis").innerHTML = kpi(sel.WAPE.toFixed(1)+"%",`forecast error (WAPE), ${r.selected}`,`old rule: ${base.WAPE.toFixed(1)}%`,"up")
    + kpi(sel.MAE.toFixed(1),"mean abs. error, units per 7 days") + kpi(sel.R2.toFixed(3),"R² on test weeks")
    + kpi(ct.ROC_AUC.toFixed(3),`stock-out ROC-AUC, ${c.selected}`) + kpi(pct(ct.Recall),"of stock-outs caught (recall)") + kpi(pct(ct.Precision),"of alerts were real (precision)");
  const rows=(arr,cols)=>`<thead><tr>${cols.map(([k,l],j)=>`<th class="${j?"n":""}">${l}</th>`).join("")}</tr></thead><tbody>`+
    arr.map(x=>`<tr${x.model===r.selected||x.model===c.selected?' class="sel"':''}>${cols.map(([k],j)=>`<td class="${j?"n":""}">${j?(+x[k]).toFixed(k==="R2"||k.includes("AUC")||["Accuracy","Precision","Recall","F1"].includes(k)?3:2):esc(x[k])}</td>`).join("")}</tr>`).join("")+"</tbody>";
  $("#regTable").innerHTML=rows(r.validation,[["model","Model"],["MAE","MAE"],["RMSE","RMSE"],["MAPE","MAPE %"],["WAPE","WAPE %"],["R2","R²"]]);
  $("#clfTable").innerHTML=rows(c.validation,[["model","Model"],["Precision","Precision"],["Recall","Recall"],["F1","F1"],["ROC_AUC","ROC-AUC"],["PR_AUC","PR-AUC"]]);
  const m=c.confusion_matrix, tot=m.flat().reduce((a,b)=>a+b,0), mx=Math.max(...m.flat());
  const cell=(v,l)=>`<div class="v" style="background:color-mix(in srgb, var(--teal) ${Math.round(15+75*v/mx)}%, var(--panel));color:${v/mx>.5?"#fff":"var(--ink)"}">${fmt(v)}<small>${l}</small></div>`;
  $("#cmTitle").textContent=`Confusion matrix, test weeks (alert at ${c.threshold.toFixed(2)})`;
  $("#cm").innerHTML=`<div class="h"></div><div class="h">Predicted: fine</div><div class="h">Predicted: stock-out</div>
    <div class="h">Actual: fine</div>${cell(m[0][0],"correctly left alone")}${cell(m[0][1],"false alarms")}
    <div class="h">Actual: stock-out</div>${cell(m[1][0],"missed")}${cell(m[1][1],"caught")}`;
  $("#residTitle").textContent=`Forecast residuals (actual - forecast), mean ${D.resid.mean>0?"+":""}${D.resid.mean} units`;
  const e=D.resid.edges; chart("cResid",{type:"bar",data:{labels:D.resid.counts.map((_,i)=>Math.round((e[i]+e[i+1])/2)),datasets:[{data:D.resid.counts,backgroundColor:css("--sage"),barPercentage:1,categoryPercentage:1}]},
    options:{plugins:{legend:{display:false}},scales:{x:{grid:{display:false}},y:{grid:{color:css("--rule")}}}}});
  chart("cCalib",{type:"line",data:{labels:D.calib.p,datasets:[
    {label:"Perfect calibration",data:D.calib.p.map(p=>({x:p,y:p})),borderColor:css("--rule"),pointRadius:0,borderWidth:1.5},
    {label:"StockSense",data:D.calib.p.map((p,i)=>({x:p,y:D.calib.y[i]})),borderColor:css("--teal"),backgroundColor:css("--teal"),borderWidth:2.5,pointRadius:4}]},
    options:{scales:{x:{type:"linear",min:0,max:1,title:{display:true,text:"Predicted probability",color:css("--muted")},grid:{color:css("--rule")}},y:{min:0,max:1,title:{display:true,text:"Observed rate",color:css("--muted")},grid:{color:css("--rule")}}}}});
  $("#trustNote").innerHTML=`<b>Why these numbers can be trusted.</b><br>Model choice used the validation weeks only; the test weeks were scored once. The forecast beats the old 7-day-average rule by ${(base.WAPE-sel.WAPE).toFixed(1)} points of error. Probabilities are calibrated, so a 70% risk means roughly 7 in 10 such items really ran out. The alert threshold of ${c.threshold.toFixed(2)} favours catching stock-outs, because a missed stock-out costs sales while a false alarm costs one shelf check.`;
}

function renderWhy(){
  hbar("cImpD",D.imp_d.map(x=>x[0]),D.imp_d.map(x=>x[1]),css("--teal"),v=>(+v).toFixed(2));
  hbar("cImpS",D.imp_s.map(x=>x[0]),D.imp_s.map(x=>x[1]),css("--teal"),v=>(+v).toFixed(3));
  const v=view().sort((a,b)=>b.prob-a.prob), sel=$("#itemSel"), prev=sel.value;
  sel.innerHTML=v.map(i=>`<option value="${i.idx}">${i.s} · ${esc(i.p)} (${i.risk})</option>`).join("");
  if(prev && v.some(i=>String(i.idx)===prev)) sel.value=prev;
  const draw=()=>{ const i=D.items[+sel.value]; $("#itemTag").innerHTML=tagHTML(i);
    const rd=Object.entries(i.rd).sort((a,b)=>b[1]-a[1]);
    hbar("cRd",rd.map(x=>x[0]),rd.map(x=>x[1]),rd.map(x=>x[1]>0?css("--high"):css("--low")),v=>(v>0?"+":"")+(+v).toFixed(0)+" pts");
    const dd=Object.entries(i.dd).sort((a,b)=>b[1]-a[1]);
    hbar("cDd",dd.map(x=>x[0]),dd.map(x=>x[1]),css("--teal"),v=>(v>0?"+":"")+(+v).toFixed(0)+"%"); };
  sel.onchange=draw; if(v.length) draw();
}

function renderWhatif(){
  const k=`${state.w.disc}_${state.w.delay}_${state.w.fest}`, s=D.whatif[k], v=view();
  const now={high:0,q:0,fc:0}, sc={high:0,q:0,fc:0};
  const rows=v.map(i=>{ const j=i.idx; now.high+=i.risk==="HIGH"; now.q+=i.q; now.fc+=i.fc;
    sc.high+=s.risk[j]==="HIGH"; sc.q+=s.q[j]; sc.fc+=s.fc[j];
    return {...i, p2:s.prob[j], r2:s.risk[j], q2:s.q[j], f2:s.fc[j], dp:s.prob[j]-i.prob}; });
  const dh=sc.high-now.high, dq=sc.q-now.q;
  $("#wKpis").innerHTML = kpi(sc.high,"high-risk items",(dh>=0?"+":"")+dh+" vs today",dh>0?"down":"up")
    + kpi(fmt(sc.q),"units to order",(dq>=0?"+":"")+fmt(dq)+" units") + kpi(fmt(sc.fc),"7-day demand",((sc.fc/now.fc-1)*100).toFixed(1)+"% vs today")
    + kpi(rows.filter(r=>r.r2!==r.risk).length,"items change risk band");
  rows.sort((a,b)=>Math.abs(b.dp)-Math.abs(a.dp)||(b.q2-b.q)-(a.q2-a.q));
  $("#wTable").innerHTML=`<thead><tr><th>Store</th><th>Product</th><th class="n">Demand now</th><th class="n">In scenario</th><th class="n">Prob. now</th><th class="n">In scenario</th><th>Risk now</th><th>In scenario</th><th class="n">Order now</th><th class="n">In scenario</th></tr></thead><tbody>`+
    rows.slice(0,60).map(r=>`<tr><td>${r.s}</td><td>${esc(r.p)}</td><td class="n">${fmt(r.fc)}</td><td class="n">${fmt(r.f2)}</td><td class="n">${r.prob.toFixed(2)}</td><td class="n"><b>${r.p2.toFixed(2)}</b></td>
    <td style="--risk:${riskVar(r.risk)}"><span class="dot"></span>${r.risk}</td><td style="--risk:${riskVar(r.r2)}"><span class="dot"></span><b>${r.r2}</b></td><td class="n">${fmt(r.q)}</td><td class="n"><b>${fmt(r.q2)}</b></td></tr>`).join("")+"</tbody>";
}

function renderData(){
  $("#dqTable").innerHTML=`<thead><tr><th>Table</th><th>Issue</th><th class="n">Count</th><th>Action taken</th><th>Why</th></tr></thead><tbody>`+
    D.dq.map(r=>`<tr><td>${esc(r.table)}</td><td class="wrap">${esc(r.issue)}</td><td class="n">${fmt(r.count)}</td><td class="wrap">${esc(r.action)}</td><td class="wrap">${esc(r.justification)}</td></tr>`).join("")+"</tbody>";
  $("#stats").innerHTML=D.stats.map(t=>`<div class="stat"><h3>${esc(t.question)}</h3>
    <p class="meta">${esc(t.test)} · p = ${t.p_value.toExponential(1)} · ${t.significant?"reject H0":"keep H0"}</p>
    <p><b>H0:</b> ${esc(t.h0)}<br><b>H1:</b> ${esc(t.h1)}</p><p>${esc(t.effect)}</p><p>${esc(t.interpretation)}</p></div>`).join("");
}

const RENDER={orders:renderOrders,summary:renderSummary,demand:renderDemand,risk:renderRisk,models:renderModels,why:renderWhy,whatif:renderWhatif,data:renderData};
function renderSection(){ RENDER[state.section](); }
function renderAll(){ renderSection(); }

segInit("#orderSeg",v=>{ state.orderMode=v; state.shown=12; renderOrders(); });
segInit("#wDisc",v=>{ state.w.disc=v; renderWhatif(); });
segInit("#wDelay",v=>{ state.w.delay=v; renderWhatif(); });
segInit("#wFest",v=>{ state.w.fest=v; renderWhatif(); });
$("#moreBtn").onclick=()=>{ state.shown+=12; renderOrders(); };
$("#asof").textContent=`Data to ${D.kpi.date_max}. Forecast for the next 7 days.`;
$("#railAsof").textContent=`As of ${D.asof}`;
buildNav(); renderSection();
</script>
</body>
</html>
"""


def run():
    data = build_data()
    html = TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":"), default=float))
    out = DASHBOARD / "stocksense_dashboard.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({len(html)/1024:.0f} KB)")


if __name__ == "__main__":
    run()
