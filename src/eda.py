"""Round 1 - Exploratory Data Analysis.

Every chart answers one business question from the brief. Figures go to
reports/figures/, headline numbers to reports/eda_summary.json.

Run:  python -m src.eda
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import style
from src.config import PROCESSED, FIGURES, REPORTS

style.apply()


def load():
    return pd.read_csv(PROCESSED / "master_dataset.csv", parse_dates=["date"])


def q1_category_revenue(m, out):
    """Which categories generate most revenue? (share + Pareto of products)"""
    cat = m.groupby("category").revenue.sum().sort_values(ascending=False)
    prod = m.groupby("product_name").revenue.sum().sort_values(ascending=False)
    cum = prod.cumsum() / prod.sum()
    n80 = int((cum <= 0.80).sum() + 1)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ax[0].barh(cat.index[::-1], cat.values[::-1] / 1e5, color=style.TEAL)
    for i, v in enumerate(cat.values[::-1]):
        ax[0].text(v / 1e5, i, f"  {v / cat.sum():.0%}", va="center", fontsize=9)
    ax[0].set_title("Revenue by category (Rs lakh)")
    ax[1].bar(range(len(prod)), prod.values / 1e5, color=style.SAGE)
    ax2 = ax[1].twinx()
    ax2.plot(range(len(prod)), cum.values * 100, color=style.INK, lw=2)
    ax2.axhline(80, color=style.HIGH, ls="--", lw=1)
    ax2.set_ylim(0, 105); ax2.grid(False)
    ax[1].set_title(f"Pareto: {n80} of {len(prod)} products make 80% of revenue")
    ax[1].set_xticks([])
    ax[1].set_xlabel("Products ranked by revenue")
    fig.tight_layout(); fig.savefig(FIGURES / "01_category_revenue_pareto.png"); plt.close(fig)
    out["category_revenue"] = {k: round(float(v), 0) for k, v in cat.items()}
    out["pareto_products_for_80pct"] = n80
    out["top_products"] = {k: round(float(v), 0) for k, v in prod.head(10).items()}


def q2_store_trend(m, out):
    """Which stores are growing or declining?"""
    wk = (m.set_index("date").groupby("store_id").revenue.resample("W-SUN").sum()
            .reset_index())
    wk = wk[wk.date < m.date.max()]          # drop the partial last week
    fig, ax = plt.subplots(figsize=(13, 4.4))
    growth = {}
    for i, (s, g) in enumerate(wk.groupby("store_id")):
        ax.plot(g.date, g.revenue / 1e3, label=s, color=style.SERIES[i % 7], lw=2)
        first, last = g.revenue.iloc[:4].mean(), g.revenue.iloc[-4:].mean()
        growth[s] = round(float(last / first - 1), 4)
    ax.set_title("Weekly revenue by store (Rs thousand)")
    ax.legend(ncol=6, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(FIGURES / "02_store_weekly_trend.png"); plt.close(fig)
    out["store_growth_first4_vs_last4_weeks"] = growth
    out["weekly_store_revenue"] = {
        s: {"dates": g.date.dt.strftime("%Y-%m-%d").tolist(),
            "revenue": g.revenue.round(0).tolist()} for s, g in wk.groupby("store_id")}


def q3_promotion(m, out):
    """Do promotions increase units sold? (lift per category, in-stock days only)"""
    d = m[m.stockout_flag == 0]
    lift = (d.groupby(["category", "promotion_flag"]).units_sold.mean().unstack())
    lift["lift"] = lift[1] / lift[0] - 1
    lift = lift.sort_values("lift")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(lift.index, lift.lift * 100, color=style.TAG, edgecolor=style.INK)
    for i, v in enumerate(lift.lift):
        ax.text(v * 100, i, f"  +{v:.0%}", va="center")
    ax.set_title("Promotion lift in average daily units, by category")
    ax.set_xlabel("% more units on promotion days")
    fig.tight_layout(); fig.savefig(FIGURES / "03_promotion_lift.png"); plt.close(fig)
    overall = d.groupby("promotion_flag").units_sold.mean()
    out["promotion_lift_overall"] = round(float(overall[1] / overall[0] - 1), 4)
    out["promotion_lift_by_category"] = {k: round(float(v), 4) for k, v in lift.lift.items()}


def q4_weekend(m, out):
    """How does weekend behaviour differ?"""
    dow = m.groupby([m.date.dt.dayofweek, "store_type"]).units_sold.mean().unstack()
    dow.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    dow.plot(ax=ax, marker="o", lw=2)
    ax.axvspan(4.5, 6.4, color=style.TAG, alpha=0.25, lw=0)
    ax.set_title("Average units per product per day, by weekday")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIGURES / "04_weekday_weekend.png"); plt.close(fig)
    w = m.groupby("weekend").units_sold.mean()
    out["weekend_uplift"] = round(float(w[1] / w[0] - 1), 4)


def q5_volatility(m, out):
    """Which products are volatile? (coefficient of variation of daily demand)"""
    cv = (m.groupby("product_name").units_sold.agg(["mean", "std"])
            .assign(cv=lambda x: x["std"] / x["mean"]).sort_values("cv", ascending=False))
    top = cv.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(top.index, top.cv, color=style.TEAL)
    ax.set_title("Most volatile products (coefficient of variation)")
    ax.set_xlabel("Std / mean of daily units (higher = harder to plan)")
    fig.tight_layout(); fig.savefig(FIGURES / "05_volatility_cv.png"); plt.close(fig)
    out["most_volatile_products"] = {k: round(float(v), 3) for k, v in cv.cv.head(8).items()}


def q6_stockout_heatmap(m, out):
    """Which stores repeatedly stock out?"""
    h = m.pivot_table(index="store_id", columns="category", values="stockout_flag",
                      aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 4.2))
    im = ax.imshow(h.values * 100, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(h.shape[1])); ax.set_xticklabels(h.columns, rotation=20)
    ax.set_yticks(range(h.shape[0])); ax.set_yticklabels(h.index)
    ax.grid(False)
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            v = h.values[i, j] * 100
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                    color="white" if v > np.nanmax(h.values) * 100 * 0.7 else style.INK, fontsize=9)
    ax.set_title("Stock-out rate (% of days with empty shelf), store x category")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(); fig.savefig(FIGURES / "06_stockout_heatmap.png"); plt.close(fig)
    out["stockout_heatmap"] = {s: {c: round(float(v), 4) for c, v in r.items()}
                               for s, r in h.iterrows()}
    out["stockout_rate_by_store"] = {k: round(float(v), 4) for k, v in
                                     m.groupby("store_id").stockout_flag.mean().items()}


def q7_weather(m, out):
    """Does heat drive beverage and frozen demand?"""
    d = m[m.category.isin(["Beverages", "Frozen"]) & (m.stockout_flag == 0)].copy()
    d["temp_band"] = pd.cut(d.temp_c, [0, 28, 30, 32, 34, 50],
                            labels=["<28", "28-30", "30-32", "32-34", "34+"])
    t = d.groupby(["temp_band", "category"], observed=True).units_sold.mean().unstack()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    t.plot(kind="bar", ax=ax, color=[style.TEAL, style.SAGE], rot=0)
    ax.set_title("Heat effect: average daily units by temperature (C)")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIGURES / "07_temperature_effect.png"); plt.close(fig)


def kpis(m, out):
    """Headline KPIs defined in the brief."""
    last = m.date.max()
    cur = m[m.date > last - pd.Timedelta(days=28)]
    prev = m[(m.date <= last - pd.Timedelta(days=28)) & (m.date > last - pd.Timedelta(days=56))]
    avg_inv_value = m.groupby("date").inventory_value.sum().mean()
    days = m.date.nunique()
    # lost sales: on stock-out days demand ~ trailing 7-day mean, fulfilled = units sold
    m = m.sort_values(["store_id", "product_id", "date"])
    rm7 = (m.groupby(["store_id", "product_id"]).units_sold
             .transform(lambda x: x.shift(1).rolling(7, min_periods=3).mean()))
    lost_units = ((rm7 - m.units_sold).clip(lower=0) * m.stockout_flag).fillna(0)
    lost_value = (lost_units * m.avg_selling_price).sum()
    out["kpi"] = {
        "revenue_total": round(float(m.revenue.sum()), 0),
        "units_total": int(m.units_sold.sum()),
        "revenue_last_28d": round(float(cur.revenue.sum()), 0),
        "revenue_growth_28d": round(float(cur.revenue.sum() / prev.revenue.sum() - 1), 4),
        "stockout_rate": round(float(m.stockout_flag.mean()), 4),
        "inventory_turnover_annualised": round(float(m.cogs.sum() / avg_inv_value * 365 / days), 2),
        "inventory_value_latest": round(float(m[m.date == last].inventory_value.sum()), 0),
        "days_of_inventory_median": round(float(
            (m[m.date == last].closing_stock / (rm7[m.date == last] + 1e-6)).median()), 1),
        "estimated_lost_sales": round(float(lost_value), 0),
        "estimated_lost_units": int(lost_units.sum()),
        "date_min": m.date.min().strftime("%Y-%m-%d"),
        "date_max": last.strftime("%Y-%m-%d"),
    }
    daily = m.groupby("date").agg(revenue=("revenue", "sum"), units=("units_sold", "sum"),
                                  stockout=("stockout_flag", "mean")).reset_index()
    out["daily"] = {"dates": daily.date.dt.strftime("%Y-%m-%d").tolist(),
                    "revenue": daily.revenue.round(0).tolist(),
                    "units": daily.units.astype(int).tolist(),
                    "stockout_rate": daily.stockout.round(4).tolist()}
    ct = (m.set_index("date").groupby("category").units_sold.resample("W-SUN").sum()
            .reset_index())
    ct = ct[ct.date < last]
    out["weekly_category_units"] = {c: {"dates": g.date.dt.strftime("%Y-%m-%d").tolist(),
                                        "units": g.units_sold.astype(int).tolist()}
                                    for c, g in ct.groupby("category")}


def run():
    m = load()
    out = {}
    for f in (q1_category_revenue, q2_store_trend, q3_promotion, q4_weekend,
              q5_volatility, q6_stockout_heatmap, q7_weather, kpis):
        f(m, out)
    (REPORTS / "eda_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["kpi"], indent=1))
    print("promo lift", out["promotion_lift_overall"], "weekend", out["weekend_uplift"])
    print("growth", out["store_growth_first4_vs_last4_weeks"])
    return out


if __name__ == "__main__":
    run()
