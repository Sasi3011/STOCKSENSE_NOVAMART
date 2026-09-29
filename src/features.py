"""Round 2 - Feature engineering.

Prediction setting: at the END of day t the manager asks
  "how much will sell over t+1 .. t+7, and will the shelf go empty in that window?"
Every feature below uses information available at the end of day t only.
The single exception is the festival/holiday calendar for t+1..t+7, which is
published in advance and therefore legitimately known at prediction time.

Run:  python -m src.features
"""
import numpy as np
import pandas as pd

from src.config import PROCESSED, RAW, HORIZON
from src.data_cleaning import _parse_dates

KEY = ["store_id", "product_id"]

NUMERIC_FEATURES = [
    # time
    "day_of_week", "weekend_flag", "month", "week_no", "festival_flag", "holiday_flag",
    "festival_next_7", "holiday_next_7", "weekend_days_next_7",
    # lag / rolling
    "lag_1", "lag_7", "lag_14", "rolling_mean_7", "rolling_mean_14", "rolling_mean_28",
    "rolling_std_7", "rolling_std_28", "sales_growth_7v28", "cv_28",
    # inventory
    "current_stock", "days_of_inventory", "inventory_to_demand_ratio", "reorder_gap",
    "below_reorder_flag", "received_last_7", "stockout_days_last_14", "stockout_days_last_28",
    "lead_time_days", "days_since_last_receipt", "last_receipt_cover", "cover_minus_lead",
    "projected_stock_at_lead", "projected_stock_7d",
    # price / promo
    "promotion_flag", "discount_pct", "promo_days_last_7", "price_change",
    # external
    "temp_c", "temp_mean_7", "rain_mm", "local_event",
    # store / product
    "avg_daily_customers", "floor_area_sqft", "shelf_life_days", "mrp", "margin_pct",
    "history_days",
]
CATEGORICAL_FEATURES = ["store_type", "category", "region"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

REG_TARGET = "next_7_day_demand"
CLF_TARGET = "stockout_next_7d"

# Features computed from each item's own sales history (need fallback for new items)
HISTORY_COLS = ["lag_1", "lag_7", "lag_14", "rolling_mean_7", "rolling_mean_14",
                "rolling_mean_28", "rolling_std_7", "rolling_std_28"]


def _future_calendar():
    """Count festivals / holidays / weekend days in the next 7 days per city."""
    e = pd.read_csv(RAW / "external_factors.csv")
    e["date"] = _parse_dates(e.date)
    e["city"] = e.city.str.strip().str.title()
    e = e.sort_values(["city", "date"])
    out = []
    for city, g in e.groupby("city"):
        g = g.set_index("date")
        fwd = lambda s: s[::-1].rolling(HORIZON, min_periods=1).sum()[::-1].shift(-1)
        out.append(pd.DataFrame({
            "city": city,
            "festival_next_7": fwd(g.festival),
            "holiday_next_7": fwd(g.holiday),
            "weekend_days_next_7": fwd(g.weekend),
        }).reset_index())
    return pd.concat(out)


def build_features(m: pd.DataFrame) -> pd.DataFrame:
    m = m.sort_values(KEY + ["date"]).copy()
    g = m.groupby(KEY)
    u = g.units_sold

    # ---------------- time ----------------
    m["day_of_week"] = m.date.dt.dayofweek
    m["weekend_flag"] = (m.day_of_week >= 5).astype(int)
    m["month"] = m.date.dt.month
    m["week_no"] = m.date.dt.isocalendar().week.astype(int)
    m["festival_flag"] = m.festival
    m["holiday_flag"] = m.holiday
    m = m.merge(_future_calendar(), on=["city", "date"], how="left")
    m = m.sort_values(KEY + ["date"])
    g = m.groupby(KEY)
    u = g.units_sold

    # ---------------- lag / rolling (window ends at t, inclusive) --------------
    m["history_days"] = g.cumcount() + 1
    m["lag_1"] = m.units_sold
    m["lag_7"] = u.shift(6)
    m["lag_14"] = u.shift(13)
    for w in (7, 14, 28):
        m[f"rolling_mean_{w}"] = u.transform(lambda x: x.rolling(w, min_periods=1).mean())
    m["rolling_std_7"] = u.transform(lambda x: x.rolling(7, min_periods=2).std())
    m["rolling_std_28"] = u.transform(lambda x: x.rolling(28, min_periods=2).std())

    # fallback for sparse history (new products):
    #  1) missing lags -> the item's own recent mean
    #  2) no usable history at all -> look-alike items (same store + sub-category, else category)
    m["lag_7"] = m.lag_7.fillna(m.rolling_mean_7)
    m["lag_14"] = m.lag_14.fillna(m.rolling_mean_14)
    sparse = m.history_days < 14
    mature = m[~sparse]
    for level in (["date", "store_id", "sub_category"], ["date", "store_id", "category"]):
        ref = mature.groupby(level)[["rolling_std_7", "rolling_std_28", "rolling_mean_7"]].mean()
        ref["cv7"] = ref.rolling_std_7 / ref.rolling_mean_7.clip(lower=0.1)
        ref["cv28"] = ref.rolling_std_28 / ref.rolling_mean_7.clip(lower=0.1)
        j = m[level].merge(ref[["cv7", "cv28"]], left_on=level, right_index=True, how="left")
        for col, cv in (("rolling_std_7", "cv7"), ("rolling_std_28", "cv28")):
            need = sparse & m[col].isna()
            m.loc[need, col] = (j.loc[need, cv] * m.loc[need, "rolling_mean_7"]).values
        # sparse items: blend 28-day mean with look-alike std-based estimate
    m["rolling_std_7"] = m.rolling_std_7.fillna(m.rolling_mean_7 * 0.4)
    m["rolling_std_28"] = m.rolling_std_28.fillna(m.rolling_std_7)
    m["sparse_history_flag"] = sparse.astype(int)

    m["sales_growth_7v28"] = m.rolling_mean_7 / m.rolling_mean_28.clip(lower=0.1) - 1
    m["cv_28"] = m.rolling_std_28 / m.rolling_mean_28.clip(lower=0.1)

    # ---------------- inventory ----------------
    m["current_stock"] = m.closing_stock
    daily = m.rolling_mean_7.clip(lower=0.1)
    m["days_of_inventory"] = (m.closing_stock / daily).clip(upper=120)
    m["inventory_to_demand_ratio"] = (m.closing_stock / (daily * HORIZON)).clip(upper=20)
    m["reorder_gap"] = m.closing_stock - m.reorder_level
    m["below_reorder_flag"] = (m.closing_stock <= m.reorder_level).astype(int)
    m["received_last_7"] = g.received_qty.transform(lambda x: x.rolling(7, min_periods=1).sum())
    m["stockout_days_last_14"] = g.stockout_flag.transform(
        lambda x: x.rolling(14, min_periods=1).sum())

    m["stockout_days_last_28"] = g.stockout_flag.transform(
        lambda x: x.rolling(28, min_periods=1).sum())
    # replenishment rhythm: how long since the last delivery and how big it was
    got = m.date.where(m.received_qty > 0)
    m["_last_rcv_date"] = got.groupby([m.store_id, m.product_id]).ffill()
    m["days_since_last_receipt"] = ((m.date - m._last_rcv_date).dt.days
                                    .fillna(m.history_days).clip(upper=60))
    last_qty = m.received_qty.where(m.received_qty > 0).groupby([m.store_id, m.product_id]).ffill()
    m["last_receipt_cover"] = (last_qty.fillna(0) / daily).clip(upper=60)
    # projected shelf position if nothing arrives
    m["cover_minus_lead"] = m.days_of_inventory - m.lead_time_days
    m["projected_stock_at_lead"] = m.closing_stock - m.rolling_mean_7 * m.lead_time_days
    m["projected_stock_7d"] = m.closing_stock - m.rolling_mean_7 * HORIZON
    m = m.drop(columns="_last_rcv_date")

    # ---------------- price / promo ----------------
    m["promo_days_last_7"] = g.promotion_flag.transform(lambda x: x.rolling(7, min_periods=1).sum())
    ref_price = g.avg_selling_price.transform(lambda x: x.rolling(28, min_periods=1).mean())
    m["price_change"] = m.avg_selling_price / ref_price - 1

    # ---------------- external ----------------
    m["temp_mean_7"] = g.temp_c.transform(lambda x: x.rolling(7, min_periods=1).mean())

    # ---------------- targets (future: t+1 .. t+7) ----------------
    fwd_sum = lambda x: x[::-1].rolling(HORIZON, min_periods=HORIZON).sum()[::-1].shift(-1)
    fwd_max = lambda x: x[::-1].rolling(HORIZON, min_periods=HORIZON).max()[::-1].shift(-1)
    m[REG_TARGET] = g.units_sold.transform(fwd_sum)
    m[CLF_TARGET] = g.stockout_flag.transform(fwd_max)

    # baseline forecast from the brief: previous 7-day mean x 7
    m["baseline_forecast"] = m.rolling_mean_7 * HORIZON
    return m.sort_values(["date"] + KEY).reset_index(drop=True)


def run():
    m = pd.read_csv(PROCESSED / "master_dataset.csv", parse_dates=["date"])
    f = build_features(m)
    f.to_csv(PROCESSED / "features.csv", index=False)
    lab = f.dropna(subset=[REG_TARGET])
    print("feature table:", f.shape, "| labelled rows:", len(lab))
    print("stockout_next_7d prevalence:", round(lab[CLF_TARGET].mean(), 3))
    print("NaN in features:", int(f[FEATURES].isna().sum().sum()))
    return f


if __name__ == "__main__":
    run()
