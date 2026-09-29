"""Round 3 - Intelligence layer.

Turns the two models into a manager action list for the most recent date:
  forecast -> stock-out probability -> risk band -> reasons -> reorder qty -> action

Explanations use *driver occlusion*: for each business driver we reset its
features to a neutral value (no promotion, no festival, normal stock cover...)
and measure how much the prediction changes. This is permutation-importance
logic applied to one row, and reads naturally: "Promotion running: +18 pts".

Run:  python -m src.recommend
"""
import json
from itertools import product as iproduct

import joblib
import numpy as np
import pandas as pd

from src.config import (HIGH_RISK, HORIZON, MEDIUM_RISK, MODELS, PERISHABLE_SHELF_LIFE,
                        PROCESSED, REPORTS, SERVICE_Z)
from src.features import FEATURES

CASE_PACK = {"Beverages": 12, "Snacks": 12, "Dairy": 6, "Groceries": 6,
             "Personal Care": 6, "Household": 6, "Frozen": 6}


# --------------------------------------------------------------------------- #
def load_models():
    reg = joblib.load(MODELS / "demand_model.joblib")
    clf = joblib.load(MODELS / "stockout_model.joblib")
    return reg, clf


def predict(X, reg, clf):
    f = np.clip(reg.predict(X[FEATURES]), 0, None)
    raw = clf["model"].predict_proba(X[FEATURES])[:, 1]
    return f, np.clip(clf["calibrator"].predict(raw), 0, 1)


def risk_band(p):
    return np.where(p >= HIGH_RISK, "HIGH", np.where(p >= MEDIUM_RISK, "MEDIUM", "LOW"))


def _recompute_inventory(X):
    d = X.rolling_mean_7.clip(lower=0.1)
    X["days_of_inventory"] = (X.current_stock / d).clip(upper=120)
    X["inventory_to_demand_ratio"] = (X.current_stock / (d * HORIZON)).clip(upper=20)
    X["reorder_gap"] = X.current_stock - X.reorder_level
    X["below_reorder_flag"] = (X.current_stock <= X.reorder_level).astype(int)
    X["cover_minus_lead"] = X.days_of_inventory - X.lead_time_days
    X["projected_stock_at_lead"] = X.current_stock - X.rolling_mean_7 * X.lead_time_days
    X["projected_stock_7d"] = X.current_stock - X.rolling_mean_7 * HORIZON
    return X


# --------------------------------------------------------------------------- #
# Drivers: manager-readable name -> function that neutralises it
# --------------------------------------------------------------------------- #
def _neutral_promo(X, ref):
    X[["promotion_flag", "discount_pct", "promo_days_last_7", "price_change"]] = 0
    return X


def _neutral_festival(X, ref):
    X[["festival_next_7", "holiday_next_7", "festival_flag", "holiday_flag"]] = 0
    return X


def _neutral_growth(X, ref):
    base = X.rolling_mean_28
    for c in ("lag_1", "lag_7", "rolling_mean_7", "rolling_mean_14"):
        X[c] = base
    X["sales_growth_7v28"] = 0
    return _recompute_inventory(X)


def _neutral_stock(X, ref):
    X["current_stock"] = X.rolling_mean_7 * ref["days_of_inventory"]
    return _recompute_inventory(X)


def _neutral_history(X, ref):
    X[["stockout_days_last_14", "stockout_days_last_28"]] = 0
    return X


def _neutral_lead(X, ref):
    X["lead_time_days"] = ref["lead_time_days"]
    return _recompute_inventory(X)


def _neutral_weather(X, ref):
    X["temp_c"] = ref["temp_c"]
    X["temp_mean_7"] = ref["temp_c"]
    return X


def _neutral_volatility(X, ref):
    X["rolling_std_7"] = X.rolling_mean_7 * ref["cv"]
    X["rolling_std_28"] = X.rolling_mean_28 * ref["cv"]
    X["cv_28"] = ref["cv"]
    return X


def _neutral_delivery(X, ref):
    X["days_since_last_receipt"] = ref["days_since_last_receipt"]
    X["last_receipt_cover"] = ref["last_receipt_cover"]
    X["received_last_7"] = X.rolling_mean_7 * ref["received_ratio"]
    return X


def _neutral_event(X, ref):
    X["local_event"] = 0
    return X


DRIVERS = {
    "Low stock cover vs demand": _neutral_stock,
    "Promotion running": _neutral_promo,
    "Festival / holiday in next 7 days": _neutral_festival,
    "Recent sales growth": _neutral_growth,
    "Repeated recent stock-outs": _neutral_history,
    "Long supplier lead time": _neutral_lead,
    "Hot weather": _neutral_weather,
    "Unpredictable demand": _neutral_volatility,
    "No recent delivery": _neutral_delivery,
    "Local event nearby": _neutral_event,
}


def reference_values(feat):
    ref = feat[feat.history_days >= 14]
    return {
        "days_of_inventory": float(ref.days_of_inventory.median()),
        "lead_time_days": float(ref.lead_time_days.median()),
        "temp_c": 30.0,
        "cv": float((ref.rolling_std_28 / ref.rolling_mean_28.clip(lower=0.1)).median()),
        "days_since_last_receipt": float(ref.days_since_last_receipt.median()),
        "last_receipt_cover": float(ref.last_receipt_cover.median()),
        "received_ratio": float((ref.received_last_7 / ref.rolling_mean_7.clip(lower=0.1)).median()),
    }


def explain(X, reg, clf, ref):
    """Per-row driver impact on risk (probability points) and on demand (%)."""
    f0, p0 = predict(X, reg, clf)
    risk_imp, dem_imp = {}, {}
    for name, fn in DRIVERS.items():
        Xn = fn(X.copy(), ref)
        f1, p1 = predict(Xn, reg, clf)
        risk_imp[name] = (p0 - p1) * 100
        dem_imp[name] = np.where(f1 > 0.5, (f0 - f1) / np.maximum(f1, 0.5) * 100, 0)
    return pd.DataFrame(risk_imp, index=X.index), pd.DataFrame(dem_imp, index=X.index)


# --------------------------------------------------------------------------- #
def reorder_logic(df, incoming=None):
    """Brief's rule: Recommended stock = forecast + safety stock;
    Reorder = max(0, Recommended - Current - Incoming), with two refinements:
    perishables are capped by shelf life and orders are rounded up to case packs."""
    df = df.copy()
    df["incoming_stock"] = 0 if incoming is None else incoming
    z = df.risk_level.map(SERVICE_Z)
    sigma = np.maximum(df.rolling_std_7, np.sqrt(df.forecast_7d / HORIZON))
    df["safety_stock"] = np.ceil(z * sigma * np.sqrt(df.lead_time_days))
    df["recommended_stock"] = np.ceil(df.forecast_7d + df.safety_stock)
    perish = df.shelf_life_days <= PERISHABLE_SHELF_LIFE
    cap = np.ceil(df.forecast_7d / HORIZON * df.shelf_life_days + df.safety_stock)
    df["perishable_cap_applied"] = (perish & (cap < df.recommended_stock)).astype(int)
    df.loc[perish, "recommended_stock"] = np.minimum(df.loc[perish, "recommended_stock"],
                                                     cap[perish])
    raw = (df.recommended_stock - df.current_stock - df.incoming_stock).clip(lower=0)
    pack = df.category.map(CASE_PACK).fillna(6)
    df["reorder_qty"] = np.where(raw > 0, np.ceil(raw / pack) * pack, 0).astype(int)
    df["case_pack"] = pack.astype(int)
    daily_fc = (df.forecast_7d / HORIZON).clip(lower=0.1)
    df["days_of_cover"] = (df.current_stock / daily_fc).round(1)
    shortfall = (df.forecast_7d - df.current_stock - df.incoming_stock).clip(lower=0)
    df["revenue_at_risk"] = (shortfall * df.avg_selling_price * df.stockout_probability).round(0)
    # overstock: stock beyond two ordering cycles (14 days) of forecast demand, or beyond shelf life
    keep_days = np.minimum(14, df.shelf_life_days)
    df["excess_units"] = (df.current_stock - daily_fc * keep_days).clip(lower=0).round(0)
    df["excess_value"] = (df.excess_units * df.cost_price).round(0)
    return df


def action_text(r):
    lead = int(r.lead_time_days)
    if r.risk_level == "HIGH":
        a = "Raise replenishment order today"
        if lead >= 3:
            a += f" - {r.supplier_id} needs {lead} days, ask for express delivery"
        if r.perishable_cap_applied:
            a += "; split into daily drops (short shelf life)"
        return a
    if r.risk_level == "MEDIUM":
        return ("Order within 2 days and recheck the shelf tomorrow" if r.reorder_qty > 0
                else "Monitor daily - stock just covers the forecast")
    if r.excess_units > 0:
        if r.shelf_life_days <= 30 and r.days_of_cover > r.shelf_life_days:
            return "Expiry risk - move stock forward, run a markdown"
        return "Hold orders - overstocked; consider a promotion to release cash"
    if r.reorder_qty > 0:
        return "Add to routine order"
    return "No action needed"


DEMAND_DRIVERS = ["Promotion running", "Festival / holiday in next 7 days", "Recent sales growth",
                  "Hot weather", "Local event nearby"]


def apply_guardrails(X, prob):
    """Transparent business rules on top of the model.
    The model has learned that empty shelves are often refilled soon, but it cannot
    see open purchase orders. Until it can, two situations are always escalated."""
    prob = prob.copy()
    basis = np.array(["Model"] * len(X), dtype=object)
    daily = (X.forecast_7d / HORIZON).clip(lower=0.1)
    cover = X.current_stock / daily
    empty = (X.current_stock <= 0) & (X.forecast_7d >= 1)
    before = (~empty) & (cover < X.lead_time_days) & (X.forecast_7d >= 1)
    prob[empty] = np.maximum(prob[empty], 0.95)
    prob[before] = np.maximum(prob[before], HIGH_RISK)
    basis[empty] = "Rule: shelf already empty"
    basis[before & (basis == "Model")] = "Rule: runs out before a new order can arrive"
    return prob, basis


def reason_text(risk_row, dem_row, level, basis="Model", k=3):
    items = []
    if basis != "Model":
        items.append((basis.replace("Rule: ", "").capitalize(), "rule"))
    dem_row = dem_row[[c for c in DEMAND_DRIVERS if c in dem_row.index]]
    if level in ("HIGH", "MEDIUM"):
        top = risk_row[risk_row > 1.0].sort_values(ascending=False).head(k - len(items))
        items += [(n, f"+{v:.0f} pts risk") for n, v in top.items()]
    dtop = dem_row[dem_row > 3].sort_values(ascending=False)
    for n, v in dtop.items():
        if len(items) >= k:
            break
        if n not in [i[0] for i in items]:
            items.append((n, f"+{v:.0f}% demand"))
    return items


# --------------------------------------------------------------------------- #
def score_latest(feat=None, reg=None, clf=None, scenario=None, explain_rows=True):
    feat = feat if feat is not None else pd.read_csv(PROCESSED / "features.csv",
                                                     parse_dates=["date"])
    if reg is None:
        reg, clf = load_models()
    asof = feat.date.max()
    X = feat[feat.date == asof].copy().reset_index(drop=True)
    scenario = scenario or {}
    # demand reacts only to demand-side levers; supply delays change risk, not demand
    Xd = apply_scenario(X, {k: v for k, v in scenario.items() if k != "supplier_delay_days"})
    X = apply_scenario(X, scenario)
    fc = np.clip(reg.predict(Xd[FEATURES]), 0, None)
    _, prob = predict(X, reg, clf)
    fc = fc * (1 + scenario.get("demand_shock_pct", 0) / 100)
    X["forecast_7d"] = np.round(fc, 1)
    X["model_probability"] = np.round(prob, 4)
    prob, basis = apply_guardrails(X, prob)
    X["risk_basis"] = basis
    X["stockout_probability"] = np.round(prob, 4)
    X["risk_level"] = risk_band(prob)
    X = reorder_logic(X)
    X["manager_action"] = X.apply(action_text, axis=1)
    if explain_rows:
        ref = reference_values(feat)
        rimp, dimp = explain(X, reg, clf, ref)
        X["reasons"] = [json.dumps(reason_text(rimp.loc[i], dimp.loc[i], X.risk_level[i],
                                                X.risk_basis[i]))
                        for i in X.index]
        X["risk_drivers"] = [json.dumps({k: round(float(v), 1) for k, v in rimp.loc[i].items()})
                             for i in X.index]
        X["demand_drivers"] = [json.dumps({k: round(float(v), 1) for k, v in dimp.loc[i].items()})
                               for i in X.index]
    X["priority_score"] = X.revenue_at_risk + X.stockout_probability * 100
    X["as_of_date"] = asof.strftime("%Y-%m-%d")
    return X.sort_values(["priority_score"], ascending=False).reset_index(drop=True)


def apply_scenario(X, s):
    X = X.copy()
    if s.get("discount_pct", 0) > 0:
        X["promotion_flag"] = 1
        X["discount_pct"] = s["discount_pct"]
        X["promo_days_last_7"] = X.promo_days_last_7.clip(lower=1)
        # a discount is a price cut against the item's recent reference price
        X["price_change"] = -s["discount_pct"] / 100
    if s.get("supplier_delay_days", 0) > 0:
        X["lead_time_days"] = X.lead_time_days + s["supplier_delay_days"]
        X = _recompute_inventory(X)
    if s.get("festival", 0) > 0:
        X["festival_next_7"] = X.festival_next_7 + 1
    return X


OUTPUT_COLS = ["as_of_date", "store_id", "city", "product_id", "product_name", "category",
               "supplier_id", "current_stock", "incoming_stock", "forecast_7d",
               "model_probability", "stockout_probability", "risk_level", "risk_basis", "safety_stock", "recommended_stock",
               "reorder_qty", "case_pack", "days_of_cover", "lead_time_days", "shelf_life_days",
               "revenue_at_risk", "excess_units", "excess_value", "sparse_history_flag",
               "manager_action", "reasons", "risk_drivers", "demand_drivers"]


def run():
    feat = pd.read_csv(PROCESSED / "features.csv", parse_dates=["date"])
    reg, clf = load_models()
    rec = score_latest(feat, reg, clf)
    rec[OUTPUT_COLS].to_csv(PROCESSED / "recommendations.csv", index=False)

    # pre-computed what-if grid for the static HTML dashboard
    grid = []
    for disc, delay, fest in iproduct([0, 10, 20], [0, 2], [0, 1]):
        s = dict(discount_pct=disc, supplier_delay_days=delay, festival=fest)
        r = score_latest(feat, reg, clf, scenario=s, explain_rows=False)
        r = r.assign(discount_pct_s=disc, supplier_delay_s=delay, festival_s=fest)
        grid.append(r[["store_id", "product_id", "forecast_7d", "stockout_probability",
                       "risk_level", "reorder_qty", "discount_pct_s", "supplier_delay_s",
                       "festival_s"]])
    pd.concat(grid).to_csv(PROCESSED / "whatif_grid.csv", index=False)

    c = rec.risk_level.value_counts()
    print(f"as of {rec.as_of_date[0]}: HIGH {c.get('HIGH', 0)} | MEDIUM {c.get('MEDIUM', 0)} | "
          f"LOW {c.get('LOW', 0)} | units to order {rec.reorder_qty.sum():,}")
    print(rec[["store_id", "product_name", "current_stock", "forecast_7d",
               "stockout_probability", "risk_level", "reorder_qty", "manager_action"]]
          .head(10).to_string(index=False))
    print(rec.reasons.head(5).tolist())
    return rec


if __name__ == "__main__":
    run()
