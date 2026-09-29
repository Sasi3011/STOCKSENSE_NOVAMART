"""Round 1 - Data cleaning & integration.

Produces:
  data/processed/master_dataset.csv   (grain: ONE DATE x ONE STORE x ONE PRODUCT)
  reports/data_quality_report.md      (issue, count, action, justification)

Run:  python -m src.data_cleaning
"""
import numpy as np
import pandas as pd

from src.config import RAW, PROCESSED, REPORTS

CATEGORY_MAP = {
    "beverage": "Beverages", "beverages": "Beverages",
    "dairy": "Dairy",
    "snack": "Snacks", "snacks": "Snacks",
    "personal care": "Personal Care", "personal-care": "Personal Care",
    "household": "Household",
    "frozen": "Frozen",
    "grocery": "Groceries", "groceries": "Groceries",
}


class QualityLog:
    """Collects one row per data-quality issue for the report."""

    def __init__(self):
        self.rows = []

    def add(self, table, issue, count, action, why):
        self.rows.append(dict(table=table, issue=issue, count=int(count),
                              action=action, justification=why))

    def frame(self):
        return pd.DataFrame(self.rows)


def _parse_dates(s: pd.Series) -> pd.Series:
    """Accept both ISO (2026-08-01) and day-first (01-08-2026) formats."""
    iso = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    dmy = pd.to_datetime(s, format="%d-%m-%Y", errors="coerce")
    return iso.fillna(dmy)


def _clean_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()


# --------------------------------------------------------------------------- #
def clean_products(log):
    p = pd.read_csv(RAW / "products.csv")
    raw_cats = p.category.nunique()
    p["product_id"] = _clean_id(p.product_id)
    p["category"] = (p.category.str.strip().str.lower().map(CATEGORY_MAP)
                     .fillna(p.category.str.strip().str.title()))
    log.add("products", "Category spelling/case inconsistency "
            f"({raw_cats} raw labels -> {p.category.nunique()})",
            raw_cats - p.category.nunique(), "Mapped to 7 canonical categories",
            "Groupings like 'beverage' vs 'BEVERAGES' would split revenue and model encodings.")
    miss = p.shelf_life_days.isna().sum()
    p["shelf_life_days"] = p.shelf_life_days.fillna(
        p.groupby("sub_category").shelf_life_days.transform("median")).fillna(
        p.groupby("category").shelf_life_days.transform("median"))
    log.add("products", "Missing shelf_life_days", miss,
            "Imputed with sub-category (else category) median",
            "Shelf life is a stable product attribute; siblings are a reliable proxy.")
    p["margin_pct"] = (p.mrp - p.cost_price) / p.mrp * 100
    return p


def clean_stores(log):
    s = pd.read_csv(RAW / "stores.csv")
    bad = (s.city != s.city.str.strip().str.title()).sum()
    s["store_id"] = _clean_id(s.store_id)
    s["city"] = s.city.str.strip().str.title()
    log.add("stores", "City name whitespace/case issues", bad,
            "Trimmed and title-cased", "City is the join key to external_factors.csv.")
    return s


def clean_external(log):
    e = pd.read_csv(RAW / "external_factors.csv")
    e["date"] = _parse_dates(e.date)
    e["city"] = e.city.str.strip().str.title()
    e = e.sort_values(["city", "date"])
    mt, mr = e.temp_c.isna().sum(), e.rain_mm.isna().sum()
    e["temp_c"] = e.groupby("city").temp_c.transform(
        lambda x: x.interpolate(limit_direction="both"))
    e["rain_mm"] = e.rain_mm.fillna(0.0)
    log.add("external_factors", "Missing temperature", mt,
            "Linear interpolation within city",
            "Temperature changes smoothly day to day; neighbours are the best estimate.")
    log.add("external_factors", "Missing rainfall", mr, "Filled with 0 mm",
            "Most days are dry (median = 0); a missing reading most likely means no rain.")
    return e


def clean_transactions(log, products):
    t = pd.read_csv(RAW / "transactions.csv")
    n0 = len(t)

    dups = t.duplicated().sum()
    t = t.drop_duplicates()
    id_dups = t.transaction_id.duplicated().sum()
    t = t.drop_duplicates("transaction_id")
    log.add("transactions", "Duplicate rows (same transaction_id repeated)", dups + id_dups,
            "Removed, kept first occurrence",
            "A repeated transaction_id is a double-posting; keeping it inflates demand.")

    bad_ids = (~t.store_id.astype(str).str.fullmatch(r"S\d{2}")).sum()
    t["store_id"], t["product_id"] = _clean_id(t.store_id), _clean_id(t.product_id)
    log.add("transactions", "Store IDs with lower-case / trailing spaces", bad_ids,
            "Trimmed and upper-cased", "Otherwise 's01 ' and 'S01' become two stores.")

    raw_dates = t.date.astype(str)
    alt = (~raw_dates.str.match(r"^\d{4}-\d{2}-\d{2}$")).sum()
    t["date"] = _parse_dates(t.date)
    log.add("transactions", "Non-ISO date format (dd-mm-yyyy)", alt,
            "Parsed both formats into one datetime", "Mixed formats silently mis-sort time series.")

    pm = (~t.payment_mode.isin(["UPI", "Card", "Cash"])).sum()
    t["payment_mode"] = t.payment_mode.str.strip().str.upper().replace({"CARD": "Card", "CASH": "Cash"})
    log.add("transactions", "Payment mode case inconsistency", pm, "Standardised to UPI/Card/Cash",
            "Needed for clean payment-mix reporting.")

    neg = (t.quantity <= 0).sum()
    t = t[t.quantity > 0]
    log.add("transactions", "Impossible quantity (<= 0)", neg,
            "Removed from demand, logged",
            "Negative sale quantities are returns/keying errors, not customer demand.")

    # keying errors: quantity far beyond anything plausible for one basket
    cap = t.groupby("product_id").quantity.transform(lambda x: x.quantile(0.999))
    extreme = (t.quantity > np.maximum(cap * 5, 20)).sum()
    t = t[~(t.quantity > np.maximum(cap * 5, 20))]
    log.add("transactions", "Extreme basket quantity (e.g. 150-400 units)", extreme,
            "Removed (quantity > 5x product 99.9th percentile)",
            "Inventory sold_qty for these days does not support them; clear keying errors.")

    mp = t.selling_price.isna().sum()
    mrp = t.product_id.map(products.set_index("product_id").mrp)
    t["selling_price"] = t.selling_price.fillna((mrp * (1 - t.discount_pct / 100)).round(2))
    log.add("transactions", "Missing selling_price", mp,
            "Imputed as MRP x (1 - discount_pct)", "Matches the pricing rule seen in valid rows.")
    log.add("transactions", "Rows retained", len(t), f"{len(t)/n0:.1%} of raw rows kept", "")
    return t


def clean_inventory(log):
    inv = pd.read_csv(RAW / "inventory.csv")
    # accept the short headers shown in the brief's sample table as well as full names
    inv = inv.rename(columns={"store": "store_id", "product": "product_id", "opening": "opening_stock",
                              "received": "received_qty", "sold": "sold_qty", "closing": "closing_stock",
                              "reorder_lvl": "reorder_level", "lead_days": "lead_time_days"})
    inv["date"] = _parse_dates(inv.date)
    inv["store_id"], inv["product_id"] = _clean_id(inv.store_id), _clean_id(inv.product_id)
    d = inv.duplicated(["date", "store_id", "product_id"]).sum()
    inv = inv.drop_duplicates(["date", "store_id", "product_id"])
    log.add("inventory", "Duplicate date x store x product rows", d, "Removed",
            "The grain must be unique before merging.")

    mr = inv.reorder_level.isna().sum()
    inv["reorder_level"] = inv.reorder_level.fillna(
        inv.groupby(["store_id", "product_id"]).reorder_level.transform("median"))
    log.add("inventory", "Missing reorder_level", mr,
            "Filled with the store x product median", "Reorder levels are fixed policy values.")

    expected = inv.opening_stock + inv.received_qty - inv.sold_qty
    mism = expected != inv.closing_stock
    inv["inventory_mismatch_flag"] = mism.astype(int)
    inv["recorded_closing_stock"] = inv.closing_stock
    inv.loc[mism, "closing_stock"] = expected[mism].clip(lower=0)
    log.add("inventory", "closing != opening + received - sold", mism.sum(),
            "Recomputed closing from the equation and flagged",
            "Opening/received/sold reconcile with the next day's opening; the closing count "
            "is the field most likely mis-keyed.")
    return inv


# --------------------------------------------------------------------------- #
def build_master(tx, inv, products, stores, ext):
    daily = (tx.assign(revenue=tx.quantity * tx.selling_price)
               .groupby(["date", "store_id", "product_id"])
               .agg(units_sold=("quantity", "sum"), revenue=("revenue", "sum"),
                    n_transactions=("transaction_id", "count"),
                    avg_selling_price=("selling_price", "mean"),
                    discount_pct=("discount_pct", "max"),
                    promotion_flag=("promotion_flag", "max"))
               .reset_index())

    m = inv.merge(daily, on=["date", "store_id", "product_id"], how="left")
    zero = m.units_sold.isna()
    m.loc[zero, ["units_sold", "revenue", "n_transactions"]] = 0
    m = m.merge(products, on="product_id", how="left").merge(stores, on="store_id", how="left")

    # promotion / price on zero-sale days: carry within store-product
    m = m.sort_values(["store_id", "product_id", "date"])
    g = m.groupby(["store_id", "product_id"])
    m["avg_selling_price"] = g.avg_selling_price.transform(lambda x: x.ffill().bfill())
    m["avg_selling_price"] = m.avg_selling_price.fillna(m.mrp)
    m["discount_pct"] = m.discount_pct.fillna(0)
    m["promotion_flag"] = m.promotion_flag.fillna(0).astype(int)

    m = m.merge(ext, on=["date", "city"], how="left")

    # reconciliation between POS and inventory system
    m["pos_vs_inventory_gap"] = m.sold_qty - m.units_sold

    # operational stock-out definition
    m["stockout_flag"] = (m.closing_stock <= 0).astype(int)
    m["inventory_value"] = m.closing_stock * m.cost_price
    m["cogs"] = m.units_sold * m.cost_price
    return m.sort_values(["date", "store_id", "product_id"]).reset_index(drop=True)


def write_quality_report(log, master):
    df = log.frame()
    lines = ["# Data Quality Report - StockSense (NovaMart)", "",
             "Every issue found while auditing the five raw files, how many records were "
             "affected, what we did and why.", "",
             "| Table | Issue | Count | Action taken | Justification |",
             "|---|---|---:|---|---|"]
    for r in df.itertuples():
        lines.append(f"| {r.table} | {r.issue} | {r.count:,} | {r.action} | {r.justification} |")
    new = master.groupby("product_id").date.nunique()
    sparse = new[new < 14]
    lines += ["", "## Sparse history (new products)", "",
              f"{len(sparse)} products have fewer than 14 days of history: "
              + ", ".join(f"{p} ({d} days)" for p, d in sparse.items()) + ".",
              "They are excluded from model training (lags/rolling windows are undefined) but are "
              "still scored. For scoring, missing lag and rolling features are filled with the "
              "average of the same sub-category in the same store on the same date "
              "(a 'look-alike product' fallback). Their forecasts carry lower confidence and are "
              "marked in the manager output.", "",
              "## Known limitation: censored demand", "",
              "Sales are capped by available stock. On stock-out days the true demand is higher "
              "than recorded sales, so the forecast target slightly under-states demand for "
              "items that stock out often. The safety-stock term in the reorder logic partly "
              "offsets this.", "",
              "## Master dataset", "",
              f"- Grain: one row = one date x one store x one product",
              f"- Rows: {len(master):,} | Columns: {master.shape[1]}",
              f"- Date range: {master.date.min():%d %b %Y} to {master.date.max():%d %b %Y}",
              f"- Stores: {master.store_id.nunique()} | Products: {master.product_id.nunique()}",
              f"- Stock-out rate (closing stock = 0): {master.stockout_flag.mean():.1%}",
              f"- Remaining missing values: {int(master.isna().sum().sum())}"]
    (REPORTS / "data_quality_report.md").write_text("\n".join(lines))
    df.to_csv(REPORTS / "data_quality_log.csv", index=False)


def run():
    log = QualityLog()
    products = clean_products(log)
    stores = clean_stores(log)
    ext = clean_external(log)
    tx = clean_transactions(log, products)
    inv = clean_inventory(log)
    master = build_master(tx, inv, products, stores, ext)
    master.to_csv(PROCESSED / "master_dataset.csv", index=False)
    tx.to_csv(PROCESSED / "transactions_clean.csv", index=False)
    write_quality_report(log, master)
    print(log.frame()[["table", "issue", "count"]].to_string(index=False))
    print("master:", master.shape)
    return master


if __name__ == "__main__":
    run()
