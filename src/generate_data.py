"""Generate NovaMart raw data that follows the exact schemas in the IntelliData
2026 brief, including the six intentional data-quality traps.

If the organisers provide real CSVs, drop them into data/raw/ and skip this
script - everything downstream reads only from data/raw/.

Run:  python -m src.generate_data
"""
import numpy as np
import pandas as pd

from src.config import RAW, RANDOM_STATE

rng = np.random.default_rng(RANDOM_STATE)

START, END = pd.Timestamp("2026-04-01"), pd.Timestamp("2026-08-31")
EXT_END = pd.Timestamp("2026-09-07")          # weather/calendar known 7 days ahead
DATES = pd.date_range(START, END, freq="D")

# --------------------------------------------------------------------------- #
# Stores
# --------------------------------------------------------------------------- #
STORES = pd.DataFrame([
    ["S01", "Coimbatore", "Supermarket", 8500, 1250, "West", 1.00, 0.0008],
    ["S02", "Chennai", "Hypermarket", 18000, 2850, "North", 2.10, 0.0022],
    ["S03", "Madurai", "Express", 4200, 720, "South", 0.60, -0.0012],
    ["S04", "Salem", "Supermarket", 7600, 1080, "Central", 0.90, 0.0004],
    ["S05", "Tiruppur", "Express", 3900, 650, "West", 0.55, 0.0015],
    ["S06", "Trichy", "Supermarket", 8100, 1150, "Central", 0.95, -0.0004],
], columns=["store_id", "city", "store_type", "floor_area_sqft",
            "avg_daily_customers", "region", "_scale", "_trend"])

# --------------------------------------------------------------------------- #
# Products  (id, category, sub_category, brand, name, mrp, cost, shelf, supplier,
#            base daily units, weather sensitivity, launch date)
# --------------------------------------------------------------------------- #
P = [
    ("P101", "Dairy", "Milk", "Aavin", "Milk 1L", 60, 52, 3, "SUP01", 42, 0.0),
    ("P102", "Dairy", "Curd", "Aavin", "Curd 500g", 40, 31, 5, "SUP01", 22, 0.4),
    ("P103", "Dairy", "Paneer", "Milky Mist", "Paneer 200g", 95, 72, 10, "SUP02", 9, 0.0),
    ("P104", "Dairy", "Butter", "Amul", "Butter 100g", 58, 47, 180, "SUP02", 8, 0.0),
    ("P105", "Dairy", "Cheese", "Amul", "Cheese Slices 200g", 145, 112, 180, "SUP02", 4, 0.0),
    ("P106", "Dairy", "Ghee", "Aavin", "Ghee 500ml", 340, 285, 270, "SUP01", 3, 0.0),
    ("P110", "Groceries", "Bread", "Britannia", "Bread 400g", 45, 34, 4, "SUP03", 30, 0.0),
    ("P111", "Groceries", "Rice", "India Gate", "Basmati Rice 1kg", 165, 128, 540, "SUP05", 10, 0.0),
    ("P112", "Groceries", "Atta", "Aashirvaad", "Atta 5kg", 285, 238, 120, "SUP05", 7, 0.0),
    ("P113", "Groceries", "Oil", "Gold Winner", "Sunflower Oil 1L", 155, 131, 270, "SUP05", 11, 0.0),
    ("P114", "Groceries", "Dal", "Tata Sampann", "Toor Dal 1kg", 180, 149, 365, "SUP05", 8, 0.0),
    ("P115", "Groceries", "Sugar", "Parrys", "Sugar 1kg", 52, 44, 365, "SUP05", 12, 0.0),
    ("P116", "Groceries", "Eggs", "Suguna", "Eggs 6pc", 48, 38, 14, "SUP03", 18, 0.0),
    ("P117", "Groceries", "Salt", "Tata", "Salt 1kg", 28, 21, 730, "SUP05", 6, 0.0),
    ("P330", "Beverages", "Soft Drink", "FizzUp", "Cola 750ml", 50, 31, 180, "SUP04", 20, 1.0),
    ("P331", "Beverages", "Soft Drink", "FizzUp", "Lemon Soda 600ml", 40, 25, 180, "SUP04", 12, 1.0),
    ("P332", "Beverages", "Juice", "Real", "Mango Juice 1L", 110, 78, 240, "SUP04", 8, 0.8),
    ("P333", "Beverages", "Water", "Aquafina", "Water 1L", 20, 11, 365, "SUP04", 28, 0.9),
    ("P334", "Beverages", "Tea", "3 Roses", "Tea 250g", 145, 112, 365, "SUP07", 7, -0.3),
    ("P335", "Beverages", "Coffee", "Narasus", "Filter Coffee 200g", 135, 101, 365, "SUP07", 6, -0.3),
    ("P336", "Beverages", "Energy", "Voltz", "Energy Drink 250ml", 99, 64, 365, "SUP04", 5, 0.6),
    ("P440", "Snacks", "Chips", "Crunchy", "Potato Chips 90g", 30, 19, 120, "SUP06", 16, 0.0),
    ("P441", "Snacks", "Namkeen", "Haldiram", "Mixture 200g", 55, 38, 150, "SUP06", 10, 0.0),
    ("P442", "Snacks", "Biscuits", "Crispo", "Cream Biscuits 150g", 35, 22, 240, "SUP06", 18, 0.0),
    ("P443", "Snacks", "Chocolate", "Cocoa Bay", "Milk Chocolate 50g", 45, 30, 300, "SUP06", 11, -0.2),
    ("P444", "Snacks", "Noodles", "Yippee", "Instant Noodles 4pk", 56, 42, 270, "SUP06", 13, 0.0),
    ("P445", "Snacks", "Cookies", "Crispo", "Butter Cookies 200g", 60, 40, 240, "SUP06", 7, 0.0),
    ("P205", "Personal Care", "Shampoo", "GlowCare", "Shampoo 180ml", 120, 79, 730, "SUP08", 5, 0.0),
    ("P206", "Personal Care", "Soap", "Medimix", "Bath Soap 4pk", 140, 98, 730, "SUP08", 7, 0.0),
    ("P207", "Personal Care", "Toothpaste", "Colgate", "Toothpaste 150g", 99, 70, 540, "SUP08", 8, 0.0),
    ("P208", "Personal Care", "Hair Oil", "Parachute", "Coconut Hair Oil 200ml", 105, 74, 540, "SUP08", 5, 0.0),
    ("P209", "Personal Care", "Face Wash", "GlowCare", "Face Wash 100ml", 150, 96, 540, "SUP08", 3, 0.2),
    ("P210", "Personal Care", "Deodorant", "Zest", "Deodorant 150ml", 199, 128, 730, "SUP08", 2, 0.3),
    ("P520", "Household", "Detergent", "Surf", "Detergent 1kg", 145, 112, 730, "SUP09", 8, 0.0),
    ("P521", "Household", "Dishwash", "Vim", "Dishwash Bar 3pk", 60, 44, 730, "SUP09", 9, 0.0),
    ("P522", "Household", "Floor Cleaner", "Lizol", "Floor Cleaner 1L", 210, 158, 730, "SUP09", 4, 0.0),
    ("P523", "Household", "Tissue", "Origami", "Tissue Roll 4pk", 180, 126, 1095, "SUP09", 3, 0.0),
    ("P524", "Household", "Agarbatti", "Cycle", "Agarbatti 100g", 65, 41, 1095, "SUP09", 6, 0.0),
    ("P525", "Household", "Mosquito", "GoodKnight", "Mosquito Refill", 85, 61, 730, "SUP09", 4, -0.4),
    ("P630", "Frozen", "Ice Cream", "Arun", "Vanilla Ice Cream 1L", 190, 128, 180, "SUP10", 6, 1.2),
    ("P631", "Frozen", "Ice Cream", "Arun", "Choco Bar", 30, 18, 180, "SUP10", 14, 1.3),
    ("P632", "Frozen", "Frozen Veg", "Safal", "Green Peas 500g", 95, 70, 365, "SUP10", 4, 0.0),
    ("P633", "Frozen", "Ready Snack", "McCain", "French Fries 420g", 145, 104, 365, "SUP10", 3, 0.0),
    ("P634", "Frozen", "Frozen Meat", "Licious", "Chicken Nuggets 250g", 199, 150, 180, "SUP10", 3, 0.0),
    ("P635", "Frozen", "Kulfi", "Arun", "Kulfi Stick", 25, 15, 180, "SUP10", 8, 1.1),
    # New products launched late -> sparse-history trap
    ("P160", "Beverages", "Juice", "Real", "Guava Juice 1L", 110, 79, 240, "SUP04", 7, 0.7),
    ("P161", "Snacks", "Protein Bar", "Yoga Bar", "Protein Bar 60g", 80, 52, 180, "SUP06", 6, 0.0),
]
PRODUCTS = pd.DataFrame(P, columns=["product_id", "category", "sub_category", "brand",
                                    "product_name", "mrp", "cost_price", "shelf_life_days",
                                    "supplier_id", "_base", "_weather"])
PRODUCTS["_launch"] = START
PRODUCTS.loc[PRODUCTS.product_id.isin(["P160", "P161"]), "_launch"] = pd.Timestamp("2026-08-27")

SUPPLIER_LEAD = {"SUP01": 1, "SUP02": 2, "SUP03": 1, "SUP04": 3, "SUP05": 4,
                 "SUP06": 3, "SUP07": 5, "SUP08": 4, "SUP09": 5, "SUP10": 2}

# --------------------------------------------------------------------------- #
# External factors  (Tamil Nadu calendar, summer -> SW monsoon)
# --------------------------------------------------------------------------- #
HOLIDAYS = {"2026-04-03", "2026-04-14", "2026-05-01", "2026-06-26", "2026-08-15"}
FESTIVALS_ALL = {"2026-04-14", "2026-08-03", "2026-09-04"}   # Puthandu, Aadi Perukku, Krishna Jayanthi
FESTIVALS_CITY = {("Madurai", "2026-04-29"), ("Madurai", "2026-08-25"),
                  ("Coimbatore", "2026-08-26"), ("Trichy", "2026-05-12"),
                  ("Chennai", "2026-07-17"), ("Salem", "2026-07-24"),
                  ("Tiruppur", "2026-06-05")}
CITY_TEMP = {"Coimbatore": -2.5, "Chennai": 2.0, "Madurai": 2.5, "Salem": 0.5,
             "Tiruppur": -1.0, "Trichy": 2.2}


def make_external():
    rows = []
    for city, off in CITY_TEMP.items():
        for d in pd.date_range(START, EXT_END):
            doy = d.dayofyear
            base = 33.5 - 4.5 * np.clip((doy - 135) / 60, 0, 1)   # hot Apr-May, cooler Jun-Aug
            temp = round(base + off + rng.normal(0, 1.1), 1)
            monsoon = 1 if 150 <= doy <= 250 else 0
            rain_p = 0.12 + (0.25 if monsoon else 0) + (0.15 if city == "Coimbatore" else 0)
            rain = round(rng.gamma(1.6, 6.0), 1) if rng.random() < rain_p else 0.0
            ds = d.strftime("%Y-%m-%d")
            rows.append([ds, city, temp, rain, int(ds in HOLIDAYS),
                         int(ds in FESTIVALS_ALL or (city, ds) in FESTIVALS_CITY),
                         int(d.dayofweek >= 5),
                         int(rng.random() < 0.06)])
    ext = pd.DataFrame(rows, columns=["date", "city", "temp_c", "rain_mm", "holiday",
                                      "festival", "weekend", "local_event"])
    return ext


# --------------------------------------------------------------------------- #
# Promotion calendar: multi-day windows per store x product
# --------------------------------------------------------------------------- #
def make_promos():
    promo = {}
    for s in STORES.store_id:
        for p in PRODUCTS.product_id:
            flags = np.zeros(len(DATES), dtype=int)
            disc = np.zeros(len(DATES))
            i = int(rng.integers(0, 20))
            while i < len(DATES):
                if rng.random() < 0.30:
                    L = int(rng.integers(3, 8))
                    d = float(rng.choice([5, 10, 15, 20, 25], p=[.2, .3, .25, .15, .1]))
                    flags[i:i + L] = 1
                    disc[i:i + L] = d
                    i += L + int(rng.integers(10, 25))
                else:
                    i += int(rng.integers(7, 18))
            promo[(s, p)] = (flags, disc)
    return promo


# --------------------------------------------------------------------------- #
# Demand + inventory simulation
# --------------------------------------------------------------------------- #
def simulate(ext, promo):
    ext_idx = ext.assign(date=pd.to_datetime(ext.date)).set_index(["city", "date"])
    inv_rows, sales = [], []
    for _, st in STORES.iterrows():
        e = ext_idx.loc[st.city].reindex(DATES)
        wk_boost = 1.18 if st.store_type != "Express" else 1.08
        for _, pr in PRODUCTS.iterrows():
            flags, disc = promo[(st.store_id, pr.product_id)]
            prod_noise = rng.normal(1, 0.15)
            mean = pr._base * st._scale * prod_noise
            lead = SUPPLIER_LEAD[pr.supplier_id]
            reorder_level = int(np.ceil(mean * (lead + 1.2)))
            if pr.shelf_life_days <= 14:
                cover = rng.uniform(1.5, min(4.0, pr.shelf_life_days))
            else:
                cover = rng.uniform(6, 26)          # some items are clearly overstocked
            order_up_to = int(np.ceil(reorder_level + mean * cover))
            stock = int(order_up_to * rng.uniform(0.7, 1.0))
            pipeline = {}           # arrival_day_index -> qty
            launched = False
            for t, d in enumerate(DATES):
                if d < pr._launch:
                    continue
                if not launched:
                    launched = True
                    stock = int(mean * 6)
                t_rel = (d - START).days
                mu = mean * (1 + st._trend) ** t_rel
                if d.dayofweek >= 5:
                    mu *= wk_boost
                if flags[t]:
                    mu *= 1 + 0.9 * disc[t] / 100 + 0.20
                if e.festival.iloc[t]:
                    mu *= 1.45 if pr.category in ("Dairy", "Groceries", "Snacks", "Beverages") else 1.15
                if e.holiday.iloc[t]:
                    mu *= 1.15
                if e.local_event.iloc[t]:
                    mu *= 1.12
                if e.rain_mm.iloc[t] > 8:
                    mu *= 0.88
                mu *= 1 + pr._weather * 0.045 * (e.temp_c.iloc[t] - 30)
                mu = max(mu, 0.2)
                k = 6.0
                demand = rng.negative_binomial(k, k / (k + mu))

                opening = stock
                received = pipeline.pop(t, 0)
                available = opening + received
                sold = int(min(demand, available))
                closing = available - sold

                # simple rule-based replenishment (what NovaMart does today)
                outstanding = sum(pipeline.values())
                if closing + outstanding <= reorder_level:
                    qty = int((order_up_to - closing - outstanding) * rng.uniform(0.55, 1.05))
                    delay = lead + int(rng.choice([0, 0, 0, 1, 2]))
                    if qty > 0:
                        pipeline[t + delay] = pipeline.get(t + delay, 0) + qty
                stock = closing

                inv_rows.append([d.strftime("%Y-%m-%d"), st.store_id, pr.product_id, opening,
                                 received, sold, closing, reorder_level, lead])
                if sold > 0:
                    sales.append((d, st.store_id, pr.product_id, sold, pr.mrp,
                                  disc[t], flags[t]))
    inv = pd.DataFrame(inv_rows, columns=["date", "store_id", "product_id", "opening_stock",
                                          "received_qty", "sold_qty", "closing_stock",
                                          "reorder_level", "lead_time_days"])
    return inv, sales


def explode_transactions(sales):
    """Split daily sold units into individual customer baskets."""
    recs = []
    tid = 100000
    hours = np.arange(7, 23)
    hp = np.array([2, 5, 7, 6, 5, 5, 4, 4, 5, 7, 9, 10, 9, 7, 5, 3], dtype=float)
    hp /= hp.sum()
    for d, s, p, sold, mrp, disc, flag in sales:
        left = sold
        while left > 0:
            q = int(min(left, rng.choice([1, 1, 1, 2, 2, 3, 4])))
            left -= q
            tid += 1
            price = round(mrp * (1 - disc / 100), 2)
            recs.append([f"T{tid}", d.strftime("%Y-%m-%d"), s, p, q, price, disc, flag,
                         f"C{int(rng.integers(10000, 60000))}",
                         rng.choice(["UPI", "Card", "Cash"], p=[.55, .2, .25]),
                         int(rng.choice(hours, p=hp))])
    return pd.DataFrame(recs, columns=["transaction_id", "date", "store_id", "product_id",
                                       "quantity", "selling_price", "discount_pct",
                                       "promotion_flag", "customer_id", "payment_mode", "hour"])


# --------------------------------------------------------------------------- #
# Inject the intentional data-quality traps
# --------------------------------------------------------------------------- #
def inject_traps(tx, products, stores, inv, ext):
    tx = tx.copy()
    n = len(tx)
    # 1. duplicate transactions (same transaction_id repeated)
    dup = tx.sample(frac=0.008, random_state=1)
    tx = pd.concat([tx, dup]).sample(frac=1, random_state=2).reset_index(drop=True)
    # 2. impossible quantities
    bad = tx.sample(n=int(n * 0.0015), random_state=3).index
    tx.loc[bad, "quantity"] = -rng.integers(1, 6, len(bad))
    big = tx.sample(n=40, random_state=4).index
    tx.loc[big, "quantity"] = rng.integers(150, 400, len(big))      # keying errors
    # 3. inconsistent IDs, payment modes, date formats, missing price
    idx = tx.sample(frac=0.01, random_state=5).index
    tx.loc[idx, "store_id"] = tx.loc[idx, "store_id"].str.lower() + " "
    idx = tx.sample(frac=0.02, random_state=6).index
    tx.loc[idx, "payment_mode"] = tx.loc[idx, "payment_mode"].str.lower()
    idx = tx.sample(frac=0.01, random_state=7).index
    tx.loc[idx, "date"] = pd.to_datetime(tx.loc[idx, "date"]).dt.strftime("%d-%m-%Y")
    idx = tx.sample(frac=0.004, random_state=8).index
    tx.loc[idx, "selling_price"] = np.nan

    # 4. category inconsistency
    products = products.copy()
    variants = {"Beverages": ["beverage", "BEVERAGES", "Beverages "],
                "Dairy": ["dairy", "DAIRY"], "Snacks": ["snack", "SNACKS"],
                "Personal Care": ["personal care", "Personal-Care"],
                "Household": ["household", "HouseHold"], "Frozen": ["frozen", "FROZEN"],
                "Groceries": ["grocery", "GROCERIES"]}
    for i in products.sample(frac=0.35, random_state=9).index:
        products.loc[i, "category"] = rng.choice(variants[products.loc[i, "category"]])
    products.loc[products.sample(n=2, random_state=10).index, "shelf_life_days"] = np.nan

    stores = stores.copy()
    stores.loc[stores.store_id == "S02", "city"] = " chennai"

    # 5. inventory arithmetic mismatches + duplicates + missing reorder levels
    inv = inv.copy()
    mm = inv.sample(frac=0.012, random_state=11).index
    inv.loc[mm, "closing_stock"] = inv.loc[mm, "closing_stock"] + rng.integers(-15, 16, len(mm))
    inv.loc[mm, "closing_stock"] = inv.loc[mm, "closing_stock"].clip(lower=0)
    inv = pd.concat([inv, inv.sample(n=120, random_state=12)]).reset_index(drop=True)
    inv.loc[inv.sample(n=60, random_state=13).index, "reorder_level"] = np.nan

    # 6. missing weather
    ext = ext.copy()
    ext.loc[ext.sample(frac=0.03, random_state=14).index, "temp_c"] = np.nan
    ext.loc[ext.sample(frac=0.015, random_state=15).index, "rain_mm"] = np.nan
    return tx, products, stores, inv, ext


def main():
    ext = make_external()
    promo = make_promos()
    inv, sales = simulate(ext, promo)
    tx = explode_transactions(sales)
    products = PRODUCTS.drop(columns=["_base", "_weather", "_launch"])
    stores = STORES.drop(columns=["_scale", "_trend"])
    tx, products, stores, inv, ext = inject_traps(tx, products, stores, inv, ext)

    tx.to_csv(RAW / "transactions.csv", index=False)
    products.to_csv(RAW / "products.csv", index=False)
    stores.to_csv(RAW / "stores.csv", index=False)
    inv.to_csv(RAW / "inventory.csv", index=False)
    ext.to_csv(RAW / "external_factors.csv", index=False)
    print(f"transactions {len(tx):,} | products {len(products)} | stores {len(stores)} | "
          f"inventory {len(inv):,} | external {len(ext):,}")
    print("stock-out (closing==0) share:", round((inv.closing_stock == 0).mean(), 3))


if __name__ == "__main__":
    main()
