"""Generates the three round notebooks (run from project root).
    python notebooks/_build_notebooks.py
"""
import nbformat as nbf
from pathlib import Path

NB = Path(__file__).parent
SETUP = """import sys, json, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT))
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from IPython.display import Image, Markdown, display
pd.set_option('display.max_columns', 40); pd.set_option('display.width', 200)
FIG = ROOT / 'reports' / 'figures'"""


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


def save(cells, name):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, NB / name)


# --------------------------------------------------------------------------- #
nb1 = [
    md("""# StockSense - Round 1: Data Understanding, Preparation & EDA
**NovaMart Retail Pvt. Ltd. | IntelliData 2026**

Business question: *can we predict what will sell, spot products likely to go out of stock, and tell
store managers what to do before it happens?*

This notebook covers the audit of the five raw files, cleaning and integration into one master table
(grain: **one date x one store x one product**), EDA where every chart answers a business question,
and four hypothesis tests."""),
    code(SETUP),
    md("## 1. Audit the raw files\nData types, missing values, duplicates and obviously invalid values, before touching anything."),
    code("""RAW = ROOT / 'data' / 'raw'
raw = {n: pd.read_csv(RAW / f'{n}.csv') for n in ['transactions','products','stores','inventory','external_factors']}
audit = pd.DataFrame({n: {'rows': len(d), 'columns': d.shape[1], 'missing cells': int(d.isna().sum().sum()),
                          'duplicate rows': int(d.duplicated().sum())} for n, d in raw.items()}).T
audit"""),
    code("""t = raw['transactions']
print('Duplicate transaction_id :', t.transaction_id.duplicated().sum())
print('Quantity <= 0            :', (t.quantity <= 0).sum())
print('Quantity > 100           :', (t.quantity > 100).sum())
print('Non-standard store IDs   :', (~t.store_id.astype(str).str.fullmatch(r'S\\d{2}')).sum())
print('Non-ISO dates            :', (~t.date.astype(str).str.match(r'^\\d{4}-\\d{2}-\\d{2}$')).sum())
print('\\nRaw category labels:', sorted(raw['products'].category.unique()))
inv = raw['inventory']
print('\\nInventory rows where closing != opening + received - sold:',
      int((inv.opening_stock + inv.received_qty - inv.sold_qty != inv.closing_stock).sum()))"""),
    md("## 2. Clean and integrate\nAll rules live in `src/data_cleaning.py` so the notebook and the pipeline can never disagree."),
    code("""from src import data_cleaning
master = data_cleaning.run()"""),
    md("### Data Quality Report\nOne row per issue: count, action taken and justification (also saved to `reports/data_quality_report.md`)."),
    code("""pd.read_csv(ROOT / 'reports' / 'data_quality_log.csv')"""),
    code("""print(master.shape)
print('Unique grain check:', not master.duplicated(['date','store_id','product_id']).any())
master.head()"""),
    md("""**Stock-out definition (operational):** a store-product day is a stock-out day when the closing stock
is zero, meaning the shelf was empty at the end of trading. For modelling, the target is whether
*any* day in the next 7 days is a stock-out day."""),
    md("## 3. Exploratory Data Analysis\nEach figure is produced by `src/eda.py`."),
    code("""from src import eda
summary = eda.run()"""),
    md("### Q1. Which categories generate the most revenue?"),
    code("display(Image(FIG / '01_category_revenue_pareto.png'))\nprint('Products making 80% of revenue:', summary['pareto_products_for_80pct'])"),
    md("""Dairy and Groceries dominate revenue and are also the fastest movers. A small head of products
drives most of the revenue (Pareto), so forecast accuracy on those items matters most."""),
    md("### Q2. Which stores are growing or declining?"),
    code("display(Image(FIG / '02_store_weekly_trend.png'))\npd.Series(summary['store_growth_first4_vs_last4_weeks']).sort_values().map('{:+.1%}'.format)"),
    md("""S02 Chennai and S05 Tiruppur are growing fast, while S03 Madurai is declining. Growing stores are
where fixed reorder levels break first."""),
    md("### Q3. Do promotions increase units sold?"),
    code("display(Image(FIG / '03_promotion_lift.png'))"),
    md("### Q4. How does weekend behaviour differ?"),
    code("display(Image(FIG / '04_weekday_weekend.png'))\nprint(f\"Weekend uplift: {summary['weekend_uplift']:+.1%}\")"),
    md("### Q5. Which products are volatile?"),
    code("display(Image(FIG / '05_volatility_cv.png'))"),
    md("""High coefficient-of-variation items (ice cream, cold drinks, low-volume personal care) need
proportionally more safety stock. That is why the reorder logic uses each item's own demand
standard deviation."""),
    md("### Q6. Which stores repeatedly stock out?"),
    code("display(Image(FIG / '06_stockout_heatmap.png'))"),
    md("""S02 stocks out most across almost every category. Combined with Q2, this is the headline finding:
**the fastest-growing store has the emptiest shelves**, because its replenishment rules were never
re-sized."""),
    md("### Q7. Does heat drive beverage and frozen demand?"),
    code("display(Image(FIG / '07_temperature_effect.png'))"),
    md("### Headline KPIs"),
    code("pd.Series(summary['kpi'])"),
    md("## 4. Statistical reasoning\nBusiness question, H0, H1, test, p-value and business interpretation for each."),
    code("""from src import stats_tests
res = stats_tests.run()
pd.DataFrame(res)[['question','test','statistic','p_value','significant','effect']]"""),
    code("""for r in res:
    display(Markdown(f"**{r['question']}**  \\nH0: {r['h0']}  \\nH1: {r['h1']}  \\n"
                     f"Test: {r['test']} - {r['why_test']}  \\np = {r['p_value']:.3g}  \\n"
                     f"*Interpretation:* {r['interpretation']}"))"""),
]

# --------------------------------------------------------------------------- #
nb2 = [
    md("""# StockSense - Round 2: Feature Engineering & Machine Learning

**Prediction setting.** At the end of day *t*, predict:
1. `next_7_day_demand`, the units sold over *t+1 ... t+7* (regression)
2. `stockout_next_7d`, whether the shelf goes empty on any day in *t+1 ... t+7* (classification)

**Leakage rule.** Every feature uses information available at the end of day *t*. The only
forward-looking inputs are the festival and holiday calendar, which is published in advance.

Only permitted algorithms are used: Linear Regression, Decision Tree, SVM, XGBoost, KNN, Naive Bayes, Random Forest."""),
    code(SETUP),
    md("## 1. Feature engineering"),
    code("""from src import features
from src.features import NUMERIC_FEATURES, CATEGORICAL_FEATURES, FEATURES, REG_TARGET, CLF_TARGET
f = features.run()
groups = {
 'Time': ['day_of_week','weekend_flag','month','week_no','festival_flag','holiday_flag','festival_next_7','holiday_next_7','weekend_days_next_7'],
 'Lag / rolling': ['lag_1','lag_7','lag_14','rolling_mean_7','rolling_mean_14','rolling_mean_28','rolling_std_7','rolling_std_28','sales_growth_7v28','cv_28'],
 'Inventory': ['current_stock','days_of_inventory','inventory_to_demand_ratio','reorder_gap','below_reorder_flag','received_last_7','stockout_days_last_14','stockout_days_last_28','lead_time_days','days_since_last_receipt','last_receipt_cover','cover_minus_lead','projected_stock_at_lead','projected_stock_7d'],
 'Price / promo': ['promotion_flag','discount_pct','promo_days_last_7','price_change'],
 'External': ['temp_c','temp_mean_7','rain_mm','local_event'],
 'Store / product': ['avg_daily_customers','floor_area_sqft','shelf_life_days','mrp','margin_pct','history_days'] + CATEGORICAL_FEATURES}
pd.DataFrame([(g, len(c), ', '.join(c)) for g, c in groups.items()], columns=['Group','Count','Features'])"""),
    md("""### Leakage check
Rolling windows end at day *t* inclusive and targets start at *t+1*. The check below recomputes one
row by hand to prove it."""),
    code("""row = f[(f.store_id=='S01') & (f.product_id=='P101')].reset_index(drop=True)
i = 60
print('date                  :', row.date[i].date())
print('rolling_mean_7 (feat) :', round(row.rolling_mean_7[i], 3), '= mean of units t-6..t:', round(row.units_sold[i-6:i+1].mean(), 3))
print('next_7_day_demand     :', row[REG_TARGET][i], '= sum of units t+1..t+7:', row.units_sold[i+1:i+8].sum())"""),
    md("""### Sparse-history fallback
New products (P160, P161) have 5 days of history. They are excluded from training. At scoring time,
missing lags fall back to their own short average, and missing volatility falls back to look-alike
products (same store and sub-category, otherwise category)."""),
    code("f[f.product_id.isin(['P160','P161']) & (f.date==f.date.max())][['store_id','product_name','history_days','rolling_mean_7','lag_14','rolling_std_7','sparse_history_flag']]"),
    md("## 2. Time-aware, purged split"),
    code("""from src.train_models import load_splits
_, lab, train, valid, test = load_splits()
pd.DataFrame({'rows':[len(train),len(valid),len(test)],
              'from':[train.date.min().date(), valid.date.min().date(), test.date.min().date()],
              'to':[train.date.max().date(), valid.date.max().date(), test.date.max().date()],
              'stock-out prevalence':[train[CLF_TARGET].mean(), valid[CLF_TARGET].mean(), test[CLF_TARGET].mean()]},
             index=['train','validation','test']).round(3)"""),
    md("""A 7-day gap sits before validation and before test. Without it, the last training rows would have
targets that overlap the validation period. That is a subtle form of leakage that inflates scores."""),
    md("""## 3. Train and compare models
Training all 12 models takes about 7 minutes on one CPU. Set `RETRAIN = True` to re-run; otherwise
the saved results from `python -m src.train_models` are loaded."""),
    code("""RETRAIN = False
if RETRAIN:
    from src import train_models
    meta = train_models.run()
else:
    meta = json.loads((ROOT / 'models' / 'model_metadata.json').read_text())"""),
    md("### Model 1: 7-day demand forecast"),
    code("""print('Validation:')
display(pd.DataFrame(meta['regression']['validation']).drop(columns='split').round(3))
print('Test (selected model vs the brief\\'s baseline):')
display(pd.DataFrame(meta['regression']['test']).drop(columns='split').round(3))"""),
    md("""**Metric that matters most: WAPE (and MAE in units).** Orders are placed in units. MAPE explodes
on slow movers: forecasting 3 against an actual of 1 is a 200% error, but only 2 units. WAPE
weights errors by volume, which is how the business feels them.

Linear Regression wins narrowly over Random Forest and XGBoost. With strong lag and rolling
features the relationship is close to linear, and the simpler model is faster, more stable and
easier to explain. It beats the baseline by about 6 WAPE points."""),
    code("display(Image(FIG / '08_forecast_vs_actual.png')); display(Image(FIG / '09_residuals.png'))"),
    md("### Model 2: stock-out risk"),
    code("""display(pd.DataFrame(meta['classification']['validation']).drop(columns='split').round(3))
c = meta['classification']
print('Selected:', c['selected'], '| threshold:', c['threshold'], '| XGBoost scale_pos_weight:', c['pos_weight'])
display(pd.DataFrame(c['test']).drop(columns='split').round(3))"""),
    md("""**Class imbalance.** About 26% of windows are positive. The models use `class_weight='balanced'`
(trees and SVM) and `scale_pos_weight` (XGBoost), and are ranked by PR-AUC, which is more honest
than accuracy under imbalance.

**Calibration.** Weighting distorts probabilities, so the winner is isotonic-calibrated on the
validation weeks. That makes the brief's fixed bands (0.70 / 0.40) mean what they say.

**Threshold.** The binary alert threshold is chosen on validation to maximise F2 (recall weighted
double), subject to precision of at least 45% so managers keep trusting the alerts."""),
    code("display(Image(FIG / '10_classifier_evaluation.png')); display(Image(FIG / '11_calibration.png'))"),
    md("## 4. Global explainability: permutation importance"),
    code("display(Image(FIG / '12_importance_demand.png')); display(Image(FIG / '13_importance_stockout.png'))"),
    md("""Recent sales levels drive the demand forecast. Stock cover, delivery rhythm and recent stock-outs
drive risk. Both match how an experienced store manager thinks, which is a good sign the models
learned real structure rather than noise."""),
]

# --------------------------------------------------------------------------- #
nb3 = [
    md("""# StockSense - Round 3: Intelligence Layer, Recommendations & What-If

Pipeline for every store x product on the latest date:

`forecast -> stock-out probability -> risk band -> reasons -> reorder quantity -> manager action`"""),
    code(SETUP),
    code("""from src.recommend import score_latest, load_models
from src.config import SERVICE_Z
feat = pd.read_csv(ROOT / 'data' / 'processed' / 'features.csv', parse_dates=['date'])
reg, clf = load_models()
rec = score_latest(feat, reg, clf)
print('As of', rec.as_of_date[0], '| forecast window: next 7 days')
rec.risk_level.value_counts()"""),
    md("""## 1. Business logic

| Item | Rule |
|---|---|
| Risk | HIGH if p >= 0.70, MEDIUM if 0.40 <= p < 0.70, else LOW |
| Safety stock | z x daily demand std x sqrt(lead time), z = 1.65 / 1.28 / 1.00 for High / Medium / Low |
| Recommended stock | 7-day forecast + safety stock (capped at shelf life for perishables) |
| Reorder quantity | max(0, recommended - current - incoming), rounded up to the case pack |
| Guard-rails | Empty shelf -> p >= 0.95; stock runs out before a new order can land -> p >= 0.70 |

The guard-rails exist because the model cannot see open purchase orders. It learned that empty
shelves are often refilled soon, which is statistically true but not actionable. Every escalated
item is labelled `Rule: ...` so managers can see exactly why."""),
    md("## 2. Manager-ready output (format from the brief)"),
    code("""out = rec[['store_id','product_name','current_stock','forecast_7d','stockout_probability','risk_level','reorder_qty','manager_action']]
out.columns = ['Store','Product','Current Stock','7-Day Forecast','Stock-out Prob.','Risk','Recommended Order','Action']
out.head(15)"""),
    md("## 3. Recommendation card"),
    code("""def card(r):
    reasons = ', '.join(n if v == 'rule' else f'{n} ({v})' for n, v in json.loads(r.reasons))
    print(f"STORE {r.store_id} - {r.product_name.upper()}")
    print(f"Predicted 7-day demand: {r.forecast_7d:,.0f} units")
    print(f"Current stock: {r.current_stock:,} units | Incoming stock: {r.incoming_stock} units")
    print(f"Stock-out probability: {r.stockout_probability:.0%} (model {r.model_probability:.0%}) | Risk: {r.risk_level}")
    print(f"Safety stock: {r.safety_stock:.0f} | Recommended stock: {r.recommended_stock:.0f}")
    print(f"Recommended additional replenishment: {r.reorder_qty} units (case pack {r.case_pack})")
    print(f"Why? {reasons}")
    print(f"MANAGER ACTION: {r.manager_action}\\n")
for _, r in rec[rec.risk_level=='HIGH'].head(3).iterrows(): card(r)
card(rec[rec.risk_level=='MEDIUM'].iloc[0])"""),
    md("""## 4. Local explanations
For each item, each business driver is reset to a neutral value and both models are re-run. The
change in prediction is that driver's contribution. This is permutation logic applied to a single
row, and it reads naturally to a manager ("Promotion running: +10 pts risk")."""),
    code("""r = rec[rec.risk_level=='HIGH'].iloc[0]
pd.DataFrame({'risk impact (prob. points)': json.loads(r.risk_drivers),
              'demand impact (%)': json.loads(r.demand_drivers)}).sort_values('risk impact (prob. points)', ascending=False)"""),
    md("## 5. Overstock: the other half of the problem"),
    code("""over = rec[rec.excess_units > 0].sort_values('excess_value', ascending=False)
print(f"{len(over)} items hold more than 14 days of stock (or more than their shelf life): Rs {over.excess_value.sum():,.0f} at cost")
over[['store_id','product_name','current_stock','forecast_7d','days_of_cover','excess_value','manager_action']].head(10)"""),
    md("## 6. What-if analysis"),
    code("""base = rec
rows = []
for name, s in [('10% promotion', dict(discount_pct=10)), ('20% promotion', dict(discount_pct=20)),
                ('Supplier delay +2 days', dict(supplier_delay_days=2)), ('Local festival', dict(festival=1)),
                ('Demand shock +20%', dict(demand_shock_pct=20))]:
    x = score_latest(feat, reg, clf, scenario=s, explain_rows=False)
    rows.append(dict(scenario=name, demand_change=f"{x.forecast_7d.sum()/base.forecast_7d.sum()-1:+.1%}",
                     high_risk_items=int((x.risk_level=='HIGH').sum()), change=int((x.risk_level=='HIGH').sum()-(base.risk_level=='HIGH').sum()),
                     units_to_order=int(x.reorder_qty.sum())))
pd.DataFrame(rows)"""),
    md("""Supplier delay has by far the largest effect on risk: two extra days push roughly 60 more items
into High. Promotions raise demand about 5-9% and lift orders with it. The festival effect is
probably under-estimated, because there were only a handful of festival days to learn from. This
is a known limitation, listed in the README."""),
    md("## 7. Save outputs"),
    code("""from src.recommend import OUTPUT_COLS
rec[OUTPUT_COLS].to_csv(ROOT / 'data' / 'processed' / 'recommendations.csv', index=False)
print('Saved data/processed/recommendations.csv  ->  open dashboard/stocksense_dashboard.html or run: streamlit run dashboard/app.py')"""),
]

save(nb1, "01_data_understanding_eda.ipynb")
save(nb2, "02_feature_engineering_ml.ipynb")
save(nb3, "03_intelligence_layer_recommendations.ipynb")
print("notebooks written")
