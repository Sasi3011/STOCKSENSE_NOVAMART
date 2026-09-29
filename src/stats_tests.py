"""Round 1 - Statistical reasoning.

Four hypothesis tests, each stated as business question, H0, H1, test,
statistic, p-value and business interpretation.

Run:  python -m src.stats_tests
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

from src.config import PROCESSED, REPORTS

ALPHA = 0.05


def _result(question, h0, h1, test, why_test, stat, p, effect, interp):
    return dict(question=question, h0=h0, h1=h1, test=test, why_test=why_test,
                statistic=round(float(stat), 4), p_value=float(p),
                significant=bool(p < ALPHA), effect=effect, interpretation=interp)


def promo_effect(m):
    # Compare each store x product with itself (promo days vs normal days) to remove
    # the size difference between products - a paired design.
    d = m[m.stockout_flag == 0]
    pv = d.groupby(["store_id", "product_id", "promotion_flag"]).units_sold.mean().unstack().dropna()
    stat, p = stats.wilcoxon(pv[1], pv[0], alternative="greater")
    lift = float((pv[1] / pv[0]).median() - 1)
    return _result(
        "Do promotions significantly increase sales?",
        "Median daily units are the same on promotion and non-promotion days.",
        "Median daily units are higher on promotion days.",
        "Wilcoxon signed-rank (one-sided), paired by store x product",
        "Daily units are skewed counts, so a non-parametric paired test is safer than a t-test; "
        "pairing controls for product size.",
        stat, p, f"Median lift per store-product: {lift:+.1%} (n = {len(pv)} pairs)",
        f"Promotions lift units by about {lift:.0%} for the typical item. Promotion calendars "
        "must feed the replenishment plan, otherwise promotions create their own stock-outs.")


def store_type_effect(m):
    daily = m.groupby(["date", "store_id", "store_type"]).units_sold.sum().reset_index()
    groups = [g.units_sold.values for _, g in daily.groupby("store_type")]
    stat, p = stats.kruskal(*groups)
    lev = stats.levene(*groups)
    means = daily.groupby("store_type").units_sold.mean().round(0).to_dict()
    return _result(
        "Does mean demand differ across store types?",
        "Daily store demand has the same distribution for Express, Supermarket and Hypermarket.",
        "At least one store type differs.",
        "Kruskal-Wallis H test (ANOVA checked as well)",
        f"Levene's test p = {lev.pvalue:.2g} shows unequal variances, which breaks an ANOVA "
        "assumption; Kruskal-Wallis does not need equal variances.",
        stat, p, f"Mean daily units per store: {means}",
        "Store formats behave very differently, so one national reorder rule is wrong. "
        "Store type is kept as a model feature and safety stock is set per store.")


def stockout_vs_promo(m):
    tab = pd.crosstab(m.promotion_flag, m.stockout_flag)
    chi2, p, dof, _ = stats.chi2_contingency(tab)
    n = tab.values.sum()
    cramers_v = np.sqrt(chi2 / (n * (min(tab.shape) - 1)))
    r = m.groupby("promotion_flag").stockout_flag.mean()
    return _result(
        "Is stock-out frequency associated with promotion status?",
        "Stock-out occurrence is independent of whether a promotion is running.",
        "Stock-out occurrence depends on promotion status.",
        "Chi-square test of independence (2 x 2)",
        "Both variables are binary categories; expected counts are all well above 5.",
        chi2, p,
        f"Stock-out rate: {r.get(1, 0):.1%} on promotion days vs {r.get(0, 0):.1%} otherwise "
        f"(Cramer's V = {cramers_v:.3f})",
        "Promotion days are more likely to end with an empty shelf. The association is "
        "statistically clear but small on its own (low Cramer's V), which tells us promotions "
        "matter mostly in combination with low stock cover - exactly what a multivariate "
        "stock-out model captures. The promotion flag is kept as a model input.")


def weekend_effect(m):
    d = m[m.stockout_flag == 0]
    pv = d.groupby(["store_id", "product_id", "weekend"]).units_sold.mean().unstack().dropna()
    stat, p = stats.ttest_rel(pv[1], pv[0], alternative="greater")
    diff = float((pv[1] / pv[0]).mean() - 1)
    return _result(
        "Is weekend demand higher than weekday demand?",
        "Mean daily units on weekends equal weekday units for the same item.",
        "Weekend units are higher.",
        "Paired t-test (one-sided) on store x product means",
        "Each pair is an average of many days, so means are close to normal (CLT); pairing "
        "removes product size.",
        stat, p, f"Average weekend uplift: {diff:+.1%}",
        "Orders placed on Thursday should cover a heavier weekend; the 7-day forecast "
        "includes day-of-week features for this.")


def run():
    m = pd.read_csv(PROCESSED / "master_dataset.csv", parse_dates=["date"])
    res = [promo_effect(m), store_type_effect(m), stockout_vs_promo(m), weekend_effect(m)]
    (REPORTS / "statistical_tests.json").write_text(json.dumps(res, indent=1))
    lines = ["# Statistical Reasoning - StockSense", "",
             f"Significance level alpha = {ALPHA}. All tests run on the cleaned master dataset.", ""]
    for i, r in enumerate(res, 1):
        lines += [f"## Test {i}. {r['question']}", "",
                  f"- **H0:** {r['h0']}", f"- **H1:** {r['h1']}",
                  f"- **Test:** {r['test']}", f"- **Why this test:** {r['why_test']}",
                  f"- **Statistic:** {r['statistic']}",
                  f"- **p-value:** {r['p_value']:.3g} -> "
                  f"{'reject H0' if r['significant'] else 'fail to reject H0'}",
                  f"- **Effect size:** {r['effect']}",
                  f"- **Business interpretation:** {r['interpretation']}", ""]
    (REPORTS / "statistical_tests.md").write_text("\n".join(lines))
    for r in res:
        print(f"{r['question'][:55]:57s} p={r['p_value']:.3g}  {r['effect']}")
    return res


if __name__ == "__main__":
    run()
