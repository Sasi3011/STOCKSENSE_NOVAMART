# Statistical Reasoning - StockSense

Significance level alpha = 0.05. All tests run on the cleaned master dataset.

## Test 1. Do promotions significantly increase sales?

- **H0:** Median daily units are the same on promotion and non-promotion days.
- **H1:** Median daily units are higher on promotion days.
- **Test:** Wilcoxon signed-rank (one-sided), paired by store x product
- **Why this test:** Daily units are skewed counts, so a non-parametric paired test is safer than a t-test; pairing controls for product size.
- **Statistic:** 34050.0
- **p-value:** 6.08e-42 -> reject H0
- **Effect size:** Median lift per store-product: +30.8% (n = 264 pairs)
- **Business interpretation:** Promotions lift units by about 31% for the typical item. Promotion calendars must feed the replenishment plan, otherwise promotions create their own stock-outs.

## Test 2. Does mean demand differ across store types?

- **H0:** Daily store demand has the same distribution for Express, Supermarket and Hypermarket.
- **H1:** At least one store type differs.
- **Test:** Kruskal-Wallis H test (ANOVA checked as well)
- **Why this test:** Levene's test p = 2.6e-59 shows unequal variances, which breaks an ANOVA assumption; Kruskal-Wallis does not need equal variances.
- **Statistic:** 751.7539
- **p-value:** 5.74e-164 -> reject H0
- **Effect size:** Mean daily units per store: {'Express': 278.0, 'Hypermarket': 1125.0, 'Supermarket': 465.0}
- **Business interpretation:** Store formats behave very differently, so one national reorder rule is wrong. Store type is kept as a model feature and safety stock is set per store.

## Test 3. Is stock-out frequency associated with promotion status?

- **H0:** Stock-out occurrence is independent of whether a promotion is running.
- **H1:** Stock-out occurrence depends on promotion status.
- **Test:** Chi-square test of independence (2 x 2)
- **Why this test:** Both variables are binary categories; expected counts are all well above 5.
- **Statistic:** 17.6366
- **p-value:** 2.67e-05 -> reject H0
- **Effect size:** Stock-out rate: 7.2% on promotion days vs 5.6% otherwise (Cramer's V = 0.021)
- **Business interpretation:** Promotion days are more likely to end with an empty shelf. The association is statistically clear but small on its own (low Cramer's V), which tells us promotions matter mostly in combination with low stock cover - exactly what a multivariate stock-out model captures. The promotion flag is kept as a model input.

## Test 4. Is weekend demand higher than weekday demand?

- **H0:** Mean daily units on weekends equal weekday units for the same item.
- **H1:** Weekend units are higher.
- **Test:** Paired t-test (one-sided) on store x product means
- **Why this test:** Each pair is an average of many days, so means are close to normal (CLT); pairing removes product size.
- **Statistic:** 9.4719
- **p-value:** 5.92e-19 -> reject H0
- **Effect size:** Average weekend uplift: +14.3%
- **Business interpretation:** Orders placed on Thursday should cover a heavier weekend; the 7-day forecast includes day-of-week features for this.
