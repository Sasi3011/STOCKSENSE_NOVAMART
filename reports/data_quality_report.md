# Data Quality Report - StockSense (NovaMart)

Every issue found while auditing the five raw files, how many records were affected, what we did and why.

| Table | Issue | Count | Action taken | Justification |
|---|---|---:|---|---|
| products | Category spelling/case inconsistency (18 raw labels -> 7) | 11 | Mapped to 7 canonical categories | Groupings like 'beverage' vs 'BEVERAGES' would split revenue and model encodings. |
| products | Missing shelf_life_days | 2 | Imputed with sub-category (else category) median | Shelf life is a stable product attribute; siblings are a reliable proxy. |
| stores | City name whitespace/case issues | 1 | Trimmed and title-cased | City is the join key to external_factors.csv. |
| external_factors | Missing temperature | 29 | Linear interpolation within city | Temperature changes smoothly day to day; neighbours are the best estimate. |
| external_factors | Missing rainfall | 14 | Filled with 0 mm | Most days are dry (median = 0); a missing reading most likely means no rain. |
| transactions | Duplicate rows (same transaction_id repeated) | 2,013 | Removed, kept first occurrence | A repeated transaction_id is a double-posting; keeping it inflates demand. |
| transactions | Store IDs with lower-case / trailing spaces | 2,516 | Trimmed and upper-cased | Otherwise 's01 ' and 'S01' become two stores. |
| transactions | Non-ISO date format (dd-mm-yyyy) | 2,514 | Parsed both formats into one datetime | Mixed formats silently mis-sort time series. |
| transactions | Payment mode case inconsistency | 5,031 | Standardised to UPI/Card/Cash | Needed for clean payment-mix reporting. |
| transactions | Impossible quantity (<= 0) | 375 | Removed from demand, logged | Negative sale quantities are returns/keying errors, not customer demand. |
| transactions | Extreme basket quantity (e.g. 150-400 units) | 40 | Removed (quantity > 5x product 99.9th percentile) | Inventory sold_qty for these days does not support them; clear keying errors. |
| transactions | Missing selling_price | 1,007 | Imputed as MRP x (1 - discount_pct) | Matches the pricing rule seen in valid rows. |
| transactions | Rows retained | 251,186 | 99.0% of raw rows kept |  |
| inventory | Duplicate date x store x product rows | 120 | Removed | The grain must be unique before merging. |
| inventory | Missing reorder_level | 60 | Filled with the store x product median | Reorder levels are fixed policy values. |
| inventory | closing != opening + received - sold | 472 | Recomputed closing from the equation and flagged | Opening/received/sold reconcile with the next day's opening; the closing count is the field most likely mis-keyed. |

## Sparse history (new products)

2 products have fewer than 14 days of history: P160 (5 days), P161 (5 days).
They are excluded from model training (lags/rolling windows are undefined) but are still scored. For scoring, missing lag and rolling features are filled with the average of the same sub-category in the same store on the same date (a 'look-alike product' fallback). Their forecasts carry lower confidence and are marked in the manager output.

## Known limitation: censored demand

Sales are capped by available stock. On stock-out days the true demand is higher than recorded sales, so the forecast target slightly under-states demand for items that stock out often. The safety-stock term in the reorder logic partly offsets this.

## Master dataset

- Grain: one row = one date x one store x one product
- Rows: 41,370 | Columns: 41
- Date range: 01 Apr 2026 to 31 Aug 2026
- Stores: 6 | Products: 47
- Stock-out rate (closing stock = 0): 5.7%
- Remaining missing values: 0