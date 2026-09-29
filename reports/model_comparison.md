# Model Comparison & Selection - StockSense

Splits (purged by 7 days at each boundary): train ['2026-04-14', '2026-07-08', 23220], validation ['2026-07-16', '2026-07-27', 3240], test ['2026-08-04', '2026-08-24', 5670].

## Model 1 - 7-day demand forecast (validation)

| Model | MAE | RMSE | MAPE % | WAPE % | R² | Fit (s) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (7-day mean x 7) | 18.74 | 31.97 | 27.2 | 23.24 | 0.856 | 0.0 |
| Linear Regression | 14.06 | 22.72 | 21.0 | 17.43 | 0.927 | 0.5 |
| Decision Tree | 15.82 | 25.70 | 23.2 | 19.62 | 0.907 | 0.8 |
| Random Forest | 14.24 | 23.40 | 21.1 | 17.66 | 0.923 | 5.9 |
| XGBoost | 14.30 | 23.76 | 20.7 | 17.74 | 0.921 | 2.0 |
| KNN | 18.89 | 29.98 | 31.5 | 23.42 | 0.874 | 0.1 |
| SVM (SVR, RBF) | 20.46 | 46.13 | 26.8 | 25.37 | 0.701 | 5.0 |

**Selected: Linear Regression** (lowest validation WAPE).

Test window (untouched until the end):

| Model | MAE | RMSE | MAPE % | WAPE % | R² |
|---|---:|---:|---:|---:|---:|
| Linear Regression | 13.16 | 22.41 | 21.4 | 16.36 | 0.935 |
| Baseline (7-day mean x 7) | 17.84 | 31.74 | 26.2 | 22.17 | 0.870 |

**Which metric matters most to the business?** WAPE (and MAE in units). Reorder quantities are placed in units, so an error of 10 units costs the same whether the item sells 20 or 200 a week; WAPE expresses total unit error as a share of total volume and is not distorted by slow movers the way MAPE is (a forecast of 3 against an actual of 1 is a 200% MAPE error but only 2 units). RMSE is reported to show that large misses (festival spikes) are also controlled.

## Model 2 - stock-out risk (validation, threshold 0.5 before calibration)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Naive Bayes | 0.716 | 0.471 | 0.510 | 0.490 | 0.740 | 0.490 |
| Decision Tree | 0.593 | 0.369 | 0.740 | 0.493 | 0.701 | 0.435 |
| Random Forest | 0.724 | 0.482 | 0.409 | 0.442 | 0.748 | 0.489 |
| XGBoost | 0.681 | 0.442 | 0.724 | 0.549 | 0.766 | 0.498 |
| KNN | 0.741 | 0.525 | 0.316 | 0.395 | 0.733 | 0.471 |
| SVM (RBF) | 0.735 | 0.508 | 0.230 | 0.316 | 0.737 | 0.459 |

**Selected: XGBoost** (highest validation PR-AUC; PR-AUC is the right ranking metric when positives are the minority class).

Class imbalance handling: class weights (`balanced`) for trees/SVM and `scale_pos_weight = 2.834` for XGBoost. Probabilities are then isotonic-calibrated on the validation window so that the brief's fixed risk bands (>= 0.70 High, 0.40-0.70 Medium) correspond to real observed frequencies.

Alert threshold = 0.36, chosen on validation by maximising F2 subject to precision >= 0.45 (recall weighted 2x precision: a missed stock-out loses sales and customers, while a false alert only costs a manager a shelf check).

Test window results (calibrated):

- Accuracy 0.722 | Precision 0.481 | Recall 0.670 | F1 0.559 | ROC-AUC 0.793 | PR-AUC 0.542
- Confusion matrix [[TN, FP], [FN, TP]] = [[3093, 1082], [494, 1001]]

## Why these results can be trusted

- Time-ordered, purged splits: no future day is ever used to train a model that is scored on an earlier day, and targets never overlap across boundaries.
- Model selection used validation only; the test window was scored once.
- Both winners clearly beat the naive baseline required by the brief.
- Permutation importance (reports/figures/12-13) shows the models rely on business-sensible drivers: recent sales, stock cover, promotions, festivals.