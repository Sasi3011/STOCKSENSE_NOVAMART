"""StockSense - central configuration.

Every path, date boundary and business threshold lives here so the whole
pipeline can be re-run on the organisers' real CSVs by changing one file.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
DASHBOARD = ROOT / "dashboard"

for _p in (RAW, PROCESSED, MODELS, REPORTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# ---- Forecast horizon -------------------------------------------------------
HORIZON = 7                       # predict the next 7 days

# ---- Time-aware split boundaries (dates are the "as-of" prediction date) ----
TRAIN_END = "2026-07-15"          # train:      history start .. TRAIN_END
VALID_END = "2026-08-03"          # validation: TRAIN_END+1 .. VALID_END
# test: VALID_END+1 .. last date that still has a full 7-day future target
MIN_HISTORY_DAYS = 14             # rows with less history are excluded from training

# ---- Stock-out risk bands (from the challenge brief) ------------------------
HIGH_RISK = 0.70
MEDIUM_RISK = 0.40

# ---- Replenishment policy ----------------------------------------------------
SERVICE_Z = {"HIGH": 1.65, "MEDIUM": 1.28, "LOW": 1.00}   # ~95% / 90% / 84% service
PERISHABLE_SHELF_LIFE = 7         # days; perishables are capped by shelf life
