"""One command to rebuild everything from data/raw/.

    python -m src.run_pipeline              # use the CSVs already in data/raw/
    python -m src.run_pipeline --generate   # first regenerate the synthetic raw data

Steps: clean -> EDA -> statistics -> features -> train -> recommend -> HTML dashboard.
Training takes roughly 5-8 minutes on a laptop CPU.
"""
import sys
import time

from src import (build_html_dashboard, data_cleaning, eda, features, recommend, stats_tests,
                 train_models)


def main():
    steps = []
    if "--generate" in sys.argv:
        from src import generate_data
        steps.append(("Generate synthetic raw data", generate_data.main))
    steps += [("Round 1 - clean & integrate", data_cleaning.run), ("Round 1 - EDA", eda.run),
              ("Round 1 - statistical tests", stats_tests.run),
              ("Round 2 - feature engineering", features.run),
              ("Round 2 - train & validate models", train_models.run),
              ("Round 3 - recommendations & what-if", recommend.run),
              ("Round 3 - HTML dashboard", build_html_dashboard.run)]
    t0 = time.time()
    for name, fn in steps:
        print(f"\n=== {name} ===")
        s = time.time()
        fn()
        print(f"--- done in {time.time() - s:.0f}s")
    print(f"\nPipeline finished in {time.time() - t0:.0f}s. "
          "Launch the app with:  streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
