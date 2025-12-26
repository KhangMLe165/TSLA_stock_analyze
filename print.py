import csv
import numpy as np
from datetime import datetime, timedelta
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

dates = []
prices = []
calendar_dates = []

def get_data(filename):
    dates.clear()
    prices.clear()
    calendar_dates.clear()

    with open(filename, 'r') as csvfile:
        r = csv.reader(csvfile)
        next(r)  # Date,Open,High,Low,Close,Volume

        first_date = None
        for row in r:
            d = datetime.strptime(row[0], "%Y-%m-%d")
            if first_date is None:
                first_date = d

            day_index = (d - first_date).days
            calendar_dates.append(d)
            dates.append(day_index)
            prices.append(float(row[4]))  # Close

    return first_date

def train_model(dates, prices):
    X = np.array(dates).reshape(-1, 1)
    y = np.array(prices)

    svr_rbf = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=1e3, gamma="scale"))
    svr_rbf.fit(X, y)
    return svr_rbf

def print_predictions_from(model, first_date, start_date_str="2025-12-26", horizon_days=30):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    start_idx = (start_date - first_date).days

    # Predict for start date through +30 days (inclusive => 31 rows)
    day_indices = np.arange(start_idx, start_idx + horizon_days + 1).reshape(-1, 1)
    preds = model.predict(day_indices)

    # Map actual close if start_date exists in CSV
    actual_map = {d.strftime("%Y-%m-%d"): p for d, p in zip(calendar_dates, prices)}
    actual_start = actual_map.get(start_date_str)

    print(f"\nModel printout from {start_date_str} to {(start_date + timedelta(days=horizon_days)).strftime('%Y-%m-%d')}")
    if actual_start is not None:
        print(f"(Actual Close on {start_date_str}: {actual_start:.2f})")
    else:
        print(f"(No actual Close found for {start_date_str} in CSV — printing model outputs only)")

    print("\nDate        | Model Predicted Close")
    print("------------|----------------------")
    for i, pred in enumerate(preds):
        d = start_date + timedelta(days=i)
        print(f"{d.strftime('%Y-%m-%d')} | {pred:>20.2f}")

# ---- Run ----
first_date = get_data("tsla.csv")
model = train_model(dates, prices)
print_predictions_from(model, first_date, start_date_str="2025-12-26", horizon_days=30)
