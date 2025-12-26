import csv
import numpy as np
from datetime import datetime, timedelta
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

dates = []
prices = []
volumes = []
calendar_dates = []

def get_data(filename):
    dates.clear()
    prices.clear()
    volumes.clear()
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
            prices.append(float(row[4]))     # Close
            volumes.append(float(row[5]))    # Volume

    return first_date

def predict_and_plot(dates, prices, volumes, calendar_dates, first_date, horizon=30):
    X = np.array(dates).reshape(-1, 1)
    y = np.array(prices)

    svr = make_pipeline(
        StandardScaler(),
        SVR(kernel='rbf', C=1e3, gamma='scale')
    )
    svr.fit(X, y)

    # ---- Build future timeline ----
    last_day = max(dates)
    future_days = np.arange(last_day + 1, last_day + horizon + 1).reshape(-1, 1)
    future_prices = svr.predict(future_days)

    future_calendar_dates = [
        first_date + timedelta(days=int(d))
        for d in future_days.flatten()
    ]

    # ---- Plot ----
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Closing price
    ax1.plot(calendar_dates, prices, label="Close (Historical)", color="black")
    ax1.plot(future_calendar_dates, future_prices,
             label="Predicted Close (+30 days)", color="red", linestyle="--")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Close Price")
    ax1.legend(loc="upper left")

    # Volume (secondary axis)
    ax2 = ax1.twinx()
    ax2.bar(calendar_dates, volumes, alpha=0.25, label="Volume", color="blue")
    ax2.set_ylabel("Volume")
    ax2.legend(loc="upper right")

    plt.title("TSLA Closing Price & Volume with 30-Day Extrapolation")
    plt.tight_layout()
    plt.show()

    return list(zip(future_calendar_dates, future_prices))

# ---- Run ----
first_date = get_data("tsla.csv")

predictions = predict_and_plot(
    dates,
    prices,
    volumes,
    calendar_dates,
    first_date,
    horizon=30
)

print("30-day extrapolated prices:")
for d, p in predictions:
    print(d.strftime("%Y-%m-%d"), round(p, 2))
