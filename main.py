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

# def predict_and_plot(dates, prices, volumes, calendar_dates, first_date, horizon=30):
#     X = np.array(dates).reshape(-1, 1)
#     y = np.array(prices)

#     svr = make_pipeline(
#         StandardScaler(),
#         SVR(kernel='rbf', C=1e3, gamma='scale')
#     )
#     svr.fit(X, y)

#     # ---- Build future timeline ----
#     last_day = max(dates)
#     future_days = np.arange(last_day + 1, last_day + horizon + 1).reshape(-1, 1)
#     future_prices = svr.predict(future_days)

#     future_calendar_dates = [
#         first_date + timedelta(days=int(d))
#         for d in future_days.flatten()
#     ]

#     # ---- Plot ----
#     fig, ax1 = plt.subplots(figsize=(12, 6))

#     # Closing price
#     ax1.plot(calendar_dates, prices, label="Close (Historical)", color="black")
#     ax1.plot(future_calendar_dates, future_prices,
#              label="Predicted Close (+30 days)", color="red", linestyle="--")
#     ax1.set_xlabel("Date")
#     ax1.set_ylabel("Close Price")
#     ax1.legend(loc="upper left")

#     # Volume (secondary axis)
#     ax2 = ax1.twinx()
#     ax2.bar(calendar_dates, volumes, alpha=0.25, label="Volume", color="blue")
#     ax2.set_ylabel("Volume")
#     ax2.legend(loc="upper right")

#     plt.title("TSLA Closing Price & Volume with 30-Day Extrapolation")
#     plt.tight_layout()
#     plt.show()

#     return list(zip(future_calendar_dates, future_prices))

def predict_returns_and_plot(prices, volumes, calendar_dates, window=10, horizon=30):
    prices = np.array(prices, dtype=float)
    volumes = np.array(volumes, dtype=float)

    # --- compute returns ---
    returns = (prices[1:] - prices[:-1]) / prices[:-1]

    X = []
    y = []

    for i in range(window, len(returns) - 1):
        r_t = returns[i]
        volatility = np.std(returns[i-window:i])
        momentum = np.mean(returns[i-window:i])
        vol_feat = np.log1p(volumes[i + 1])  # align volume with return day

        X.append([r_t, volatility, momentum, vol_feat])
        y.append(returns[i + 1])  # next-day return

    X = np.array(X)
    y = np.array(y)

    model = make_pipeline(
        StandardScaler(),
        SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.001)
    )
    model.fit(X, y)

    # --- 30-day recursive projection ---
    last_price = prices[-1]
    last_date = calendar_dates[-1]

    recent_returns = list(returns[-window:])   # seed rolling window
    last_vol_feat = float(np.log1p(volumes[-1]))

    pred_dates = []
    pred_prices = []

    print("\nDate        | Predicted Close")
    print("------------|----------------")

    current_price = last_price
    current_date = last_date

    for step in range(1, horizon + 1):
        features = np.array([[
            recent_returns[-1],
            float(np.std(recent_returns)),
            float(np.mean(recent_returns)),
            last_vol_feat
        ]])

        r_pred = model.predict(features)[0]
        current_price *= (1 + r_pred)
        current_date += timedelta(days=1)

        recent_returns.append(r_pred)
        recent_returns.pop(0)

        pred_dates.append(current_date)
        pred_prices.append(current_price)

        print(f"{current_date.strftime('%Y-%m-%d')} | {current_price:>14.2f}")

    # --- plot 30-day predicted prices only ---
    plt.figure(figsize=(10, 5))
    plt.plot(pred_dates, pred_prices, marker="o", linestyle="--", color="red")
    plt.title("TSLA 30-Day Predicted Price Path (Return-Based Model)")
    plt.xlabel("Date")
    plt.ylabel("Predicted Close Price")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- optional plot of historical prices (context) ---
    plt.figure(figsize=(12, 5))
    plt.plot(calendar_dates, prices, label="Close Price")
    plt.title("TSLA Closing Prices (Return-Based Model)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    first_date = get_data("tsla.csv")  # <-- THIS loads prices/volumes/dates

    predict_returns_and_plot(
        prices,
        volumes,
        calendar_dates,
        window=10,
        horizon=30
    )

