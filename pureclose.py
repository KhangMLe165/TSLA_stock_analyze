import csv
import numpy as np
from datetime import datetime
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

dates = []
prices = []

def get_data(filename):
    dates.clear()
    prices.clear()

    with open(filename, 'r') as csvfile:
        r = csv.reader(csvfile)
        header = next(r)  # Date,Open,High,Low,Close,Volume

        first_date = None
        for row in r:
            d = datetime.strptime(row[0], "%Y-%m-%d")
            if first_date is None:
                first_date = d

            day_index = (d - first_date).days
            dates.append(day_index)

            close_price = float(row[4])  # <-- CLOSE (your file has Close at index 4)
            prices.append(close_price)

def predict_price(dates, prices, x_day_index):
    X = np.array(dates).reshape(-1, 1)
    y = np.array(prices)

    svr_lin  = make_pipeline(StandardScaler(), SVR(kernel='linear', C=1e3))
    svr_poly = make_pipeline(StandardScaler(), SVR(kernel='poly',   C=1e3, degree=2))
    svr_rbf  = make_pipeline(StandardScaler(), SVR(kernel='rbf',    C=1e3, gamma='scale'))

    svr_lin.fit(X, y)
    svr_poly.fit(X, y)
    svr_rbf.fit(X, y)

    # Faster plotting: sample points
    idx = np.linspace(0, len(X) - 1, 600).astype(int)
    Xp, yp = X[idx], y[idx]

    plt.scatter(Xp, yp, s=10, label='Data (sampled)')
    plt.plot(Xp, svr_lin.predict(Xp), label='Linear')
    plt.plot(Xp, svr_poly.predict(Xp), label='Poly')
    plt.plot(Xp, svr_rbf.predict(Xp), label='RBF')
    plt.xlabel('Days since first date')
    plt.ylabel('Close')
    plt.title('Stock Price Prediction (TSLA Close)')
    plt.legend()
    plt.show()

    x = np.array([[x_day_index]])
    return svr_lin.predict(x)[0], svr_poly.predict(x)[0], svr_rbf.predict(x)[0]

get_data('tsla.csv')

future_day = max(dates) + 30   # predict 30 days after the last row in your CSV
predicted = predict_price(dates, prices, future_day)

print("Predicted (linear, poly, rbf):", predicted)
