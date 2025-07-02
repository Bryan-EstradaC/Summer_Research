import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# Load data
csv_path = 'ds/aqidaily.csv'
data = pd.read_csv(csv_path)
pd.to_datetime(data["Date"])
data["Date"] = pd.to_datetime(data["Date"])

data["DayIndex"] = np.arange(len(data))
last_day = data["DayIndex"].max()
future_days = np.arange(last_day + 1, last_day + 31)
print(future_days.shape)

#y = data["Overall AQI Value"].values

kernel = RBF(length_scale = 5.0)
noise = 1e-3
B = 2.0
gp = GaussianProcessRegressor(kernel=kernel, alpha = noise, normalize_y = True)

for x in range(0, 5):    
    gp.fit(data["DayIndex"].values.reshape(-1,1), data["Overall AQI Value"].values)
    mu, std = gp.predict(future_days.reshape(-1, 1), return_std = True)
    ucb = mu + np.sqrt(B) * std

    x_next = future_days[np.argmax(ucb)]
    y_next = mu[np.argmax(ucb)]

    data = pd.concat([data, pd.DataFrame([{"DayIndex": x_next, "Overall AQI Value": y_next}])], ignore_index = True)

    # Create a new figure for each iteration
    plt.figure(figsize=(12, 6))

    # GP mean prediction
    plt.plot(future_days, mu, 'b-', label='GP Mean')

    # Confidence interval
    plt.fill_between(
        future_days,
        mu - 2*std,
        mu + 2*std,
        color='blue',
        alpha=0.2,
        label='±2σ Confidence Interval'
    )

    # UCB acquisition function
    plt.plot(future_days, ucb, 'g--', label='UCB')

    # Observed data points
    plt.scatter(
        data["DayIndex"].values,
        data["Overall AQI Value"].values,
        c='red',
        label='Observations'
    )

    # Mark the selected x_next
    plt.axvline(
        x=x_next,
        color='purple',
        linestyle=':',
        linewidth=2,
        label=f'Selected Day {int(x_next)}'
    )

    # Labels and title
    plt.xlabel('Day Index')
    plt.ylabel('AQI')
    plt.title(f'GP-UCB Iteration {x+1}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

