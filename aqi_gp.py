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
x_obs = data["DayIndex"].values
last_day = data["DayIndex"].max()
x_pred = np.arange(1150, last_day + 2)
print(x_pred)

y_obs = data["Overall AQI Value"].values

kernel = RBF(length_scale = 20.0)
noise = 1.0
B = 2.0
gp = GaussianProcessRegressor(kernel=kernel, alpha = noise, normalize_y = True)

for x in range(1150, last_day + 2):    
    gp.fit(x_obs.reshape(-1,1), y_obs)
    mu, std = gp.predict(x_pred.reshape(-1, 1), return_std = True)
    ucb = mu + np.sqrt(B) * std

    future_days_copy = x_pred.copy()

    x_next = x_pred[np.argmax(ucb)]
    # Remove selected x_next from future_days
    x_pred = x_pred[x_pred != x_next]

    y_next = np.random.normal(mu[np.argmax(ucb)], std[np.argmax(ucb)])

    data = pd.concat([data, pd.DataFrame([{"DayIndex": x_next, "Overall AQI Value": y_next}])], ignore_index = True)
    print(f"Iteration {x+1}: Selected Day {x_next}, Simulated AQI {y_next:.2f}")

# Create a new figure for each iteration
plt.figure(figsize=(12, 6))

# GP mean prediction
plt.plot(future_days_copy, mu, 'b-', label='GP Mean')

# Confidence interval
plt.fill_between(
    future_days_copy,
    mu - 2*std,
    mu + 2*std,
    color='blue',
    alpha=0.2,
    label='±2σ Confidence Interval'
)

# UCB acquisition function
plt.plot(future_days_copy, ucb, 'g--', label='UCB')

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
plt.title(f'GP-UCB Iteration{x+1}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

