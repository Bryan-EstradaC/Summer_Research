import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# True function, We don't know it
def f(x):
    return np.exp(-((x-60)**2)/100) + 0.1*np.sin(0.3 * x)

noise = 1e-5

x_s = np.random.uniform(50, 100, size=5)
y_s = f(x_s) + np.random.normal(0, noise, size=len(x_s))

# Create test inputs for prediction
x_pred = np.linspace(50, 100, 200).reshape(-1, 1)

kernel = RBF(length_scale = 5.0)

B = 2.0

for x in range(0, 5):
    gp = GaussianProcessRegressor(kernel=kernel, alpha=noise, normalize_y=True)

    gp.fit(x_s.reshape(-1,1), y_s)

    mu, std = gp.predict(x_pred, return_std=True)

    ucb = mu + np.sqrt(B) * std

    # Sort UCB indices in descending order (highest first)
    sorted_indices = np.argsort(-ucb.ravel())

    # Initialize a flag
    found_new_point = False

    # Go through candidates in order of decreasing UCB
    for idx in sorted_indices:
        candidate = x_pred[idx]

        # Check if candidate is close to any existing point
        if not np.any(np.isclose(x_s, candidate, rtol=0, atol=1e-3)):
            x_next = candidate
            y_next = f(x_next)
            found_new_point = True
            break

    # If no new point was found (all were duplicates), skip adding
    if not found_new_point:
        print("Warning: No unique x_next found, stopping early.")
        break

    x_s = np.append(x_s, x_next)
    y_s = np.append(y_s, y_next)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_pred, f(x_pred), 'k--', label="True Function")
    plt.scatter(x_s, y_s, color='red', label="Observations")
    plt.plot(x_pred, mu, 'b-', label="GP Mean")
    plt.plot(x_pred, ucb, 'g--', linewidth = 1.5, label = "UCB")
    plt.axvline(x = x_next, color = 'green', linestyle = ':', linewidth = 2, label = "Next Sample")
    plt.fill_between(x_pred.ravel(), mu - 2*std, mu + 2*std, color='blue', alpha=0.2, label='Confidence Interval')
    plt.xlabel("Temperature")
    plt.ylabel("Yield")
    plt.title("GP Fit to Initial Observations")
    plt.legend()
    plt.grid(True)
    plt.show()
