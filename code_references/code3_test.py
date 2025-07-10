import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# Download and read Airfoil dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat"
df = pd.read_csv(url, sep="\t", header=None)
df.columns = [
    "Frequency", "AngleAttack", "ChordLength", "FreeStreamVelocity", "SuctionThickness", "SoundPressure"
]

# Use Frequency and Angle of Attack as inputs (2D) and normalize
X = df[["Frequency", "AngleAttack"]].values
X_min, X_max = X.min(axis=0), X.max(axis=0)
X_norm = (X - X_min) / (X_max - X_min)
y = df["SoundPressure"].values

# Bandit "arms" = all points in the normalized dataset (or subsample for speed)
n_arms = 50  # reduce for speed if you like
idx = np.random.choice(len(X_norm), n_arms, replace=False)
X_arms = X_norm[idx]      # (n_arms, 2)
y_arms = y[idx]

noise = np.std(y_arms) * 0.08
beta = 2.0
n_rounds = 40
np.random.seed(42)

X_hist, y_hist, regret, eigval_records = [], [], [], []

for t in range(n_rounds):
    if len(X_hist) > 0:
        kernel = Matern(length_scale=0.2, length_scale_bounds=(0.01, 2.0), nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=noise**2, normalize_y=True, n_restarts_optimizer=3)
        gp.fit(np.array(X_hist), np.array(y_hist))
        mu, sigma = gp.predict(X_arms, return_std=True)
        ucb = mu + np.sqrt(beta) * sigma
        K = kernel(np.array(X_hist))
        eigvals = np.linalg.eigvalsh(K)
        eigval_records.append(eigvals)
    else:
        ucb = np.ones(n_arms) * np.inf
        mu = np.zeros(n_arms)
        sigma = np.ones(n_arms)
        eigval_records.append(np.array([1.0]))

    # Pick arm with max UCB
    if np.any(np.isinf(ucb)):
        chosen_arm = np.random.choice(n_arms)
    else:
        chosen_arm = np.argmax(ucb)
    X_hist.append(X_arms[chosen_arm])
    reward = y_arms[chosen_arm] + np.random.normal(0, noise)
    y_hist.append(reward)

    # Regret (difference from best possible)
    best_mean = np.max(y_arms)
    regret.append(best_mean - y_arms[chosen_arm])

    # --- Visualization at final round ---
    if t == n_rounds - 1:
        plt.figure(figsize=(13,5))
        # (1) 2D scatter plot of all arms colored by GP mean
        plt.subplot(131)
        plt.title("GP Mean at Sampled Points (2D)")
        plt.scatter(X_arms[:,0], X_arms[:,1], c=mu, cmap='viridis', s=60, label='All Bandits')
        hist_arr = np.array(X_hist)
        plt.scatter(hist_arr[:,0], hist_arr[:,1], c='red', s=40, label="Sampled")
        plt.xlabel("Frequency (normalized)")
        plt.ylabel("Angle of Attack (normalized)")
        plt.colorbar(label="GP Mean Sound Pressure")
        plt.legend()
        
        # (2) GP Reward curve along Frequency, holding AngleAttack at mean value
        mean_angle = np.mean(X_arms[:,1])
        freq_grid = np.linspace(0, 1, 100)
        freq_X_grid = np.column_stack([freq_grid, np.ones(100)*mean_angle])
        mu_grid, sigma_grid = gp.predict(freq_X_grid, return_std=True)
        # Get "true" reward at closest matching arms for the 1D curve
        idx_closest = np.abs(X_arms[:,1] - mean_angle).argmin()
        # All X_arms at (close to) mean AngleAttack, true y values
        y_true_curve = y_arms[X_arms[:,1].argsort()]

        plt.subplot(132)
        plt.title("GP Reward vs. Frequency\n(Angle fixed at mean)")
        plt.plot(freq_grid, mu_grid, 'b-', label="GP Mean")
        plt.fill_between(freq_grid, mu_grid-2*sigma_grid, mu_grid+2*sigma_grid, color='b', alpha=0.2, label="95% Conf")
        plt.scatter(X_arms[:,0], y_arms, c='k', s=30, alpha=0.4, label="All Bandits (raw)")
        plt.xlabel("Frequency (normalized)")
        plt.ylabel("Sound Pressure")
        plt.legend()
        
        # (3) Regret plot
        plt.subplot(133)
        plt.title("Cumulative Regret Over Time (GP-UCB)")
        plt.plot(np.cumsum(regret), 'r-', lw=2)
        plt.xlabel("Iteration")
        plt.ylabel("Cumulative Regret")
        plt.tight_layout()
        plt.show()

# Kernel eigenvalues over rounds
min_eigs = [np.min(e) for e in eigval_records if len(e) > 0]
plt.figure(figsize=(6,4))
plt.plot(min_eigs, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Minimum Eigenvalue')
plt.title('Minimum Kernel Matrix Eigenvalue over Time')
plt.grid(True)
plt.show()
