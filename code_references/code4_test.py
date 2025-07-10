import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# ---------------------------------------------
# 1. LOAD DATA AND PREPROCESS
# ---------------------------------------------
# Airfoil dataset: 3 features for input, 1 output (SoundPressure)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat"
df = pd.read_csv(url, sep="\t", header=None)
df.columns = [
    "Frequency", "AngleAttack", "ChordLength", "FreeStreamVelocity", "SuctionThickness", "SoundPressure"
]

# Use Frequency, AngleAttack, ChordLength as features
X = df[["Frequency", "AngleAttack", "ChordLength"]].values
X_min, X_max = X.min(axis=0), X.max(axis=0)
X_norm = (X - X_min) / (X_max - X_min)
y = df["SoundPressure"].values

# For demo, downsample for speed (change as needed)
n_arms = 25
idx = np.random.choice(len(X_norm), n_arms, replace=False)
X_arms = X_norm[idx]
y_arms = y[idx]

# ---------------------------------------------
# 2. GP-UCB PARAMETERS AND SETUP
# ---------------------------------------------
noise = np.std(y_arms) * 0.08
beta = 2.0
n_rounds = 40
np.random.seed(42)

X_hist = []
y_hist = []
regret = []
eigval_records = []

# ---------------------------------------------
# 3. GP-UCB ACTIVE LEARNING LOOP
# ---------------------------------------------
for t in range(n_rounds):
    if len(X_hist) > 0:
        kernel = Matern(length_scale=0.2, length_scale_bounds=(0.05, 2.0), nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=noise**2, normalize_y=True, n_restarts_optimizer=3)
        gp.fit(np.array(X_hist), np.array(y_hist))
        mu, sigma = gp.predict(X_arms, return_std=True)

        # UCB for maximization, LCB for minimization (flip sign as needed)
        ucb = mu + np.sqrt(beta) * sigma

        # Kernel matrix diagnostics
        K = kernel(np.array(X_hist))
        eigvals = np.linalg.eigvalsh(K)
        eigval_records.append(eigvals)
    else:
        ucb = np.ones(n_arms) * np.inf
        mu = np.zeros(n_arms)
        sigma = np.ones(n_arms)
        eigval_records.append(np.array([1.0]))

    # Pick next point
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

    # --- Visualizations at last round ---
    if t == n_rounds - 1:
        plt.figure(figsize=(12,5))
        # 1D slice: vary one input, fix others at mean, plot GP mean
        var_idx = 0  # 0=Frequency, 1=AngleAttack, 2=ChordLength
        grid = np.linspace(0, 1, 100)
        X_slice = np.mean(X_arms, axis=0)[None, :].repeat(len(grid), axis=0)
        X_slice[:, var_idx] = grid
        mu_slice, sigma_slice = gp.predict(X_slice, return_std=True)

        plt.subplot(121)
        plt.title("GP Mean (1D Slice, Others Fixed)")
        plt.plot(grid, mu_slice, 'b-', label="GP Mean")
        plt.fill_between(grid, mu_slice-2*sigma_slice, mu_slice+2*sigma_slice, color='b', alpha=0.2, label="95% Conf")
        plt.scatter(X_arms[:,var_idx], y_arms, c='k', s=40, alpha=0.6, label="Samples")
        plt.xlabel(df.columns[var_idx] + " (normalized)")
        plt.ylabel("Sound Pressure")
        plt.legend()

        plt.subplot(122)
        plt.title("Cumulative Regret Over Time (GP-UCB)")
        plt.plot(np.cumsum(regret), 'r-', lw=2)
        plt.xlabel("Iteration")
        plt.ylabel("Cumulative Regret")

        plt.tight_layout()
        plt.show()

# ---------------------------------------------
# 4. MINIMUM EIGENVALUE OVER TIME
# ---------------------------------------------
min_eigs = [np.min(e) for e in eigval_records if len(e) > 0]
plt.figure(figsize=(6,4))
plt.plot(min_eigs, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Minimum Eigenvalue')
plt.title('Minimum Kernel Matrix Eigenvalue over Time')
plt.grid(True)
plt.show()
