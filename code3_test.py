import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# Load the Airfoil Self-Noise dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat"
df = pd.read_csv(url, sep="\t", header=None)
df.columns = [
    "Frequency", "AngleAttack", "ChordLength", "FreeStreamVelocity", "SuctionThickness", "SoundPressure"
]

# Select two NEW input features and the output to minimize
X1 = df["ChordLength"].values
X2 = df["SuctionThickness"].values
Y = df["SoundPressure"].values

# Normalize inputs to [0, 1]
X1_norm = (X1 - X1.min()) / (X1.max() - X1.min())
X2_norm = (X2 - X2.min()) / (X2.max() - X2.min())
X_all = np.column_stack([X1_norm, X2_norm])

# Optional: Downsample for speed
n_points = 200
if len(X_all) > n_points:
    idx = np.random.choice(len(X_all), n_points, replace=False)
    X_all = X_all[idx]
    Y = Y[idx]
else:
    idx = np.arange(len(X_all))

noise = np.std(Y) * 0.04
beta = 2.0
n_rounds = 40

X_hist = []
y_hist = []
regret = []
eigval_records = []

for t in range(n_rounds):
    if len(X_hist) > 0:
        kernel = Matern(length_scale=0.3, length_scale_bounds=(0.1, 2.0), nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=noise**2, normalize_y=True, n_restarts_optimizer=3)
        gp.fit(np.array(X_hist), np.array(y_hist))
        mu, sigma = gp.predict(X_all, return_std=True)
        lcb = mu - np.sqrt(beta) * sigma

        K = kernel(np.array(X_hist))
        eigvals = np.linalg.eigvalsh(K)
        eigval_records.append(eigvals)
    else:
        lcb = np.ones(X_all.shape[0]) * -np.inf
        mu = np.zeros(X_all.shape[0])
        sigma = np.ones(X_all.shape[0])
        eigval_records.append(np.array([1.0]))

    if np.any(np.isinf(lcb)):
        chosen_idx = np.random.choice(len(X_all))
    else:
        chosen_idx = np.argmin(lcb)
    X_hist.append(X_all[chosen_idx])
    reward = Y[chosen_idx] + np.random.normal(0, noise)
    y_hist.append(reward)

    # Regret for minimization
    best_mean = np.min(Y)
    regret.append(y_hist[-1] - best_mean)

    # Visualization at last iteration
    if t == n_rounds - 1:
        plt.figure(figsize=(14,6))
        # 2D color plot of GP mean
        plt.subplot(121)
        grid_size = 40
        c_grid = np.linspace(X1_norm.min(), X1_norm.max(), grid_size)
        s_grid = np.linspace(X2_norm.min(), X2_norm.max(), grid_size)
        c_mesh, s_mesh = np.meshgrid(c_grid, s_grid)
        grid_pts = np.column_stack([c_mesh.ravel(), s_mesh.ravel()])
        gp_mean, gp_std = gp.predict(grid_pts, return_std=True)
        Z = gp_mean.reshape(grid_size, grid_size)
        plt.contourf(c_grid, s_grid, Z, levels=20, cmap="viridis")
        plt.colorbar(label="Predicted Sound Pressure")
        hist_arr = np.array(X_hist)
        plt.scatter(hist_arr[:,0], hist_arr[:,1], c='r', s=30, label="Samples")
        plt.xlabel("Chord Length (normalized)")
        plt.ylabel("Suction Side Thickness (normalized)")
        plt.title("GP Mean Sound Pressure Field (Minimization)")
        plt.legend()

        # Cumulative regret
        plt.subplot(222)
        plt.plot(np.cumsum(regret), 'r-', lw=2)
        plt.xlabel("Iteration")
        plt.ylabel("Cumulative Regret")
        plt.title("Cumulative Regret (GP-Min-UCB)")

        # Kernel matrix eigenvalues (last round)
        plt.subplot(224)
        eigs = eigval_records[-1]
        plt.semilogy(np.arange(1, len(eigs)+1), eigs[::-1], 'o-')
        plt.xlabel("Eigenvalue Index (sorted)")
        plt.ylabel("Eigenvalue (log scale)")
        plt.title("Kernel Matrix Eigenvalues (Last Iteration)")
        plt.tight_layout()
        plt.show()
