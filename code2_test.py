import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

def terrain(X):
    return np.sin(X[:, 0]) * np.cos(X[:, 1]) + 0.1 * X[:, 0] + 0.1 * X[:, 1]

def compute_ucb(gp, X, beta=2.0):
    mu, sigma = gp.predict(X, return_std=True)
    return mu + np.sqrt(beta) * sigma, mu, sigma

def plot_gp_step_3d(X, y, x1_grid, x2_grid, mu, step, elev=40, azim=120):
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x1_grid, x2_grid, mu, cmap='terrain', alpha=0.9, rstride=1, cstride=1, antialiased=True)
    ax.scatter(X[:, 0], X[:, 1], y, c='red', s=50, label='Samples')
    ax.set_title(f'GP Mean Surface (Terrain, Matern ν=2.5) - Iter {step}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Predicted Elevation')
    ax.view_init(elev=elev, azim=azim)
    fig.colorbar(surf, shrink=0.5, aspect=10, label='GP Mean')
    if len(y) > 1:
        ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()

def run_gp_ucb_terrain_matern(
    n_iter=25,
    beta=2.0,
    noise=0.1,
    random_seed=42
):
    np.random.seed(random_seed)
    grid_size = 40
    x1 = np.linspace(0, 10, grid_size)
    x2 = np.linspace(0, 10, grid_size)
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    X_grid = np.column_stack([x1_grid.ravel(), x2_grid.ravel()])

    # Standard Matern kernel (ν=2.5)
    kernel_obj = Matern(length_scale=1.0, length_scale_bounds=(1e-8, 1e3), nu=2.5)

    x0 = np.random.uniform(4.0, 6.0, size=2).reshape(1, -1)
    X = x0
    y = terrain(X) + np.random.normal(0, noise, size=X.shape[0])

    for step in range(1, n_iter + 1):
        gp = GaussianProcessRegressor(kernel=kernel_obj, alpha=noise**2, normalize_y=True, n_restarts_optimizer=5)
        gp.fit(X, y)
        ucb, mu, sigma = compute_ucb(gp, X_grid, beta)

# Avoid duplicate sampling
        ucb_mask = np.ones(len(X_grid), dtype=bool)
        for x_seen in X:
            dist = np.linalg.norm(X_grid - x_seen, axis=1)
            ucb_mask &= dist > 1e-3
        X_grid_masked = X_grid[ucb_mask]

        # Pick next point as max UCB among unsampled
        x_next = X_grid_masked[np.argmax(ucb[ucb_mask])].reshape(1, -1)
        y_next = terrain(x_next) + np.random.normal(0, noise)

        X = np.vstack((X, x_next))
        y = np.append(y, y_next)

        # Reshape mean for surface plot
        mu_grid = mu.reshape(x1_grid.shape)
        plot_gp_step_3d(X, y, x1_grid, x2_grid, mu_grid, step)

    return X, y

# Run the 3D visualization for the terrain/hills function using standard Matern kernel
X_samples, y_samples = run_gp_ucb_terrain_matern(n_iter=25, beta=2.0, noise=0.1, random_seed=40)
