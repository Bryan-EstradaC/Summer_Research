# Loads librarys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern

# Toy function: f(x) = sin(x)
def f(x):
    return np.sin(x)

#2nd code block
def compute_ucb(gp, X, beta=2.0):
    """Compute Upper Confidence Bound for GP."""
    mu, sigma = gp.predict(X, return_std=True)
    return mu + np.sqrt(beta) * sigma, mu, sigma

def plot_gp_step(X, y, X_grid, mu, sigma, ucb, step, show=True):
    plt.figure(figsize=(10, 5))
    plt.fill_between(X_grid.ravel(), mu - sigma, mu + sigma, alpha=0.2, label='Posterior Std')
    plt.plot(X_grid, f(X_grid), 'g--', label='$f(x) = \sin(x)$')
    plt.plot(X_grid, mu, 'b', label='Posterior Mean')
    plt.plot(X_grid, ucb, 'r:', label='UCB')
    plt.scatter(X, y, c='black', s=50, label='Samples')
    plt.title(f'GP-UCB Iteration {step}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    if show:
        plt.show()

#3rd code block
def run_gp_ucb(
    n_iter=50, 
    beta=2.0, 
    noise=0.1, 
    kernel='RBF', 
    random_seed=42
):
    np.random.seed(random_seed)
    # Discretize the input space
    X_grid = np.linspace(0, 10, 500).reshape(-1, 1)
    
    # Select kernel
    if kernel == 'RBF':
        kernel_obj = RBF(length_scale=1.0)
    elif kernel == 'Matern':
        kernel_obj = Matern(length_scale=1.0, nu=2.5)
    else:
        raise ValueError("Kernel must be 'RBF' or 'Matern'")
    
    # Initial data (can start empty or with a single random sample)
    X = np.array([[0.0]])
    print(X)
    y = f(X).ravel() + np.random.normal(0, noise, size=X.shape[0])
    
    for step in range(1, n_iter + 1):
        gp = GaussianProcessRegressor(kernel=kernel_obj, alpha=noise**2, normalize_y=True)
        gp.fit(X, y)
        ucb, mu, sigma = compute_ucb(gp, X_grid, beta)
        
        # Select x with highest UCB
        x_next = X_grid[np.argmax(ucb)]
        y_next = f(x_next) + np.random.normal(0, noise)
        
        # Update dataset
        X = np.vstack((X, [x_next]))
        print(X)
        y = np.append(y, y_next)
        
        # Plot step
        plot_gp_step(X, y, X_grid, mu, sigma, ucb, step)
        
    return X, y

#4rth code block Visualizations
	# Example: Run with RBF kernel, beta=2, 10 iterations, noise=0.1
X_samples, y_samples = run_gp_ucb(n_iter=10, beta=2.0, noise=0.1, kernel='RBF')
