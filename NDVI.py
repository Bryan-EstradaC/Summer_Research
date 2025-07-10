import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

# Load the data
df = pd.read_csv('ds/Costa.csv')
df1 = df.copy(deep=True)
df1 = df1.dropna()

df1[['longitude', 'latitude']] = df1['geometry'].str.extract(r'POINT \((-?\d+\.\d+) (-?\d+\.\d+)\)')

# Convert to float
df1['longitude'] = pd.to_numeric(df1['longitude'])
df1['latitude'] = pd.to_numeric(df1['latitude'])

predictors = ['RED', 'NIR', 'longitude', 'latitude']
target = 'ndvi'

X = df1[predictors].values
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)
y = df1[target].values

np.random.seed(7)

n_arms = 1000  # reduce for speed if you like
idx = np.random.choice(len(X_norm), n_arms, replace=False)
X_arms = X_norm[idx]
y_arms = y[idx]

beta = 4.0
noise = np.std(y_arms) * 0.08
n_rounds = 40

X_hist, y_hist, regret, eigval_records  = [], [], [], []

kernel = Matern(length_scale= 10.0, length_scale_bounds = (1e-8, 1e3))
gp = GaussianProcessRegressor(kernel=kernel, alpha=noise**2, normalize_y=True, n_restarts_optimizer=10)

for t in range(n_rounds):
    if len(X_hist) > 0:
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
    
# Regret plot
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