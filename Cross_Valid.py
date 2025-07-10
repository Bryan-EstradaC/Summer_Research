import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('ds/Costa.csv')
df1 = df.copy(deep=True)
df1 = df1.dropna()

#Separate geometry to longitude and latitude
df1[['longitude', 'latitude']] = df1['geometry'].str.extract(r'POINT \((-?\d+\.\d+) (-?\d+\.\d+)\)')
df1['longitude'] = pd.to_numeric(df1['longitude'])
df1['latitude'] = pd.to_numeric(df1['latitude'])

predictors = ['RED', 'NIR', 'longitude', 'latitude']
target = 'ndvi'

X = df1[predictors].values
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)
y = df1[target].values

n_arms = 1000  # reduce for speed if you like
np.random.seed(7)
idx = np.random.choice(len(X_norm), n_arms, replace=False)
X_arms = X_norm[idx]
y_arms = y[idx]

noise = np.std(y_arms) * 0.08

kernel = Matern(length_scale=10.0, nu=2.5) + WhiteKernel(noise_level= 1.0)
gp = GaussianProcessRegressor(kernel=kernel, alpha=noise**2, normalize_y=True, n_restarts_optimizer=3)

K = 5  # 5-fold is standard
kf = KFold(n_splits=K, shuffle=True)
mse_scores = []

# --- 3. Cross-validation loop ---
for fold, (train_idx, test_idx) in enumerate(kf.split(X_arms)):
    X_train, X_test = X_arms[train_idx], X_arms[test_idx]
    y_train, y_test = y_arms[train_idx], y_arms[test_idx]
   
    # Fit GP to training data
    gp.fit(X_train, y_train)
   
    # Predict on held-out fold
    y_pred, sigma = gp.predict(X_test, return_std=True)
   
    # Compute mean squared error for this fold
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)
    print(f"Fold {fold+1}: MSE = {mse:.6f}")


# --- 4. Summarize results ---
avg_mse = np.mean(mse_scores)
std_mse = np.std(mse_scores)
print(f"\nAverage cross-validated MSE: {avg_mse:.6f} ± {std_mse:.6f}")


# --- 5. (Optional) Visualize predictions for last fold ---
plt.figure(figsize=(8,5))
plt.errorbar(y_test, y_pred, yerr=2*sigma, fmt='o', color='b', alpha=0.7, label='Prediction ±2σ')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', label='Perfect Prediction')
plt.xlabel("True NDVI")
plt.ylabel("Predicted NDVI")
plt.title("GP Regression: True vs Predicted (Last Fold)")
plt.legend()
plt.tight_layout()
plt.show()