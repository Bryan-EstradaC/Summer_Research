import numpy as np
import matplotlib.pyplot as plt

# RBF Kernel function
def rbf_kernel(x1, x2, length_scale=1.0, sigma_f=1.0):
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    sqdist = np.sum(x1**2, 1).reshape(-1, 1) + \
             np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
    return sigma_f**2 * np.exp(-0.5 / length_scale**2 * sqdist)


# Sample points (training)
X_train = np.array([[-4.0], [-3.0], [-1.0], [0.0], [2.0]])
y_train = np.sin(X_train).ravel()  # True function (noisy or not)

# Test points
X_test = np.linspace(-5, 5, 100).reshape(-1, 1)

# Compute covariance matrices
K = rbf_kernel(X_train, X_train)
K_s = rbf_kernel(X_train, X_test)
K_ss = rbf_kernel(X_test, X_test)
noise = 1e-8  # for numerical stability

# Compute GP posterior mean and covariance
K_inv = np.linalg.inv(K + noise * np.eye(len(X_train)))
mu_s = K_s.T @ K_inv @ y_train
cov_s = K_ss - K_s.T @ K_inv @ K_s

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(X_train, y_train, 'ro', label='Training points')
plt.plot(X_test, mu_s, 'b-', label='GP mean')
plt.fill_between(X_test.ravel(),
                 mu_s - 1.96 * np.sqrt(np.diag(cov_s)),
                 mu_s + 1.96 * np.sqrt(np.diag(cov_s)),
                 alpha=0.3, color='blue', label='Confidence interval')
plt.title('Gaussian Process Regression')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()
