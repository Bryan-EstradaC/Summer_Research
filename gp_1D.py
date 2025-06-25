import numpy as np
import matplotlib.pyplot as plt

def k(xs, ys, sigma=1, l=1):
    dx = np.expand_dims(xs, 1) - np.expand_dims(ys, 0)
    return (sigma ** 2) * np.exp(-((dx / l) ** 2) / 2)

def m(x):
    return np.zeros_like(x)

n = 100
xs = np.linspace(-5, 5, n)
K = k(xs, xs)
mu = m(xs)

plt.figure(figsize=(10, 6))
for _ in range(5):
    ys = np.random.multivariate_normal(mu, K)
    plt.plot(xs, ys)
plt.title("Samples from the GP prior")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.show()

# coefs[i] is the coefficient of x^i
coefs = [6, -2.5, -2.4, -0.1, 0.2, 0.03]

def f(x):
    total = 0
    for exp, coef in enumerate(coefs):
        total += coef * (x ** exp)
    return total

xs = np.linspace(-5.0, 3.5, 100)
ys = f(xs)

# Plot the result
plt.figure(figsize=(8, 5))
plt.plot(xs, ys, label='Polynomial Curve', color='blue')
plt.title('Polynomial Plot')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.legend()
plt.show()

x_obs = np.array([-4, -1.5, 0, 1.5, 2.5, 2.7])
y_obs = f(x_obs)

x_s = np.linspace(-8, 7, 80)

K = k(x_obs, x_obs)
K_s = k(x_obs, x_s)
K_ss = k(x_s, x_s)

K_sTKinv = np.matmul(K_s.T, np.linalg.pinv(K))

mu_s = m(x_s) + np.matmul(K_sTKinv, y_obs - m(x_obs))
Sigma_s = K_ss - np.matmul(K_sTKinv, K_s)

stds = np.sqrt(np.diag(Sigma_s))

# Plot
plt.figure(figsize=(12, 6))

# True function
y_true = f(x_s)
plt.plot(x_s, y_true, 'k--', linewidth=2, alpha=0.4, label='True f(x)')

# Observations
plt.scatter(x_obs, y_obs, marker='x', s=80, color='red', label='Training data')

# Confidence interval (±2σ)
plt.fill_between(x_s, mu_s - 2*stds, mu_s + 2*stds,
                 color='gray', alpha=0.2, label='Uncertainty')

# GP mean
plt.plot(x_s, mu_s, color='blue', linewidth=2.5, alpha=0.7, label='Mean')

# GP samples
for i in range(3):
    sample = np.random.multivariate_normal(mu_s, Sigma_s)
    plt.plot(x_s, sample, linewidth=1, alpha=0.7)

plt.ylim(-7, 8)
plt.title("Gaussian Process Posterior")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()