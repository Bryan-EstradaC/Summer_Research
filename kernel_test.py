import numpy as np
import matplotlib.pyplot as plt

# Define the RBF (squared exponential) kernel function
def rbf_kernel(x1, x2, length_scale=1.0, sigma_f=1.0):
    sqdist = np.subtract.outer(x1, x2)**2
    return sigma_f**2 * np.exp(-0.5 / length_scale**2 * sqdist)

# Create input points
x = np.linspace(-5, 5, 100).reshape(-1)  # 1D array

# Compute the kernel matrix
K = rbf_kernel(x, x)

# Plot the kernel matrix
plt.figure(figsize=(6, 5))
plt.imshow(K, cmap='hot', extent=[-5, 5, -5, 5])
plt.title('RBF Kernel Matrix')
plt.xlabel('x')
plt.ylabel('x\'')
plt.colorbar(label='Covariance')
plt.show()

