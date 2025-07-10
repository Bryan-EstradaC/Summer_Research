import numpy as np
import matplotlib.pyplot as plt

# Fast-decaying eigenvalues kernel matrix (nearly rank 1)
K_fast = np.array([
    [1.00, 0.95, 0.90],
    [0.95, 1.00, 0.95],
    [0.90, 0.95, 1.00]
])

# Slow-decaying eigenvalues kernel matrix (more variation)
K_slow = np.array([
    [1.00, 0.50, 0.30],
    [0.50, 1.00, 0.60],
    [0.30, 0.60, 1.00]
])

# Compute eigenvalues
eigvals_fast = np.linalg.eigvalsh(K_fast)[::-1]  # descending order
eigvals_slow = np.linalg.eigvalsh(K_slow)[::-1]

# Plot eigenvalue decay
plt.figure(figsize=(10, 5))
plt.plot(eigvals_fast, 'o-', label='Fast Decay (Smooth Kernel)', linewidth=2)
plt.plot(eigvals_slow, 's--', label='Slow Decay (Wiggly Kernel)', linewidth=2)
plt.title('Eigenvalue Decay of Kernel Matrices')
plt.xlabel('Eigenvalue Index')
plt.ylabel('Eigenvalue Magnitude')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

eigvals_fast, eigvals_slow
