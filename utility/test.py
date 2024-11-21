import numpy as np
import matplotlib.pyplot as plt

# Generate 1000 values switching between +3.14 and -3.14 with noise
angles = np.linspace(-3.14, 3.14, 1000)
# Add noise to the angles
noise = np.random.normal(0, 0.5, angles.shape)  # small noise
angles[::2] = 3.14 + noise[::2]  # Add noise to values around +pi
angles[1::2] = -3.14 + noise[1::2]  # Add noise to values around -pi

# Use np.unwrap to smooth out the jumps at the +-pi boundary
angles_unwrapped = np.unwrap(angles)

# Plot the original noisy and unwrapped angles
plt.figure(figsize=(10, 5))
plt.plot(angles, label='Original Noisy Angles (Switching)', alpha=0.5)
plt.plot(angles_unwrapped, label='Unwrapped Angles (Smooth)', linestyle='--')
plt.title('Angle Unwrapping for Noisy Data')
plt.xlabel('Sample Index')
plt.ylabel('Angle (radians)')
plt.legend()
plt.show()
