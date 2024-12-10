import numpy as np
import matplotlib.pyplot as plt

# Load the .npy file
data = np.load('/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/11_25_24/pos__0.0_0.0_-0.05_rot_0.0_0.0_0.0_delta_v_0.0_0.0_0.0_mrv_w_0.0_0.0_0.0_client_w_0.0_0.0_0.020241125-184918/mrv_peg_force_trj.npy')

# Extract the forces (first three columns)
forces = data[:, 0:3]

# Time axis (assuming equally spaced time steps)
time = np.arange(len(forces))

# Plot forces over time
plt.figure(figsize=(10, 6))
plt.plot(time, forces[:, 0], label='Force X')
plt.plot(time, forces[:, 1], label='Force Y')
plt.plot(time, forces[:, 2], label='Force Z')
plt.xlabel('Time')
plt.ylabel('Force (N)')
plt.title('Forces over Time')
plt.legend()
plt.grid(True)
plt.show()
