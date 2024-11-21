import numpy as np
import matplotlib.pyplot as plt

load_path = '/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/11_18_24/pos__0.1_0.1_0.0_rot_0.0_0.0_0.0_delta_v_0.0_0.0_0.0_mrv_w_0.0_0.0_0.0_client_w_0.0_0.0_0.020241118-144144/'

run_times = np.load(load_path + 'solve_times.npy', allow_pickle=True)
# run_times = run_times[4:]

# Define the time step
time_step = 0.01
# Create the time array
time_array = np.arange(len(run_times)) * time_step

fig = plt.figure()

plt.plot(time_array, run_times, label='MPC Solver Run Time')
plt.axhline(y=0.01, xmin=0, color='r', linestyle='--', label='Control Loop Period (0.01s)')
plt.xlabel('Time (seconds)')
plt.ylabel('Run Time (seconds)')
#plt.legend()

fig.set_tight_layout(True)
plt.show()
