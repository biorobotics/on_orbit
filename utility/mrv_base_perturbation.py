import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# Load data
load_path1 = '/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/12_2_24/tvlqr/no_orientation_cost/pos__0.0_0.0_-0.05_rot_0.0_0.0_0.0_delta_v_0.0_0.0_0.0_mrv_w_0.0_0.0_0.0_client_w_0.0_0.0_0.020241202-154712'
#load_path1 = '/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/12_3_24/tvlqr/no_orientation_cost/pos__0.1_0.1_0.0_rot_0.0_0.0_0.0_delta_v_0.0_0.0_0.0_mrv_w_0.0_0.0_0.0_client_w_0.0_0.0_0.020241203-081155'
load_path1= '/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/12_3_24/tvlqr/only_orientation_cost/pos__0.0_0.0_-0.05_rot_0.0_0.0_0.0_delta_v_0.0_0.0_0.0_mrv_w_0.0_0.0_0.0_client_w_0.0_0.0_0.020241203-102516'
load_path1= '/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/12_4_24/tvlqr/special_perturbation_only/pos__0.0_0.0_-0.05_rot_0.0_0.0_0.0_delta_v_0.0_0.0_0.0_mrv_w_0.0_0.0_0.0_client_w_0.0_0.0_0.020241204-150551'

# load_path1 = '/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/12_2_24/tvlqr/orientation_cost/pos__0.0_0.0_-0.05_rot_0.0_0.0_0.0_delta_v_0.0_0.0_0.0_mrv_w_0.0_0.0_0.0_client_w_0.0_0.0_0.020241202-160847'
load_path2 = '/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/12_2_24/tvlqr/no_orientation_cost/pos__0.0_0.0_-0.05_rot_0.0_0.0_0.0_delta_v_0.0_0.0_0.0_mrv_w_0.0_0.0_0.0_client_w_0.0_0.0_0.020241202-154712'
load_path2= '/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/12_3_24/tvlqr/4_plots/both_cost/pos__0.0_0.0_-0.05_rot_0.0_0.0_0.0_delta_v_0.0_0.0_0.0_mrv_w_0.0_0.0_0.0_client_w_0.0_0.0_0.020241203-111928'
plot_single_path = False

sw_x_trj1 = np.load(load_path1 + '/' + 'sw_x_trj.npy', allow_pickle=True)
sim_ts1 = np.load(load_path1 + '/' + 'sim_ts.npy', allow_pickle=True)

if not plot_single_path:
    sw_x_trj2 = np.load(load_path2 + '/' + 'sw_x_trj.npy', allow_pickle=True)
    sim_ts2 = np.load(load_path2 + '/' + 'sim_ts.npy', allow_pickle=True)
    
min_len = min(14200,len(sw_x_trj1)) if plot_single_path else min(len(sw_x_trj1), len(sw_x_trj2), len(sim_ts1), len(sim_ts2), 1420)
sw_x_trj1 = sw_x_trj1[:min_len]
sim_ts1 = sim_ts1[:min_len]

if not plot_single_path:
    sw_x_trj2 = sw_x_trj2[:min_len]
    sim_ts2 = sim_ts2[:min_len]

sw_mrv_quat1 = sw_x_trj1[:, 3:7]
sw_mrv_rot1 = R.from_quat(sw_mrv_quat1).as_rotvec()
sw_mrv_rot_deg1 = np.degrees(sw_mrv_rot1)

if not plot_single_path:
    sw_mrv_quat2 = sw_x_trj2[:, 3:7]
    sw_mrv_rot2 = R.from_quat(sw_mrv_quat2).as_rotvec()
    sw_mrv_rot_deg2 = np.degrees(sw_mrv_rot2)

rotation_magnitude_deg1 = np.linalg.norm(sw_mrv_rot_deg1, axis=1)
if not plot_single_path:
    rotation_magnitude_deg2 = np.linalg.norm(sw_mrv_rot_deg2, axis=1)

plt.figure(figsize=(12, 6))
colors = ['r', 'g', 'b']  # RGB colors for x, y, z axes
for i, axis in enumerate(['x', 'y', 'z']):
    plt.plot(sim_ts1, sw_mrv_rot_deg1[:, i], label=f'Orientation Cost {axis}-axis', linestyle='-', color=colors[i])
    if not plot_single_path:
        plt.plot(sim_ts2, sw_mrv_rot_deg2[:, i], label=f'No Orientation Cost {axis}-axis', linestyle='--', color=colors[i])

plt.xlabel('Simulation Time (s)')
plt.ylabel('Rotation Along Axis (degrees)')
plt.title('Rotation Components Along Each Axis (Degrees)')
#plt.legend()
#plt.grid()
plt.tight_layout()
plt.show()

# Plot total rotation magnitudes in degrees
plt.figure(figsize=(12, 6))
plt.plot(sim_ts1, rotation_magnitude_deg1, label='Orientation Cost Rotation Magnitude', linestyle='-', color='k')
if not plot_single_path:
    plt.plot(sim_ts2, rotation_magnitude_deg2, label='No Orientation Cost Rotation Magnitude', linestyle='--', color='k')

plt.xlabel('Simulation Time (s)')
plt.ylabel('Rotation Magnitude (degrees)')
plt.title('Total Rotation Magnitudes (Degrees)')
#plt.legend()
#plt.grid()
plt.tight_layout()
plt.show()
