import numpy as np
import matplotlib.pyplot as plt

load_path_2 = '/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/11_11_24/charecterizing_time/'
load_path = '/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/11_18_24/mpc_real_pose/'
ipopt_ee_pos = np.load(load_path + 'ref_ee_pos_world_real.npy', allow_pickle=True)
sim_ee_pos = np.load(load_path + 'sw_peg_pos_trj.npy', allow_pickle=True)
times = np.load(load_path + 'sim_ts.npy', allow_pickle=True)
times = times[0:]
ref_len = np.load(load_path_2 + 'sw_peg_pos_trj.npy', allow_pickle=True).shape[0]

truncate_to_shortest = True

if truncate_to_shortest:
    min_length = min(len(ipopt_ee_pos), len(sim_ee_pos), ref_len)
    ipopt_ee_pos = ipopt_ee_pos[:min_length]
    sim_ee_pos = sim_ee_pos[:min_length]
    times = times[:min_length]

print(ipopt_ee_pos.shape)
print(sim_ee_pos.shape)

fig = plt.figure()
axes = fig.subplots(3, 1)

for i in range(3):
    axes[i].plot(times,ipopt_ee_pos[:, i], label='reference')
    axes[i].plot(times, sim_ee_pos[:, i], label='tracked')
    axes[i].set_title(f'ee_pos{i}')
    #axes[i].legend()
    axes[i].grid(True)

fig.set_tight_layout(True)
plt.show()



