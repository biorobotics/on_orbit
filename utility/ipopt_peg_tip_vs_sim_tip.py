import numpy as np
import matplotlib.pyplot as plt

load_path = '/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/11_11_24/charecterizing_time/'

ipopt_ee_pos = np.load(load_path + 'ref_ee_pos_trj.npy', allow_pickle=True)
sim_ee_pos = np.load(load_path + 'sw_peg_pos_trj.npy', allow_pickle=True)

print(ipopt_ee_pos.shape)
print(sim_ee_pos.shape)

fig = plt.figure()
axes = fig.subplots(3, 1)

for i in range(3):
    axes[i].plot(ipopt_ee_pos[:, i], label='ipopt')
    axes[i].plot(sim_ee_pos[:, i], label='sim')
    axes[i].set_title(f'ee_pos{i}')
    axes[i].legend()
    axes[i].grid(True)

fig.set_tight_layout(True)
plt.show()



