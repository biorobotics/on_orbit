import numpy as np
import matplotlib.pyplot as plt

load_path = '/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/11_1_24/'

ipopt_qs = np.load(load_path + 'ipopt_qs.npy', allow_pickle=True)
ipopt_vs = np.load(load_path + 'ipopt_vs.npy', allow_pickle=True)
sim_qs = np.load(load_path + 'sim_qs.npy', allow_pickle=True)
sim_vs = np.load(load_path + 'sim_vs.npy', allow_pickle=True)
print(ipopt_qs.shape)
print(ipopt_vs.shape)
print(sim_qs.shape)
print(sim_vs.shape)
qfig=plt.figure()

qaxes=qfig.subplots(6,4)
print(len(qaxes))

for i in range(21):
    qaxes[i//4,i%4].plot(ipopt_qs[:,i], label='ipopt')
    qaxes[i//4,i%4].plot(sim_qs[:,i], label='sim')
    qaxes[i//4,i%4].set_title(f'q{i}')
    qaxes[i//4,i%4].legend()
    qaxes[i//4,i%4].grid(True)
qfig.set_tight_layout(True)
vfig=plt.figure()

vaxes=vfig.subplots(6,4)

for i in range(19):
    vaxes[i//4,i%4].plot(ipopt_vs[:,i], label='ipopt')
    vaxes[i//4,i%4].plot(sim_vs[:,i], label='sim')
    vaxes[i//4,i%4].set_title(f'v{i}')
    vaxes[i//4,i%4].legend()
    vaxes[i//4,i%4].grid(True)
vfig.set_tight_layout(True)

plt.show()
    

