import numpy as np
import time
import pinocchio as pin
import roboticstoolbox as rtb
from spatialmath import SE3
from pinocchio.robot_wrapper import RobotWrapper
from scipy.spatial.transform import Rotation as R
from holodeck_interface import HolodeckInterface

import rospy
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped, WrenchStamped, PointStamped, Point
from std_msgs.msg import String, Float32, Float32MultiArray 
import pickle

from on_orbit.on_orbit_bindings import IKMoveEEBehindBarrier

import os

from visualization_msgs.msg import Marker

import matplotlib.pyplot as plt

a= pickle.load(open('/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_launchpad/scripts/data.pkl', 'rb'))
print(a.keys()) 
mrv_del_angs= a['del_angles_mrv']
xx=[]
y=[]
xy=[]
xz=[]
for i in range(1, len(mrv_del_angs)):
    xx.append(mrv_del_angs[i][0])
    y.append(i)
    xy.append(mrv_del_angs[i][1])
    xz.append(mrv_del_angs[i][2])    

client_del_angs= a['del_angles_client']
xcx=[]
yc=[]
xcy=[]
xcz=[]
for i in range(1, len(client_del_angs)):
    xcx.append(client_del_angs[i][0])
    yc.append(i)
    xcy.append(client_del_angs[i][1])
    xcz.append(client_del_angs[i][2])

plt.plot(y,xx, color='blue', label='mrv_del_angles_x')
plt.plot(y,xy, color='green', label='mrv_del_angles_y')
plt.plot(y,xz, color='purple', label='mrv_del_angles_z')

plt.plot(yc,xcx, color='red', label='client_del_angles_x')
plt.plot(yc,xcy, color='orange', label='client_del_angles_y')
plt.plot(yc,xcz, color='black', label='client_del_angles_z')
plt.legend()
plt.show()














# xx=[]
# yx=[]
# xy=[]
# yy=[]
# xz=[]
# yz=[]
# t=12000
# a= np.load('mrv_orientation.npy')
# for i in range(0, t):
#     xx.append(a[i][0])
#     yx.append(i)
#     xy.append(a[i][1])
#     xz.append(a[i][2])    
# a_nt = np.load('mrv_orientation_nt.npy')
# xtx=[]
# ytx=[]
# xty=[]
# xtz=[]
# for i in range(0, t):
#     xtx.append(a_nt[i][0])
#     ytx.append(i)
#     xty.append(a_nt[i][1])
#     xtz.append(a_nt[i][2])
# plt.plot(yx,xx, color='blue', label='mrv_orientation_x')
# # plt.plot(ytx,xtx, color='red', label='mrv_orientation_nt_x')
# plt.plot(yx,xy, color='green', label='mrv_orientation_y')
# # plt.plot(ytx,xty, color='orange', label='mrv_orientation_nt_y')
# plt.plot(yx,xz, color='purple', label='mrv_orientation_z')
# # plt.plot(ytx,xtz, color='black', label='mrv_orientation_nt_z')
# plt.legend()



# nozzle_poses = np.load('nozzle_poses.npy')
# nozzle_twists = np.load('nozzle_twists.npy')
# peg_is_close = np.load('peg_is_close.npy')
# peg_force = np.load('peg_force.npy')


# nozzle_x = []
# nozzle_y = []
# nozzle_z = []

# nozzle_vx = []
# nozzle_vy = []
# nozzle_vz = []

# peg_force_x = []
# peg_force_y = []
# peg_force_z = []

# peg_close = []


# y=[]

# for i in range(1000,nozzle_poses.shape[0]):
#     y.append(i)
#     nozzle_x.append(nozzle_poses[i][0])
#     nozzle_y.append(nozzle_poses[i][1])
#     nozzle_z.append(nozzle_poses[i][2])

#     nozzle_vx.append(nozzle_twists[i][0])
#     nozzle_vy.append(nozzle_twists[i][1])
#     nozzle_vz.append(nozzle_twists[i][2])

#     peg_force_x.append(peg_force[i][0]/50)
#     peg_force_y.append(peg_force[i][1]/50)
#     peg_force_z.append(peg_force[i][2]/50)

#     if peg_is_close[i]:
#         peg_close.append(0.02)
#     else:
#         peg_close.append(0)
# #     time.sleep(0.1)

# plt.plot(y,nozzle_x, color='blue', label='nozzle_x')
# plt.plot(y,nozzle_y, color='green', label='nozzle_y')
# plt.plot(y,nozzle_z, color='purple', label='nozzle_z')

# # plt.plot(y,nozzle_vx, color='orange', label='nozzle_vx')
# # plt.plot(y,nozzle_vy, color='red', label='nozzle_vy')
# # plt.plot(y,nozzle_vz, color='black', label='nozzle_vz')

# plt.plot(y,peg_force_x, color='yellow', label='peg_force_x')
# plt.plot(y,peg_force_y, color='cyan', label='peg_force_y')
# plt.plot(y,peg_force_z, color='magenta', label='peg_force_z')

# plt.plot(y,peg_close, color='brown', label='peg_close')

# plt.legend()

# print("final nozzle pose: ", nozzle_poses[-1])
plt.show()