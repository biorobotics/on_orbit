#!/usr/bin/env python3
import numpy as np
import time
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import WrenchStamped
from holodeck_interface import HolodeckInterface

import rospy
import rospkg
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from ros_utils import ur_controllers

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)

dt = 0.01
rospy.init_node('calibrate')

sensor_array = None

def sensor_callback(data):
  global sensor_array
  global ft_bias

  if sensor_array is None:
    sensor_array = np.zeros(6)
  sensor_array[0] = data.wrench.force.x
  sensor_array[1] = data.wrench.force.y
  sensor_array[2] = data.wrench.force.z
  sensor_array[3] = data.wrench.torque.x
  sensor_array[4] = data.wrench.torque.y
  sensor_array[5] = data.wrench.torque.z

numbers = np.random.choice(10, size=(5,))
passcode = ''.join([str(n) for n in numbers])
entry = input('Type ' + passcode + ' and press enter once you have made sure the peg is far away from the nozzle: ')
if entry != passcode:
  print('Passcode incorrect')
  quit()




holo_cntrl = HolodeckInterface(dt=0.01)
# print(repr(ctrl.jstate_right))
# quit()
mrv_home_angles = np.array([-3.229931179677145, -2.183319231072897, -1.407126545906067, -2.705010076562399, -1.6619661490069788, 5.393088340759277]) # Peg roughly horizontal
# mrv_arm_home_angles = np.array([ 1.63799095, -1.52442773, -2.37254238, -0.80219872,  4.6899662 , -5.54509145]) # Peg roughly vertical
holo_cntrl.ur_idle_mode('mrv')
holo_cntrl.ur_joint_move('mrv', mrv_home_angles)
time.sleep(5)

sensorSub = rospy.Subscriber('/netft_3_data', WrenchStamped, sensor_callback)

rate = rospy.Rate(1/dt)
while sensor_array is None:
  if rospy.is_shutdown():
    quit()
  print('waiting for force/torque data')
  rate.sleep()

rospack = rospkg.RosPack()
rospath = rospack.get_path('on_orbit')

# Initialize pinocchio model (used for forward and inverse kinematics)
urdf_file = rospath + '/urdf/mrv_ur_arm.urdf'
pin_model = RobotWrapper.BuildFromURDF(urdf_file).model
pin_data = pin.Data(pin_model)

jidx_mrv_arm = pin_model.getJointId('ur_3_shoulder_pan_joint')
qidx_mrv_arm = pin_model.idx_qs[jidx_mrv_arm]
vidx_mrv_arm = pin_model.idx_vs[jidx_mrv_arm]

# Get end-effector and F/T sensor frame ids, used to retreive their
# poses during forward kinematics
ft_fid = pin_model.getFrameId('ft_sensor')

q = pin.neutral(pin_model)
v = np.zeros(pin_model.nv)
ft_twist_ctrl = np.zeros(6) # Commanded local-world-aligned twist for mrv_arm end-effector

# Limits for linear and angular velocity of the mrv_arm end-effector
lin_vel_limit = 0.05
ang_vel_limit = 0.15

# Limit on the joint velocities
max_joint_vel = np.pi/8*np.ones(6)
max_joint_acc = np.pi/4*np.ones(6)

rate = rospy.Rate(1/dt)

# Get mass of everything after the ft sensor

def move_to_waypoint(ft_pos_d, ft_rmat_d):
  mrv_hil_js = holo_cntrl.get_mrv_hil_js()
  q[qidx_mrv_arm:qidx_mrv_arm + 6] = mrv_hil_js.position[1:7]
  v[vidx_mrv_arm:vidx_mrv_arm + 6] = mrv_hil_js.velocity[1:7]

  pin.forwardKinematics(pin_model, pin_data, q)
  pin.updateFramePlacement(pin_model, pin_data, ft_fid)
  ft_pos = pin_data.oMf[ft_fid].translation
  ft_rmat = pin_data.oMf[ft_fid].rotation

  pos_err = np.linalg.norm(ft_pos - ft_pos_d)
  rot_err = np.linalg.norm(pin.log3(ft_rmat_d.transpose()@ft_rmat))
  holo_cntrl.ur_idle_mode('mrv')
  holo_cntrl.ur_velocity_mode('mrv')
  while not rospy.is_shutdown() and (np.linalg.norm(pos_err) > 1e-3 or np.linalg.norm(rot_err) > 0.1*np.pi/180):
    # Get joint positions and velocities
    mrv_hil_js = holo_cntrl.get_mrv_hil_js()
    q[qidx_mrv_arm:qidx_mrv_arm + 6] = mrv_hil_js.position[1:7]
    v[vidx_mrv_arm:vidx_mrv_arm + 6] = mrv_hil_js.velocity[1:7]

    # Forward kinematics: get poses and Jacobians
    pin.forwardKinematics(pin_model, pin_data, q)
    pin.computeJointJacobians(pin_model, pin_data)
    pin.updateFramePlacement(pin_model, pin_data, ft_fid)
    ft_pos = pin_data.oMf[ft_fid].translation
    ft_rmat = pin_data.oMf[ft_fid].rotation
    Jft = pin.getFrameJacobian(pin_model, pin_data, ft_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

    pos_kp = 2
    # rot_kp = 0.2
    rot_kp = 1

    pos_err = ft_pos - ft_pos_d
    rot_err = ft_rmat@pin.log3(ft_rmat_d.transpose()@ft_rmat)
    ft_twist_ctrl[:3] = -pos_kp*pos_err
    ft_twist_ctrl[3:] = -rot_kp*rot_err

    # Clip the twist
    ft_twist_ctrl[:3] = np.clip(ft_twist_ctrl[:3], -lin_vel_limit, lin_vel_limit)
    ft_twist_ctrl[3:] = np.clip(ft_twist_ctrl[3:], -ang_vel_limit, ang_vel_limit)

    # We have end-effector velocity commands. Now we solve
    # for joint velocity commands using inverse kinematics
    v_mrv_arm_cmd = np.linalg.lstsq(Jft, ft_twist_ctrl)[0]

    # Clip joint velocities to satisfy vel limits
    v_mrv_arm_cmd = np.clip(v_mrv_arm_cmd, -max_joint_vel, max_joint_vel)

    # Clip joint velocities to satisfy acc limits
    acc_induced_min_vel_mrv_arm = v - max_joint_acc*dt
    acc_induced_max_vel_mrv_arm = v + max_joint_acc*dt

    v_mrv_arm_cmd = np.clip(v_mrv_arm_cmd, acc_induced_min_vel_mrv_arm, acc_induced_max_vel_mrv_arm)
    acc_norm_mrv_arm = np.linalg.norm((v_mrv_arm_cmd - v)/dt, ord=np.inf)
    holo_cntrl.cmd_ur_velocity('mrv', v_mrv_arm_cmd)

    rate.sleep()

num_samples_per_waypoint = 5000

def get_av_wrench():
  global sensor_array

  av_wrench = 0
  for i in range(num_samples_per_waypoint):
    av_wrench = (i*av_wrench + sensor_array)/(i + 1)
    rate.sleep()

  return av_wrench

mrv_hil_js = holo_cntrl.get_mrv_hil_js()
q[qidx_mrv_arm:qidx_mrv_arm + 6] = mrv_hil_js.position[1:7]
v[vidx_mrv_arm:vidx_mrv_arm + 6] = mrv_hil_js.velocity[1:7]
pin.forwardKinematics(pin_model, pin_data, q)
pin.updateFramePlacement(pin_model, pin_data, ft_fid)
ft_pos = pin_data.oMf[ft_fid].translation

# Keep the peg horizontal and roll it, measuring average wrench at each stopping point
'''
ft_pos_d = np.copy(ft_pos)
wrenches = []
num_angles = 32
for i in range(num_angles):
  ft_rmat_d = pin.exp3(np.array([0., 0., 2*np.pi*i/num_angles]))
  move_to_waypoint(ft_pos_d, ft_rmat_d)
  wrenches.append(get_av_wrench())

np.save(rospath + '/data/09_10_24/ft_calibration/z/wrenches.npy', wrenches)

quit()
'''

# Keep the peg vertical and roll it, measuring average wrench at each stopping point
'''
ft_pos_d = np.copy(ft_pos)
wrenches = []
num_angles = 32
for i in range(num_angles):
  ft_rmat_d = pin.exp3(np.array([np.pi/2, 0., 0.]))@pin.exp3(np.array([0., 0., -2*np.pi*i/num_angles]))
  move_to_waypoint(ft_pos_d, ft_rmat_d)
  wrenches.append(get_av_wrench())

np.save(rospath + '/data/09_10_24/ft_calibration/xy/wrenches.npy', wrenches)

quit()
'''

# Take wrench measurements at 0, 90, 180, 270 degrees to determine gravity comp terms

# Take measurement with peg horizontal
ft_pos_d = np.copy(ft_pos)
ft_rmat_d = pin.exp3(np.array([0., np.pi/2, 0.]))
move_to_waypoint(ft_pos_d, ft_rmat_d)
av_wrench_1 = get_av_wrench()

# Take measurement with peg horizontal but rotated 180 deg about long axis
ft_rmat_d = pin.exp3(np.array([0., np.pi/2, 0.]))@pin.exp3(np.array([0., 0., np.pi]))
move_to_waypoint(ft_pos_d, ft_rmat_d)
av_wrench_2 = get_av_wrench()

print(av_wrench_1 - av_wrench_2)

# Take measurement with peg horizontal but rotated 90 deg about long axis
ft_rmat_d = pin.exp3(np.array([0., np.pi/2, 0.]))@pin.exp3(np.array([0., 0., np.pi/2]))
move_to_waypoint(ft_pos_d, ft_rmat_d)
av_wrench_3 = get_av_wrench()

# Take measurement with peg horizontal but rotated -90 deg about long axis
ft_rmat_d = pin.exp3(np.array([0., np.pi/2, 0.]))@pin.exp3(np.array([0., 0., -np.pi/2]))
move_to_waypoint(ft_pos_d, ft_rmat_d)
av_wrench_4 = get_av_wrench()

print(av_wrench_3 - av_wrench_4)

grav_force1 = np.abs((av_wrench_1[1] - av_wrench_2[1])/2)
grav_force2 = np.abs((av_wrench_3[0] - av_wrench_4[0])/2)
grav_force = (grav_force1 + grav_force2)/2
g = 9.81
mass = grav_force/g
print('Mass = %f' %(mass))

grav_torque1 = np.abs((av_wrench_1[3] - av_wrench_2[3])/2)
grav_torque2 = np.abs((av_wrench_3[4] - av_wrench_4[4])/2)
grav_torque = (grav_torque1 + grav_torque2)/2
com = grav_torque/grav_force
print('CoM = %f' %(com))
