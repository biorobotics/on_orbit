#!/usr/bin/env python3
import numpy as np
import time
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from mrv_client_sim import MRVClientSim
from scipy.spatial.transform import Rotation as R

import rospy
import rospkg
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped, WrenchStamped, PointStamped, Point

import os
from visualization_msgs.msg import Marker

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)

rospy.init_node('replay_node', anonymous=True)

dt = 0.01

np.random.seed(0)

rospack = rospkg.RosPack()
rospath = rospack.get_path('on_orbit')

dist_centering_waypoint_from_goal = rospy.get_param('dist_centering_waypoint_from_goal')
dist_nozzle_opening_from_goal = rospy.get_param('dist_nozzle_opening_from_goal')
dist_centering_waypoint_from_nozzle_opening = dist_centering_waypoint_from_goal - dist_nozzle_opening_from_goal

cone_slope = rospy.get_param('cone_slope')

# Initialize pinocchio model (used for forward and inverse kinematics)
urdf_file = rospath + '/urdf/on_orbit.urdf'
robot = RobotWrapper.BuildFromURDF(urdf_file)
pin_model = robot.model
pin_data = robot.data

mrv_pin_model = RobotWrapper.BuildFromURDF(rospath + '/urdf/robot.urdf', root_joint=pin.JointModelFreeFlyer()).model
mrv_pin_data = pin.Data(mrv_pin_model)
mrv_tip_fid = mrv_pin_model.getFrameId('ee_tip')

mrv_cv_urdf_file = rospath + '/urdf/robot_cv_detached.urdf'
mrv_cv_pin_model = RobotWrapper.BuildFromURDF(mrv_cv_urdf_file).model
mrv_cv_pin_data = pin.Data(mrv_cv_pin_model)
# mrv_cv_pin_geom = pin.buildGeomFromUrdf(mrv_cv_pin_model, mrv_cv_urdf_file, rospath + '/meshes/', pin.GeometryType.COLLISION)
mrv_cv_tip_fid = mrv_cv_pin_model.getFrameId('ee_tip')
mrv_cv_goal_fid = mrv_cv_pin_model.getFrameId('goal')

mrv_jidx = mrv_cv_pin_model.getJointId('world_to_base')
mrv_qidx = mrv_cv_pin_model.idx_qs[mrv_jidx]
mrv_vidx = mrv_cv_pin_model.idx_vs[mrv_jidx]

cv_jidx = mrv_cv_pin_model.getJointId('world_to_client')
cv_qidx = mrv_cv_pin_model.idx_qs[cv_jidx]
cv_vidx = mrv_cv_pin_model.idx_vs[cv_jidx]

mrv_nq = mrv_cv_pin_model.nq - 7
mrv_nv = mrv_cv_pin_model.nv - 6

'''
tip_geom_id = mrv_cv_pin_geom.getGeometryId('ee_peg_0')
for i in range(8):
  nozzle_geom_id = mrv_cv_pin_geom.getGeometryId('nozzle_geom' + str(i) + '_0')
  mrv_cv_pin_geom.addCollisionPair(pin.CollisionPair(tip_geom_id, nozzle_geom_id))

  hole_geom_id = mrv_cv_pin_geom.getGeometryId('tube_geom' + str(i) + '_0')
  mrv_cv_pin_geom.addCollisionPair(pin.CollisionPair(tip_geom_id, hole_geom_id))

mrv_cv_geom_data = pin.GeometryData(mrv_cv_pin_geom)
'''

# The pinocchio model stacks the UR16 joint angles and UR5 joint angles
# into one large vector q. Similarly, it stacks the derivatives of these quantities into one large vector v.
# Here, we find the starting indices of each arm's joint angles within q,
# and the starting indices of their derivatives in v.
jidx_ur16 = pin_model.getJointId('ur_3_shoulder_pan_joint')
qidx_ur16 = pin_model.idx_qs[jidx_ur16]
vidx_ur16 = pin_model.idx_vs[jidx_ur16]

# Here, we find the starting index of the UR5's joint angles within q,
# and the starting index of the UR5's joint velocities within v
jidx_ur5 = pin_model.getJointId('ur_4_shoulder_pan_joint')
qidx_ur5 = pin_model.idx_qs[jidx_ur5]
vidx_ur5 = pin_model.idx_vs[jidx_ur5]

# Publish joint states for visualization
hw_joints_pub = rospy.Publisher('/on_orbit/hw_joint_state', JointState, queue_size=1)
hw_joints_msg = JointState()
for j in range(1, pin_model.njoints):
  hw_joints_msg.name.append(pin_model.names[j])
  hw_joints_msg.position.append(0)
  hw_joints_msg.velocity.append(0)
  hw_joints_msg.effort.append(0)

sw_joints_pub = rospy.Publisher('/on_orbit/sw_joint_state', JointState, queue_size=1)
sw_joints_msg = JointState()
for j in range(7):
  sw_joints_msg.name.append('joint' + str(j + 1))
  sw_joints_msg.position.append(0)
  sw_joints_msg.velocity.append(0)
  sw_joints_msg.effort.append(0)

contacts_pub = rospy.Publisher('/on_orbit/contacts', Marker, queue_size=1)
contacts_marker = Marker()
contacts_marker.type = Marker.LINE_STRIP
contacts_marker.action = Marker.ADD
contacts_marker.pose.position.x = 0
contacts_marker.pose.position.y = 0
contacts_marker.pose.position.z = 0
contacts_marker.pose.orientation.w = 1
contacts_marker.pose.orientation.x = 0
contacts_marker.pose.orientation.y = 0
contacts_marker.pose.orientation.z = 0
contacts_marker.scale.x = 0.01
'''
contacts_marker.color.r = 1
contacts_marker.color.g = 20/255
contacts_marker.color.b = 147/255
'''
contacts_marker.color.r = 1
contacts_marker.color.g = 1
contacts_marker.color.b = 0
contacts_marker.color.a = 1
contacts_marker.header.frame_id = "client"
contacts_marker.ns = "docking_node"

tf_br = TransformBroadcaster()

sw_base_link_tf_msg = TransformStamped()
sw_base_link_tf_msg.header.frame_id = 'world'
sw_base_link_tf_msg.child_frame_id = 'base_link'

sw_client_tf_msg = TransformStamped()
sw_client_tf_msg.header.frame_id = 'world'
sw_client_tf_msg.child_frame_id = 'client'

sw_client_est_tf_msg = TransformStamped()
sw_client_est_tf_msg.header.frame_id = 'world'
sw_client_est_tf_msg.child_frame_id = 'client_estimate'

waypoint1_tf_msg = TransformStamped()
waypoint1_tf_msg.header.frame_id = 'world'
waypoint1_tf_msg.child_frame_id = 'Waypoint 1'

waypoint2_tf_msg = TransformStamped()
waypoint2_tf_msg.header.frame_id = 'world'
waypoint2_tf_msg.child_frame_id = 'Waypoint 2'

waypoint3_tf_msg = TransformStamped()
waypoint3_tf_msg.header.frame_id = 'world'
waypoint3_tf_msg.child_frame_id = 'Waypoint 3'

# load_path = rospath + '/experiment_logs/03_09_24/joint_space_interp_testing_hw/pos__0.0_0.0_-0.1_rot_0.0_0.0_0.0_delta_v_0.0_0.0_0.0_mrv_w_0.0_0.0_0.0_client_w_0.017453292519943295_0.0_0.0/control/20240309-172743/'
# load_path = rospath + '/experiment_logs/04_16_24/cb_center_with_contact/pos__0.0_0.0_-0.05_rot_0.0_0.0_0.0_delta_v_0.0_0.0_0.0_mrv_w_0.0_0.0_0.0_client_w_0.25_0.0_0.0/control/20240416-133317'
#load_path = rospath + '/experiment_logs/04_16_24/cb_center_with_contact/pos__0.0_0.0_-0.05_rot_0.0_0.0_0.0_delta_v_0.0_0.0_0.0_mrv_w_0.0_0.0_0.0_client_w_0.45000000000000007_0.0_0.0/control/20240416-133452'
#load_path = '/home/biorobotics/Documents/sr_ws/src/on_orbit/experiment_logs/05_06_24/test_lib/pos_0.0_0.0_0.0_rot_0.0_0.0_0.0_delta_v_0.0_0.0_0.0_client_w_0.0_0.0_0.0/control/20240506-200254'
# load_path = '/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/11_15_24/pos__0.0_0.0_-0.05_rot_0.0_0.0_0.0_delta_v_0.0_0.0_0.0_mrv_w_0.0_0.0_0.0_client_w_0.0_0.0_0.020241115-123934'
load_path = '/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/11_15_24/pos__0.0_0.0_-0.05_rot_0.0_0.0_0.0_delta_v_0.0_0.0_0.0_mrv_w_0.0_0.0_0.0_client_w_0.0_0.0_0.020241115-123407'

replay_from_xs = False
replay_from_init_xs = True
if replay_from_xs:
  if replay_from_init_xs:
    x_trj = np.load(load_path + '/init_xs.npy')
  else:
    x_trj = np.load(load_path + '/xs.npy')
  sw_base_pos_trj = x_trj[:, mrv_qidx:mrv_qidx + 3]
  sw_base_rmat_trj = np.array([R.from_quat(quat).as_matrix() for quat in x_trj[:, mrv_qidx + 3:mrv_qidx + 7]])
  sw_joint_angles_trj = x_trj[:, mrv_qidx + 7:mrv_qidx + mrv_nq]
  sw_client_pos_trj = x_trj[:, cv_qidx:cv_qidx + 3]
  sw_client_rmat_trj = np.array([R.from_quat(quat).as_matrix() for quat in x_trj[:, cv_qidx + 3:cv_qidx + 7]])

  sw_nozzle_pos_trj = []
  sw_nozzle_rmat_trj = []
  nozzle_fid = mrv_cv_pin_model.getFrameId('nozzle')
  for q in x_trj[:, :mrv_cv_pin_model.nq]:
    pin.forwardKinematics(mrv_cv_pin_model, mrv_cv_pin_data, q)
    pin.updateFramePlacement(mrv_cv_pin_model, mrv_cv_pin_data, nozzle_fid)
    sw_nozzle_pos_trj.append(mrv_cv_pin_data.oMf[nozzle_fid].translation)
    sw_nozzle_rmat_trj.append(mrv_cv_pin_data.oMf[nozzle_fid].rotation)
else:
  sw_base_pos_trj = np.load(load_path + '/sw_base_pos_trj.npy')
  sw_base_rmat_trj = np.load(load_path + '/sw_base_rmat_trj.npy')
  sw_joint_angles_trj = np.load(load_path + '/sw_joint_angles_trj.npy')

  sw_client_pos_trj = np.load(load_path + '/sw_client_pos_trj.npy')
  sw_client_rmat_trj = np.load(load_path + '/sw_client_rmat_trj.npy')

  # ur16_joint_angles_trj = np.load(load_path + '/ur16_joint_angles_trj.npy')
  # ur5_joint_angles_trj = np.load(load_path + '/ur5_joint_angles_trj.npy')

  sw_nozzle_pos_trj = np.load(load_path + '/sw_nozzle_pos_trj.npy')
  sw_nozzle_rmat_trj = np.load(load_path + '/sw_nozzle_rmat_trj.npy')

rate = rospy.Rate(1/dt)

# trj_idx = len(sw_nozzle_pos_trj) - 1
trj_idx = 0
start_time = rospy.Time.now().to_sec()
while not rospy.is_shutdown():
  do_step = rospy.Time.now().to_sec() - start_time > 2
  # do_step = False
  # Get joint positions and velocities
  '''
  q = pin.neutral(pin_model)
  q[qidx_ur16:qidx_ur16 + 6] = ur16_joint_angles_trj[trj_idx]
  q[qidx_ur5:qidx_ur5 + 6] = ur5_joint_angles_trj[trj_idx]

  for i in range(pin_model.nv):
    hw_joints_msg.position[i] = q[i]
  hw_joints_msg.header.stamp = rospy.Time.now()
  hw_joints_pub.publish(hw_joints_msg)
  '''

  for i in range(len(sw_joint_angles_trj[0])):
    sw_joints_msg.position[i] = sw_joint_angles_trj[trj_idx, i]
  sw_joints_msg.header.stamp = rospy.Time.now()
  sw_joints_pub.publish(sw_joints_msg)

  base_link_pos = sw_base_pos_trj[trj_idx]
  base_link_quat = R.from_matrix(sw_base_rmat_trj[trj_idx]).as_quat()

  q_mrv = np.concatenate((base_link_pos, base_link_quat, sw_joint_angles_trj[trj_idx]))
  pin.forwardKinematics(mrv_pin_model, mrv_pin_data, q_mrv)
  pin.computeJointJacobians(mrv_pin_model, mrv_pin_data)
  pin.updateFramePlacement(mrv_pin_model, mrv_pin_data, mrv_tip_fid)
  Jpeg_sw = pin.getFrameJacobian(mrv_pin_model, mrv_pin_data, mrv_tip_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
  Ag_sw = pin.computeCentroidalMap(mrv_pin_model, mrv_pin_data, q_mrv)
  Jpeg_sw_gen = Jpeg_sw[:, 6:] - Jpeg_sw[:, :6]@np.linalg.solve(Ag_sw[:, :6], Ag_sw[:, 6:])

  cond = np.linalg.cond(Jpeg_sw_gen)
  #print('Cond = %f' %(cond))

  sw_base_link_tf_msg.transform.translation.x = base_link_pos[0]
  sw_base_link_tf_msg.transform.translation.y = base_link_pos[1]
  sw_base_link_tf_msg.transform.translation.z = base_link_pos[2]

  sw_base_link_tf_msg.transform.rotation.x = base_link_quat[0]
  sw_base_link_tf_msg.transform.rotation.y = base_link_quat[1]
  sw_base_link_tf_msg.transform.rotation.z = base_link_quat[2]
  sw_base_link_tf_msg.transform.rotation.w = base_link_quat[3]

  sw_base_link_tf_msg.header.stamp = rospy.Time.now()
  tf_br.sendTransform(sw_base_link_tf_msg)

  client_pos = sw_client_pos_trj[trj_idx]
  client_quat = R.from_matrix(sw_client_rmat_trj[trj_idx]).as_quat()

  sw_client_tf_msg.transform.translation.x = client_pos[0]
  sw_client_tf_msg.transform.translation.y = client_pos[1]
  sw_client_tf_msg.transform.translation.z = client_pos[2]

  sw_client_tf_msg.transform.rotation.x = client_quat[0]
  sw_client_tf_msg.transform.rotation.y = client_quat[1]
  sw_client_tf_msg.transform.rotation.z = client_quat[2]
  sw_client_tf_msg.transform.rotation.w = client_quat[3]

  sw_client_tf_msg.header.stamp = rospy.Time.now()
  tf_br.sendTransform(sw_client_tf_msg)

  sw_nozzle_pos = sw_nozzle_pos_trj[trj_idx]
  sw_nozzle_rmat = sw_nozzle_rmat_trj[trj_idx]

  pos_goal = sw_nozzle_pos + sw_nozzle_rmat[:, 2]*dist_nozzle_opening_from_goal
  rmat_goal = np.copy(sw_nozzle_rmat)

  pos_waypoint1 = pos_goal - rmat_goal[:, 2]*dist_centering_waypoint_from_goal
  rmat_waypoint1 = np.copy(rmat_goal)
  quat_waypoint1 = R.from_matrix(rmat_waypoint1).as_quat()

  pos_waypoint2 = pos_goal + rmat_goal[:, 2]*0.01
  rmat_waypoint2 = np.copy(rmat_goal)
  quat_waypoint2 = R.from_matrix(rmat_waypoint2).as_quat()

  waypoint1_tf_msg.transform.translation.x = pos_waypoint1[0]
  waypoint1_tf_msg.transform.translation.y = pos_waypoint1[1]
  waypoint1_tf_msg.transform.translation.z = pos_waypoint1[2]

  waypoint1_tf_msg.transform.rotation.x = quat_waypoint1[0]
  waypoint1_tf_msg.transform.rotation.y = quat_waypoint1[1]
  waypoint1_tf_msg.transform.rotation.z = quat_waypoint1[2]
  waypoint1_tf_msg.transform.rotation.w = quat_waypoint1[3]

  waypoint1_tf_msg.header.stamp = rospy.Time.now()
  tf_br.sendTransform(waypoint1_tf_msg)

  waypoint2_tf_msg.transform.translation.x = pos_waypoint2[0]
  waypoint2_tf_msg.transform.translation.y = pos_waypoint2[1]
  waypoint2_tf_msg.transform.translation.z = pos_waypoint2[2]

  waypoint2_tf_msg.transform.rotation.x = quat_waypoint2[0]
  waypoint2_tf_msg.transform.rotation.y = quat_waypoint2[1]
  waypoint2_tf_msg.transform.rotation.z = quat_waypoint2[2]
  waypoint2_tf_msg.transform.rotation.w = quat_waypoint2[3]

  waypoint2_tf_msg.header.stamp = rospy.Time.now()
  tf_br.sendTransform(waypoint2_tf_msg)

  waypoint3_tf_msg.transform.translation.x = pos_waypoint1[0]
  waypoint3_tf_msg.transform.translation.y = pos_waypoint1[1]
  waypoint3_tf_msg.transform.translation.z = pos_waypoint1[2] + 0.30

  waypoint3_tf_msg.transform.rotation.x = quat_waypoint1[0]
  waypoint3_tf_msg.transform.rotation.y = quat_waypoint1[1]
  waypoint3_tf_msg.transform.rotation.z = quat_waypoint1[2]
  waypoint3_tf_msg.transform.rotation.w = quat_waypoint1[3]

  waypoint3_tf_msg.header.stamp = rospy.Time.now()
  tf_br.sendTransform(waypoint3_tf_msg)


  q = pin.neutral(mrv_cv_pin_model)
  q[mrv_qidx:mrv_qidx + 3] = base_link_pos
  q[mrv_qidx + 3:mrv_qidx + 7] = base_link_quat
  q[mrv_qidx + 7:mrv_qidx + mrv_nq] = sw_joint_angles_trj[trj_idx]
  q[cv_qidx:cv_qidx + 3] = client_pos
  q[cv_qidx + 3:cv_qidx + 7] = client_quat
  pin.forwardKinematics(mrv_cv_pin_model, mrv_cv_pin_data, q)
  pin.updateFramePlacement(mrv_cv_pin_model, mrv_cv_pin_data, mrv_cv_goal_fid)
  pin.updateFramePlacement(mrv_cv_pin_model, mrv_cv_pin_data, mrv_cv_tip_fid)
  # if pin.computeCollisions(mrv_cv_pin_model, mrv_cv_pin_data, mrv_cv_pin_geom, mrv_cv_geom_data, q, True):
  if False:
    print('Collision')
    for res in mrv_cv_geom_data.collisionResults:
      contacts = res.getContacts()
      if len(contacts) > 0:
        pt_wrt_client = R.from_quat(client_quat).inv().apply(contacts[0].pos - client_pos)
        contacts_marker.points.append(Point(pt_wrt_client[0], pt_wrt_client[1], pt_wrt_client[2]))
        break
    '''
    pin.forwardKinematics(mrv_cv_pin_model, mrv_cv_pin_data, q)
    pin.updateFramePlacement(mrv_cv_pin_model, mrv_cv_pin_data, mrv_cv_tip_fid)
    tip_wrt_client = R.from_quat(client_quat).inv().apply(mrv_cv_pin_data.oMf[mrv_cv_tip_fid].translation - client_pos)
    contacts_marker.points.append(Point(tip_wrt_client[0], tip_wrt_client[1], tip_wrt_client[2]))
    '''
    # break
  contacts_marker.header.stamp = rospy.Time.now()
  contacts_pub.publish(contacts_marker)

  if do_step:
    trj_idx += 1

  if trj_idx > len(sw_base_pos_trj) - 1:
    break

  rate.sleep()
