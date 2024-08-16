#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import JointState
import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import rospkg
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R

rospy.init_node('visualize_urdf', anonymous=True)
joints_pub = rospy.Publisher('/on_orbit/sw_joint_state', JointState, queue_size=1)

rospack = rospkg.RosPack()
rospath = rospack.get_path('on_orbit')
urdf_file = rospath + '/urdf/robot_cv_detached.urdf'
pin_model = RobotWrapper.BuildFromURDF(urdf_file).model

mrv_jidx = pin_model.getJointId('world_to_base')
mrv_qidx = pin_model.idx_qs[mrv_jidx]
mrv_vidx = pin_model.idx_vs[mrv_jidx]

cv_jidx = pin_model.getJointId('world_to_client')
cv_qidx = pin_model.idx_qs[cv_jidx]
cv_vidx = pin_model.idx_vs[cv_jidx]

num_rotary = pin_model.nv - 12

joint_angles = np.array([-0.7047788387305209, -1.6486664575555459, -0.0007934622405855179, 2.1705127935725677, 3.1399998540089658, 0.5218470737159768, -2.4354950340230865])
# joint_angles = np.zeros(num_rotary)

joints_msg = JointState()
for jidx in range(mrv_jidx + 1, mrv_jidx + 1 + num_rotary):
  joints_msg.name.append(pin_model.names[jidx])
  joints_msg.position.append(joint_angles[jidx - mrv_jidx - 1])
  joints_msg.velocity.append(0)
  joints_msg.effort.append(0)

base_pos = np.array([4.38570453e-06, 5.27465730e-06, -7.71069928e+00])
# base_pos = np.array([0.3182, -0.28137, -8.6957])
base_quat = np.array([0., 0., 0., 1.])

sw_base_link_tf_msg = TransformStamped()
sw_base_link_tf_msg.header.frame_id = 'world'
sw_base_link_tf_msg.child_frame_id = 'base_link'
sw_base_link_tf_msg.transform.translation.x = base_pos[0]
sw_base_link_tf_msg.transform.translation.y = base_pos[1]
sw_base_link_tf_msg.transform.translation.z = base_pos[2]
sw_base_link_tf_msg.transform.rotation.x = base_quat[0]
sw_base_link_tf_msg.transform.rotation.y = base_quat[1]
sw_base_link_tf_msg.transform.rotation.z = base_quat[2]
sw_base_link_tf_msg.transform.rotation.w = base_quat[3]

client_pos = np.zeros(3)
client_quat = np.array([0., 0., 0., 1.])

sw_client_tf_msg = TransformStamped()
sw_client_tf_msg.header.frame_id = 'world'
sw_client_tf_msg.child_frame_id = 'client'

sw_client_tf_msg.transform.translation.x = client_pos[0]
sw_client_tf_msg.transform.translation.y = client_pos[1]
sw_client_tf_msg.transform.translation.z = client_pos[2]
sw_client_tf_msg.transform.rotation.x = client_quat[0]
sw_client_tf_msg.transform.rotation.y = client_quat[1]
sw_client_tf_msg.transform.rotation.z = client_quat[2]
sw_client_tf_msg.transform.rotation.w = client_quat[3]

tf_br = TransformBroadcaster()

rate = rospy.Rate(50)
ros_time = None
while not rospy.is_shutdown():
  if rospy.Time.now() != ros_time:
    ros_time = rospy.Time.now()
    joints_msg.header.stamp = ros_time
    joints_pub.publish(joints_msg)

    sw_base_link_tf_msg.header.stamp = rospy.Time.now()
    tf_br.sendTransform(sw_base_link_tf_msg)

    sw_client_tf_msg.header.stamp = rospy.Time.now()
    tf_br.sendTransform(sw_client_tf_msg)

  rate.sleep()
