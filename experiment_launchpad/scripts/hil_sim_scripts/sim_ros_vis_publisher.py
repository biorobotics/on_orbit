import numpy as np
import time
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

from scipy.spatial.transform import Rotation as R

import rospy
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

from visualization_msgs.msg import Marker

class SimROSVisPublisher(object):
  def __init__(self, rospath):
    pin_model = RobotWrapper.BuildFromURDF(rospath + '/urdf/robot.urdf', root_joint=pin.JointModelFreeFlyer()).model
    num_rotary = pin_model.nv - 6
    self.sw_joints_pub = rospy.Publisher('/on_orbit/sw_joint_state', JointState, queue_size=1)
    self.sw_joints_msg = JointState()
    joint1_idx = pin_model.getJointId('joint1')
    for j in range(joint1_idx, joint1_idx + num_rotary):
      self.sw_joints_msg.name.append(pin_model.names[j])
      self.sw_joints_msg.position.append(0)
      self.sw_joints_msg.velocity.append(0)
      self.sw_joints_msg.effort.append(0)

    self.tf_br = TransformBroadcaster()

    self.sw_base_link_tf_msg = TransformStamped()
    self.sw_base_link_tf_msg.header.frame_id = 'world'
    self.sw_base_link_tf_msg.child_frame_id = 'base_link'

    self.sw_client_tf_msg = TransformStamped()
    self.sw_client_tf_msg.header.frame_id = 'world'
    self.sw_client_tf_msg.child_frame_id = 'client'

    self.traj_tf_msg = TransformStamped()
    self.traj_tf_msg.header.frame_id = 'world'
    self.traj_tf_msg.child_frame_id = 'traj'

    self.filter_marker_pub = rospy.Publisher('/on_orbit/filter_marker', Marker, queue_size=1000)
    self.filter_marker = Marker()
    self.filter_marker.type = Marker.SPHERE
    self.filter_marker.action = Marker.ADD
    self.filter_marker.pose.position.x = 0
    self.filter_marker.pose.position.y = 0
    self.filter_marker.pose.position.z = 0
    self.filter_marker.pose.orientation.w = 1
    self.filter_marker.pose.orientation.x = 0
    self.filter_marker.pose.orientation.y = 0
    self.filter_marker.pose.orientation.z = 0
    self.filter_marker.scale.x = 0.01
    self.filter_marker.scale.y = 0.01
    self.filter_marker.scale.z = 0.01
    self.filter_marker.color.a = 1.0
    self.filter_marker.color.r = 1.0
    self.filter_marker.color.g = 0.0
    self.filter_marker.color.b = 0.0
    self.filter_marker.header.frame_id = 'world'
    self.filter_marker.ns = 'filter_marker'

  def publish(self, sw_joint_angles, sw_base_pos, sw_base_rmat, sw_client_pos, sw_client_rmat, traj_pos, traj_rmat):
    for i, angle in enumerate(sw_joint_angles):
      self.sw_joints_msg.position[i] = sw_joint_angles[i]
    self.sw_joints_msg.header.stamp = rospy.Time.now()
    self.sw_joints_pub.publish(self.sw_joints_msg)

    self.sw_base_link_tf_msg.transform.translation.x = sw_base_pos[0]
    self.sw_base_link_tf_msg.transform.translation.y = sw_base_pos[1]
    self.sw_base_link_tf_msg.transform.translation.z = sw_base_pos[2]

    sw_base_quat = R.from_matrix(sw_base_rmat).as_quat()

    self.sw_base_link_tf_msg.transform.rotation.x = sw_base_quat[0]
    self.sw_base_link_tf_msg.transform.rotation.y = sw_base_quat[1]
    self.sw_base_link_tf_msg.transform.rotation.z = sw_base_quat[2]
    self.sw_base_link_tf_msg.transform.rotation.w = sw_base_quat[3]

    self.sw_base_link_tf_msg.header.stamp = rospy.Time.now()
    self.tf_br.sendTransform(self.sw_base_link_tf_msg)

    self.sw_client_tf_msg.transform.translation.x = sw_client_pos[0]
    self.sw_client_tf_msg.transform.translation.y = sw_client_pos[1]
    self.sw_client_tf_msg.transform.translation.z = sw_client_pos[2]

    sw_client_quat = R.from_matrix(sw_client_rmat).as_quat()

    self.sw_client_tf_msg.transform.rotation.x = sw_client_quat[0]
    self.sw_client_tf_msg.transform.rotation.y = sw_client_quat[1]
    self.sw_client_tf_msg.transform.rotation.z = sw_client_quat[2]
    self.sw_client_tf_msg.transform.rotation.w = sw_client_quat[3]

    self.sw_client_tf_msg.header.stamp = rospy.Time.now()
    self.tf_br.sendTransform(self.sw_client_tf_msg)

    # Trajectory point
    self.traj_tf_msg.transform.translation.x = traj_pos[0]
    self.traj_tf_msg.transform.translation.y = traj_pos[1]
    self.traj_tf_msg.transform.translation.z = traj_pos[2]

    traj_quat = R.from_matrix(traj_rmat).as_quat()
    self.traj_tf_msg.transform.rotation.x = traj_quat[0]
    self.traj_tf_msg.transform.rotation.y = traj_quat[1]
    self.traj_tf_msg.transform.rotation.z = traj_quat[2]
    self.traj_tf_msg.transform.rotation.w = traj_quat[3]

    self.traj_tf_msg.header.stamp = rospy.Time.now()
    self.tf_br.sendTransform(self.traj_tf_msg)

  def delete_filter_marker(self):
      delete_marker = Marker()
      delete_marker.action = Marker.DELETEALL
      delete_marker.header.frame_id = 'world'
      self.filter_marker_pub.publish(delete_marker)

  def publish_filter_marker(self, positions, is_red = True):
    for i , pos in enumerate(positions):
            self.filter_marker.id = i
            self.filter_marker.pose.position.x = pos[0]
            self.filter_marker.pose.position.y = pos[1]
            self.filter_marker.pose.position.z = pos[2]
            self.filter_marker.header.stamp = rospy.Time.now()
            
            if is_red:
                self.filter_marker.color.r = 1.0
                self.filter_marker.color.g = 0.0
                self.filter_marker.color.b = 0.0
            else:
                # Blue
                self.filter_marker.color.r = 0.0
                self.filter_marker.color.g = 0.0
                self.filter_marker.color.b = 1.0
            
            self.filter_marker_pub.publish(self.filter_marker)


    


