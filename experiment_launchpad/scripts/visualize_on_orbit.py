#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import JointState
from holodeck_interface import HolodeckInterface
import rospkg
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

class VisualizeOnOrbit:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('visualize_on_orbit', anonymous=True)
        
        # Initialize the holodeck interface
        self.holo_control = HolodeckInterface()
        
        self.num_joints = 7

        self.ft_bias = np.zeros(6)
        self.sensor_array = None

        # Get the urdf path
        rospack = rospkg.RosPack()
        urdf_file = rospack.get_path('on_orbit') + '/urdf/on_orbit.urdf'
        self.pin_model = RobotWrapper.BuildFromURDF(urdf_file).model
        self.pin_data = pin.Data(self.pin_model)

        self.jidx_mrv_hil = self.pin_model.getJointId('carriage_3') 
        self.qidx_mrv_hil = self.pin_model.idx_qs[self.jidx_mrv_hil]
        self.vidx_mrv_hil = self.pin_model.idx_vs[self.jidx_mrv_hil]

        self.jidx_client_hil = self.pin_model.getJointId('carriage_4') 
        self.qidx_client_hil = self.pin_model.idx_qs[self.jidx_client_hil]
        self.vidx_client_hil = self.pin_model.idx_vs[self.jidx_client_hil]

        # Publish joint states for visualization
        self.hw_joints_pub = rospy.Publisher('/on_orbit/hw_joint_state', JointState, queue_size=1)
        self.hw_joints_msg = JointState()
        for j in range(1, self.pin_model.njoints):
            self.hw_joints_msg.name.append(self.pin_model.names[j])
            self.hw_joints_msg.position.append(0)
            self.hw_joints_msg.velocity.append(0)
            self.hw_joints_msg.effort.append(0)

    def publish_hw_joints_for_viz(self):
        # Retrieve joint states from Holodeck interface
        mrv_hil_js = self.holo_control.get_mrv_hil_js()  # Getting joint states for MRV
        client_hil_js = self.holo_control.get_client_hil_js()  # Getting joint states for client arm

        # Combine joint positions, velocities, and efforts
        self.hw_joints_msg.position = list(mrv_hil_js.position) + list(client_hil_js.position)
        self.hw_joints_msg.velocity = list(mrv_hil_js.velocity) + list(client_hil_js.velocity)
        self.hw_joints_msg.effort = list(mrv_hil_js.effort) + list(client_hil_js.effort)

        # Update header timestamp and publish
        self.hw_joints_msg.header.stamp = rospy.Time.now()
        self.hw_joints_pub.publish(self.hw_joints_msg)

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz loop rate
        while not rospy.is_shutdown():
            self.publish_hw_joints_for_viz()
            rate.sleep()

if __name__ == '__main__':
    viz_on_orbit = VisualizeOnOrbit()
    viz_on_orbit.run()