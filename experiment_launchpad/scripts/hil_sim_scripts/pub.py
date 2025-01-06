#!/usr/bin/env python3
import numpy as np
import time
import random

from std_msgs.msg import Float32MultiArray, Float32

from scipy.spatial.transform import Rotation as R
from get_grid import get_grid
from trajlib_util import get_trajlib_load_paths, interp_trajectories_on_initial_client_w, interp_trajectories_on_delta_pos, interp_trajectories_on_init_client_state

import rospy
import rospkg

from hil_runner import HILRunner
from holodeck_interface import HolodeckInterface
from sim_ros_vis_publisher import SimROSVisPublisher

import os
import gc

from mrv_controller import MrvController

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)

class pub:
  def __init__(self):
    self.gridpub = rospy.Publisher('/poses', Float32MultiArray, queue_size=10)

    self.grid_sub = rospy.Subscriber('/grid_type', Float32, self.grid_callback)

    self.grid_type = 0

  def grid_callback(self, data):
    msg = Float32MultiArray()
    if (data.data == 20):  
      for i in range(12):
          msg.data.append(0)
      # msg.data[2] = -0.05
      self.gridpub.publish(msg)
    if (data.data > -1 and data.data < 18):
        self.grid_type = data.data
        self.ic=get_grid(self.grid_type)
        print(self.ic)
        print("*******************")
        self.ic=self.ic[0]
        print(self.ic)
        msg.data = self.ic
        self.gridpub.publish(msg)
    elif (data.data == -1):
      # for i in range(12):
      #   msg.data.append(np.random.rand())
      #   self.gridpub.publish(msg)
      msg.data.append(random.uniform(-0.1, 0.1))
      msg.data.append(random.uniform(-0.1, 0.1))
      msg.data.append(random.uniform(-0.1, 0.0))





      # msg.data.append(0.0)
      # msg.data.append(0.0)
      # msg.data.append(0.0)
      # msg.data.append(0.0)
      # msg.data.append(0.0)
      # msg.data.append(0.0)




      msg.data.append(random.uniform(-0.0035, 0.0035))
      msg.data.append(random.uniform(-0.0035, 0.0035))
      msg.data.append(random.uniform(-0.0035, 0.0035))
      msg.data.append(random.uniform(-0.125, 0.125))
      msg.data.append(random.uniform(-0.125, 0.125))
      msg.data.append(random.uniform(-0.125, 0.125))






      msg.data.append(random.uniform(-0.125, 0.125))
      msg.data.append(random.uniform(-0.125, 0.125))
      msg.data.append(random.uniform(-0.125, 0.125))
      print(msg.data)
      self.gridpub.publish(msg)

    else:
        print("Invalid grid type")


#   def run(self):
#     rospy.init_node('experiment', anonymous=True)
#     rate = rospy.Rate(10)
#     while not rospy.is_shutdown():
#       grid = get_grid()
#       gridpub.publish(grid)
#       rate.sleep()


if __name__ == '__main__':
  rospy.init_node('publish', anonymous=True)
  pub = pub()
  rospy.spin()
  # pub.run()