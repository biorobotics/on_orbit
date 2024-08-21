import rospy
import rospkg
import roslaunch
import numpy as np
from std_msgs.msg import Float32MultiArray, Float32
from geometry_msgs.msg import TransformStamped, WrenchStamped, Pose
from ur_state_machine.srv import JointMove, PositionServo, PositionMove, PositionMoveRequest
from ur_state_machine.msg import JointMoveParams, PositionServoParams, Move
from vention_control.srv import PositionMove as VentionPositionMove
from sensor_msgs.msg import JointState
from scipy.spatial.transform import Rotation as R
import tf2_ros as tf
import threading
import copy
import pinocchio as pin
import roboticstoolbox as rtb
from spatialmath import SE3

class WaveDemo:
    def __init__(self):
        rospy.init_node('wave_demo', anonymous=True)
        rospack = rospkg.RosPack()
        rospath = rospack.get_path('on_orbit')
        self.rate = rospy.Rate(100)

        self.ur = 'UR2'
        self.carriage = f'vention{self.ur[-1]}'

        rospy.on_shutdown(self.shutdown)
        
        
        
    def run(self):
        # Services
        carriage_srv_name = f'/{self.carriage}/position_move'
        arm_srv_name = f'/{self.ur}/joint_move'
        rospy.wait_for_service(carriage_srv_name)
        rospy.wait_for_service(arm_srv_name)
        carriage_srv = rospy.ServiceProxy(carriage_srv_name, VentionPositionMove)
        arm_srv = rospy.ServiceProxy(arm_srv_name, JointMove)

        # Move to initial position
        carriage_target = [2.68]
        try:
            resp = carriage_srv(carriage_target)
            print(f'Carriage move response: {resp}')
        except rospy.ServiceException as e:
            print(f'Service call failed: {e}')


        # Wait for carriage to reach target
        rospy.sleep(3)

        # Move UR to initial position
        ur_arm_target = JointMoveParams()
        ur_arm_target.joint_angles = [3.7720515727996826, -1.1426871579936524, 1.4870312849627894, -0.8703290981105347, 0.6934359073638916, 0.007406964432448149]

        try:
            resp = arm_srv(ur_arm_target)
            print(f'Arm move response: {resp}')
        except rospy.ServiceException as e:
            print(f'Service call failed: {e}')
        
        # Wait for UR to reach target
        # rospy.sleep(4)

        while not rospy.is_shutdown():
            ur_arm_target.joint_angles = [3.7816660404205322, -1.143493877058365, 1.486244026814596, -0.9283094567111512, -0.3560064474688929, 0.007391524501144886]
            try:
                resp = arm_srv(ur_arm_target)
                print(f'Arm move response: {resp}')
            except rospy.ServiceException as e:
                print(f'Service call failed: {e}')

            ur_arm_target.joint_angles = [3.6493163108825684, -1.0521329206279297, 1.4845383802997034, -0.9167767328074952, 1.3223192691802979, 0.008949661627411842]
            try:
                resp = arm_srv(ur_arm_target)
                print(f'Arm move response: {resp}')
            except rospy.ServiceException as e:
                print(f'Service call failed: {e}')

    def shutdown(self):
        rospy.loginfo('Shutting down wave demo')
        ur_arm_target = JointMoveParams()
        carriage_srv_name = f'/{self.carriage}/position_move'
        arm_srv_name = f'/{self.ur}/joint_move'
        carriage_srv = rospy.ServiceProxy(carriage_srv_name, VentionPositionMove)
        arm_srv = rospy.ServiceProxy(arm_srv_name, JointMove)
        # # Put the ur back to initial position
        # ur_arm_target.joint_angles = [3.7720515727996826, -1.1426871579936524, 1.4870312849627894, -0.8703290981105347, 0.6934359073638916, 0.007406964432448149]
        # try:
        #     resp = arm_srv(ur_arm_target)
        #     print(f'Arm move response: {resp}')
        # except rospy.ServiceException as e:
        #     print(f'Service call failed: {e}')
        
        # Wait for UR to reach target
        # rospy.sleep(4)

        
        # Send the carriage back to home
        carriage_target = [0.01]
        try:
            resp = carriage_srv(carriage_target)
            print(f'Carriage move response: {resp}')
        except rospy.ServiceException as e:
            print(f'Service call failed: {e}')

        # Fold the arm up
        ur_arm_target.joint_angles = [3.14159265, -1.57079633,  2.8,  -1.5712853870787562, 0, 0]
        try:
            resp = arm_srv(ur_arm_target)
            print(f'Arm move response: {resp}')
        except rospy.ServiceException as e:
            print(f'Service call failed: {e}')
        

if __name__ == '__main__':
    try:
        wave = WaveDemo()
        wave.run()
    except rospy.ROSInterruptException:
        pass