import rospy
import numpy as np
from std_msgs.msg import Float32MultiArray, Float32, String
from geometry_msgs.msg import TransformStamped, WrenchStamped, Pose
from ur_state_machine.srv import JointMove, PositionServo, PositionMove, PositionMoveRequest, JointVelocityServo, JointVelocityServoResponse
from ur_state_machine.msg import JointMoveParams, PositionServoParams, Move, JointVelocityServoParams, URJointCommand
from vention_control.srv import PositionMove as VentionPositionMove
from vention_control.srv import PositionMoveResponse as VentionPositionMoveResponse
from std_srvs.srv import Trigger, TriggerResponse
from sensor_msgs.msg import JointState
from threading import Lock

class HolodeckInterface:
    def __init__(self, dt = 0.01):

        # Time step
        self.dt = dt

        # Lock for accessing shared variables
        self.lock = Lock()
        self.mrv_hil_js = JointState()
        self.client_hil_js = JointState()
        self.mrv_ur_state = ""
        self.client_ur_state = ""
        self.mrv_carriage_state = ""
        self.client_carriage_state = ""

        # Depends on which rail is being used
        self.mrv_arm_name = 'UR3'
        self.client_arm_name = 'UR4'
        self.mrv_carriage_name = f'vention{self.mrv_arm_name[-1]}'
        self.client_carriage_name = f'vention{self.client_arm_name[-1]}'

        # Initialize some variables
        self.mrv_hil_js = JointState()
        self.mrv_hil_js.position = np.zeros(7)
        self.client_hil_js = JointState()
        self.client_hil_js.position = np.zeros(7)

        self.mrv_ur_state = str()
        self.client_ur_state = str()
        self.mrv_carriage_state = str()
        self.client_carriage_state = str()

        # TODO: Fix Joint Move service there is something amis with the service
        # ROS Publishers, Subscribers
        self.mrv_arm_pub = rospy.Publisher(f'/{self.mrv_arm_name}/target_joint_velocity', URJointCommand, queue_size=1)
        self.client_arm_pub = rospy.Publisher(f'/{self.client_arm_name}/target_joint_velocity', URJointCommand, queue_size=1)
        self.mrv_carriage_pub = rospy.Publisher(f'/{self.mrv_carriage_name}/joint_velocity', Float32, queue_size=1)
        self.client_carriage_pub = rospy.Publisher(f'/{self.client_carriage_name}/joint_velocity', Float32, queue_size=1)

        self.mrv_arm_js_sub = rospy.Subscriber(f'/{self.mrv_arm_name}/joint_states', JointState, self.mrv_arm_js_callback)
        self.client_arm_js_sub = rospy.Subscriber(f'/{self.client_arm_name}/joint_states', JointState, self.client_arm_js_callback)

        self.mrv_carriage_js_sub = rospy.Subscriber(f'/{self.mrv_carriage_name}/joint_state', JointState, self.mrv_carriage_js_callback)
        self.client_carriage_js_sub = rospy.Subscriber(f'/{self.client_carriage_name}/joint_state', JointState, self.client_carriage_js_callback)

        self.mrv_ur_state_sub = rospy.Subscriber(f'/{self.mrv_arm_name}/state', String, self.mrv_ur_state_callback)
        self.client_ur_state_sub = rospy.Subscriber(f'/{self.client_arm_name}/state', String, self.client_ur_state_callback)

        self.mrv_carriage_state_sub = rospy.Subscriber(f'/{self.mrv_carriage_name}/state', String, self.mrv_carriage_state_callback)
        self.client_carriage_state_sub = rospy.Subscriber(f'/{self.client_carriage_name}/state', String, self.client_carriage_state_callback)

        # ROS Service Proxies
        self.mrv_joint_velocity_servo = rospy.ServiceProxy(f'/{self.mrv_arm_name}/joint_velocity_servo', JointVelocityServo)
        self.client_joint_velocity_servo = rospy.ServiceProxy(f'/{self.client_arm_name}/joint_velocity_servo', JointVelocityServo)

        self.mrv_joint_move = rospy.ServiceProxy(f'/{self.mrv_arm_name}/joint_move', JointMove)
        self.client_joint_move = rospy.ServiceProxy(f'/{self.client_arm_name}/joint_move', JointMove)

        self.mrv_idle = rospy.ServiceProxy(f'/{self.mrv_arm_name}/stop', Trigger)
        self.client_idle = rospy.ServiceProxy(f'/{self.client_arm_name}/stop', Trigger)


    
    # Callbacks
    def mrv_arm_js_callback(self, data):
        with self.lock:
            self.mrv_hil_js.position[1:7] = np.array(data.position)
            self.mrv_hil_js.velocity[1:7] = np.array(data.velocity)

    def client_arm_js_callback(self, data):
        with self.lock:
            self.client_hil_js.position[1:7] = np.array(data.position)
            self.client_hil_js.velocity[1:7] = np.array(data.velocity)

    def mrv_carriage_js_callback(self, data):
        with self.lock:
            self.mrv_hil_js.position[0] = data.position[0]
            self.mrv_hil_js.velocity[0] = data.velocity[0]

    def client_carriage_js_callback(self, data):
        with self.lock:
            self.client_hil_js.position[0] = data.position[0]
            self.client_hil_js.velocity[0] = data.velocity[0]

    def mrv_ur_state_callback(self, data):
        with self.lock:
            self.mrv_ur_state = data.data

    def client_ur_state_callback(self, data):
        with self.lock:
            self.client_ur_state = data.data
    
    def mrv_carriage_state_callback(self, data):
        with self.lock:
            self.mrv_carriage_state = data.data
    
    def client_carriage_state_callback(self, data):
        with self.lock:
            self.client_carriage_state = data.data

    # Getters
    def get_mrv_hil_js(self):
        with self.lock:
            return self.copy_joint_state(self.mrv_hil_js)
    
    def get_client_hil_js(self):
        with self.lock:
            return self.copy_joint_state(self.client_hil_js)
    
    def get_mrv_ur_state(self):
        with self.lock:
            return self.mrv_ur_state
    
    def get_client_ur_state(self):
        with self.lock:
            return 
    
    def get_mrv_carriage_state(self):
        with self.lock:
            return 
    
    def get_client_carriage_state(self):
        with self.lock:
            return 
    
    def copy_joint_state(self, js):
        js_new = JointState()
        js_new.header = js.header
        js_new.name = js.name
        js_new.position = js.position
        js_new.velocity = js.velocity
        js_new.effort = js.effort
        return js_new
    

        

    # UR Utility Functions

    def ur_velocity_mode(self, name, acc = 0.3):
      # Put the specified arm into velocity servo mode
        request = JointVelocityServoParams()
        request.acceleration = acc
        request.time = self.dt * 10

        try:
            if name == 'mrv':
                rospy.wait_for_service(f'/{self.mrv_arm_name}/joint_velocity_servo')
                resp = self.mrv_joint_velocity_servo(request)
            elif name == 'client':
                rospy.wait_for_service(f'/{self.client_arm_name}/joint_velocity_servo')
                resp = self.client_joint_velocity_servo(request)
            else:
                raise ValueError("Invalid arm name. Use 'mrv' or 'client'.")
            if not resp.success:
                print(f"{resp.message}")
                print(f"Failed to put {name} arm into velocity servo mode.")
                quit()

        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")
            quit()
    
    def ur_idle_mode(self, name):
        # Put the specified arm into idle mode
        try:
            if name == 'mrv':
                rospy.wait_for_service(f'/{self.mrv_arm_name}/stop')
                resp = self.mrv_idle()
            elif name == 'client':
                rospy.wait_for_service(f'/{self.client_arm_name}/stop')
                resp = self.client_idle()
            else:
                raise ValueError("Invalid arm name. Use 'mrv' or 'client'.")
            if not resp.success:
                print(f"{resp.message}")
                print(f"Failed to put {name} arm into idle mode.")
                quit()
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")
            quit()
    
    def ur_joint_move(self, name, angles):
        # rospy.loginfo(f"Moving {name} arm to {angles}")
        # Move the specified arm to the specified joint angles
        if len(angles) != 6:
            raise ValueError("Joint angles must have 6 elements.")
        request = JointMoveParams()
        request.joint_angles = angles
        try:
            if name == 'mrv':
                rospy.wait_for_service(f'/{self.mrv_arm_name}/joint_move')
                resp = self.mrv_joint_move(request)
            elif name == 'client':
                rospy.wait_for_service(f'/{self.client_arm_name}/joint_move')
                resp = self.client_joint_move(request)
            else:
                raise ValueError("Invalid arm name. Use 'mrv' or 'client'.")
            if not resp.success:
                print(f"{resp.message}")
                print(f"Failed to move {name} arm to specified joint angles.")
                quit()
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")
            quit
    
    def cmd_ur_velocity(self,name,v):
        # Send a velocity command to the specified arm
        if name == 'mrv':
            rospy.wait_for_service(f'/{self.mrv_arm_name}/joint_velocity_servo')
        elif name == 'client':
            rospy.wait_for_service(f'/{self.client_arm_name}/joint_velocity_servo')
        else:
            raise ValueError("Invalid arm name. Use 'mrv' or 'client'.")
            
        if len(v) != 6:
            raise ValueError("Velocity command must have 6 elements.")
        vel_msg = URJointCommand()
        vel_msg.base = v[0]
        vel_msg.shoulder = v[1]
        vel_msg.elbow = v[2]
        vel_msg.wrist1 = v[3]
        vel_msg.wrist2 = v[4]
        vel_msg.wrist3 = v[5]

        if name == 'mrv':
            self.mrv_arm_pub.publish(vel_msg)
        elif name == 'client':
            self.client_arm_pub.publish(vel_msg)
        
    # Vention Utility Functions
    def vention_position_move(self, name, pos):
        if name == 'mrv':
            service_name = f'/{self.mrv_carriage_name}/position_move'
            rospy.wait_for_service(service_name)
            carriage_srv = rospy.ServiceProxy(service_name, VentionPositionMove)
        elif name == 'client':
            service_name = f'/{self.client_carriage_name}/position_move'
            rospy.wait_for_service(service_name)
            carriage_srv = rospy.ServiceProxy(service_name, VentionPositionMove)
        else:
            raise ValueError("Invalid arm name. Use 'mrv' or 'client'.")
        
        try:
            resp = carriage_srv(pos)
            print(f"Response: {resp}")
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")
            rospy.signal_shutdown("Service call failed.")
            quit()
    