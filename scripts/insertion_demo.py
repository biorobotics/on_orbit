import rospy
import rospkg
import roslaunch
import numpy as np
from std_msgs.msg import Float32MultiArray, Float32, String
from geometry_msgs.msg import TransformStamped, WrenchStamped, Pose
from ur_state_machine.srv import JointMove, PositionServo, PositionMove, PositionMoveRequest
from ur_state_machine.msg import JointMoveParams, PositionServoParams, Move
from vention_control.srv import PositionMove as VentionPositionMove
from std_srvs.srv import Trigger
from sensor_msgs.msg import JointState
from scipy.spatial.transform import Rotation as R
import tf2_ros as tf
import threading
import copy
import pinocchio as pin
import roboticstoolbox as rtb
from spatialmath import SE3

class InsertionDemo:
    def __init__(self):
        rospy.init_node('gravity_demo_emulator', anonymous=True)
        rospack = rospkg.RosPack()
        rospath = rospack.get_path('on_orbit')
        self.rate = rospy.Rate(100)
        
        self.mrv_arm_name = 'UR3'
        self.client_arm_name = 'UR4'
        self.mrv_carriage_name = f'vention{self.mrv_arm_name[-1]}'
        self.client_carriage_name = f'vention{self.client_arm_name[-1]}'

        self.mrv_hil_js = JointState()
        self.mrv_hil_js.position = np.zeros(7)
        self.client_hil_js = JointState()
        self.client_hil_js.position = np.zeros(7)

        self.mrv_ur_state = str()
        self.client_ur_state = str()
        self.mrv_carriage_state = str()
        self.client_carriage_state = str()

        # ROS Publishers, Subscribers, and Services
        self.mrv_arm_pub = rospy.Publisher(f'/{self.mrv_arm_name}/joint_velocity', Float32MultiArray, queue_size=1)
        self.client_arm_pub = rospy.Publisher(f'/{self.client_arm_name}/joint_velocity', Float32MultiArray, queue_size=1)
        self.mrv_carriage_pub = rospy.Publisher(f'/{self.mrv_carriage_name}/joint_velocity', Float32, queue_size=1)
        self.client_carriage_pub = rospy.Publisher(f'/{self.client_carriage_name}/joint_velocity', Float32, queue_size=1)
        self.mrv_arm_js_sub = rospy.Subscriber(f'/{self.mrv_arm_name}/joint_states', JointState, self.mrv_arm_js_callback)
        self.client_arm_js_sub = rospy.Subscriber(f'/{self.client_arm_name}/joint_states', JointState, self.client_arm_js_callback)
        self.mrv_arm_js_sub = rospy.Subscriber(f'/{self.mrv_carriage_name}/joint_state', JointState, self.mrv_carriage_js_callback)
        self.client_arm_js_sub = rospy.Subscriber(f'/{self.client_carriage_name}/joint_state', JointState, self.client_carriage_js_callback)
        self.mrv_ur_state_sub = rospy.Subscriber(f'/{self.mrv_arm_name}/state', String, self.mrv_ur_state_callback)
        self.client_ur_state_sub = rospy.Subscriber(f'/{self.client_arm_name}/state', String, self.client_ur_state_callback)
        self.mrv_carriage_state_sub = rospy.Subscriber(f'/{self.mrv_carriage_name}/state', String, self.mrv_carriage_state_callback)
        self.client_carriage_state_sub = rospy.Subscriber(f'/{self.client_carriage_name}/state', String, self.client_carriage_state_callback)

        # TF Initialization
        self._tf_buffer = tf.Buffer()
        self._tf_listener = tf.TransformListener(self._tf_buffer)
        self._tf_broadcaster = tf.TransformBroadcaster()
        self._actual_transform_lock = threading.Lock()
        self._initial_transform = TransformStamped()

        self.mrv_ur_tcp_frame_id = self.mrv_arm_name + 'tcp'
        self.mrv_ur_base_frame_id = self.mrv_arm_name + 'base'
        self.client_ur_tcp_frame_id = self.client_arm_name + 'tcp'
        self.client_ur_base_frame_id = self.client_arm_name + 'base'
        
        try:
            self.mrv_ur_actual_transform = self._tf_buffer.lookup_transform(
                f'{self.mrv_ur_base_frame_id}', f'actual_{self.mrv_ur_tcp_frame_id}',
                rospy.Time(), timeout=rospy.Duration(2))
            self.mrv_ur_target_transform = TransformStamped()
            self.mrv_ur_target_transform.transform = self.mrv_ur_actual_transform.transform
            self.mrv_ur_target_transform.header.frame_id = self.mrv_ur_base_frame_id
            self.mrv_ur_target_transform.child_frame_id = f'desired_{self.mrv_ur_tcp_frame_id}'
            self.mrv_ur_target_transform.header.stamp = rospy.Time.now()

            self.client_ur_actual_transform = self._tf_buffer.lookup_transform(
                f'{self.client_ur_base_frame_id}', f'actual_{self.client_ur_tcp_frame_id}',
                rospy.Time(), timeout=rospy.Duration(2))
            self.client_ur_target_transform = TransformStamped()
            self.client_ur_target_transform.transform = self.client_ur_actual_transform.transform
            self.client_ur_target_transform.header.frame_id = self.client_ur_base_frame_id
            self.client_ur_target_transform.child_frame_id = f'desired_{self.client_ur_tcp_frame_id}'
            self.client_ur_target_transform.header.stamp = rospy.Time.now()

        except Exception as e:
            rospy.logerr(e)

        # Pinocchio Model Initialization
        self.model = pin.buildModelFromUrdf(rospath + '/urdf/on_orbit_rail.urdf')
        self.data = self.model.createData()

        self.mrv_arm_base_fid = self.model.getFrameId(f'ur_{self.mrv_arm_name[-1]}_base')
        self.mrv_arm_ee_fid = self.model.getFrameId(f'ur_{self.mrv_arm_name[-1]}_ee')
        # self.mrv_arm_ee_fid = self.model.getFrameId(f'peg_center')
        self.mrv_arm_joint_idx = self.model.getJointId(f'ur_{self.mrv_arm_name[-1]}_shoulder_pan_joint')
        self.mrv_carriage_joint_idx = self.model.getJointId(f'carriage_{self.mrv_arm_name[-1]}')
        self.mrv_arm_q_idx = self.model.idx_qs[self.mrv_arm_joint_idx]
        self.mrv_carriage_q_idx = self.model.idx_qs[self.mrv_carriage_joint_idx]
        self.mrv_carriage_v_idx = self.model.idx_vs[self.mrv_carriage_joint_idx]

        self.client_arm_base_fid = self.model.getFrameId(f'ur_{self.client_arm_name[-1]}_base')
        self.client_arm_ee_fid = self.model.getFrameId(f'ur_{self.client_arm_name[-1]}_ee')
        self.client_arm_joint_idx = self.model.getJointId(f'ur_{self.client_arm_name[-1]}_shoulder_pan_joint')
        self.client_carriage_q_idx = self.model.idx_qs[self.client_arm_joint_idx-1]
        
        print('arm q idx:', self.mrv_arm_q_idx)
        print('carriage q idx:', self.mrv_carriage_q_idx)
        print('arm ee frame id:', self.mrv_arm_ee_fid)

        self.fg_world = np.array([0, 0, -9.81*0.186515]) # gravity in world frame
        self.q = pin.neutral(self.model) # joint angle vector for al250l arm and rail joints
        self.ft_bias = np.zeros(3)
        self.g_WB = np.array([[-1, 0, 0],
                              [0, -np.sqrt(2)/2, np.sqrt(2)/2],
                              [0, np.sqrt(2)/2, np.sqrt(2)/2]]) 
        
        # Load EE Pose Trajectories
        path = '/experiment_logs/pos__0.1_0.1_0.0_rot_0.0_0.0_0.0_delta_v_0.0_0.0_0.0_mrv_w_0.0_0.0_0.0_client_w_0.002617993877991494_0.002617993877991494_0.0_20240628-170528/'
        self.mrv_ee_pos_trj = np.load(rospath + path + 'sw_peg_pos_trj.npy')
        self.mrv_ee_rmat_trj = np.load(rospath + path + 'sw_peg_rmat_trj.npy')
        self.client_ee_pos_trj = np.load(rospath + path + 'sw_nozzle_pos_trj.npy')
        self.client_ee_rmat_trj = np.load(rospath + path + 'sw_nozzle_rmat_trj.npy')

        # Frame transformations
        self.g_WhWo = np.eye(4) # transformation from old world frame to new (holodeck) world frame
        self.g_TB = np.eye(4) # transformation from client tool0 frame to MRV arm base frame
        # self.g_TN = 

    def mrv_arm_js_callback(self, data):
        self.mrv_hil_js.position[1:7] = np.array(data.position)
        self.mrv_hil_js.velocity[1:7] = np.array(data.velocity)

    def client_arm_js_callback(self, data):
        self.client_hil_js.position[1:7] = np.array(data.position)
        self.client_hil_js.velocity[1:7] = np.array(data.velocity)

    def mrv_carriage_js_callback(self, data):
        self.mrv_hil_js.position[0] = data.position[0]
        self.mrv_hil_js.velocity[0] = data.velocity[0]

    def client_carriage_js_callback(self, data):
        self.client_hil_js.position[0] = data.position[0]
        self.client_hil_js.velocity[0] = data.velocity[0]

    def mrv_ur_state_callback(self, data):
        self.mrv_ur_state = data.data

    def client_ur_state_callback(self, data):
        self.client_ur_state = data.data
    
    def mrv_carriage_state_callback(self, data):
        self.mrv_carriage_state = data.data
    
    def client_carriage_state_callback(self, data):
        self.client_carriage_state = data.data
        
    def update_actual_transform(self):
        try:
            with self._actual_transform_lock:
                self.mrv_ur_actual_transform = self._tf_buffer.lookup_transform(f'{self.mrv_ur_base_frame_id}', f'actual_{self.mrv_ur_tcp_frame_id}',
                    rospy.Time(), timeout=rospy.Duration(2))
                self.client_ur_actual_transform = self._tf_buffer.lookup_transform(f'{self.client_ur_base_frame_id}', f'actual_{self.client_ur_tcp_frame_id}',
                    rospy.Time(), timeout=rospy.Duration(2))
        except Exception as e:
            rospy.logerr(e)

    def initialize_hil(self):
        # Move MRV HIL to initial position
        mrv_carriage_srv_name = f'/{self.mrv_carriage_name}/position_move'
        mrv_arm_srv_name = f'/{self.mrv_arm_name}/joint_move'
        rospy.wait_for_service(mrv_carriage_srv_name)
        rospy.wait_for_service(mrv_arm_srv_name)
        mrv_carriage_srv = rospy.ServiceProxy(mrv_carriage_srv_name, VentionPositionMove)
        mrv_arm_srv = rospy.ServiceProxy(mrv_arm_srv_name, JointMove)
        mrv_arm_target = JointMoveParams()
        # mrv_arm_target.joint_angles = [-3.843987528477804, -0.8381941479495545, -2.391078472137451, -3.0148450336852015, -2.2746804396258753, 0.04980294406414032]
        mrv_arm_target.joint_angles = [-3.6132238546954554, -1.449615129535534, -2.2138543128967285, -2.505747457543844, -2.078312698994772, -0.5055072943316858]

        try:
            resp1 = mrv_carriage_srv([2.01]) 
            resp2 = mrv_arm_srv(mrv_arm_target)
            print(f"Response: {resp1}, {resp2}")
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")
            rospy.signal_shutdown('Failed to initialize MRV')

        # Move Client HIL to initial position
        client_carriage_srv_name = f'/{self.client_carriage_name}/position_move'
        client_arm_srv_name = f'/{self.client_arm_name}/joint_move'
        rospy.wait_for_service(client_carriage_srv_name)
        rospy.wait_for_service(client_arm_srv_name)
        client_carriage_srv = rospy.ServiceProxy(client_carriage_srv_name, VentionPositionMove)
        client_arm_srv = rospy.ServiceProxy(client_arm_srv_name, JointMove)
        client_arm_target = JointMoveParams()
        # client_arm_target.joint_angles = [3.2340543270111084, -1.013890103702881, 1.2362335363971155, -0.23575575769458013, -4.647035304700033, -0.023717228566304982]
        client_arm_target.joint_angles = [3.244448184967041, -1.167282060985901, 1.7743399778949183, -0.6228822034648438, -4.635807816182272, 0.05253524333238602]
        try:
            resp1 = client_carriage_srv([1.24])
            resp2 = client_arm_srv(client_arm_target)
            print(f"Response: {resp1}, {resp2}")
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")
            rospy.signal_shutdown('Failed to initialize MRV')

    def to_ur_base_frame(self, g_WN, is_mrv=True):
        '''Update the transform from world frame to MRV arm base frame so commands can be sent to the arm in its base frame'''
        self.q[self.mrv_carriage_q_idx:self.mrv_carriage_q_idx + 7] = self.mrv_hil_js.position
        self.q[self.client_carriage_q_idx:self.client_carriage_q_idx + 7] = self.client_hil_js.position
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)
        Tc_fid = self.model.getFrameId(f'ur_{self.client_arm_name[-1]}_tool0')
        Tm_fid = self.model.getFrameId(f'ur_{self.mrv_arm_name[-1]}_tool0')
        # print('T_fid:', T_fid)
        
        # g_BN = np.linalg.inv(self.data.oMf[self.client_arm_base_fid]) @ g_WN
        # g_NT = np.linalg.inv(self.data.oMf[self.client_arm_ee_fid]) @ self.data.oMf[T_fid] # constant transform from client tool0 frame to client EE frame
        # g_TB = np.linalg.inv(g_BN @ g_NT) 

        if is_mrv:
            g_WB = np.copy(self.data.oMf[self.mrv_arm_base_fid]) # transform MRV base to world frame
            g_WP = g_WN
            g_PT = np.linalg.inv(self.data.oMf[self.mrv_arm_ee_fid]) @ self.data.oMf[Tm_fid] # constant transform from MRV tool0 frame to MRV EE frame
        else:
            g_WB = np.copy(self.data.oMf[self.client_arm_base_fid])
            g_WP = g_WN
            g_PT = np.linalg.inv(self.data.oMf[self.client_arm_ee_fid]) @ self.data.oMf[Tc_fid]

        # np.linalg.inv(self.data.oMf[self.client_arm_base_fid]) @ g_WN 

        return np.linalg.inv(g_WB) @ g_WP @ g_PT
        
    def calibrate_hil_sim_transform(self, trj_idx):
        '''Align the end effector poses of MRV HIL with the trajectory'''
        # update self.q with the current joint states
        print('mrv_hil_js:', self.mrv_hil_js.position)
        self.q[self.mrv_carriage_q_idx:self.mrv_carriage_q_idx + 7] = self.mrv_hil_js.position
        
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)
        g_WhMe = np.copy(self.data.oMf[self.mrv_arm_ee_fid]) # transform MRV EE to holodeck world frame
        g_WsMs = np.eye(4) # transform from a sim EE pose (which is coincident with the MRV EE frame) to the sim world frame
        g_WsMs[:3, 3] = self.mrv_ee_pos_trj[trj_idx]
        g_WsMs[:3, :3] = self.mrv_ee_rmat_trj[trj_idx]
        print('sw peg:', g_WsMs[:3, 3])
        # self.g_WhWo = g_WhMe @ np.linalg.inv(g_WoM) # transformation from old world frame to new (holodeck) world frame

        # self.g_WhWo = np.linalg.inv(g_WhMe) @ g_WoM # transformation from old world frame to new (holodeck) world frame

        # g_MeM = np.linalg.inv(g_WhMe) @ g_WoM



        self.g_WhWo = g_WhMe @ np.linalg.inv(g_WsMs) # transformation from old world frame to new (holodeck) world frame

        print('peg_world:', (self.g_WhWo @ g_WsMs)[:3, 3]) 
        print('peg_base:', self.to_ur_base_frame(self.g_WhWo @ g_WsMs, is_mrv=True)[:3, 3])

        

    def align_client_hil(self, trj_idx):
        '''Align the end effector poses of Client HIL with the trajectory'''
        print('Aligning Client HIL...')
        sw_nozzle_pose = np.eye(4)
        sw_nozzle_pose[:3, 3] = self.client_ee_pos_trj[trj_idx]
        sw_nozzle_pose[:3, :3] = self.client_ee_rmat_trj[trj_idx]
        print('sw nozzle position:', sw_nozzle_pose[:3, 3])
        hw_nozzle_pose = self.g_WhWo @ sw_nozzle_pose # transform from new world frame to client EE frame
        
        
        print('hw nozzle position:', hw_nozzle_pose[:3, 3])


        g_TB = self.to_ur_base_frame(hw_nozzle_pose, is_mrv=False)

        self.update_actual_transform()
        current_pose = np.eye(4)
        current_pose[:3, 3] = [self.client_ur_actual_transform.transform.translation.x,
                                self.client_ur_actual_transform.transform.translation.y,
                                  self.client_ur_actual_transform.transform.translation.z]
        current_pose[:3, :3] = R.from_quat([self.client_ur_actual_transform.transform.rotation.x,
                                            self.client_ur_actual_transform.transform.rotation.y,
                                            self.client_ur_actual_transform.transform.rotation.z,
                                            self.client_ur_actual_transform.transform.rotation.w]).as_matrix()
        traj = self.interpolate_poses(current_pose, g_TB)
        
        # cont = input('Press Enter to continue...')
        self.call_position_servo()
        rospy.sleep(0.2)
        print('send it')
        idx = 0
        while not rospy.is_shutdown():
            self.client_ur_target_transform.header.stamp = rospy.Time.now()
            self.client_ur_target_transform.transform.translation.x = traj[idx].A[0, 3]
            self.client_ur_target_transform.transform.translation.y = traj[idx].A[1, 3]
            self.client_ur_target_transform.transform.translation.z = traj[idx].A[2, 3]
            r = R.from_matrix(traj[idx].A[:3, :3])
            self.client_ur_target_transform.transform.rotation.x = r.as_quat()[0]
            self.client_ur_target_transform.transform.rotation.y = r.as_quat()[1]
            self.client_ur_target_transform.transform.rotation.z = r.as_quat()[2]
            self.client_ur_target_transform.transform.rotation.w = r.as_quat()[3]
            self._tf_broadcaster.sendTransform(self.client_ur_target_transform)
            idx += 1
            if idx == len(traj):
                break
            self.rate.sleep()

    def interpolate_poses(self, pose1, pose2):
        '''Interpolate between two poses'''
        traj = rtb.tools.trajectory.ctraj(SE3(pose1), SE3(pose2), 150)
        return traj

    def call_position_servo(self):
        srv_name1 = f'/{self.mrv_arm_name}/position_servo'
        rospy.wait_for_service(srv_name1)       
        try:
            position_servo = rospy.ServiceProxy(srv_name1, PositionServo)
            params = PositionServoParams()
            # gain: 300, lookahead_time: 0.05, max_position_step: 1, max_orientation_step: 0.1
            params.gain = 300
            params.lookahead_time = 0.05
            params.max_position_step = 1
            params.max_orientation_step = 0.1
            response = position_servo(params)
            print(f"Service call to {srv_name1} succeeded")
            print(f"Response: {response}") 
            # position_servo.spin()          
        except rospy.ServiceException as e:
            print(f"Service call to {srv_name1} failed: {e}")

        srv_name2 = f'/{self.client_arm_name}/position_servo'
        rospy.wait_for_service(srv_name1)       
        try:
            position_servo = rospy.ServiceProxy(srv_name2, PositionServo)
            params = PositionServoParams()
            # gain: 300, lookahead_time: 0.05, max_position_step: 1, max_orientation_step: 0.1
            params.gain = 300
            params.lookahead_time = 0.05
            params.max_position_step = 1
            params.max_orientation_step = 0.1
            response = position_servo(params)
            print(f"Service call to {srv_name2} succeeded")
            print(f"Response: {response}") 
            # position_servo.spin()          
        except rospy.ServiceException as e:
            print(f"Service call to {srv_name2} failed: {e}")

    def move_peg_out_of_hole(self):
        srv_name = f'/{self.mrv_carriage_name}/position_move'
        rospy.wait_for_service(srv_name)
        try:
            position_move = rospy.ServiceProxy(srv_name, VentionPositionMove)
            response = position_move([self.mrv_hil_js.position[0] - 0.4])
            print(f"Service call to {srv_name} succeeded")
            print(f"Response: {response}")
        except rospy.ServiceException as e:
            print(f"Service call to {srv_name} failed: {e}")

        try:
            stop1 = rospy.ServiceProxy(f'/{self.mrv_arm_name}/stop', Trigger)
            response = stop1()
            print(f"Service call to stop succeeded")
            print(f"Response: {response}")

            stop2 = rospy.ServiceProxy(f'/{self.client_arm_name}/stop', Trigger)
            response = stop2()
            print(f"Service call to stop succeeded")
            print(f"Response: {response}")

        except rospy.ServiceException as e:
            print(f"Service call to stop failed: {e}")

    def run(self):
        trj_idx = 0 # where in the pose trajectory to start from
        while not rospy.is_shutdown():
            self.initialize_hil()
            for i in range(250):
                if (self.mrv_ur_state == 'IDLE' and self.client_ur_state == 'IDLE' and self.mrv_carriage_state == 'READY_TO_MOVE' and self.client_carriage_state == 'READY_TO_MOVE'):
                    break
                if i == 99:
                    rospy.signal_shutdown('Failed to initialize HIL')
                rospy.sleep(0.1)
            rospy.sleep(1)
            rospy.logwarn('HIL initialized. Calibrating HIL-Sim transform...')
            self.calibrate_hil_sim_transform(trj_idx)
            rospy.logwarn('Calibration complete. Aligning Client HIL...')
            self.align_client_hil(trj_idx)
            rospy.logwarn('Client HIL aligned.')
            rospy.sleep(1)
            # self.call_position_servo()
            rospy.logwarn('Servo called. Starying insetion trajectory...')
            for i in range(len(self.mrv_ee_pos_trj[trj_idx:])):
                self.update_actual_transform()
                # Update MRV HIL target transform
                mrv_sim_pose = np.eye(4)
                mrv_sim_pose[:3, 3] = self.mrv_ee_pos_trj[trj_idx + i]
                mrv_sim_pose[:3, :3] = self.mrv_ee_rmat_trj[trj_idx + i]
                mrv_hw_pose = self.g_WhWo @ mrv_sim_pose
                g_TB = self.to_ur_base_frame(mrv_hw_pose, is_mrv=True)
                self.mrv_ur_target_transform.header.stamp = rospy.Time.now()
                self.mrv_ur_target_transform.transform.translation.x = g_TB[0, 3]
                self.mrv_ur_target_transform.transform.translation.y = g_TB[1, 3]
                self.mrv_ur_target_transform.transform.translation.z = g_TB[2, 3]
                r = R.from_matrix(g_TB[:3, :3])
                self.mrv_ur_target_transform.transform.rotation.x = r.as_quat()[0]
                self.mrv_ur_target_transform.transform.rotation.y = r.as_quat()[1]
                self.mrv_ur_target_transform.transform.rotation.z = r.as_quat()[2]
                self.mrv_ur_target_transform.transform.rotation.w = r.as_quat()[3]
                self._tf_broadcaster.sendTransform(self.mrv_ur_target_transform)

                # # Update Client HIL target transform
                client_sim_pose = np.eye(4)
                client_sim_pose[:3, 3] = self.client_ee_pos_trj[trj_idx + i]
                client_sim_pose[:3, :3] = self.client_ee_rmat_trj[trj_idx + i]
                client_hw_pose = self.g_WhWo @ client_sim_pose
                g_TB_c = self.to_ur_base_frame(client_hw_pose, is_mrv=False)
                self.client_ur_target_transform.header.stamp = rospy.Time.now()
                self.client_ur_target_transform.transform.translation.x = g_TB_c[0, 3]
                self.client_ur_target_transform.transform.translation.y = g_TB_c[1, 3]
                self.client_ur_target_transform.transform.translation.z = g_TB_c[2, 3]
                r = R.from_matrix(g_TB_c[:3, :3])
                self.client_ur_target_transform.transform.rotation.x = r.as_quat()[0]
                self.client_ur_target_transform.transform.rotation.y = r.as_quat()[1]
                self.client_ur_target_transform.transform.rotation.z = r.as_quat()[2]
                self.client_ur_target_transform.transform.rotation.w = r.as_quat()[3]
                self._tf_broadcaster.sendTransform(self.client_ur_target_transform)

                self.rate.sleep()
            self.move_peg_out_of_hole()
            print('Trajectory complete. Moving peg out of hole...')
            for i in range(100):
                if (self.mrv_ur_state == 'IDLE' and self.client_ur_state == 'IDLE' and self.mrv_carriage_state == 'READY_TO_MOVE' and self.client_carriage_state == 'READY_TO_MOVE'):
                    break
                if i == 99:
                    rospy.signal_shutdown('Failed to initialize HIL')
                rospy.sleep(0.1)
            rospy.sleep(1)
            

if __name__ == "__main__":
    try:
        demo = InsertionDemo()
        demo.run()
    except rospy.ROSInterruptException:
        pass