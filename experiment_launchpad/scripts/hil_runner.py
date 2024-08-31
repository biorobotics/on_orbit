import numpy as np
import time
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from scipy.spatial.transform import Rotation as R
from holodeck_interface import HolodeckInterface

import rospy
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped, WrenchStamped, PointStamped, Point
from std_msgs.msg import String, Float32, Float32MultiArray


# from on_orbit.on_orbit_bindings import IKMoveEEBehindBarrier

import os

from visualization_msgs.msg import Marker
# UR 16 (jstate_right) has peg. Make sure to assign names appropraitely. UR 5 has the nozzle (jstate_left)
class HILRunner(object):
  def __init__(self, rospath, mrv_hil_home_angles, client_hil_home_angles , dt):
    urdf_file = rospath + '/urdf/on_orbit.urdf' 
    


    # Initialize the holodeck interface
    self.mrv_arm_name = 'UR3'
    self.client_arm_name = 'UR4'
    self.mrv_carriage_name = f'vention{self.mrv_arm_name[-1]}'
    self.client_carriage_name = f'vention{self.client_arm_name[-1]}'
    self.holo_control = HolodeckInterface(dt=dt)
    self.mrv_arm_pub = rospy.Publisher(f'/{self.mrv_arm_name}/joint_velocity', Float32MultiArray, queue_size=1)
    self.client_arm_pub = rospy.Publisher(f'/{self.client_arm_name}/joint_velocity', Float32MultiArray, queue_size=1)
    self.mrv_carriage_pub = rospy.Publisher(f'/{self.mrv_carriage_name}/joint_velocity', Float32, queue_size=1)
    self.client_carriage_pub = rospy.Publisher(f'/{self.client_carriage_name}/joint_velocity', Float32, queue_size=1)

    self.num_joints = 7

    self.ft_bias = np.zeros(6)
    self.sensor_array = None

    # Initialize pinocchio model (used for forward and inverse kinematics)
    self.pin_model = RobotWrapper.BuildFromURDF(urdf_file).model
    self.pin_data = pin.Data(self.pin_model)

    # The pinocchio model stacks the mrv joint angles and client joint angles
    # into one large vector q. This vector has 7 elements. The first
    # element is the carriage and the next
    # 6 are the joint angles.
    # Similarly, it stacks the derivatives of these quantities into one large vector v.
    # Here, we find the starting indices of each robots angles within q,
    # and the starting indices of their derivatives in v.
    self.jidx_mrv_hil = self.pin_model.getJointId('carriage_3') 
    self.qidx_mrv_hil = self.pin_model.idx_qs[self.jidx_mrv_hil]
    self.vidx_mrv_hil = self.pin_model.idx_vs[self.jidx_mrv_hil]

    # Here, we find the starting index of the client_arm's joint angles within q,
    # and the starting index of the client_arm's joint velocities within v
    self.jidx_client_hil = self.pin_model.getJointId('carriage_4') 
    self.qidx_client_hil = self.pin_model.idx_qs[self.jidx_client_hil]
    self.vidx_client_hil = self.pin_model.idx_vs[self.jidx_client_hil]

    # Get end-effector and F/T sensor frame ids, used to retreive their
    # poses during forward kinematics
    self.nozzle_fid = self.pin_model.getFrameId('ur_4_ee') 
    self.peg_fid = self.pin_model.getFrameId('ur_3_ee')
    self.ft_fid = self.pin_model.getFrameId('ft_sensor')

    # Limits for linear and angular velocity of the client_arm end-effector
    # ang_vel_limit also applies to the mrv_arm end-effector
    self.lin_vel_limit = 0.2
    self.ang_vel_limit = 0.45

    # Limit on the joint velocities
    self.max_joint_vel = np.pi/8*np.ones(6)
    self.max_joint_acc = np.pi*np.ones(6)

    # Publish joint states for visualization
    self.hw_joints_pub = rospy.Publisher('/on_orbit/hw_joint_state', JointState, queue_size=1)
    self.hw_joints_msg = JointState()
    for j in range(1, self.pin_model.njoints):
      self.hw_joints_msg.name.append(self.pin_model.names[j])
      self.hw_joints_msg.position.append(0)
      self.hw_joints_msg.velocity.append(0)
      self.hw_joints_msg.effort.append(0)

    self.ee_mrv_arm_path_pub = rospy.Publisher('/on_orbit/ee_mrv_arm_path', Marker, queue_size=1)
    self.ee_mrv_arm_path_marker = Marker()
    self.ee_mrv_arm_path_marker.type = Marker.LINE_STRIP
    self.ee_mrv_arm_path_marker.action = Marker.ADD
    self.ee_mrv_arm_path_marker.pose.position.x = 0
    self.ee_mrv_arm_path_marker.pose.position.y = 0
    self.ee_mrv_arm_path_marker.pose.position.z = 0
    self.ee_mrv_arm_path_marker.pose.orientation.w = 1
    self.ee_mrv_arm_path_marker.pose.orientation.x = 0
    self.ee_mrv_arm_path_marker.pose.orientation.y = 0
    self.ee_mrv_arm_path_marker.pose.orientation.z = 0
    self.ee_mrv_arm_path_marker.scale.x = 0.01
    self.ee_mrv_arm_path_marker.color.r = 0
    self.ee_mrv_arm_path_marker.color.g = 1
    self.ee_mrv_arm_path_marker.color.b = 0
    self.ee_mrv_arm_path_marker.color.a = 1
    self.ee_mrv_arm_path_marker.header.frame_id = "world"
    self.ee_mrv_arm_path_marker.ns = "control_node"

    self.ee_client_arm_path_pub = rospy.Publisher('/on_orbit/ee_client_arm_path', Marker, queue_size=1)
    self.ee_client_arm_path_marker = Marker()
    self.ee_client_arm_path_marker.type = Marker.LINE_STRIP
    self.ee_client_arm_path_marker.action = Marker.ADD
    self.ee_client_arm_path_marker.pose.position.x = 0
    self.ee_client_arm_path_marker.pose.position.y = 0
    self.ee_client_arm_path_marker.pose.position.z = 0
    self.ee_client_arm_path_marker.pose.orientation.w = 1
    self.ee_client_arm_path_marker.pose.orientation.x = 0
    self.ee_client_arm_path_marker.pose.orientation.y = 0
    self.ee_client_arm_path_marker.pose.orientation.z = 0
    self.ee_client_arm_path_marker.scale.x = 0.01
    self.ee_client_arm_path_marker.color.r = 0
    self.ee_client_arm_path_marker.color.g = 1
    self.ee_client_arm_path_marker.color.b = 0
    self.ee_client_arm_path_marker.color.a = 1
    self.ee_client_arm_path_marker.header.frame_id = "world"
    self.ee_mrv_arm_path_marker.ns = "control_node"

    
    self.client_arm_home_angles = client_hil_home_angles[1:7]
    self.mrv_arm_home_angles = mrv_hil_home_angles[1:7]
    self.client_carriage_home_angles = client_hil_home_angles[0]
    self.mrv_carriage_home_angles = mrv_hil_home_angles[0]

    self.sensorSub = rospy.Subscriber('/netft_node_data', WrenchStamped, self.ft_sensor_callback)

    ''' Use this to move the robot beyond the safety barrier
    Unsafe but was necessary for debugging'''
    self.override_barriers = True

    self.ur_barrier_dist = 0.0003 # 2x the sum of client_arm repeatability and mrv_arm repeatability

    q = pin.neutral(self.pin_model)
    v = np.zeros(self.pin_model.nv)

    
    pin.forwardKinematics(self.pin_model, self.pin_data, q)
    pin.updateFramePlacements(self.pin_model, self.pin_data)

    # Matrix that transforms vectors from F/T frame to the peg frame
    self.rmat_ft_peg = self.pin_data.oMf[self.peg_fid].rotation.transpose()@self.pin_data.oMf[self.ft_fid].rotation
    self.t_ft_peg = self.pin_data.oMf[self.peg_fid].actInv(self.pin_data.oMf[self.ft_fid].translation)

    self.peg_mass = rospy.get_param('peg_mass', None)
    self.peg_com = rospy.get_param('peg_com', None)

    if (self.peg_mass is None) or (self.peg_com is None):
      raise Exception("peg_mass and peg_com must be specified in experiment.xml")

    g = 9.81
    self.fg_world = np.array([0., 0., -self.peg_mass*g])

    self.need_to_reinitialize = True

    self.dt = dt

    #self.mrv_arm_init_controller = IKMoveEEBehindBarrier(urdf_file, -np.inf*np.ones(self.pin_model.nv), np.inf*np.ones(self.pin_model.nv), np.concatenate((self.max_joint_vel, self.max_joint_vel)), np.concatenate((self.max_joint_acc, self.max_joint_acc)), self.dt, 'ur_3_ee')
    #self.client_arm_init_controller = IKMoveEEBehindBarrier(urdf_file, -np.inf*np.ones(self.pin_model.nv), np.inf*np.ones(self.pin_model.nv), np.concatenate((self.max_joint_vel, self.max_joint_vel)), np.concatenate((self.max_joint_acc, self.max_joint_acc)), self.dt, 'ur_4_ee')
    self.visualize_while_simulating = True

    '''Rotation matrix and vector with world frame of software in bottom right, world frame of hardware in top left'''
    self.rmat_sw_hw = None
    self.t_sw_hw = None

  def calibrate_ft_bias(self):
    pin_model = self.pin_model
    pin_data = self.pin_data
    peg_fid = self.peg_fid
    ft_fid = self.ft_fid
    holo_control = self.holo_control
    qidx_mrv_hil = self.qidx_mrv_hil

    # Get force/torque sensor bias
    num_bias_samples = 100
    bias_estimate = np.zeros(6)
    self.ft_bias = np.zeros(6)
    num_data_points_so_far = 0
    rate = rospy.Rate(100)
    q = pin.neutral(pin_model)
    for i in range(num_bias_samples):
      mrv_hil_js = holo_control.get_mrv_hil_js()
      q[qidx_mrv_hil:qidx_mrv_hil + 7] = mrv_hil_js.position
      pin.forwardKinematics(pin_model, pin_data, q)
      pin.updateFramePlacement(pin_model, pin_data, peg_fid)
      pin.updateFramePlacement(pin_model, pin_data, ft_fid)

      # Gravity compensation for force sensor
      ft_compensated = self.get_ft_compensated(pin_data, ft_fid)
      if ft_compensated is None:
        print("Quitting because F/T data is unavailable.")
        quit()

      bias_estimate = (num_data_points_so_far*bias_estimate + ft_compensated)/(num_data_points_so_far + 1)
      num_data_points_so_far += 1
      rate.sleep()

    self.ft_bias = bias_estimate

  def clip_client_arm_cmd(self,v_client_arm_curr,v_client_arm_cmd):
    ''' Clip client_arm joint velocities and accelerations to satisfy limits'''
    v_client_arm_cmd = np.clip(v_client_arm_cmd, -self.max_joint_vel, self.max_joint_vel)
    acc_induced_min_vel_client_arm = v_client_arm_curr - self.max_joint_acc*self.dt
    acc_induced_max_vel_client_arm = v_client_arm_curr + self.max_joint_acc*self.dt
    return np.clip(v_client_arm_cmd, acc_induced_min_vel_client_arm, acc_induced_max_vel_client_arm)
    
  def clip_mrv_arm_cmd(self,v_mrv_arm_curr,v_mrv_arm_cmd):
    ''' Clip mrv_arm joint velocities and accelerations to satisfy limits'''
    v_mrv_arm_cmd = np.clip(v_mrv_arm_cmd, -self.max_joint_vel, self.max_joint_vel)
    acc_induced_min_vel_mrv_arm = v_mrv_arm_curr - self.max_joint_acc*self.dt 
    acc_induced_max_vel_mrv_arm = v_mrv_arm_curr + self.max_joint_acc*self.dt

    return np.clip(v_mrv_arm_cmd, acc_induced_min_vel_mrv_arm, acc_induced_max_vel_mrv_arm)
  
  def clip_twist(self,twist):
    twist[:3] = np.clip(twist[:3], -self.lin_vel_limit, self.lin_vel_limit)
    twist[3:] = np.clip(twist[3:], -self.ang_vel_limit, self.ang_vel_limit)
    return twist

  def ft_sensor_callback(self, data):
    if self.sensor_array is None:
      self.sensor_array = np.zeros(6)
    self.sensor_array[0] = data.wrench.force.x - self.ft_bias[0]
    self.sensor_array[1] = data.wrench.force.y - self.ft_bias[1]
    self.sensor_array[2] = data.wrench.force.z - self.ft_bias[2]
    self.sensor_array[3] = data.wrench.torque.x - self.ft_bias[3]
    self.sensor_array[4] = data.wrench.torque.y - self.ft_bias[4]
    self.sensor_array[5] = data.wrench.torque.z - self.ft_bias[5]


  def reset_to_home_angles(self, check_for_continue=True):

    ''' First move the carriages to their home configuration'''
    self.holo_control.vention_position_move('mrv', [self.mrv_carriage_home_angles])
    self.holo_control.vention_position_move('client', [self.client_carriage_home_angles])


    '''Arms to home configuration'''
    self.holo_control.ur_joint_move('mrv', self.mrv_arm_home_angles)
    self.holo_control.ur_joint_move('client', self.client_arm_home_angles)

    # self.calibrate_ft_bias() 


  # TODO: I think what needs to happen here is a replacement where we no longer assume that hb and world frames are aligned becuase they are not and
  # this will change the transforms that we send to our UR arm end effectors

  def move_peg_out_of_hole(self,visualize_before_moving=True):
    qidx_mrv_arm = self.qidx_mrv_hil + 1
    qidx_client_arm = self.qidx_client_hil + 1
    vidx_mrv_arm = self.vidx_mrv_hil + 1
    vidx_client_arm = self.vidx_client_hil + 1
    holo_control = self.holo_control
    pin_model = self.pin_model
    pin_data = self.pin_data
    peg_fid = self.peg_fid
    ft_fid = self.ft_fid
    dt = self.dt
    ee_mrv_arm_path_marker = self.ee_mrv_arm_path_marker
    ee_mrv_arm_path_pub = self.ee_mrv_arm_path_pub

    move_out_time = 6 # s
    move_out_speed = 0.015 # m/s
    move_out_steps = int(move_out_time/self.dt)

    # Ensure that the arms are stationary
    holo_control.ur_idle_mode('mrv')
    holo_control.ur_idle_mode('client')

    # Arms into velocity servo mode
    holo_control.ur_velocity_mode('mrv', 0.3)
    holo_control.ur_velocity_mode('client', 0.3)

    # Visualize
    stop_vis = False
    while not stop_vis:
      rate = rospy.Rate(5/dt)
      q_vis = pin.neutral(pin_model)
      
      mrv_hil_js = holo_control.get_mrv_hil_js()
      client_hil_js = holo_control.get_client_hil_js()
      q_vis[qidx_mrv_arm:qidx_mrv_arm + 6] = np.copy(mrv_hil_js.position[1:7])
      q_vis[qidx_client_arm:qidx_client_arm + 6] = np.copy(client_hil_js.position[1:7])
      v_vis = np.zeros(pin_model.nv)
      ee_mrv_arm_path_marker.points.clear()
      for step in range(move_out_steps):
        if rospy.is_shutdown():
          break
        # Forward kinematics: get poses and Jacobians
        pin.forwardKinematics(pin_model, pin_data, q_vis)
        pin.computeJointJacobians(pin_model, pin_data)
        pin.updateFramePlacement(pin_model, pin_data, peg_fid)
        t_peg_w = pin_data.oMf[peg_fid].translation
        rmat_peg_w = pin_data.oMf[peg_fid].rotation
        Jpeg = pin.getFrameJacobian(pin_model, pin_data, peg_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:, vidx_mrv_arm:vidx_mrv_arm + 6]

        peg_twist_ctrl = np.zeros(6)
        peg_twist_ctrl[:3] = -move_out_speed*rmat_peg_w[:, 2]

        peg_twist_ctrl = self.clip_twist(peg_twist_ctrl)

        # Solve instantaneous inverse kinematics
        v_mrv_arm_cmd = np.linalg.lstsq(Jpeg, peg_twist_ctrl, rcond=None)[0]
        v_mrv_arm_cmd = self.clip_mrv_arm_cmd(v_vis[self.vidx_mrv_hil:self.vidx_mrv_hil + 6],v_mrv_arm_cmd)

        q_vis[qidx_mrv_arm:qidx_mrv_arm + 6] += v_mrv_arm_cmd*dt
        v_vis[vidx_mrv_arm:vidx_mrv_arm + 6] = np.copy(v_mrv_arm_cmd)

        ee_mrv_arm_path_marker.points.append(Point(t_peg_w[0], t_peg_w[1], t_peg_w[2]))

        ee_mrv_arm_path_marker.header.stamp = rospy.Time.now()

        ee_mrv_arm_path_pub.publish(ee_mrv_arm_path_marker)

        if self.visualize_while_simulating:
          self.publish_hw_joints_for_viz(pin_model.nv,q_vis)

        rate.sleep()

      if visualize_before_moving:
        stop_vis = self.check_for_continue_with_visualization()
      else:
        stop_vis = True

    rate = rospy.Rate(1/dt)
    q = pin.neutral(pin_model)
    v = np.zeros(pin_model.nv)
    
    for step in range(move_out_steps):
      if rospy.is_shutdown():
        break
      # Get joint positions and velocities
      mrv_hil_js = holo_control.get_mrv_hil_js()
      client_hil_js = holo_control.get_client_hil_js()
      q[qidx_mrv_arm:qidx_mrv_arm + 6] = mrv_hil_js.position[1:7]
      v[vidx_mrv_arm:vidx_mrv_arm + 6] = mrv_hil_js.velocity[1:7]

      # Forward kinematics: get poses and Jacobians
      pin.forwardKinematics(pin_model, pin_data, q)
      pin.computeJointJacobians(pin_model, pin_data)
      pin.updateFramePlacement(pin_model, pin_data, peg_fid)
      t_peg_w = pin_data.oMf[peg_fid].translation
      rmat_peg_w = pin_data.oMf[peg_fid].rotation
      Jpeg = pin.getFrameJacobian(pin_model, pin_data, peg_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:, vidx_mrv_arm:vidx_mrv_arm + 6]

      pin.updateFramePlacement(pin_model, pin_data, ft_fid)

      # ft_compensated = self.get_ft_compensated(pin_data, ft_fid)
      # if ft_compensated is None:
      #   print("Stopping because F/T data is unavailable.")
      #   quit()

      # if self.is_ft_excessive(ft_compensated):
      #   print('Stopping because of excessive force on F/T sensor.')
      #   quit()

      peg_twist_ctrl = np.zeros(6)
      peg_twist_ctrl[:3] = -move_out_speed*rmat_peg_w[:, 2]

      peg_twist_ctrl = self.clip_twist(peg_twist_ctrl)

      # We have end-effector velocity commands. Now we solve
      # for joint velocity commands using inverse kinematics
      v_mrv_arm_cmd = np.linalg.lstsq(Jpeg, peg_twist_ctrl, rcond=None)[0]
      v_mrv_arm_cmd = self.clip_mrv_arm_cmd(v[self.vidx_mrv_hil:self.vidx_mrv_hil + 6],v_mrv_arm_cmd)

      # acc_norm_mrv_arm = np.linalg.norm((v_mrv_arm_cmd - v[vidx_mrv_arm:vidx_mrv_arm + 6])/dt, ord=np.inf)
      holo_control.cmd_ur_velocity('mrv', v_mrv_arm_cmd)

      rate.sleep()
    holo_control.ur_idle_mode('mrv')
  
 

  def publish_hw_joints_for_viz(self,nv,q_vis): 
    for i in range(nv):
      self.hw_joints_msg.position[i] = q_vis[i]
      self.hw_joints_msg.header.stamp = rospy.Time.now()
      self.hw_joints_pub.publish(self.hw_joints_msg)

  def create_data_for_saving(self):
    self.ee_mrv_pos_trj = []
    self.ee_mrv_rmat_trj = []
    self.ee_mrv_twist_trj = []

    self.ee_client_pos_trj = []
    self.ee_client_rmat_trj = []
    self.ee_client_twist_trj = []

    self.mrv_arm_joint_angles_trj = []
    self.mrv_arm_joint_vels_trj = []
    self.mrv_arm_joint_vel_cmds_trj = []

    self.client_arm_joint_angles_trj = []
    self.client_arm_joint_vels_trj = []
    self.client_arm_joint_vel_cmds_trj = []

    self.grav_comp_force_trj = []
    self.grav_comp_torque_trj = []

    self.hw_ts = []

    self.moving_hw = []

    self.ft_compensated_trj = []

  def get_des_peg_and_nozzle_kinematics(self, sw_peg_pos, sw_peg_rmat, sw_peg_twist, sw_nozzle_pos, sw_nozzle_rmat, sw_nozzle_twist):
    rmat_sw_hw = self.rmat_sw_hw
    t_sw_hw = self.t_sw_hw 
    hw_peg_pos_d = rmat_sw_hw@sw_peg_pos + t_sw_hw
    hw_peg_rmat_d = rmat_sw_hw@sw_peg_rmat
    hw_peg_twist_d = np.zeros(6)
    hw_peg_twist_d[:3] = rmat_sw_hw@sw_peg_twist[:3]
    hw_peg_twist_d[3:] = rmat_sw_hw@sw_peg_twist[3:]

    hw_nozzle_pos_d = rmat_sw_hw@sw_nozzle_pos + t_sw_hw
    hw_nozzle_rmat_d = rmat_sw_hw@sw_nozzle_rmat
    hw_nozzle_twist_d = np.zeros(6)
    hw_nozzle_twist_d[:3] = rmat_sw_hw@sw_nozzle_twist[:3]
    hw_nozzle_twist_d[3:] = rmat_sw_hw@sw_nozzle_twist[3:]

    return hw_peg_pos_d, \
           hw_peg_rmat_d, \
           hw_peg_twist_d, \
           hw_nozzle_pos_d, \
           hw_nozzle_rmat_d, \
           hw_nozzle_twist_d

  def initialize_arms_to_ee_poses(self, \
                                  sw_peg_pos, \
                                  sw_peg_rmat, \
                                  sw_peg_twist, \
                                  sw_nozzle_pos, \
                                  sw_nozzle_rmat, \
                                  sw_nozzle_twist, \
                                  verify_trajectory_visually):
    pin_data = self.pin_data
    nozzle_fid = self.nozzle_fid

    t_peg_hb, _, t_nozzle_hb, _ = self.update_pin_model()

    # The following blocks assume the robot has already been moved to its initial home configuration via a call to self.reset_to_home_angles
    ''' Move nozzle and peg to have identity rotation matrices '''
    print("Modifying arm pose.")
    pos_offset = np.array([0,0.050,0])
    self.move_mrv_arm_and_client_arm_to_pose(t_peg_hb + pos_offset,np.eye(3),t_nozzle_hb + pos_offset,np.eye(3),False)
    self.update_pin_model()

    self.calibrate_ft_bias() 
    
    # TODO: This is thw sw_hw transformation that is constant. It should be calculated once and stored
    # Now that the arms are in their home configuraiton, determine the position and rotation of the hardware base with respect to the world frame
    self.rmat_sw_hw = pin_data.oMf[nozzle_fid].rotation@sw_nozzle_rmat.transpose()
    self.t_sw_hw = pin_data.oMf[nozzle_fid].translation - self.rmat_sw_hw@sw_nozzle_pos

    hw_peg_pos_d, \
    hw_peg_rmat_d, \
    _,\
    hw_nozzle_pos_d,\
    hw_nozzle_rmat_d,\
    _ = self.get_des_peg_and_nozzle_kinematics(sw_peg_pos, \
                                                              sw_peg_rmat, \
                                                              sw_peg_twist, \
                                                              sw_nozzle_pos, \
                                                              sw_nozzle_rmat, \
                                                              sw_nozzle_twist)

    print("Initializing mrv_arm and client_arm to pose.")
    self.move_mrv_arm_and_client_arm_to_pose(hw_peg_pos_d,hw_peg_rmat_d,hw_nozzle_pos_d,hw_nozzle_rmat_d,verify_trajectory_visually)
    print("Initialized client_arm and mrv_arm poses.")

    self.need_to_reinitialize = False

  def update_pin_model(self):
    '''Update the UR arms' Pinnochio model using the current UR arm joint values and return the peg/nozzle position and orientation.
    
    Outputs: 
      t_peg_hb - position of peg in hardware base frame
      rmat_peg_hb - rotation matrix of peg in hardware base frame
      t_nozzle_hb - position of nozzle in hardware base frame
      rmat_nozzle_hb - rotation matrix of nozzle in hardware base frame'''
    
    pin_model = self.pin_model
    pin_data = self.pin_data
    holo_control = self.holo_control
    peg_fid = self.peg_fid
    nozzle_fid = self.nozzle_fid
    qidx_mrv_arm = self.qidx_mrv_hil + 1 
    qidx_client_arm = self.qidx_client_hil + 1
    
    mrv_hil_js = holo_control.get_mrv_hil_js()
    client_hil_js = holo_control.get_client_hil_js()
    q = pin.neutral(pin_model)
    q[qidx_mrv_arm:qidx_mrv_arm + 6] = mrv_hil_js.position[1:7]
    q[qidx_client_arm:qidx_client_arm + 6] = client_hil_js.position[1:7]
    pin.forwardKinematics(pin_model, pin_data, q)
    pin.updateFramePlacement(pin_model, pin_data, peg_fid)
    pin.updateFramePlacement(pin_model, pin_data, nozzle_fid)
    t_peg_hb = pin_data.oMf[peg_fid].translation
    rmat_peg_hb = pin_data.oMf[peg_fid].rotation
    t_nozzle_hb = pin_data.oMf[nozzle_fid].translation
    rmat_nozzle_hb = pin_data.oMf[nozzle_fid].rotation
  
    return t_peg_hb, rmat_peg_hb, t_nozzle_hb, rmat_nozzle_hb

  def emulate(self, sw_peg_pos, sw_peg_rmat, sw_peg_twist, sw_nozzle_pos, sw_nozzle_rmat, sw_nozzle_twist, sw_dist_nozzle_opening_from_goal):
    '''
    Outputs: 
      fail_reason: ? 
      wrench_peg_peg: wrench with force followed by torque applied to the peg's tip, expressed the peg tip frame
    '''

    pin_model = self.pin_model
    pin_data = self.pin_data
    holo_control = self.holo_control
    peg_fid = self.peg_fid
    nozzle_fid = self.nozzle_fid
    ft_fid = self.ft_fid
    rmat_ft_peg = self.rmat_ft_peg
    qidx_mrv_arm = self.qidx_mrv_hil + 1
    qidx_client_arm = self.qidx_client_hil + 1
    vidx_mrv_arm = self.vidx_mrv_hil + 1
    vidx_client_arm = self.vidx_client_hil + 1
    dt = self.dt
    mrv_arm_pub = self.mrv_arm_pub
    client_arm_pub = self.client_arm_pub

    peg_twist_ctrl = np.zeros(6)
    nozzle_twist_ctrl = np.zeros(6)

    # Get joint positions and velocities
    q = pin.neutral(pin_model)
    v = np.zeros(pin_model.nv)
    mrv_hil_js = holo_control.get_mrv_hil_js()
    client_hil_js = holo_control.get_client_hil_js()
    q[qidx_mrv_arm:qidx_mrv_arm + 6] = mrv_hil_js.position[1:7]
    q[qidx_client_arm:qidx_client_arm + 6] = client_hil_js.position[1:7]
    v[vidx_mrv_arm:vidx_mrv_arm + 6] = mrv_hil_js.velocity[1:7]
    v[vidx_client_arm:vidx_client_arm + 6] = client_hil_js.velocity[1:7]

    # Forward kinematics: get poses and Jacobians of the peg (mrv_arm end-effector) and nozzle (client_arm end-effector)
    pin.forwardKinematics(pin_model, pin_data, q)
    pin.computeJointJacobians(pin_model, pin_data)
    pin.updateFramePlacement(pin_model, pin_data, nozzle_fid)
    pin.updateFramePlacement(pin_model, pin_data, peg_fid)
    pin.updateFramePlacement(pin_model, pin_data, ft_fid)

    nozzle_pos = pin_data.oMf[nozzle_fid].translation
    nozzle_rmat = pin_data.oMf[nozzle_fid].rotation
    Jnozzle = pin.getFrameJacobian(pin_model, pin_data, nozzle_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:, vidx_client_arm:vidx_client_arm + 6]
    #nozzle_twist = Jnozzle@v[vidx_client_arm:vidx_client_arm + 6]

    t_peg_hb = pin_data.oMf[peg_fid].translation
    rmat_peg_hb = pin_data.oMf[peg_fid].rotation
    Jpeg = pin.getFrameJacobian(pin_model, pin_data, peg_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:, vidx_mrv_arm:vidx_mrv_arm + 6]

    self.ee_mrv_pos_trj.append(np.copy(t_peg_hb))
    self.ee_mrv_rmat_trj.append(np.copy(rmat_peg_hb))
    self.ee_mrv_twist_trj.append(np.copy(Jpeg@v[vidx_mrv_arm:vidx_mrv_arm + 6]))

    self.ee_client_pos_trj.append(np.copy(nozzle_pos))
    self.ee_client_rmat_trj.append(np.copy(nozzle_rmat))
    self.ee_client_twist_trj.append(np.copy(Jnozzle@v[vidx_client_arm:vidx_client_arm + 6]))

    self.mrv_arm_joint_angles_trj.append(np.copy(q[qidx_mrv_arm:qidx_mrv_arm + 6]))
    self.mrv_arm_joint_vels_trj.append(np.copy(v[vidx_mrv_arm:vidx_mrv_arm + 6]))

    self.client_arm_joint_angles_trj.append(np.copy(q[qidx_client_arm:qidx_client_arm + 6]))
    self.client_arm_joint_vels_trj.append(np.copy(v[vidx_client_arm:vidx_client_arm + 6]))

    self.hw_ts.append(rospy.Time.now().to_sec())

    # Bias and gravity compensation for force sensor
    ft_compensated = self.get_ft_compensated(pin_data, ft_fid)
    if ft_compensated is None:
      print("Stopping because F/T data is unavailable.")
      quit()

    self.ft_compensated_trj.append(np.copy(ft_compensated))
    self.grav_comp_force_trj.append(np.copy(ft_compensated[:3]))
    self.grav_comp_torque_trj.append(np.copy(ft_compensated[3:]))

    # Calculate F/T in frame of peg
    wrench_peg_peg = np.zeros(6)
    # Don't apply a force to the simulation unless the measured force is significant
    if np.abs(ft_compensated[0]) > 0.1 or np.abs(ft_compensated[1]) > 0.1 or np.abs(ft_compensated[2]) > 2: 
      f_peg_hb = rmat_peg_hb@rmat_ft_peg@ft_compensated[:3]
      wrench_peg_peg[:3] = rmat_peg_hb.transpose()@f_peg_hb

      #self.t_ft_peg is position of f/t sensor frame wrt peg frame, expressed in hardware peg frame
      tau_ft_hb = rmat_peg_hb@rmat_ft_peg@ft_compensated[3:]
      wrench_peg_peg[3:] = rmat_peg_hb.transpose()@tau_ft_hb + np.cross(-1*self.t_ft_peg, wrench_peg_peg[:3])

    if self.is_ft_excessive(ft_compensated):
      print('Stopping because of excessive force')
      fail_reason = 'Excessive force'

      return fail_reason, wrench_peg_peg

    sw_peg_pos_wrt_nozzle = sw_nozzle_rmat.transpose()@(sw_peg_pos - sw_nozzle_pos) #expressed in nozzle frame

    self.moving_hw.append(not self.need_to_reinitialize)

    hw_peg_pos_d, \
    hw_peg_rmat_d, \
    hw_peg_twist_d,\
    hw_nozzle_pos_d,\
    hw_nozzle_rmat_d,\
    hw_nozzle_twist_d = self.get_des_peg_and_nozzle_kinematics(sw_peg_pos, \
                                                               sw_peg_rmat, \
                                                               sw_peg_twist, \
                                                               sw_nozzle_pos, \
                                                               sw_nozzle_rmat, \
                                                               sw_nozzle_twist)

    # Compute desired client_arm end-effector twist
    pos_kp = 10
    rot_kp = 10
    nozzle_twist_ctrl[:3] = hw_nozzle_twist_d[:3] - pos_kp*(nozzle_pos - hw_nozzle_pos_d)
    nozzle_twist_ctrl[3:] = hw_nozzle_twist_d[3:] + rot_kp*pin.log3(hw_nozzle_rmat_d@nozzle_rmat.transpose())

    nozzle_twist_ctrl = self.clip_twist(nozzle_twist_ctrl)

    # Compute desired mrv_arm end-effector twist
    pos_kp = 10
    rot_kp = 10
    peg_twist_ctrl[:3] = hw_peg_twist_d[:3] - pos_kp*(t_peg_hb - hw_peg_pos_d)
    peg_twist_ctrl[3:] = hw_peg_twist_d[3:] + rot_kp*pin.log3(hw_peg_rmat_d@rmat_peg_hb.transpose())

    peg_twist_ctrl = self.clip_twist(peg_twist_ctrl)

    # We have end-effector velocity commands. Now we solve
    # for joint velocity commands using inverse kinematics
    v_client_arm_cmd = np.linalg.lstsq(Jnozzle, nozzle_twist_ctrl, rcond= None)[0]
    v_mrv_arm_cmd = np.linalg.lstsq(Jpeg, peg_twist_ctrl, rcond=None)[0]
    
    client_hil_js = holo_control.get_client_hil_js()
    mrv_hil_js = holo_control.get_mrv_hil_js()
    v_client_arm_cmd = self.clip_client_arm_cmd(client_hil_js.velocity[1:7],v_client_arm_cmd)
    v_mrv_arm_cmd = self.clip_mrv_arm_cmd(mrv_hil_js.velocity[1:7],v_mrv_arm_cmd)

    # Only send commands to hardware if peg and nozzle in sw sim are within certain proximity
    if not self.override_barriers:
      if np.abs(sw_peg_pos_wrt_nozzle[0]) > 0.11 \
         or np.abs(sw_peg_pos_wrt_nozzle[1]) > 0.11 \
         or sw_peg_pos_wrt_nozzle[2] < -0.11 \
         or sw_peg_pos_wrt_nozzle[2] > 1.1*sw_dist_nozzle_opening_from_goal:
      
        self.need_to_reinitialize = True
        v_mrv_arm_cmd[:] = 0
        v_client_arm_cmd[:] = 0

    client_hil_js = holo_control.get_client_hil_js()
    mrv_hil_js = holo_control.get_mrv_hil_js()
    acc_norm_client_arm = np.linalg.norm((v_client_arm_cmd - client_hil_js.velocity[1:7])/dt, ord=np.inf)
    acc_norm_mrv_arm = np.linalg.norm((v_mrv_arm_cmd - mrv_hil_js.velocity[1:7])/dt, ord=np.inf)
    
    self.mrv_arm_joint_vel_cmds_trj.append(np.copy(v_mrv_arm_cmd))
    self.client_arm_joint_vel_cmds_trj.append(np.copy(v_client_arm_cmd))

    # Send commands to hardware
    if not self.need_to_reinitialize:
      # Annnnnd another missed bit here. This needs to be using the updated state machine not the ctrl.speedj
      holo_control.cmd_ur_velocity('mrv', v_mrv_arm_cmd)
      holo_control.cmd_ur_velocity('client', v_client_arm_cmd)
    else: 
      print("Need to reinitialize arms.")

    if self.visualize_while_simulating:
      self.publish_hw_joints_for_viz(pin_model.nv,q)

    return 'nothing', wrench_peg_peg
  
  @staticmethod
  def is_ft_excessive(ft_np_array):
    ''' Check if F/T sensor value exceeds safety limits

    Inputs:
      ft_array: length 6 numpy array with forces followed by torqes
    Outputs: 
      Bool indicating true if F/T limits are exceeded, False otherwise
      Which index of ft_np_array for which the force/torque was exceeded
      '''
  
    # SI-125-3 Nano 25 F/T sensor 
    # Sensing range in Fx,Fy <125N, Fz <500N, Tx,Ty,Tz is 3Nm. Overload values are much higher but we want to be safe and make sure sensing is accurate.

    if ft_np_array[0] > 125:
      print("Fx force exceeded.")
      return True, 0
    elif ft_np_array[1] > 125:
      print("Fy force exceeded.")
      return True, 1
    elif ft_np_array[2] > 500:
      print("Fz force exceeded.")
      return True, 2
    elif ft_np_array[3] > 3: 
      print("Tx torque exceeded.")
      return True, 3
    elif ft_np_array[4] > 3: 
      print("Ty torque exceeded.")
      return True, 4
    elif ft_np_array[5] > 3:
      print("Tz torque exceeded.")
      return True, 5
    else:
      return False
  
  def get_ft_compensated(self, pin_data, ft_fid):
    '''Get a length 6 numpy array with force followed by torque that is compensated for gravity and unbiased. Expressed in frame of F/T sensor.'''
    ft_compensated = None # return None if no data is available

    if self.sensor_array is not None:
      ft_compensated = np.copy(self.sensor_array)

      ft_compensated[:3] -= pin_data.oMf[ft_fid].rotation.transpose()@self.fg_world # Subtract force due to gravity
      ft_compensated[3:] -= pin_data.oMf[ft_fid].rotation.transpose()@np.cross(pin_data.oMf[ft_fid].rotation[:, 2]*self.peg_com, self.fg_world) # Subtract torque due to gravity
      # Our ATI Nano25 is particularly noisy along the z-axis (but it's still in spec)
      # If we see less than 3 N force along that axis, just assume zero force along that axis
      # 
      # We should consider removing this, because it creates discontinuities in the z-axis force
      if abs(ft_compensated[2]) < 3.0:
        ft_compensated[2] = 0

    return ft_compensated
  
  @staticmethod
  def check_for_continue_with_visualization():
    print('Press y, then press enter if you want to execute the visualized trajectory. Press r to replay. Press n to exit.') 
    choice = input()
    choice = 'y'
    if choice == 'y':
      return True
    elif choice == 'n':
      quit()
    if rospy.is_shutdown():
      quit()

  @staticmethod
  def check_for_continue_without_visualization():
    print('Press y, then press enter to continue. Press n to exit.') 
    choice = input()
    if choice == 'y':
      return True
    elif choice == 'n':
      quit()
    if rospy.is_shutdown():
      quit()


  def save(self, save_path):
    np.save(save_path + '/ee_client_pos_trj.npy', self.ee_mrv_pos_trj)
    np.save(save_path + '/ee_client_rmat_trj.npy', self.ee_mrv_rmat_trj)
    np.save(save_path + '/ee_client_twist_trj.npy', self.ee_mrv_twist_trj)

    np.save(save_path + '/ee_client_pos_trj.npy', self.ee_client_pos_trj)
    np.save(save_path + '/ee_client_rmat_trj.npy', self.ee_client_rmat_trj)
    np.save(save_path + '/ee_client_twist_trj.npy', self.ee_client_twist_trj)

    np.save(save_path + '/mrv_arm_joint_angles_trj.npy', self.mrv_arm_joint_angles_trj)
    np.save(save_path + '/mrv_arm_joint_vels_trj.npy', self.mrv_arm_joint_vels_trj)
    np.save(save_path + '/mrv_arm_joint_vel_cmds_trj.npy', self.mrv_arm_joint_vel_cmds_trj)

    np.save(save_path + '/client_arm_joint_angles_trj.npy', self.client_arm_joint_angles_trj)
    np.save(save_path + '/client_arm_joint_vels_trj.npy', self.client_arm_joint_vels_trj)
    np.save(save_path + '/client_arm_joint_vel_cmds_trj.npy', self.client_arm_joint_vel_cmds_trj)

    np.save(save_path + '/grav_comp_force_trj.npy', self.grav_comp_force_trj)
    np.save(save_path + '/grav_comp_torque_trj.npy', self.grav_comp_torque_trj)

    np.save(save_path + '/hw_ts.npy', self.hw_ts)

    np.save(save_path + '/moving_hw.npy', self.moving_hw)

    np.save(save_path + '/ft_compensated_trj.npy', self.ft_compensated_trj)
    print(self.ft_compensated_trj)

    # Compute max time interval
    if len(self.hw_ts) > 1:
        hw_dts = np.diff(self.hw_ts)
        argmax = np.argmax(hw_dts)
        print('Max time interval: %f' %(hw_dts[argmax]))
        print('Max time interval index: %d' %(argmax))

