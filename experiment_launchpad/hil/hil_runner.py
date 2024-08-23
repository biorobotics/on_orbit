import numpy as np
import time
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

from ur_state_machine.srv import JointVelocityServo, JointVelocityServoResponse
from ur_state_machine.msg import JointVelocityServoParams, URJointCommand
from vention_control.srv import PositionMove, PositionMoveResponse
from std_srvs.srv import Trigger, TriggerResponse

from scipy.spatial.transform import Rotation as R

import rospy
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped, WrenchStamped, PointStamped, Point
from std_msgs.msg import String, Float32, Float32MultiArray


from ros_utils import ur_controllers

from on_orbit.on_orbit_bindings import IKMoveEEBehindBarrierIKMoveEEBehindBarrier

import os

from visualization_msgs.msg import Marker
# UR 16 (jstate_right) has peg. Make sure to assign names appropraitely. UR 5 has the nozzle (jstate_left)
class HILRunner(object):
  def __init__(self, rospath, mrv_arm_home_angles, client_arm_home_angles):
    urdf_file = rospath + '/urdf/on_orbit.urdf' 
    
    # naming convention: 'mrv_hil' includes the mrv_arm and the mrv_carriage
    self.mrv_arm_name = 'UR3'
    self.client_arm_name = 'UR4'
    self.mrv_carriage_name = f'vention{self.mrv_arm_name[-1]}'
    self.client_carriage_name = f'vention{self.client_arm_name[-1]}'

    self.num_joints = 7

    self.ft_bias = np.zeros(6)
    self.sensor_array = None

    self.mrv_hil_js = JointState()
    self.client_hil_js = JointState() # order: [carriage, base, shoulder, elbow, wrist, wrist_2, wrist_3]

    ## Setup subscripers and publishers ##
    self.mrv_arm_pub = rospy.Publisher(f'/{self.mrv_arm_name}/joint_velocity', URJointCommand, queue_size=1)
    self.client_arm_pub = rospy.Publisher(f'/{self.client_arm_name}/joint_velocity', URJointCommand, queue_size=1)
    self.mrv_carriage_pub = rospy.Publisher(f'/{self.mrv_carriage_name}/joint_velocity', URJointCommand, queue_size=1)
    self.client_carriage_pub = rospy.Publisher(f'/{self.client_carriage_name}/joint_velocity', URJointCommand, queue_size=1)

    self.mrv_arm_js_sub = rospy.Subscriber(f'/{self.mrv_arm_name}/joint_states', JointState, self.mrv_arm_js_callback)
    self.client_arm_js_sub = rospy.Subscriber(f'/{self.client_arm_name}/joint_states', JointState, self.client_arm_js_callback)
    self.mrv_arm_js_sub = rospy.Subscriber(f'/{self.mrv_carriage_name}/joint_states', JointState, self.mrv_carriage_js_callback)
    self.client_arm_js_sub = rospy.Subscriber(f'/{self.client_carriage_name}/joint_states', JointState, self.client_carriage_js_callback)

    # Service Proxies
    self.mrv_joint_velocity_servo = rospy.ServiceProxy(f'/{self.mrv_arm_name}/joint_velocity_servo', JointVelocityServo)
    self.client_joint_velocity_servo = rospy.ServiceProxy(f'/{self.client_arm_name}/joint_velocity_servo', JointVelocityServo)

    self.mrv_idle = rospy.ServiceProxy(f'/{self.mrv_arm_name}/stop', Trigger)
    self.client_idle = rospy.ServiceProxy(f'/{self.client_arm_name}/stop', Trigger)

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

    self.client_arm_home_angles = client_arm_home_angles
    self.mrv_arm_home_angles = mrv_arm_home_angles

    self.sensorSub = rospy.Subscriber('/netft_data', WrenchStamped, self.ft_sensor_callback)

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

    self.dt = 0.01

    self.mrv_arm_init_controller = IKMoveEEBehindBarrier(urdf_file, -np.inf*np.ones(self.pin_model.nv), np.inf*np.ones(self.pin_model.nv), np.concatenate((self.max_joint_vel, self.max_joint_vel)), np.concatenate((self.max_joint_acc, self.max_joint_acc)), self.dt, 'ur_3_ee')
    self.client_arm_init_controller = IKMoveEEBehindBarrier(urdf_file, -np.inf*np.ones(self.pin_model.nv), np.inf*np.ones(self.pin_model.nv), np.concatenate((self.max_joint_vel, self.max_joint_vel)), np.concatenate((self.max_joint_acc, self.max_joint_acc)), self.dt, 'ur_4_ee')
    self.visualize_while_simulating = True

    '''Rotation matrix and vector with world frame of software in bottom right, world frame of hardware in top left'''
    self.rmat_sw_hw = None
    self.t_sw_hw = None

  def ur_velocity_mode(self, ur_name, acc):
      # Put the specified arm into velocity servo mode
      request = JointVelocityServoParams()
      request.acceleration = acc
      request.time = self.dt * 10

      try:
          if ur_name == 'mrv':
              resp = self.mrv_joint_velocity_servo(request)
          elif ur_name == 'client':
              resp = self.client_joint_velocity_servo(request)
          else:
              raise ValueError("Invalid arm name. Use 'mrv' or 'client'.")
          if not resp.success:
              print(f"Failed to put {ur_name} arm into velocity servo mode.")
              quit()

      except rospy.ServiceException as e:
          print(f"Service call failed: {e}")
          quit()

  def ur_idle_mode(self, ur_name):
    # Put the specified arm into idle mode
    try:
      if ur_name == 'mrv':
        resp = self.mrv_idle()
      elif ur_name == 'client':
        resp = self.client_idle()
      else:
        raise ValueError("Invalid arm name. Use 'mrv' or 'client'.")
      if not resp.success:
        print(f"Failed to put {ur_name} arm into idle mode.")
        quit()
    except rospy.ServiceException as e:
      print(f"Service call failed: {e}")
      quit()
        
  def publish_velocity(self,ur_name, v):
    # Publish the velocity command to the specified arm
    if len(v) != 6:
      raise ValueError("Velocity command must have 6 elements.")
    vel_msg = URJointCommand()
    vel_msg.base = v[0]
    vel_msg.shoulder = v[1]
    vel_msg.elbow = v[2]
    vel_msg.wrist1 = v[3]
    vel_msg.wrist2 = v[4]
    vel_msg.wrist3 = v[5]

    if ur_name == 'mrv':
      self.mrv_arm_pub.publish(vel_msg)
    elif ur_name == 'client':
      self.client_arm_pub.publish(vel_msg)
    else:
      raise ValueError("Invalid arm name. Use 'mrv' or 'client'.")
    
    


  def calibrate_ft_bias(self):
    pin_model = self.pin_model
    pin_data = self.pin_data
    peg_fid = self.peg_fid
    ft_fid = self.ft_fid
    ctrl = self.ctrl
    qidx_mrv_hil = self.qidx_mrv_hil

    # Get force/torque sensor bias
    num_bias_samples = 100
    bias_estimate = np.zeros(6)
    self.ft_bias = np.zeros(6)
    num_data_points_so_far = 0
    rate = rospy.Rate(100)
    q = pin.neutral(pin_model)
    for i in range(num_bias_samples):
      q[qidx_mrv_hil:qidx_mrv_hil + 7] = self.mrv_hil_js
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

  def mrv_arm_js_callback(self, data):
    self.mrv_hil_js.position[1:7] = np.array(data.position)
    self.mrv_hil_js.velocity[1:7] = np.array(data.velocity)

  def client_arm_js_callback(self, data):
    self.client_hil_js.position[1:7] = np.array(data.position)
    self.client_hil_js.velocity[1:7] = np.array(data.velocity)

  def mrv_carriage_js_callback(self, data):
    self.mrv_hil_js.position[0] = np.array(data.position)
    self.mrv_hil_js.velocity[0] = np.array(data.velocity)

  def client_carriage_js_callback(self, data):
    self.client_hil_js.position[0] = np.array(data.position)
    self.client_hil_js.velocity[0] = np.array(data.velocity)

  def visualize_then_move_joints(self, q_final, check_for_continue=True):



    ctrl = self.ctrl
    hw_joints_pub = self.hw_joints_pub
    hw_joints_msg = self.hw_joints_msg
    ee_mrv_arm_path_pub = self.ee_mrv_arm_path_pub
    ee_mrv_arm_path_marker = self.ee_mrv_arm_path_marker
    ee_client_arm_path_pub = self.ee_client_arm_path_pub
    ee_client_arm_path_marker = self.ee_client_arm_path_marker
    pin_model = self.pin_model
    pin_data = self.pin_data
    qidx_mrv_arm = self.qidx_mrv_hil + 1
    qidx_client_arm = self.qidx_client_hil
    vidx_mrv_arm = self.vidx_mrv_hil + 1
    vidx_client_arm = self.vidx_client_hil
    max_joint_vel = self.max_joint_vel
    max_joint_acc = self.max_joint_acc
    dt = self.dt
    peg_fid = self.peg_fid
    nozzle_fid = self.nozzle_fid
    mrv_arm_pub = self.mrv_arm_pub
    client_arm_pub = self.client_arm_pub

    joints_kp = 2

    T_a = 1 # Initial acceleration time
    T_c = 1 # Coast (constant velocity time)
    T_d = 1 # Final deceleration time



    # Arms into velocity servo mode
    self.ur_velocity_mode('mrv', 0.3)
    self.ur_velocity_mode('client', 0.3)

    # Extend the duration until we know a trapezoidal joint profile won't violate
    # velocity or acceleration limits

    
    limit_vio = True
    while limit_vio:
      T = T_a + T_c + T_d # Total time

      # Initial configuration
      q0 = pin.neutral(pin_model)
      q0[qidx_mrv_arm:qidx_mrv_arm + 6] = np.copy(self.mrv_hil_js[1:7])
      q0[qidx_client_arm:qidx_client_arm + 6] = np.copy(self.client_hil_js[1:7])

      # Desired displacement
      dq = q_final - q0

      # Desired displacement
      a_a = dq/(T_a*(T - T_a/2 - T_d/2)) # Initial acceleration
      a_d = -dq/(T_d*(T - T_a/2 - T_d/2)) # Final acceleration

      v_c = dq/(T - T_a/2 - T_d/2)

      dq_a = 0.5*a_a*T_a**2
      dq_c = v_c*T_c

      if np.any(np.abs(v_c[:6]) > max_joint_vel) or \
         np.any(np.abs(a_a[:6]) > max_joint_acc) or \
         np.any(np.abs(v_c[6:]) > max_joint_vel) or \
         np.any(np.abs(a_a[6:]) > max_joint_acc):
        T_a *= 1.5
        T_c *= 1.5
        T_d *= 1.5
      else:
        limit_vio = False

    stop_vis = False
    while not stop_vis:
      ee_mrv_arm_path_marker.points.clear()
      ee_client_arm_path_marker.points.clear()

      q_vis = np.copy(q0)
      v_cmd = np.zeros(pin_model.nv)

      rate = rospy.Rate(5/self.dt) # Slow it down a bit to make it easier to watch the trajectory
      for t_vis in np.arange(0, T, dt):
        if t_vis < T_a:
          # Initial acceleration
          q_des = q0 + 0.5*a_a*t_vis**2
          v_des = a_a*t_vis
        elif t_vis < T_a + T_c:
          # Coast at constant velocity
          q_des = q0 + dq_a + v_c*(t_vis - T_a)
          v_des = v_c
        else:
          # Final deceleration
          q_des = q0 + dq_a + dq_c + v_c*(t_vis - T_a - T_c) + 0.5*a_d*(t_vis - T_a - T_c)**2
          v_des = v_c + a_d*(t_vis - T_a - T_c)

        q_err = q_vis - q_des
        v_mrv_arm_cmd = v_des[vidx_mrv_arm:vidx_mrv_arm + 6] - joints_kp*q_err[vidx_mrv_arm:vidx_mrv_arm + 6]
        v_client_arm_cmd = v_des[vidx_client_arm:vidx_client_arm + 6] - joints_kp*q_err[vidx_client_arm:vidx_client_arm + 6]

        v_client_arm_cmd = self.clip_client_arm_cmd(v_cmd[vidx_client_arm:vidx_client_arm + 6],v_client_arm_cmd)
        v_mrv_arm_cmd = self.clip_mrv_arm_cmd(v_cmd[vidx_mrv_arm:vidx_mrv_arm + 6],v_mrv_arm_cmd)

        v_cmd[vidx_mrv_arm:vidx_mrv_arm + 6] = v_mrv_arm_cmd
        v_cmd[vidx_client_arm:vidx_client_arm + 6] = v_client_arm_cmd

        q_vis = q_vis + v_cmd*dt

        pin.forwardKinematics(pin_model, pin_data, q_vis)

        pin.updateFramePlacement(pin_model, pin_data, peg_fid)
        t_peg_hb = pin_data.oMf[peg_fid].translation

        pin.updateFramePlacement(pin_model, pin_data, nozzle_fid)
        nozzle_pos = pin_data.oMf[nozzle_fid].translation

        ee_mrv_arm_path_marker.points.append(Point(t_peg_hb[0], t_peg_hb[1], t_peg_hb[2]))
        ee_client_arm_path_marker.points.append(Point(nozzle_pos[0], nozzle_pos[1], nozzle_pos[2]))

        ee_mrv_arm_path_marker.header.stamp = rospy.Time.now()
        ee_client_arm_path_marker.header.stamp = rospy.Time.now()

        ee_mrv_arm_path_pub.publish(ee_mrv_arm_path_marker)
        ee_client_arm_path_pub.publish(ee_client_arm_path_marker)

        for i in range(pin_model.nv):
          hw_joints_msg.position[i] = q_vis[i]
        
        hw_joints_msg.header.stamp = rospy.Time.now()
        hw_joints_pub.publish(hw_joints_msg)

        if rospy.is_shutdown():
          quit()
        rate.sleep()

      if check_for_continue:
        stop_vis = self.check_for_continue_with_visualization()
      else: 
        stop_vis = True
      
    ee_mrv_arm_path_marker.points.clear()
    ee_client_arm_path_marker.points.clear()

    


    rate = rospy.Rate(1/self.dt)
    for t_exec in np.arange(0, T, dt):
      if t_exec < T_a:
        # Initial acceleration
        q_des = q0 + 0.5*a_a*t_exec**2
        v_des = a_a*t_exec
      elif t_exec < T_a + T_c:
        # Coast at constant velocity
        q_des = q0 + dq_a + v_c*(t_exec - T_a)
        v_des = v_c
      else:
        # Final deceleration
        q_des = q0 + dq_a + dq_c + v_c*(t_exec - T_a - T_c) + 0.5*a_d*(t_exec - T_a - T_c)**2
        v_des = v_c + a_d*(t_exec - T_a - T_c)

      q = pin.neutral(pin_model)
      v = np.zeros(pin_model.nv)
      q[qidx_mrv_arm:qidx_mrv_arm + 6] = self.mrv_hil_js.position[1:7]
      q[qidx_client_arm:qidx_client_arm + 6] = self.client_hil_js.position[1:7]
      v[vidx_mrv_arm:vidx_mrv_arm + 6] = self.mrv_hil_js.velocity[1:7]
      v[vidx_client_arm:vidx_client_arm + 6] = self.client_hil_js.velocity[1:7]

      q_err = q - q_des
      v_mrv_arm_cmd = v_des[vidx_mrv_arm:vidx_mrv_arm + 6] - joints_kp*q_err[vidx_mrv_arm:vidx_mrv_arm + 6]
      v_client_arm_cmd = v_des[vidx_client_arm:vidx_client_arm + 6] - joints_kp*q_err[vidx_client_arm:vidx_client_arm + 6]

      v_client_arm_cmd = self.clip_client_arm_cmd(self.client_hil_js.velocity[1:7],v_client_arm_cmd)
      v_mrv_arm_cmd = self.clip_mrv_arm_cmd(self.mrv_hil_js.velocity[1:7],v_mrv_arm_cmd)

      # acc_norm_client_arm = np.linalg.norm((v_client_arm_cmd - self.client_hil_js.velocity[1:7])/dt, ord=np.inf)
      # acc_norm_mrv_arm = np.linalg.norm((v_mrv_arm_cmd - self.mrv_hil_js.velocity[1:7])/dt, ord=np.inf)

      
      self.publish_velocity('mrv', v_mrv_arm_cmd)
      self.publish_velocity('client', v_client_arm_cmd)

      pin.forwardKinematics(pin_model, pin_data, q)

      pin.updateFramePlacement(pin_model, pin_data, peg_fid)
      t_peg_hb = pin_data.oMf[peg_fid].translation

      pin.updateFramePlacement(pin_model, pin_data, nozzle_fid)
      nozzle_pos = pin_data.oMf[nozzle_fid].translation

      ee_mrv_arm_path_marker.points.append(Point(t_peg_hb[0], t_peg_hb[1], t_peg_hb[2]))
      ee_client_arm_path_marker.points.append(Point(nozzle_pos[0], nozzle_pos[1], nozzle_pos[2]))

      ee_mrv_arm_path_marker.header.stamp = rospy.Time.now()
      ee_client_arm_path_marker.header.stamp = rospy.Time.now()

      ee_mrv_arm_path_pub.publish(ee_mrv_arm_path_marker)
      ee_client_arm_path_pub.publish(ee_client_arm_path_marker)

      for i in range(pin_model.nv):
        hw_joints_msg.position[i] = q[i]
      hw_joints_msg.header.stamp = rospy.Time.now()
      hw_joints_pub.publish(hw_joints_msg)

      if rospy.is_shutdown():
        quit()
      rate.sleep()

    self.ur_idle_mode('mrv')
    self.ur_idle_mode('client')
    # At this point joints should be very close to desired angles
    if np.linalg.norm(q - q_final, ord=np.inf)*180/np.pi > 1:
      print('Large error in joint angles. Exiting')
      quit()

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
    qidx_mrv_arm = self.qidx_mrv_hil
    qidx_client_arm = self.qidx_client_hil

    '''For moving the arms at the same time'''
    print('Moving client_arm and mrv_arm to home configuration')
    q_final = pin.neutral(self.pin_model)
    q_final[qidx_mrv_arm:qidx_mrv_arm + 6] = self.mrv_arm_home_angles
    q_final[qidx_client_arm:qidx_client_arm + 6] = self.client_arm_home_angles
    self.visualize_then_move_joints(q_final, check_for_continue)

    self.calibrate_ft_bias() 


  # TODO: Implement this with the rtde interface
  def move_peg_out_of_hole(self,visualize_before_moving=True):
    qidx_mrv_arm = self.qidx_mrv_hil
    qidx_client_arm = self.qidx_client_hil
    vidx_mrv_arm = self.vidx_mrv_hil
    vidx_client_arm = self.vidx_client_hil
    ctrl = self.ctrl
    pin_model = self.pin_model
    pin_data = self.pin_data
    peg_fid = self.peg_fid
    ft_fid = self.ft_fid
    dt = self.dt
    ee_mrv_arm_path_marker = self.ee_mrv_arm_path_marker
    ee_mrv_arm_path_pub = self.ee_mrv_arm_path_pub
    mrv_arm_pub = self.mrv_arm_pub

    move_out_time = 6 # s
    move_out_speed = 0.015 # m/s
    move_out_steps = int(move_out_time/self.dt)

    # Arms into velocity servo mode
    self.ur_velocity_mode('mrv', 0.3)
    self.ur_velocity_mode('client', 0.3)

    # Visualize
    stop_vis = False
    while not stop_vis:
      rate = rospy.Rate(5/dt)
      q_vis = pin.neutral(pin_model)
      q_vis[qidx_mrv_arm:qidx_mrv_arm + 6] = np.copy(self.mrv_hil_js.position[1:7])
      q_vis[qidx_client_arm:qidx_client_arm + 6] = np.copy(self.client_hil_js.position[1:7])
      v_vis = np.zeros(pin_model.nv)
      ee_mrv_arm_path_marker.points.clear()
      for step in range(move_out_steps):
        if rospy.is_shutdown():
          break
        # Forward kinematics: get poses and Jacobians
        pin.forwardKinematics(pin_model, pin_data, q_vis)
        pin.computeJointJacobians(pin_model, pin_data)
        pin.updateFramePlacement(pin_model, pin_data, peg_fid)
        t_peg_hb = pin_data.oMf[peg_fid].translation
        rmat_peg_hb = pin_data.oMf[peg_fid].rotation
        Jpeg = pin.getFrameJacobian(pin_model, pin_data, peg_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:, vidx_mrv_arm:vidx_mrv_arm + 6]

        peg_twist_ctrl = np.zeros(6)
        peg_twist_ctrl[:3] = -move_out_speed*rmat_peg_hb[:, 2]

        peg_twist_ctrl = self.clip_twist(peg_twist_ctrl)

        # Solve instantaneous inverse kinematics
        v_mrv_arm_cmd = np.linalg.lstsq(Jpeg, peg_twist_ctrl)[0]
        v_mrv_arm_cmd = self.clip_mrv_arm_cmd(v_vis[self.vidx_mrv_hil:self.vidx_mrv_hil + 6],v_mrv_arm_cmd)

        q_vis[qidx_mrv_arm:qidx_mrv_arm + 6] += v_mrv_arm_cmd*dt
        v_vis[vidx_mrv_arm:vidx_mrv_arm + 6] = np.copy(v_mrv_arm_cmd)

        ee_mrv_arm_path_marker.points.append(Point(t_peg_hb[0], t_peg_hb[1], t_peg_hb[2]))

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
      q[qidx_mrv_arm:qidx_mrv_arm + 6] = self.mrv_hil_js.position[1:7]
      v[vidx_mrv_arm:vidx_mrv_arm + 6] = self.mrv_hil_js.velocity[1:7]

      # Forward kinematics: get poses and Jacobians
      pin.forwardKinematics(pin_model, pin_data, q)
      pin.computeJointJacobians(pin_model, pin_data)
      pin.updateFramePlacement(pin_model, pin_data, peg_fid)
      t_peg_hb = pin_data.oMf[peg_fid].translation
      rmat_peg_hb = pin_data.oMf[peg_fid].rotation
      Jpeg = pin.getFrameJacobian(pin_model, pin_data, peg_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:, vidx_mrv_arm:vidx_mrv_arm + 6]

      pin.updateFramePlacement(pin_model, pin_data, ft_fid)

      ft_compensated = self.get_ft_compensated(pin_data, ft_fid)
      if ft_compensated is None:
        print("Stopping because F/T data is unavailable.")
        quit()

      if self.is_ft_excessive(ft_compensated):
        print('Stopping because of excessive force on F/T sensor.')
        quit()

      peg_twist_ctrl = np.zeros(6)
      peg_twist_ctrl[:3] = -move_out_speed*rmat_peg_hb[:, 2]

      peg_twist_ctrl = self.clip_twist(peg_twist_ctrl)

      # We have end-effector velocity commands. Now we solve
      # for joint velocity commands using inverse kinematics
      v_mrv_arm_cmd = np.linalg.lstsq(Jpeg, peg_twist_ctrl)[0]
      v_mrv_arm_cmd = self.clip_mrv_arm_cmd(v[self.vidx_mrv_hil:self.vidx_mrv_hil + 6],v_mrv_arm_cmd)

      # acc_norm_mrv_arm = np.linalg.norm((v_mrv_arm_cmd - v[vidx_mrv_arm:vidx_mrv_arm + 6])/dt, ord=np.inf)
      self.publish_velocity('mrv', v_mrv_arm_cmd)

      rate.sleep()
   # Move is completed when the peg is out of the hole, transition to idle mode
    self.ur_idle_mode('mrv')
    self.ur_idle_mode('client')
  
 

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

  def get_vel_cmd_moving_ee_behind_barrier(self, q, v, pos_d, rmat_d, controller):
    pin_model = self.pin_model
    pin_data = self.pin_data
    peg_fid = self.peg_fid
    nozzle_fid = self.nozzle_fid

    x = np.concatenate((q, v))

    # TODO: References hb conflate the world frame with the hardware base frame. This is incorrect for our new setup. We 
    # need to track this down wherever itt occures referencing back to the peg_in_hole code and making sure that 
    # our HIL runner is consistent with the new setup
    pin.forwardKinematics(pin_model, pin_data, q)
    pin.updateFramePlacement(pin_model, pin_data, peg_fid)
    pin.updateFramePlacement(pin_model, pin_data, nozzle_fid)
    t_peg_hb = pin_data.oMf[peg_fid].translation
    rmat_peg_hb = pin_data.oMf[peg_fid].rotation
    nozzle_pos = pin_data.oMf[nozzle_fid].translation
    nozzle_rmat = pin_data.oMf[nozzle_fid].rotation
    
    if self.override_barriers:
      mrv_arm_barrier_pos = pos_d
      client_arm_barrier_pos = pos_d
    else: 
      mrv_arm_barrier_pos = nozzle_pos - nozzle_rmat[:, 2]*self.ur_barrier_dist
      client_arm_barrier_pos = t_peg_hb + rmat_peg_hb[:, 2]*self.ur_barrier_dist


    # TODO: Pretty sure that this should be a +7 instead of a +6 becuase its the full state vector
    if controller == self.mrv_arm_init_controller:
        v_cmd = controller.get_vel_cmd(x, \
                                     pos_d, \
                                     rmat_d, \
                                     mrv_arm_barrier_pos, \
                                     nozzle_rmat[:, 2])[self.vidx_mrv_hil:self.vidx_mrv_hil + 6]
    else:
      v_cmd = controller.get_vel_cmd(x, \
                                     pos_d, \
                                     rmat_d, \
                                     client_arm_barrier_pos, \
                                     -rmat_peg_hb[:, 2])[self.vidx_client_hil:self.vidx_client_hil + 6]
      
    pos_err_norm = controller.get_prev_pos_err_norm()
    rot_err_norm = controller.get_prev_rot_err_norm()
    return v_cmd, pos_err_norm, rot_err_norm
  
  def update_pin_model(self):
    '''Update the UR arms' Pinnochio model using the current UR arm joint values and return the peg/nozzle position and orientation.
    
    Outputs: 
      t_peg_hb - position of peg in hardware base frame
      rmat_peg_hb - rotation matrix of peg in hardware base frame
      t_nozzle_hb - position of nozzle in hardware base frame
      rmat_nozzle_hb - rotation matrix of nozzle in hardware base frame'''
    
    pin_model = self.pin_model
    pin_data = self.pin_data
    ctrl = self.ctrl
    peg_fid = self.peg_fid
    nozzle_fid = self.nozzle_fid
    qidx_mrv_arm = self.qidx_mrv_hil
    qidx_client_arm = self.qidx_client_hil
    
    q = pin.neutral(pin_model)
    q[qidx_mrv_arm:qidx_mrv_arm + 6] = self.mrv_hil_js.position[1:7]
    q[qidx_client_arm:qidx_client_arm + 6] = self.client_hil_js.position[1:7]
    pin.forwardKinematics(pin_model, pin_data, q)
    pin.updateFramePlacement(pin_model, pin_data, peg_fid)
    pin.updateFramePlacement(pin_model, pin_data, nozzle_fid)
    t_peg_hb = pin_data.oMf[peg_fid].translation
    rmat_peg_hb = pin_data.oMf[peg_fid].rotation
    t_nozzle_hb = pin_data.oMf[nozzle_fid].translation
    rmat_nozzle_hb = pin_data.oMf[nozzle_fid].rotation
  
    return t_peg_hb, rmat_peg_hb, t_nozzle_hb, rmat_nozzle_hb

  # TODO: Implement this with the rtde interface
  def move_client_arm_to_pose(self,hw_nozzle_pos_d,hw_nozzle_rmat_d, visualize_before_moving):
    '''Move the client_arm arm to a specified pose. 
    
    If visualize is set to True, the trajectory will be visualized and the user will prompted to proceed before moving'''
    client_arm_init_controller = self.client_arm_init_controller
    dt = self.dt
    pin_model = self.pin_model
    pin_data = self.pin_data
    ctrl = self.ctrl
    ft_fid = self.ft_fid
    nozzle_fid = self.nozzle_fid
    qidx_mrv_arm = self.qidx_mrv_hil
    qidx_client_arm = self.qidx_client_hil
    vidx_mrv_arm = self.vidx_mrv_hil
    vidx_client_arm = self.vidx_client_hil
    client_arm_pub = self.client_arm_pub
    ee_client_arm_path_marker = self.ee_client_arm_path_marker
    ee_client_arm_path_pub = self.ee_client_arm_path_pub

    '''Visualize client_arm trajectory'''
    stop_vis = False
    while not stop_vis:
      client_arm_init_controller.reset()
      q_vis = pin.neutral(pin_model)
      v_vis = np.zeros(pin_model.nv)
      q_vis[qidx_mrv_arm:qidx_mrv_arm + 6] = np.copy(self.mrv_hil_js.position[1:7])
      q_vis[qidx_client_arm:qidx_client_arm + 6] = np.copy(self.client_hil_js.position[1:7])
      v_vis[vidx_mrv_arm:vidx_mrv_arm + 6] = np.copy(self.mrv_hil_js.velocity[1:7])
      v_vis[vidx_client_arm:vidx_client_arm + 6] = np.copy(self.client_hil_js.velocity[1:7])
      ee_client_arm_path_marker.points.clear()
      rate = rospy.Rate(5/dt)
      while not rospy.is_shutdown():
        v_client_arm_cmd, \
        pos_err_norm, \
        rot_err_norm = self.get_vel_cmd_moving_ee_behind_barrier(q_vis, v_vis, \
                                                                  hw_nozzle_pos_d, \
                                                                  hw_nozzle_rmat_d, \
                                                                  client_arm_init_controller)
        if pos_err_norm <= 1e-3 and rot_err_norm <= 0.5*np.pi/180:
          break

        q_vis[qidx_client_arm:qidx_client_arm + 6] += v_client_arm_cmd*dt
        v_vis[vidx_client_arm:vidx_client_arm + 6] = np.copy(v_client_arm_cmd)

        nozzle_pos = pin_data.oMf[nozzle_fid].translation
        ee_client_arm_path_marker.points.append(Point(nozzle_pos[0], nozzle_pos[1], nozzle_pos[2]))
        ee_client_arm_path_marker.header.stamp = rospy.Time.now()
        ee_client_arm_path_pub.publish(ee_client_arm_path_marker)

        if self.visualize_while_simulating:
          self.publish_hw_joints_for_viz(pin_model.nv,q_vis)

        rate.sleep()

      if visualize_before_moving:
        stop_vis = self.check_for_continue_with_visualization()       
      else:
        stop_vis = True

    '''Move client_arm'''
    client_arm_init_controller.reset()
    rate = rospy.Rate(1/dt)
    q = pin.neutral(pin_model)
    v = np.zeros(pin_model.nv)
    while not rospy.is_shutdown():
      # Get joint positions and velocities
      q[qidx_mrv_arm:qidx_mrv_arm + 6] = self.mrv_hil_js.position[1:7]
      q[qidx_client_arm:qidx_client_arm + 6] = self.client_hil_js.position[1:7]
      v[vidx_mrv_arm:vidx_mrv_arm + 6] = self.mrv_hil_js.velocity[1:7]
      v[vidx_client_arm:vidx_client_arm + 6] = self.client_hil_js.velocity[1:7]

      pin.forwardKinematics(pin_model, pin_data, q)
      pin.updateFramePlacement(pin_model, pin_data, ft_fid)

      ft_compensated = self.get_ft_compensated(pin_data, ft_fid)
      if ft_compensated is None:
        print("Stopping because F/T data is unavailable.")
        quit()

      if self.is_ft_excessive(ft_compensated):
        print('Stopping because of excessive force on F/T sensor.')
        quit()

      v_client_arm_cmd, \
      pos_err_norm, \
      rot_err_norm = self.get_vel_cmd_moving_ee_behind_barrier(q, v, \
                                                                hw_nozzle_pos_d, \
                                                                hw_nozzle_rmat_d, \
                                                                client_arm_init_controller)
      if pos_err_norm <= 1e-3 and rot_err_norm <= 0.5*np.pi/180:
        break

      acc_norm_client_arm = np.linalg.norm((v_client_arm_cmd - v[vidx_client_arm:vidx_client_arm + 6])/dt, ord=np.inf)
      ctrl.speedj(v_client_arm_cmd, client_arm_pub, acc_norm_client_arm)

      rate.sleep()

  def move_mrv_arm_to_pose(self,hw_peg_pos_d,hw_peg_rmat_d,visualize_before_moving):
    mrv_arm_init_controller = self.mrv_arm_init_controller
    dt = self.dt
    pin_model = self.pin_model
    pin_data = self.pin_data
    ctrl = self.ctrl
    peg_fid = self.peg_fid
    ft_fid = self.ft_fid
    qidx_mrv_arm = self.qidx_mrv_hil
    qidx_client_arm = self.qidx_client_hil
    vidx_mrv_arm = self.vidx_mrv_hil
    vidx_client_arm = self.vidx_client_hil
    mrv_arm_pub = self.mrv_arm_pub
    ee_mrv_arm_path_marker = self.ee_mrv_arm_path_marker
    ee_mrv_arm_path_pub = self.ee_mrv_arm_path_pub
      
    '''Visualize mrv_arm trajectory'''
    stop_vis = False
    while not stop_vis:
      mrv_arm_init_controller.reset()
      q_vis = pin.neutral(pin_model)
      v_vis = np.zeros(pin_model.nv)
      q_vis[qidx_mrv_arm:qidx_mrv_arm + 6] = self.mrv_hil_js.position[1:7]
      q_vis[qidx_client_arm:qidx_client_arm + 6] = self.client_hil_js.position[1:7]
      ee_mrv_arm_path_marker.points.clear()
      rate = rospy.Rate(5/dt)
      while not rospy.is_shutdown():
        v_mrv_arm_cmd, \
        pos_err_norm, \
        rot_err_norm = self.get_vel_cmd_moving_ee_behind_barrier(q_vis, v_vis, \
                                                                 hw_peg_pos_d, \
                                                                 hw_peg_rmat_d, \
                                                                 mrv_arm_init_controller)

        if pos_err_norm <= 1e-3 and rot_err_norm <= 0.5*np.pi/180:
          break

        q_vis[qidx_mrv_arm:qidx_mrv_arm + 6] += v_mrv_arm_cmd*dt
        v_vis[vidx_mrv_arm:vidx_mrv_arm + 6] = np.copy(v_mrv_arm_cmd)

        pin.updateFramePlacement(pin_model, pin_data, peg_fid)
        t_peg_hb = pin_data.oMf[peg_fid].translation

        ee_mrv_arm_path_marker.points.append(Point(t_peg_hb[0], t_peg_hb[1], t_peg_hb[2]))
        ee_mrv_arm_path_marker.header.stamp = rospy.Time.now()
        ee_mrv_arm_path_pub.publish(ee_mrv_arm_path_marker)

        if self.visualize_while_simulating:
          self.publish_hw_joints_for_viz(pin_model.nv,q_vis)

        rate.sleep()

      if visualize_before_moving:
        stop_vis = self.check_for_continue_with_visualization()
      else:
        stop_vis = True

    '''Move mrv_arm'''
    mrv_arm_init_controller.reset()
    rate = rospy.Rate(1/dt)
    q = pin.neutral(pin_model)
    v = np.zeros(pin_model.nv)
    while not rospy.is_shutdown():
      # Get joint positions and velocities
      q[qidx_mrv_arm:qidx_mrv_arm + 6] = self.mrv_hil_js.position[1:7]
      q[qidx_client_arm:qidx_client_arm + 6] = self.client_hil_js.position[1:7]
      v[vidx_mrv_arm:vidx_mrv_arm + 6] = self.mrv_hil_js.velocity[1:7]
      v[vidx_client_arm:vidx_client_arm + 6] = self.client_hil_js.velocity[1:7]

      pin.forwardKinematics(pin_model, pin_data, q)
      pin.updateFramePlacement(pin_model, pin_data, ft_fid)

      ft_compensated = self.get_ft_compensated(pin_data, ft_fid)
      if ft_compensated is None:
        print("Stopping because F/T data is unavailable.")
        quit()

      if self.is_ft_excessive(ft_compensated):
        print('Stopping because of excessive force on F/T sensor.')
        quit()

      v_mrv_arm_cmd, \
      pos_err_norm, \
      rot_err_norm = self.get_vel_cmd_moving_ee_behind_barrier(q, v, \
                                                               hw_peg_pos_d, \
                                                               hw_peg_rmat_d, \
                                                               mrv_arm_init_controller)

      if pos_err_norm <= 1e-3 and rot_err_norm <= 0.5*np.pi/180:
        break

      acc_norm_mrv_arm = np.linalg.norm((v_mrv_arm_cmd - v[vidx_mrv_arm:vidx_mrv_arm + 6])/dt, ord=np.inf)
      ctrl.speedj(v_mrv_arm_cmd, mrv_arm_pub, acc_norm_mrv_arm)

      rate.sleep()

  def move_mrv_arm_and_client_arm_to_pose(self,hw_peg_pos_d,hw_peg_rmat_d, hw_nozzle_pos_d, hw_nozzle_rmat_d, visualize_before_moving):
    mrv_arm_init_controller = self.mrv_arm_init_controller
    client_arm_init_controller = self.client_arm_init_controller
    dt = self.dt
    pin_model = self.pin_model
    pin_data = self.pin_data
    ctrl = self.ctrl
    peg_fid = self.peg_fid
    nozzle_fid = self.nozzle_fid
    ft_fid = self.ft_fid
    qidx_mrv_arm = self.qidx_mrv_hil
    qidx_client_arm = self.qidx_client_hil
    vidx_mrv_arm = self.vidx_mrv_hil
    vidx_client_arm = self.vidx_client_hil
    mrv_arm_pub = self.mrv_arm_pub
    ee_mrv_arm_path_marker = self.ee_mrv_arm_path_marker
    ee_mrv_arm_path_pub = self.ee_mrv_arm_path_pub
    client_arm_pub = self.client_arm_pub
    ee_client_arm_path_marker = self.ee_client_arm_path_marker
    ee_client_arm_path_pub = self.ee_client_arm_path_pub

    '''Visualize trajectory'''
    stop_vis = False
    while not stop_vis:
      client_arm_init_controller.reset()
      mrv_arm_init_controller.reset()
      q_vis = pin.neutral(pin_model)
      v_vis = np.zeros(pin_model.nv)
      q_vis[qidx_mrv_arm:qidx_mrv_arm + 6] = self.mrv_hil_js.position[1:7]
      q_vis[qidx_client_arm:qidx_client_arm + 6] = self.client_hil_js.position[1:7]
      v_vis[vidx_mrv_arm:vidx_mrv_arm + 6] = np.copy(self.mrv_hil_js.velocity[1:7])
      v_vis[vidx_client_arm:vidx_client_arm + 6] = np.copy(self.client_hil_js.velocity[1:7])
      ee_client_arm_path_marker.points.clear()
      ee_mrv_arm_path_marker.points.clear()
      rate = rospy.Rate(5/dt)
      while not rospy.is_shutdown():
        v_mrv_arm_cmd, \
        pos_err_norm_mrv_arm, \
        rot_err_norm_mrv_arm = self.get_vel_cmd_moving_ee_behind_barrier(q_vis, v_vis, \
                                                                 hw_peg_pos_d, \
                                                                 hw_peg_rmat_d, \
                                                                 mrv_arm_init_controller)

        v_client_arm_cmd, \
        pos_err_norm_client_arm, \
        rot_err_norm_client_arm = self.get_vel_cmd_moving_ee_behind_barrier(q_vis, v_vis, \
                                                                  hw_nozzle_pos_d, \
                                                                  hw_nozzle_rmat_d, \
                                                                  client_arm_init_controller)

        if pos_err_norm_client_arm <= 1e-3 and rot_err_norm_client_arm <= 0.5*np.pi/180 and \
           pos_err_norm_mrv_arm <= 1e-3 and rot_err_norm_mrv_arm <= 0.5*np.pi/180:
          break

        q_vis[qidx_client_arm:qidx_client_arm + 6] += v_client_arm_cmd*dt
        v_vis[vidx_client_arm:vidx_client_arm + 6] = np.copy(v_client_arm_cmd)
        q_vis[qidx_mrv_arm:qidx_mrv_arm + 6] += v_mrv_arm_cmd*dt
        v_vis[vidx_mrv_arm:vidx_mrv_arm + 6] = np.copy(v_mrv_arm_cmd)

        pin.updateFramePlacement(pin_model, pin_data, peg_fid)

        t_peg_hb = pin_data.oMf[peg_fid].translation
        nozzle_pos = pin_data.oMf[nozzle_fid].translation

        ee_client_arm_path_marker.points.append(Point(nozzle_pos[0], nozzle_pos[1], nozzle_pos[2]))
        ee_client_arm_path_marker.header.stamp = rospy.Time.now()
        ee_client_arm_path_pub.publish(ee_client_arm_path_marker)
        ee_mrv_arm_path_marker.points.append(Point(t_peg_hb[0], t_peg_hb[1], t_peg_hb[2]))
        ee_mrv_arm_path_marker.header.stamp = rospy.Time.now()
        ee_mrv_arm_path_pub.publish(ee_mrv_arm_path_marker)

        if self.visualize_while_simulating:
          self.publish_hw_joints_for_viz(pin_model.nv,q_vis)

        rate.sleep()

      if visualize_before_moving:
        stop_vis = self.check_for_continue_with_visualization()
      else:
        stop_vis = True

    '''Move both arms'''
    mrv_arm_init_controller.reset()
    client_arm_init_controller.reset()
    rate = rospy.Rate(1/dt)
    q = pin.neutral(pin_model)
    v = np.zeros(pin_model.nv)
    while not rospy.is_shutdown():
      # Get joint positions and velocities
      q[qidx_mrv_arm:qidx_mrv_arm + 6] = self.mrv_hil_js.position[1:7]
      q[qidx_client_arm:qidx_client_arm + 6] = self.client_hil_js.position[1:7]
      v[vidx_mrv_arm:vidx_mrv_arm + 6] = self.mrv_hil_js.velocity[1:7]
      v[vidx_client_arm:vidx_client_arm + 6] = self.client_hil_js.velocity[1:7]

      pin.forwardKinematics(pin_model, pin_data, q)
      pin.updateFramePlacement(pin_model, pin_data, ft_fid)

      ft_compensated = self.get_ft_compensated(pin_data, ft_fid)
      if ft_compensated is None:
        print("Stopping because F/T data is unavailable.")
        quit()

      if self.is_ft_excessive(ft_compensated):
        print('Stopping because of excessive force on F/T sensor.')
        quit()

      v_mrv_arm_cmd, \
      pos_err_norm_mrv_arm, \
      rot_err_norm_mrv_arm = self.get_vel_cmd_moving_ee_behind_barrier(q, v, \
                                                                hw_peg_pos_d, \
                                                                hw_peg_rmat_d, \
                                                                mrv_arm_init_controller)

      v_client_arm_cmd, \
      pos_err_norm_client_arm, \
      rot_err_norm_client_arm = self.get_vel_cmd_moving_ee_behind_barrier(q, v, \
                                                                hw_nozzle_pos_d, \
                                                                hw_nozzle_rmat_d, \
                                                                client_arm_init_controller)

      if pos_err_norm_client_arm <= 1e-3 and rot_err_norm_client_arm <= 0.5*np.pi/180 and \
         pos_err_norm_mrv_arm <= 1e-3 and rot_err_norm_mrv_arm <= 0.5*np.pi/180:
        break

      acc_norm_client_arm = np.linalg.norm((v_client_arm_cmd - v[vidx_client_arm:vidx_client_arm + 6])/dt, ord=np.inf)
      acc_norm_mrv_arm = np.linalg.norm((v_mrv_arm_cmd - v[vidx_mrv_arm:vidx_mrv_arm + 6])/dt, ord=np.inf)
      ctrl.speedj(v_client_arm_cmd, client_arm_pub, acc_norm_client_arm)
      ctrl.speedj(v_mrv_arm_cmd, mrv_arm_pub, acc_norm_mrv_arm)

      rate.sleep()

  def emulate(self, sw_peg_pos, sw_peg_rmat, sw_peg_twist, sw_nozzle_pos, sw_nozzle_rmat, sw_nozzle_twist, sw_dist_nozzle_opening_from_goal):
    '''
    Outputs: 
      fail_reason: ? 
      wrench_peg_peg: wrench with force followed by torque applied to the peg's tip, expressed the peg tip frame
    '''

    pin_model = self.pin_model
    pin_data = self.pin_data
    ctrl = self.ctrl
    peg_fid = self.peg_fid
    nozzle_fid = self.nozzle_fid
    ft_fid = self.ft_fid
    rmat_ft_peg = self.rmat_ft_peg
    qidx_mrv_arm = self.qidx_mrv_hil
    qidx_client_arm = self.qidx_client_hil
    vidx_mrv_arm = self.vidx_mrv_hil
    vidx_client_arm = self.vidx_client_hil
    dt = self.dt
    mrv_arm_pub = self.mrv_arm_pub
    client_arm_pub = self.client_arm_pub

    peg_twist_ctrl = np.zeros(6)
    nozzle_twist_ctrl = np.zeros(6)

    # Get joint positions and velocities
    q = pin.neutral(pin_model)
    v = np.zeros(pin_model.nv)
    q[qidx_mrv_arm:qidx_mrv_arm + 6] = self.mrv_hil_js.position[1:7]
    q[qidx_client_arm:qidx_client_arm + 6] = self.client_hil_js.position[1:7]
    v[vidx_mrv_arm:vidx_mrv_arm + 6] = self.mrv_hil_js.velocity[1:7]
    v[vidx_client_arm:vidx_client_arm + 6] = self.client_hil_js.velocity[1:7]

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

    '''Enforce passivity'''
    if self.enforce_passivity:

      '''Get the passivity controller force using the velocity of the peg'''
      sw_peg_linear_vel_body = sw_peg_rmat.transpose()@sw_peg_twist[:3]
      force_update_peg_peg_from_peg = self.peg_passivity_enforcer.update(sw_peg_linear_vel_body, wrench_peg_peg[:3])

      '''Get the passivity controller force using the velocity of the nozzle'''
      sw_nozzle_linear_vel_body = sw_nozzle_rmat.transpose()@sw_nozzle_twist[:3]
      force_nozzle_nozzle = sw_nozzle_rmat.transpose()@sw_peg_rmat@(-1*wrench_peg_peg[:3])
      updated_force_nozzle_nozzle = self.nozzle_passivity_enforcer.update(sw_nozzle_linear_vel_body, force_nozzle_nozzle)

      force_update_peg_peg_from_nozzle = sw_peg_rmat.transpose()@sw_nozzle_rmat@(-1*updated_force_nozzle_nozzle)
      
      # Reconcile the two different stabilized forces
      force_update = force_update_peg_peg_from_nozzle
      for i in range(0,3):
        if force_update_peg_peg_from_peg[i] > force_update_peg_peg_from_nozzle[i]:
          force_update[i] = force_update_peg_peg_from_peg[i]

      print("updates")
      print(force_update_peg_peg_from_nozzle)
      print(force_update_peg_peg_from_peg)
      print(force_update)

      wrench_peg_peg[:3] = wrench_peg_peg[:3] - force_update

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
    v_client_arm_cmd = np.linalg.lstsq(Jnozzle, nozzle_twist_ctrl)[0]
    v_mrv_arm_cmd = np.linalg.lstsq(Jpeg, peg_twist_ctrl)[0]

    v_client_arm_cmd = self.clip_client_arm_cmd(self.client_hil_js.velocity[1:7],v_client_arm_cmd)
    v_mrv_arm_cmd = self.clip_mrv_arm_cmd(self.mrv_hil_js.velocity[1:7],v_mrv_arm_cmd)

    # Only send commands to hardware if peg and nozzle in sw sim are within certain proximity
    if not self.override_barriers:
      if np.abs(sw_peg_pos_wrt_nozzle[0]) > 0.11 \
         or np.abs(sw_peg_pos_wrt_nozzle[1]) > 0.11 \
         or sw_peg_pos_wrt_nozzle[2] < -0.11 \
         or sw_peg_pos_wrt_nozzle[2] > 1.1*sw_dist_nozzle_opening_from_goal:
      
        self.need_to_reinitialize = True
        v_mrv_arm_cmd[:] = 0
        v_client_arm_cmd[:] = 0

    acc_norm_client_arm = np.linalg.norm((v_client_arm_cmd - self.client_hil_js.velocity[1:7])/dt, ord=np.inf)
    acc_norm_mrv_arm = np.linalg.norm((v_mrv_arm_cmd - self.mrv_hil_js.velocity[1:7])/dt, ord=np.inf)
    
    self.mrv_arm_joint_vel_cmds_trj.append(np.copy(v_mrv_arm_cmd))
    self.client_arm_joint_vel_cmds_trj.append(np.copy(v_client_arm_cmd))

    # Send commands to hardware
    if not self.need_to_reinitialize:
      ctrl.speedj(v_client_arm_cmd, client_arm_pub, acc_norm_client_arm)
      ctrl.speedj(v_mrv_arm_cmd, mrv_arm_pub, acc_norm_mrv_arm)
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

