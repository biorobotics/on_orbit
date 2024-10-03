import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from on_orbit.on_orbit_bindings import *
from scipy.spatial.transform import Rotation as R
from scipy.optimize import linprog
from itertools import product
import cv2
from R3_SO3_ekf import R3SO3EKF
import time
import pybullet
import pybullet_data
import pybullet_utils.bullet_client as pbbc
from angle_mod import *
import mujoco_py
import time
import scipy
import copy

def in_hull(points, x):
    n_points = len(points)
    n_dim = len(x)
    c = np.zeros(n_points)
    A = np.concatenate((points.transpose(), np.ones((1, n_points))), 0)
    b = np.concatenate((x, np.ones(1)))
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success

class MRVClientSim(object):
  # If pose noise is True, we add noise to the client pose and use that as a measurement. Otherwise we generate image features
  # and add noise in image space.
  def __init__(self, mrv_cv_urdf_file, mrv_urdf_file, pybullet_mrv_urdf_file, pybullet_cv_urdf_file, joint_angle_lower_limits, joint_angle_upper_limits, joint_vel_limits, joint_acc_limits, 
               joint_torque_limits, dt, cone_slope, time_steps_between_measurements, cw_a, cw_mu, cw_orbit_dir, do_noisy_state_estimation, lock_client, lock_mrv, use_cw=True):
    
    '''Input URDFs:
    These are all simply passed direclty to MRVCLientSim
    mrv_cv_urdf_file - currently using robot_cv_detached.urdf. Ued to build pin_model
    mrv_urdf_file - currently using robot.urdf. Used to build mrv_pin_model

    These are used for the pybullet simulation:
      pybullet_mrv_urdf_file - currently using barebones_robot.urdf. 
      pybullet_cv_urdf_file - currently using cv.urdf.
    '''

    self.joint_torque_limits = joint_torque_limits
    
    self.max_diff = 0.
    self.pin_model = RobotWrapper.BuildFromURDF(mrv_cv_urdf_file).model
    self.pin_model.gravity.setZero()
    self.pin_data = pin.Data(self.pin_model)
    self.prev_pin_data = pin.Data(self.pin_model)

    self.pin_data_est = pin.Data(self.pin_model)

    self.joint_angle_lower_limits = np.copy(joint_angle_lower_limits)
    self.joint_angle_upper_limits = np.copy(joint_angle_upper_limits)
    self.joint_vel_limits = np.copy(joint_vel_limits)
    self.joint_acc_limits = np.copy(joint_acc_limits)

    self.cv_jidx = self.pin_model.getJointId('world_to_client')
    self.cv_qidx = self.pin_model.idx_qs[self.cv_jidx]
    self.cv_vidx = self.pin_model.idx_vs[self.cv_jidx]

    self.mrv_jidx = self.pin_model.getJointId('world_to_base')
    self.mrv_qidx = self.pin_model.idx_qs[self.mrv_jidx]
    self.mrv_vidx = self.pin_model.idx_vs[self.mrv_jidx]

    self.mrv_nq = self.pin_model.nq - 7
    self.mrv_nv = self.pin_model.nv - 6
    self.num_rotary = self.mrv_nv - 6

    self.sim_time = 0
    self.dt = dt
    self.cone_slope = cone_slope

    self.x = None
    self.x_est = None

    self.peg_fid = self.pin_model.getFrameId('ee_tip')
    self.nozzle_fid = self.pin_model.getFrameId('nozzle')
    self.goal_fid = self.pin_model.getFrameId('goal')
    self.wrist_fid = self.pin_model.getFrameId('wrist')
    self.hole_fid = self.pin_model.getFrameId('hole')
    self.mrv_base_fid = self.pin_model.getFrameId('base_link')
    self.tube_geom_fids = [self.pin_model.getFrameId('tube_geom' + str(i)) for i in range(8)]
    self.cone_vertex_fid = self.pin_model.getFrameId('cone_vertex')

    pin.forwardKinematics(self.pin_model, self.pin_data, pin.neutral(self.pin_model))
    pin.updateFramePlacements(self.pin_model, self.pin_data)

    # Position of tip with respect to wrist, expressed in the world frame.
    # We assume when robot is neutral, that peg frame is aligned with world frame
    self.t_tip_wrist = self.pin_data.oMf[self.peg_fid].translation - self.pin_data.oMf[self.wrist_fid].translation

    self.joint_vel_vertices = np.array(list(product((1, -1), repeat=self.num_rotary)))*np.tile(joint_vel_limits, (2**self.num_rotary, 1))
    self.joint_acc_vertices = np.array(list(product((1, -1), repeat=self.num_rotary)))*np.tile(joint_acc_limits, (2**self.num_rotary, 1))

    self.image_width = 2592
    self.image_height = 2048
    fovx = 45.9
    fovy = 35.2
    fx = 0.5*self.image_width/np.tan(0.5*fovx)
    fy = 0.5*self.image_height/np.tan(0.5*fovy)
    self.camera_matrix = np.array([[fx, 0., self.image_width/2], \
                                   [0., fy, self.image_height/2], \
                                   [0., 0., 1.]])
    client_lx = 2.209 # m
    client_ly = 2.387 # m
    client_lz = 3.341 # m
    hexagon_width = client_ly/4 # Distance from one vertex of the hexagon to its opposite
    hexagon_halfwidth = hexagon_width/2
    num_hexagon_vertices = 6
    features_3d_wrt_client = []
    for prod in product((1, -1), repeat=2):
        features_3d_wrt_client.append(np.array([prod[0]*client_lx/2, prod[1]*client_ly/2, -client_lz/2]))
    for i in range(num_hexagon_vertices):
        theta = i*2*np.pi/num_hexagon_vertices
        features_3d_wrt_client.append(np.array([hexagon_halfwidth*np.cos(theta), hexagon_halfwidth*np.sin(theta), -client_lz/2]))

    self.features_3d_wrt_client = np.array(features_3d_wrt_client)
    self.num_features = self.features_3d_wrt_client.shape[0]

    self.t_cam_base = np.array([0.889, 0.548, 1.953])

    self.h_0 = np.zeros(6)

    # self.ekf = SE3EKF()
    self.ekf = R3SO3EKF()
    self.mrv_pin_model = RobotWrapper.BuildFromURDF(mrv_urdf_file, root_joint=pin.JointModelFreeFlyer()).model
    self.mrv_pin_model.gravity.setZero()
    self.mrv_pin_data = pin.Data(self.mrv_pin_model)
    self.prev_mrv_pin_data = pin.Data(self.mrv_pin_model)

    self.mrv_peg_fid = self.mrv_pin_model.getFrameId('ee_tip')
    self.mrv_wrist_fid = self.mrv_pin_model.getFrameId('wrist')

    self.client_mass = pin.computeTotalMass(self.pin_model) - pin.computeTotalMass(self.mrv_pin_model)
    self.client_inertia = self.pin_model.inertias[self.cv_jidx].matrix()[3:, 3:]
    self.client_inertia_inv = np.linalg.inv(self.client_inertia)

    self.mrv_inertia  = self.pin_model.inertias[self.mrv_jidx].matrix()[3:, 3:] # TODO: update this to also include the joint intertias
    self.mrv_inertia_inv = np.linalg.inv(self.mrv_inertia)
    self.mrv_mass = pin.computeTotalMass(self.mrv_pin_model)

    self.time_steps_between_measurements = time_steps_between_measurements
    self.time_steps_since_measurement = 0

    self.reset_ekf()

    self.pb_peg_wrench = np.zeros(6)

    # To check if client is out of workspace
    pin.forwardKinematics(self.mrv_pin_model, self.mrv_pin_data, pin.neutral(self.mrv_pin_model))
    com = pin.centerOfMass(self.mrv_pin_model, self.mrv_pin_data, pin.neutral(self.mrv_pin_model))
    mrv_peg_fid = self.mrv_pin_model.getFrameId('ee_tip')
    pin.updateFramePlacements(self.mrv_pin_model, self.mrv_pin_data)
    self.reach = np.linalg.norm(self.mrv_pin_data.oMf[mrv_peg_fid].translation - com)

    # PyBullet stuff
    actuator_names = []

    for j in range(2, self.mrv_pin_model.njoints):
      actuator_names.append(self.mrv_pin_model.names[j])

    self.pb_client = pbbc.BulletClient(connection_mode=pybullet.DIRECT) # Or pybullet.GUI for graphical version
    self.pb_client.setAdditionalSearchPath(pybullet_data.getDataPath()) # TODO: what does this actually do? They say it's optional
    self.pb_client.setGravity(0, 0, 0)
    self.pb_client.setTimeStep(dt)

    self.pb_mrv_id = self.pb_client.loadURDF(pybullet_mrv_urdf_file, np.zeros(3), np.array([0., 0., 0., 1.]), flags=pybullet.URDF_USE_INERTIA_FROM_FILE|pybullet.URDF_USE_IMPLICIT_CYLINDER)
    self.pb_cv_id = self.pb_client.loadURDF(pybullet_cv_urdf_file, 100*np.ones(3), np.array([0., 0., 0., 1.]), flags=pybullet.URDF_USE_INERTIA_FROM_FILE|pybullet.URDF_USE_IMPLICIT_CYLINDER)

    for j in range(self.pb_client.getNumJoints(self.pb_mrv_id)):
      self.pb_client.changeDynamics(self.pb_mrv_id, j, lateralFriction=0.0, spinningFriction=0.0, restitution=0.0, rollingFriction=0.0)
    for j in range(self.pb_client.getNumJoints(self.pb_cv_id)):
      self.pb_client.changeDynamics(self.pb_cv_id, j, lateralFriction=0.0, spinningFriction=0.0, restitution=0.0, rollingFriction=0.0)

    self.pb_joint_indices = []
    for actuator_name in actuator_names:
      for j in range(self.pb_client.getNumJoints(self.pb_mrv_id)):
        joint_info = self.pb_client.getJointInfo(self.pb_mrv_id, j)
        if joint_info[1].decode('UTF-8') == actuator_name:
          self.pb_joint_indices.append(j)
          break

    self.initial_client_pos = []
    self.initial_nozzle_q = []
    self.initial_nozzle_pos = []

    '''Simulation options'''
    self.use_solar_panels = False
    self.use_ee_compliance = False

    '''This will reset the pose of the client and/or MRV at each step of the simulation.'''
    self.lock_client = lock_client
    self.lock_mrv = lock_mrv

    self.simulate_angular_velocity_disturbance = False
    self.angular_velocity_disturbance = np.zeros(3)
    self.K_ang_vel_dist = 500.0*np.identity(3) #gains for tracking angular velocity disturbance

    self.compliance_fraction = 0.03
    self.compliance_config = np.zeros(3)
    self.compliance_vel = np.zeros(3)
    self.compliance_force = np.zeros(3)

    self.Kf = 0.556
      
    self.pb_solar_indices = []
    for j in range(self.pb_client.getNumJoints(self.pb_mrv_id)):
      joint_info = self.pb_client.getJointInfo(self.pb_mrv_id, j)
      if 'solar' in joint_info[1].decode('UTF-8'):
        self.pb_solar_indices.append(j)

    self.pb_compliance_indices = []
    for j in range(self.pb_client.getNumJoints(self.pb_mrv_id)):
      joint_info = self.pb_client.getJointInfo(self.pb_mrv_id, j)
      if 'compliance' in joint_info[1].decode('UTF-8'):
        self.pb_compliance_indices.append(j)

    self.pb_peg_id = None
    for j in range(self.pb_client.getNumJoints(self.pb_mrv_id)):
      joint_info = self.pb_client.getJointInfo(self.pb_mrv_id, j)
      if 'ee_peg_joint' in joint_info[1].decode('UTF-8'):
        self.pb_peg_id = j

    self.pb_nozzle_id = None
    for j in range(self.pb_client.getNumJoints(self.pb_cv_id)):
      joint_info = self.pb_client.getJointInfo(self.pb_cv_id, j)
      if 'nozzle' in joint_info[1].decode('UTF-8'):
        self.pb_nozzle_id = j

    # For some reason I need this line to enable torque control
    self.pb_client.setJointMotorControlArray(self.pb_mrv_id, self.pb_joint_indices, pybullet.VELOCITY_CONTROL, forces=np.zeros(len(self.pb_joint_indices)))
    self.pb_client.setJointMotorControlArray(self.pb_mrv_id, self.pb_solar_indices, pybullet.VELOCITY_CONTROL, forces=np.zeros(len(self.pb_solar_indices)))

    # PyBullet damps link linear and angular velocities by default. Disable this
    for j in range(self.pb_client.getNumJoints(self.pb_mrv_id)):
      self.pb_client.changeDynamics(self.pb_mrv_id, j, linearDamping=0, angularDamping=0)

    self.pb_client.changeDynamics(self.pb_mrv_id, -1, linearDamping=0, angularDamping=0)
    self.pb_client.changeDynamics(self.pb_cv_id, -1, linearDamping=0, angularDamping=0)

    # CW stuff
    self.cw_a = cw_a
    self.cw_mu = cw_mu
    self.cw_n = np.sqrt(cw_mu/cw_a**3)
    self.cw_orbit_dir = cw_orbit_dir
    self.use_cw = use_cw

    self.rw_freq = 1 # Hz
    self.rw_period = 1/self.rw_freq
    self.rw_delay_steps = int(self.rw_period/dt)
    self.rw_counter = 0
    self.rw_tau = np.zeros(3)

    self.nozzle_geom_idx = -1
    self.contact_pos_wrt_peg = np.nan*np.ones(3)

    '''Simulate noise on client pose'''
    self.do_noisy_state_estimation = do_noisy_state_estimation

    '''Max for constant offset in position and orientation'''
    self.pose_noise_pos_std = 0.02 #m
    self.pose_noise_rot_std = 0.5*np.pi/180. #rad

    '''Total amount of work done by joints, computed by sampling joint torques'''
    self.joint_work = 0

  @staticmethod
  def hat3(w):
    '''Returns the 3x3 skew symmmetric matrix for a length 3 vector'''
    what = np.zeros((3,3))
    what[0,1] = -w[2]
    what[0,2] = w[1]
    what[1,0] = w[2]
    what[1,2] = -w[0]
    what[2,0] = -w[1]
    what[2,1] = w[0]
    return what
  
  def get_wrist_jacobian(self):
    return copy.deepcopy(pin.getFrameJacobian(self.mrv_pin_model, self.mrv_pin_data, self.mrv_wrist_fid, pin.ReferenceFrame.LOCAL))

  def get_mrv_config(self):
    return copy.deepcopy(self.x[self.mrv_qidx:self.mrv_qidx + self.mrv_pin_model.nq])
  
  def get_mrv_joint_angles(self):
    mrv_config = self.get_mrv_config()
    return mrv_config[7:]
  
  def get_mrv_joint_angles_limit_ratio(self):
    '''Return a vector with values from 0 to 1 where 0 indicates the joint is at its lower limit and 1 indicates the joint is at its upper limit'''
    return np.divide(self.get_mrv_joint_angles() - self.joint_angle_lower_limits,(self.joint_angle_upper_limits - self.joint_angle_lower_limits))
  
  def get_mrv_config_dot(self):
    return copy.deepcopy(self.x[self.pin_model.nq + self.mrv_vidx:self.pin_model.nq + self.mrv_vidx + self.mrv_pin_model.nv])
  
  def get_full_config(self):
    return copy.deepcopy(self.x[:self.pin_model.nq])
  
  def get_full_config_deriv(self):
    return copy.deepcopy(self.x[self.pin_model.nq:])

  def get_mrv_generalized_jacobian(self): 
    ''' Return the generalized Jacobian for a "local world aligned" frame at the probe tip
    
    This assumes that self.mrv_pin_data has been updated already, and computeJointJacobians and updateFrames has been called
    References: 
    Yoshida 1993 "Control of Space Manipulators with Generalized Jacobian Matrix"
    Nenchev 1991 "Analysis, design and control of free-flying space robots using fixed-attitude restricted Jacobina Matrix"
    '''
    J = pin.getFrameJacobian(self.pin_model, self.pin_data, self.peg_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

    pin.computeJointJacobiansTimeVariation(self.pin_model, self.pin_data, self.get_full_config(), self.get_full_config_deriv())
    J_dot = pin.getFrameJacobianTimeVariation(self.pin_model, self.pin_data, self.peg_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    Jm = J[:,6:13]
    Jm_dot = J_dot[:,6:13]

    use_regular_jacobian = False
    if use_regular_jacobian:
      return Jm, Jm_dot
    else:
      Ag = pin.computeCentroidalMap(self.pin_model, self.pin_data, self.get_full_config())

      Ab = Ag[:,:6]
      Ab_inv = np.linalg.inv(Ab)
      Atheta = Ag[:,6:13]
      Jb = J[:,:6] #Jacobian for the base frame
      Jstar = Jm - Jb@Ab_inv@Atheta
      
      #Identical way to compute Jstar
      #Jstar = J[:, 6:13] - J[:, :6]@np.linalg.solve(Ag[:, :6], Ag[:, 6:13])

      # TODO: no need to recompute this, an access via data struct
      dAg = pin.computeCentroidalMapTimeVariation(self.pin_model, self.pin_data, self.get_full_config(), self.get_full_config_deriv())
      dAb = dAg[:,:6]
      dAb_inv = -Ab_inv@dAb@Ab_inv
      Jb_dot = J_dot[:,:6]
      Atheta_dot = dAg[:,6:13]
      Jstar_dot = Jm_dot - Jb_dot@Ab_inv@Atheta -Jb@dAb_inv@Atheta - Jb@Ab_inv@Atheta_dot

      return Jstar, Jstar_dot

  def disable_pybullet_collisions(self):
    for i in range(self.pb_client.getNumJoints(self.pb_cv_id)):
      for j in range(self.pb_client.getNumJoints(self.pb_mrv_id)):
        self.pb_client.setCollisionFilterGroupMask(i, j, 0, 0)

  def set_mrv_pose(self, mrv_pos, mrv_quat):
    self.pb_client.resetBasePositionAndOrientation(self.pb_mrv_id, mrv_pos, mrv_quat)

  def set_client_pose(self, client_position, client_q):
    self.pb_client.resetBasePositionAndOrientation(self.pb_cv_id, client_position, client_q)

  def set_client_velocity(self, client_v, client_w):
    ''' Set the client linear and angular velocity
    Inputs:
      client_v: length 3 list for linear velocity in world frame
      client_w: length 3 list for angular velocity in world frame
    '''
    self.pb_client.resetBaseVelocity(self.pb_cv_id, client_v, client_w)

  def reset_ekf(self):
    self.x_est = None

    # Process noise: increasing this means you trust process more
    self.ekf.kf_Q = np.eye(self.ekf.ndx)
    Q_scale = 0.125
    R_scale = 1.0
    self.ekf.kf_Q[:3] *= Q_scale*5e-7 # position
    self.ekf.kf_Q[3:] *= Q_scale*9e-8 # orientation, linear velocity, angular velocity
    self.ekf.kf_R = np.eye(self.ekf.ndz)
    self.ekf.kf_R[:3] *= R_scale*(0.02/3)**2
    self.ekf.kf_R[3:] *= R_scale*(0.5/3*np.pi/180)**2
    self.ekf.kf_cov0[:3, :3] = np.copy(self.ekf.kf_R[:3, :3])
    self.ekf.kf_cov0[3:6, 3:6] = np.copy(self.ekf.kf_R[3:6, 3:6])
    self.ekf.kf_cov0[6:9, 6:9] = 0.5*(0.014/3)**2*np.eye(3)
    self.ekf.kf_cov0[9:12, 9:12] = 0.5*(1.0/3)**2*np.eye(3)

    self.est_err_trj = []
    self.cov_trj = []

  def set_client_angular_velocity_disturbance(self, client_w):
    '''Track a desired velocity in the client by applying a wrench to it. '''
    self.angular_velocity_disturbance = client_w

  # Resets state to given state, resets pybullet, and resets controllers
  def reset(self, initial_client_pos, initial_nozzle_q, initial_client_v, initial_client_w, 
            initial_mrv_pos, initial_mrv_quat, initial_joint_angles, initial_mrv_vel, initial_mrv_w, initial_joint_velocities, rng):
    
    self.x = np.concatenate((pin.neutral(self.pin_model), np.zeros(self.pin_model.nv)))
    self.x[self.mrv_qidx:self.mrv_qidx + 3] = initial_mrv_pos
    self.x[self.mrv_qidx + 3:self.mrv_qidx + 7] = initial_mrv_quat
    self.x[self.mrv_qidx + 7:self.mrv_qidx + 7 + self.num_rotary] = initial_joint_angles

    self.x[self.cv_qidx:self.cv_qidx + 3] = initial_client_pos
    self.x[self.cv_qidx + 3:self.cv_qidx + 7] = initial_nozzle_q
    self.x[self.pin_model.nq + self.mrv_vidx:self.pin_model.nq + self.mrv_vidx + 3] = initial_mrv_vel
    self.x[self.pin_model.nq + self.mrv_vidx + 3:self.pin_model.nq + self.mrv_vidx + 6] = initial_mrv_w
    self.x[self.pin_model.nq + self.mrv_vidx + 6:self.pin_model.nq + self.mrv_vidx + 6 + self.num_rotary] = initial_joint_velocities
    self.x[self.pin_model.nq + self.cv_vidx:self.pin_model.nq + self.cv_vidx + 3] = initial_client_v
    self.x[self.pin_model.nq + self.cv_vidx + 3:self.pin_model.nq + self.cv_vidx + 6] = initial_client_w

    self.h_0 = pin.computeCentroidalMap(self.pin_model, self.pin_data, self.x[:self.pin_model.nq])@self.x[self.pin_model.nq:]

    # Set MRV pose and velocity
    self.set_mrv_pose(initial_mrv_pos, initial_mrv_quat)
    self.initial_mrv_pos = initial_mrv_pos
    self.initial_mrv_quat = initial_mrv_quat
    
    mrv_R = R.from_quat(initial_mrv_quat)
    self.pb_client.resetBaseVelocity(self.pb_mrv_id, mrv_R.apply(np.array(initial_mrv_vel)), mrv_R.apply(np.array(initial_mrv_w)))

    self.set_client_pose(initial_client_pos, initial_nozzle_q)
    self.initial_client_pos = initial_client_pos 
    self.initial_nozzle_q = initial_nozzle_q  

    # I do not know why this rotation was ever applied. I'm leaving it here as a backup but it created a discrepency between the trajopt and the Pybullet simulation.
    #self.set_client_velocity(mrv_R.apply(np.array(initial_client_v)), mrv_R.apply(np.array(initial_client_w)))
    self.set_client_velocity(np.array(initial_client_v), np.array(initial_client_w))

    for j in range(len(initial_joint_angles)):
      self.pb_client.resetJointState(self.pb_mrv_id, self.pb_joint_indices[j], initial_joint_angles[j], initial_joint_velocities[j])

    for idx in self.pb_solar_indices:
      self.pb_client.resetJointState(self.pb_mrv_id, idx, 0, 0)

    for idx in self.pb_compliance_indices:
      self.pb_client.resetJointState(self.pb_mrv_id, idx, 0, 0)

    cv_R = R.from_quat(initial_nozzle_q)
    client_rmat = cv_R.as_matrix()
    self.cw_xproj = client_rmat[:, 2]
    
    if self.cw_orbit_dir == 'x':
      self.cw_yproj = client_rmat[:, 0]
    else:
      self.cw_yproj = client_rmat[:, 1]

    self.cw_zproj = np.cross(self.cw_xproj, self.cw_yproj)

    # Note: the code below is greyed out in Visual Studio, seemingly due to the line above setting self.cw_yproj.

    self.reset_rw()

    self.do_jitter = False
    self.num_rw = 4
    self.rw_locs = np.array([[0.6309, 0., 0.], \
                             [0., 0.6309, 0.], \
                             [-0.6309, 0., 0.], \
                             [0., -0.6309, 0.]])

    # The above rw placement makes the tip of the pyramid point along the body z axis.
    # We actually want it to point perpendicular to the orbital plane
    if self.cw_orbit_dir == 'x':
      self.rw_locs = self.rw_locs@pin.exp3(np.array([-np.pi/2, 0., 0.])).transpose()
    else:
      self.rw_locs = self.rw_locs@pin.exp3(np.array([0., np.pi/2, 0.])).transpose()

    static_imbalance = 0.48/(1000*100) # 0.48 g cm converted to kg m
    dynamic_imbalance = 15.4/(1000*100*100) # 15.4 g cm^2 converted to kg m^2
    self.rw_masses = 12*np.ones(self.num_rw)
    self.rw_inertias = np.array([np.array([[0.0796, 0., 2.0e-4], \
                                           [0., 0.0430, 0.], \
                                           [2.0e-4, 0., 0.0430]]) for rw_idx in range(self.num_rw)])
    self.rw_inertias *= dynamic_imbalance/np.linalg.norm(self.rw_inertias[0, :2, 2])
    self.rw_com_offsets = static_imbalance/self.rw_masses

    self.rw_rmats = []
    for rw_idx in range(self.num_rw):
      rmat = pin.exp3(np.array([0., 0., rw_idx*np.pi/2]))@pin.exp3(np.array([0., -35.26*np.pi/180, 0.]))
      self.rw_rmats.append(rmat)
    self.rw_rmats = np.array(self.rw_rmats)

    if self.cw_orbit_dir == 'x':
      self.rw_rmats = np.array([pin.exp3(np.array([-np.pi/2, 0., 0.]))@rmat for rmat in self.rw_rmats])
    else:
      self.rw_rmats = np.array([pin.exp3(np.array([0., np.pi/2, 0.]))@rmat for rmat in self.rw_rmats])

    # (angle, angular velocity)
    self.rw_states = np.array([[0., 628.3], \
                               [0., 628.3], \
                               [0., 628.3], \
                               [0., 628.3]])

    self.solar_panel_angles = np.zeros(2)
    self.solar_panel_vels = np.zeros(2)

    self.compliance_config = np.zeros(3)
    self.compliance_vel = np.zeros(3)

    self.rng = rng

    self.sw_is_in_throat_trj = []
    self.sw_is_in_nozzle_trj = []
    self.sw_peg_pos_trj = []
    self.sw_peg_rmat_trj = []
    self.sw_peg_twist_trj = []

    self.sw_nozzle_pos_trj = []
    self.sw_nozzle_rmat_trj = []
    self.sw_nozzle_twist_trj = []

    self.mrv_peg_force_trj = []
    self.mrv_peg_torque_trj = []

    self.sw_base_pos_trj = []
    self.sw_base_rmat_trj = []
    self.sw_joint_angles_trj = []
    self.sw_base_v_trj = []
    self.sw_base_w_trj = []
    self.sw_joint_vels_trj = []
    self.sw_joint_cmd_trj = []

    self.sw_client_pos_trj = []
    self.sw_client_rmat_trj = []
    self.sw_client_v_trj = []
    self.sw_client_w_trj = []

    self.sw_x_trj = []

    self.sw_x_est_trj = []

    self.solar_panel_angles_trj = []
    self.solar_panel_vels_trj = []

    self.kinetic_energy_trj = []
    self.joint_work_trj = []

    self.sim_ts = []

    self.num_geoms_ncp = 8
    self.ncp_normal_forces = np.zeros(self.num_geoms_ncp)
    self.ncp_penetration_slack = np.zeros(self.num_geoms_ncp)
    self.nozzle_geom_fids = [self.pin_model.getFrameId('nozzle_geom' + str(i)) for i in range(self.num_geoms_ncp)]

    self.joint_work = 0

    self.update_kinematics()
    self.initial_nozzle_pos = self.get_nozzle_pos()

  def get_state_in_pieces(self):
    sw_base_pos = self.get_base_pos()
    sw_base_rmat = R.from_quat(self.x[self.mrv_qidx + 3:self.mrv_qidx + 7]).as_matrix()
    sw_joint_angles = self.get_joint_angles()
    sw_base_v = np.copy(self.x[self.pin_model.nq + self.mrv_vidx:self.pin_model.nq + self.mrv_vidx + 3])
    sw_base_w = np.copy(self.x[self.pin_model.nq + self.mrv_vidx + 3:self.pin_model.nq + self.mrv_vidx + 6])
    sw_joint_vels = np.copy(self.x[self.pin_model.nq + self.mrv_vidx + 6:self.pin_model.nq + self.mrv_vidx + 6 + self.num_rotary])

    sw_client_pos = np.copy(self.x[self.cv_qidx:self.cv_qidx + 3])
    sw_client_rmat = R.from_quat(self.x[self.cv_qidx + 3:self.cv_qidx + 7]).as_matrix()
    sw_client_v = self.get_client_linear_vel()
    sw_client_w = self.get_client_angular_vel()

    return sw_base_pos, sw_base_rmat, sw_joint_angles, sw_base_v, sw_base_w, sw_joint_vels, sw_client_pos, sw_client_rmat, sw_client_v, sw_client_w
  
  def get_client_pos(self):
    return np.copy(self.x[self.cv_qidx:self.cv_qidx + 3])
  
  def get_joint_angles(self):
    return np.copy(self.x[self.mrv_qidx + 7:self.mrv_qidx + 7 + self.num_rotary])

  def get_base_pos(self):
    return np.copy(self.x[self.mrv_qidx:self.mrv_qidx + 3])

  def enable_mujoco(self, xml_file):
    mj_path = mujoco_py.utils.discover_mujoco()
    self.mjmodel = mujoco_py.load_model_from_path(xml_file)
    self.mjmodel.opt.timestep = self.dt
    self.mjsim = mujoco_py.MjSim(self.mjmodel)
    self.mj_peg_geom_id = self.mjmodel.geom_name2id('ee_peg')

  def reset_rw(self):
    self.rw_counter = 0
    self.rw_tau = np.zeros(3)

  def update_gt_state_trj(self):
    sw_peg_pos, sw_peg_rmat, sw_peg_twist, sw_nozzle_pos, sw_nozzle_rmat, sw_nozzle_twist = self.get_peg_and_nozzle_info()

    self.sw_is_in_throat_trj.append(self.is_peg_in_throat())
    self.sw_is_in_nozzle_trj.append(self.is_peg_past_nozzle_opening())
    self.sw_peg_pos_trj.append(sw_peg_pos)
    self.sw_peg_rmat_trj.append(sw_peg_rmat)
    self.sw_peg_twist_trj.append(sw_peg_twist)

    self.sw_nozzle_pos_trj.append(sw_nozzle_pos)
    self.sw_nozzle_rmat_trj.append(sw_nozzle_rmat)
    self.sw_nozzle_twist_trj.append(sw_nozzle_twist)

    sw_base_pos, sw_base_rmat, sw_joint_angles, sw_base_v, sw_base_w, sw_joint_vels, sw_client_pos, sw_client_rmat, sw_client_v, sw_client_w = self.get_state_in_pieces()
    #sw_base_ang = scipy.linalg.logm(sw_base_rmat) 

    self.sw_base_pos_trj.append(sw_base_pos)
    self.sw_base_rmat_trj.append(sw_base_rmat)
    self.sw_joint_angles_trj.append(sw_joint_angles)
    self.sw_base_v_trj.append(sw_base_v)
    self.sw_base_w_trj.append(sw_base_w)
    self.sw_joint_vels_trj.append(sw_joint_vels)

    self.sw_client_pos_trj.append(sw_client_pos)
    self.sw_client_rmat_trj.append(sw_client_rmat)
    self.sw_client_v_trj.append(sw_client_v)
    self.sw_client_w_trj.append(sw_client_w)

    self.sw_x_trj.append(np.copy(self.x))

    self.kinetic_energy_trj.append(self.kinetic_energy())
    self.joint_work_trj.append(self.joint_work)

    self.sim_ts.append(self.sim_time)

  def update_wrench_trj(self, wrench_peg_peg):
    self.mrv_peg_force_trj.append(np.copy(wrench_peg_peg[:3]))
    self.mrv_peg_torque_trj.append(np.copy(wrench_peg_peg[3:]))

  def unforced_eulers_eqns_for_client(self, use_est):
    if use_est:
      client_w = self.x_est[self.pin_model.nq + self.cv_vidx + 3:self.pin_model.nq + self.cv_vidx + 6]
    else:
      client_w = self.x[self.pin_model.nq + self.cv_vidx + 3:self.pin_model.nq + self.cv_vidx + 6]
    return -self.client_inertia_inv@(np.cross(client_w, self.client_inertia@client_w))
  
  def unforced_eulers_eqns_for_mrv(self, use_est):
    if use_est:
      mrv_w = self.x_est[self.pin_model.nq + self.mrv_vidx + 3:self.pin_model.nq + self.mrv_vidx + 6]
    else:
      mrv_w = self.x[self.pin_model.nq + self.mrv_vidx + 3:self.pin_model.nq + self.mrv_vidx + 6]
    return -self.mrv_inertia_inv@(np.cross(mrv_w, self.mrv_inertia@mrv_w))

  def compute_cw_force(self, use_est):
    # Get CW states from sim state
    if use_est:
      mrv_q = self.x_est[self.mrv_qidx:self.mrv_qidx + self.mrv_pin_model.nq]
      mrv_v = self.x_est[self.pin_model.nq + self.mrv_vidx:self.pin_model.nq + self.mrv_vidx + self.mrv_pin_model.nv]
    else:
      mrv_q = self.x[self.mrv_qidx:self.mrv_qidx + self.mrv_pin_model.nq]
      mrv_v = self.x[self.pin_model.nq + self.mrv_vidx:self.pin_model.nq + self.mrv_vidx + self.mrv_pin_model.nv]

    mrv_x = np.concatenate((mrv_q, mrv_v))

    com = pin.centerOfMass(self.mrv_pin_model, self.mrv_pin_data, mrv_q)
    Jcom = pin.jacobianCenterOfMass(self.mrv_pin_model, self.mrv_pin_data, mrv_q)
    vcom = Jcom@mrv_v

    if use_est:
      client_pos = self.x_est[self.cv_qidx:self.cv_qidx + 3]
      client_quat = self.x_est[self.cv_qidx + 3:self.cv_qidx + 7]
      client_v = self.x_est[self.pin_model.nq + self.cv_vidx:self.pin_model.nq + self.cv_vidx + 3]
    else:
      client_pos = self.x[self.cv_qidx:self.cv_qidx + 3]
      client_quat = self.x[self.cv_qidx + 3:self.cv_qidx + 7]
      client_v = self.x[self.pin_model.nq + self.cv_vidx:self.pin_model.nq + self.cv_vidx + 3]

    separation = com - client_pos
    cw_x = self.cw_xproj@separation 
    cw_y = self.cw_yproj@separation
    cw_z = self.cw_zproj@separation

    vdiff = vcom - R.from_quat(client_quat).apply(client_v)
    cw_xdot = self.cw_xproj@vdiff
    cw_ydot = self.cw_yproj@vdiff
    cw_zdot = self.cw_zproj@vdiff

    # Acceleration of MRV com relative to client com
    cw_xddot = 3*self.cw_n**2*cw_x + 2*self.cw_n*cw_ydot
    cw_yddot = -2*self.cw_n*cw_xdot
    cw_zddot = -self.cw_n**2*cw_z
    
    # Compute fictitious forces on the client vehicle arising from CW eqns
    cw_fx = -self.client_mass*cw_xddot
    cw_fy = -self.client_mass*cw_yddot
    cw_fz = -self.client_mass*cw_zddot

    cw_force = cw_fx*self.cw_xproj + cw_fy*self.cw_yproj + cw_fz*self.cw_zproj

    return cw_force

  def compute_jitter_force_and_torque(self):
    force = np.zeros(3)
    torque = np.zeros(3)
    for rw_idx in range(self.num_rw):
      # Calculate CoM position in wheel frame
      com_pos = np.array([self.rw_com_offsets[rw_idx]*np.cos(self.rw_states[rw_idx, 0]), \
                          self.rw_com_offsets[rw_idx]*np.sin(self.rw_states[rw_idx, 0]), \
                          0.])
      static_imbalance_force = -self.rw_masses[rw_idx]*self.rw_states[rw_idx, 1]**2*com_pos
      static_imbalance_force = self.rw_rmats[rw_idx]@static_imbalance_force
      force += static_imbalance_force
      torque += np.cross(self.rw_locs[rw_idx], static_imbalance_force)
      J23 = self.rw_inertias[rw_idx][1, 2]
      J13 = self.rw_inertias[rw_idx][0, 2]
      dynamic_imbalance_torque = self.rw_states[rw_idx, 1]**2*np.array([-J23, J13, 0.])
      dynamic_imbalance_torque = self.rw_rmats[rw_idx]@dynamic_imbalance_torque
      torque += dynamic_imbalance_torque

    return force, torque
  
  def set_pybullet_joint_velocity(self,joint_cmd):
    '''Control is a 6x1 array of control inputs, which could be torques or velocities depending on what MRVClientSim.joint_control_mode is set to.'''
    self.pb_client.setJointMotorControlArray(self.pb_mrv_id, self.pb_joint_indices, pybullet.VELOCITY_CONTROL, targetVelocities=joint_cmd, forces=self.joint_torque_limits)

  def set_pybullet_joint_torque(self,joint_torques): 
    self.pb_client.setJointMotorControlArray(self.pb_mrv_id, self.pb_joint_indices, pybullet.TORQUE_CONTROL, forces=joint_torques)

  def kinetic_energy(self): 
    '''Assumes that update_kinematics has been called'''
    return pin.computeKineticEnergy(self.pin_model,self.pin_data)

  def get_pybullet_torques(self):
    torques = []
    for i in self.pb_joint_indices:
       _,_,_,torque  = self.pb_client.getJointState(self.pb_mrv_id, i)
       torques.append(torque)
    return np.array(torques)
  
  def get_pybullet_joint_velocities(self):
    velocities = []
    for i in self.pb_joint_indices:
       _,vel,_,_  = self.pb_client.getJointState(self.pb_mrv_id, i)
       velocities.append(vel)
    return np.array(velocities)

  def step(self, joint_command, wrench_peg_peg):
    '''All future step methods should be called from here.'''
    self.step_pybullet(joint_command, wrench_peg_peg)
    self.sim_time = self.sim_time + self.dt

  def step_pybullet(self, joint_command, wrench_peg_peg):
    self.update_gt_state_trj()
    self.sw_joint_cmd_trj.append(np.copy(joint_command))

    if joint_command is not None:
      self.set_pybullet_joint_velocity(joint_command)

    if self.use_cw:
      cw_force = self.compute_cw_force(False)
      client_quat = self.x[self.cv_qidx + 3:self.cv_qidx + 7]
      
      #CW forces applied only to client because we calcualte the relative acceleration between the client and the MRV
      self.pb_client.applyExternalForce(self.pb_cv_id, -1, list(R.from_quat(client_quat).inv().apply(cw_force)), [0., 0., 0.], pybullet.LINK_FRAME)

    # DEBUGGING Create motion in the client in a particular direction 
    #self.pb_client.applyExternalForce(self.pb_cv_id, -1, list(R.from_quat(self.x[self.cv_qidx + 3:self.cv_qidx + 7]).inv().apply([-1,0,0])), [0., 0., 0.], pybullet.LINK_FRAME)

    if self.lock_client:
      self.set_client_pose(self.initial_client_pos, self.initial_nozzle_q)
    if self.lock_mrv:
      self.set_mrv_pose(self.initial_mrv_pos, self.initial_mrv_quat)

    if wrench_peg_peg is not None:

      ''' Get the force in world frame 
      Be warned, that pin_data has not yet been updated, so we're using the result from the last time step.'''
      peg_rmat = self.pin_data.oMf[self.peg_fid].rotation.transpose()
      force_peg_world = peg_rmat@wrench_peg_peg[:3]

      #Forces are applied at the tip. We express the contact location in world frame
      p_tip_world_world = self.pin_data.oMf[self.peg_fid].translation
      self.pb_client.applyExternalForce(self.pb_mrv_id, self.pb_peg_id, list(force_peg_world), p_tip_world_world, pybullet.WORLD_FRAME)
      self.pb_client.applyExternalForce(self.pb_cv_id, self.pb_nozzle_id, list(np.multiply(-1,force_peg_world)), p_tip_world_world, pybullet.WORLD_FRAME)

    prev_x = np.copy(self.x)

    if self.simulate_angular_velocity_disturbance:
        tau = self.K_ang_vel_dist@(self.angular_velocity_disturbance - self.get_client_angular_vel())
        self.pb_client.applyExternalTorque(self.pb_cv_id, -1, list(tau), pybullet.WORLD_FRAME)

    self.pb_client.stepSimulation()

    mrv_pos, mrv_quat = self.pb_client.getBasePositionAndOrientation(self.pb_mrv_id)
    mrv_R = R.from_quat(mrv_quat)
    mrv_v, mrv_w = self.pb_client.getBaseVelocity(self.pb_mrv_id)
    mrv_v = mrv_R.inv().apply(mrv_v)
    mrv_w = mrv_R.inv().apply(mrv_w)
    cv_pos, cv_quat = self.pb_client.getBasePositionAndOrientation(self.pb_cv_id)
    cv_R = R.from_quat(cv_quat)
    cv_v, cv_w = self.pb_client.getBaseVelocity(self.pb_cv_id)
    cv_v = cv_R.inv().apply(cv_v)
    cv_w = cv_R.inv().apply(cv_w)
    joint_states = self.pb_client.getJointStates(self.pb_mrv_id, self.pb_joint_indices)

    self.x[self.cv_qidx:self.cv_qidx + 3] = np.copy(cv_pos)
    self.x[self.cv_qidx + 3:self.cv_qidx + 7] = np.copy(cv_quat)
    self.x[self.mrv_qidx:self.mrv_qidx + 3] = np.copy(mrv_pos)
    self.x[self.mrv_qidx + 3:self.mrv_qidx + 7] = np.copy(mrv_quat)
    self.x[self.mrv_qidx + 7:self.mrv_qidx + self.mrv_pin_model.nq] = np.array([state[0] for state in joint_states])

    self.x[self.pin_model.nq + self.cv_vidx:self.pin_model.nq + self.cv_vidx + 3] = np.copy(cv_v)
    self.x[self.pin_model.nq + self.cv_vidx + 3:self.pin_model.nq + self.cv_vidx + 6] = np.copy(cv_w)

    self.x[self.pin_model.nq + self.mrv_vidx:self.pin_model.nq + self.mrv_vidx + 3] = np.copy(mrv_v)
    self.x[self.pin_model.nq + self.mrv_vidx + 3:self.pin_model.nq + self.mrv_vidx + 6] = np.copy(mrv_w)
    self.x[self.pin_model.nq + self.mrv_vidx + 6:self.pin_model.nq + self.mrv_vidx + self.mrv_pin_model.nv] = np.array([state[1] for state in joint_states])

    self.update_kinematics()

    pin.forwardKinematics(self.pin_model, self.prev_pin_data, prev_x[:self.pin_model.nq], prev_x[self.pin_model.nq:])
    pin.updateFramePlacement(self.pin_model, self.prev_pin_data, self.peg_fid)

    if wrench_peg_peg is None:#if we are running pure sim with contact simulation, simulate and store the contact force

      # For some reason, switching the order of pb_cv_id and pb_mrv_id made getJointInfo not return an error
      contact_points = self.pb_client.getContactPoints(self.pb_mrv_id, self.pb_cv_id)
      
      net_contact_force = np.zeros(3)
      net_contact_torque = np.zeros(3)

      self.nozzle_geom_idx = -1
      for contact_point in contact_points:
        normal = np.array(contact_point[7])
        distance = contact_point[8]
        normal_force = np.array(contact_point[9])
        friction_force1 = contact_point[10]
        friction_dir1 = np.array(contact_point[11])
        friction_force2 = contact_point[12]
        friction_dir2 = np.array(contact_point[13])
        contact_force = normal_force*normal + friction_force1*friction_dir1 + friction_force2*friction_dir2
        net_contact_force += contact_force
        contact_pos = np.array(contact_point[5])
        self.contact_pos_wrt_peg = self.pin_data.oMf[self.peg_fid].actInv(contact_pos)
        r_contact = contact_pos - self.pin_data.oMf[self.peg_fid].translation
        net_contact_torque += np.cross(r_contact, contact_force)
        joint_info = self.pb_client.getJointInfo(contact_point[2], contact_point[4])
        if normal_force != 0:
          joint_info = self.pb_client.getJointInfo(contact_point[2], contact_point[4])
          '''
          print('Distance', distance)
          print(joint_info)
          print('Normal vec and force in C frame', repr(cv_R.inv().apply(normal)), normal_force)
          print('Contact pos wrt peg', repr(self.contact_pos_wrt_peg))
          adj = np.eye(3, 6)
          peg_pos = self.prev_pin_data.oMf[self.peg_fid].translation
          adj[:, 3:] = -pin.skew(contact_pos - peg_pos)
          contact_pt_vel_peg = adj@pin.getFrameVelocity(self.pin_model, self.prev_pin_data, self.peg_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).vector
          prev_client_pos = prev_x[self.cv_qidx:self.cv_qidx + 3]
          prev_client_v = prev_x[self.pin_model.nq + self.cv_vidx:self.pin_model.nq + self.cv_vidx + 3]
          prev_client_w = prev_x[self.pin_model.nq + self.cv_vidx + 3:self.pin_model.nq + self.cv_vidx + 6]
          prev_client_rmat = R.from_quat(prev_x[self.cv_qidx + 3:self.cv_qidx + 7]).as_matrix()
          adj[:, 3:] = -pin.skew(contact_pos - prev_client_pos)@prev_client_rmat
          adj[:, :3] = prev_client_rmat
          contact_pt_vel_client = adj@np.concatenate((prev_client_v, prev_client_w))

          normal_vel = normal@(contact_pt_vel_peg - contact_pt_vel_client)
          # normal_vel -= normal@np.outer(prev_client_rmat[:, 2], prev_client_rmat[:, 2])@(contact_pt_vel_peg - contact_pt_vel_client)
          print('Normal vel', normal_vel)
          print(repr(self.pin_data.oMf[self.hole_fid].actInv(self.pin_data.oMf[self.peg_fid].translation)))
          for i, fid in enumerate(self.tube_geom_fids):
            pin.updateFramePlacement(self.pin_model, self.pin_data, fid)
            print('Contact pos wrt tube' + str(i), self.pin_data.oMf[fid].actInv(contact_pos))
          '''
          joint_info_str = joint_info[1].decode('UTF-8')
          if 'nozzle_geom' in joint_info_str:
            self.nozzle_geom_idx = int(joint_info_str[-1])

      # Transform force and moment from a frame at the wrist, aligned with the world, to the actual wrist frame
      peg_force = self.pin_data.oMf[self.peg_fid].rotation.transpose()@net_contact_force
      peg_torque = self.pin_data.oMf[self.peg_fid].rotation.transpose()@net_contact_torque

      self.pb_peg_wrench = np.zeros(6)
      self.pb_peg_wrench[:3] = peg_force
      self.pb_peg_wrench[3:] = peg_torque
      self.update_wrench_trj(self.pb_peg_wrench)
    else: 
      self.update_wrench_trj(wrench_peg_peg) 

    joint_torques = self.get_pybullet_torques()
    joint_velocities = self.get_pybullet_joint_velocities()
    self.joint_work = self.joint_work + joint_torques.transpose()@joint_velocities*self.dt

  def update_kinematics(self):
    pin.forwardKinematics(self.pin_model, self.pin_data, self.x[:self.pin_model.nq], self.x[self.pin_model.nq:])
    pin.computeJointJacobians(self.pin_model, self.pin_data)
    pin.updateFramePlacement(self.pin_model, self.pin_data, self.peg_fid)
    pin.updateFramePlacement(self.pin_model, self.pin_data, self.wrist_fid)
    pin.updateFramePlacement(self.pin_model, self.pin_data, self.nozzle_fid)
    pin.updateFramePlacement(self.pin_model, self.pin_data, self.hole_fid)
    pin.updateFramePlacement(self.pin_model, self.pin_data, self.goal_fid)

  def update_kinematics_est(self):
    pin.forwardKinematics(self.pin_model, self.pin_data_est, self.x_est[:self.pin_model.nq], self.x_est[self.pin_model.nq:])
    pin.computeJointJacobians(self.pin_model, self.pin_data_est)
    pin.updateFramePlacement(self.pin_model, self.pin_data_est, self.peg_fid)
    pin.updateFramePlacement(self.pin_model, self.pin_data_est, self.wrist_fid)
    pin.updateFramePlacement(self.pin_model, self.pin_data_est, self.nozzle_fid)
    pin.updateFramePlacement(self.pin_model, self.pin_data_est, self.hole_fid)
    pin.updateFramePlacement(self.pin_model, self.pin_data_est, self.goal_fid)

  def update_state_estimate(self, wrench_peg_peg, dt=None):

    if self.do_noisy_state_estimation:
      self.sw_x_est_trj.append(self.x_est)
      if dt is None:
        dt = self.dt
      self.x_est = np.copy(self.x)

      base_Rinv = R.from_quat(self.x[self.mrv_qidx + 3:self.mrv_qidx + 7]).inv()
      f_client_in_base_frame = -base_Rinv.apply(self.pin_data.oMf[self.peg_fid].rotation@wrench_peg_peg[:3])
      B_p_EC = base_Rinv.apply(self.pin_data.oMf[self.peg_fid].translation - self.x[self.cv_qidx:self.cv_qidx + 3])
      tau_client_in_base_frame = np.cross(B_p_EC, f_client_in_base_frame)

      cam_rmat = R.from_quat(self.x[self.mrv_qidx + 3:self.mrv_qidx + 7]).as_matrix()
      cam_pos = self.x[self.mrv_qidx:self.mrv_qidx + 3] + cam_rmat@self.t_cam_base
      cam_w = self.x[self.pin_model.nq + self.mrv_vidx + 3:self.pin_model.nq + self.mrv_vidx + 6]
      cam_v = self.x[self.pin_model.nq + self.mrv_vidx:self.pin_model.nq + self.mrv_vidx + 3] + np.cross(cam_w, self.t_cam_base)

      com = pin.centerOfMass(self.mrv_pin_model, self.mrv_pin_data, self.x[self.mrv_qidx:self.mrv_qidx + self.mrv_pin_model.nq], self.x[self.pin_model.nq + self.mrv_vidx:self.pin_model.nq + self.mrv_vidx + self.mrv_pin_model.nv])
      vcom = self.mrv_pin_data.vcom[0]
      com_wrt_cam = cam_rmat.transpose()@(com - cam_pos)
      vcom_wrt_cam = cam_rmat.transpose()@vcom

      # Predict
      if self.ekf.initialized:
        # Estimation wrt inertial frame
        #self.ekf.predict(np.zeros(3), np.zeros(3), np.zeros(3), tau_client_in_base_frame, dt)

        # Estimation wrt camera frame
        f_client_in_base_frame = np.zeros(3)
        tau_client_in_base_frame = np.zeros(3)
        self.ekf.predict(cam_v - vcom_wrt_cam, cam_w, f_client_in_base_frame, tau_client_in_base_frame, dt)

      # Generate noisy measurement
      client_pos = self.x[self.cv_qidx:self.cv_qidx + 3]
      client_rmat = R.from_quat(self.x[self.cv_qidx + 3:self.cv_qidx + 7]).as_matrix()
      measurement_succeeded = True

      client_pos_wrt_cam = cam_rmat.transpose()@(client_pos - cam_pos)
      client_rmat_wrt_cam = cam_rmat.transpose()@client_rmat

      pos_std = np.array([self.pose_noise_pos_std, self.pose_noise_pos_std, self.pose_noise_pos_std])
      rot_std = np.array([self.pose_noise_rot_std,self.pose_noise_rot_std,self.pose_noise_rot_std])
      client_pos_meas_wrt_cam = client_pos_wrt_cam + self.rng.normal(np.zeros(3), pos_std)
      client_rmat_meas_wrt_cam = client_rmat_wrt_cam@R.from_euler('ZYX', self.rng.normal(np.zeros(3), rot_std)).as_matrix()

      if measurement_succeeded:
        # Estimation wrt inertial frame
        '''
        client_pos_meas = cam_pos + cam_rmat@client_pos_meas_wrt_cam
        client_rmat_meas = cam_rmat@client_rmat_meas_wrt_cam

        if self.ekf.initialized:
          if self.time_steps_since_measurement >= self.time_steps_between_measurements:
            self.ekf.correct(client_pos_meas, R.from_matrix(client_rmat_meas).as_quat())
            self.time_steps_since_measurement = 0
        else:
          self.ekf.initialize(client_pos_meas, R.from_matrix(client_rmat_meas).as_quat(), np.zeros(3), np.zeros(3))

        self.time_steps_since_measurement += 1

        client_pos_est = self.ekf.kf_x[:3]
        client_rmat_est = R.from_quat(self.ekf.kf_x[3:7]).as_matrix()

        client_v_est = client_rmat_est.transpose()@self.ekf.kf_x[7:10]
        client_w_est = client_rmat_est.transpose()@self.ekf.kf_x[10:13]

        self.x_est[self.cv_qidx:self.cv_qidx + 3] = client_pos_est
        self.x_est[self.cv_qidx + 3:self.cv_qidx + 7] = R.from_matrix(client_rmat_est).as_quat()
        self.x_est[self.pin_model.nq + self.cv_vidx:self.pin_model.nq + self.cv_vidx + 3] = client_v_est
        self.x_est[self.pin_model.nq + self.cv_vidx + 3:self.pin_model.nq + self.cv_vidx + 6] = client_w_est
        '''
               
        # Estimation wrt camera frame
        client_pos_meas = cam_pos + cam_rmat@client_pos_meas_wrt_cam
        client_rmat_meas = cam_rmat@client_rmat_meas_wrt_cam

        client_pos_meas_wrt_com = client_pos_meas_wrt_cam - com_wrt_cam

        if self.ekf.initialized:
          if self.time_steps_since_measurement >= self.time_steps_between_measurements:
            self.ekf.correct(client_pos_meas_wrt_com, R.from_matrix(client_rmat_meas_wrt_cam).as_quat())
            self.time_steps_since_measurement = 0
        else:
          self.ekf.initialize(client_pos_meas_wrt_com, R.from_matrix(client_rmat_meas_wrt_cam).as_quat(), np.zeros(3), np.zeros(3))

        self.time_steps_since_measurement += 1

        client_pos_est_wrt_com = self.ekf.kf_x[:3]
        client_pos_est = cam_pos + cam_rmat@(client_pos_est_wrt_com + com_wrt_cam)
        client_rmat_est_wrt_cam = R.from_quat(self.ekf.kf_x[3:7]).as_matrix()
        client_rmat_est = cam_rmat@client_rmat_est_wrt_cam

        client_v_est = client_rmat_est_wrt_cam.transpose()@(self.ekf.kf_x[7:10] + vcom_wrt_cam)
        client_w_est = client_rmat_est_wrt_cam.transpose()@self.ekf.kf_x[10:13]

        # Updated state
        self.x_est[self.cv_qidx:self.cv_qidx + 3] = client_pos_est
        self.x_est[self.cv_qidx + 3:self.cv_qidx + 7] = R.from_matrix(client_rmat_est).as_quat()
        self.x_est[self.pin_model.nq + self.cv_vidx:self.pin_model.nq + self.cv_vidx + 3] = client_v_est
        self.x_est[self.pin_model.nq + self.cv_vidx + 3:self.pin_model.nq + self.cv_vidx + 6] = client_w_est

        # Collect data to see if the estimator is consistent
        _Bp_MC = client_pos_wrt_cam - com_wrt_cam
        _Bp_MC_est = client_pos_est_wrt_com

        R_BC = client_rmat_wrt_cam
        R_BC_est = client_rmat_est_wrt_cam

        _Bpdot_MC = cam_rmat.transpose()@(client_rmat@self.x[self.pin_model.nq + self.cv_vidx:self.pin_model.nq + self.cv_vidx + 3] - vcom)
        _Bpdot_MC_est = self.ekf.kf_x[7:10]

        _Bw_WC = cam_rmat.transpose()@(client_rmat@self.x[self.pin_model.nq + self.cv_vidx + 3:self.pin_model.nq + self.cv_vidx + 6])
        _Bw_WC_est = self.ekf.kf_x[10:13]

        est_err = np.concatenate((_Bp_MC - _Bp_MC_est, \
                                  pin.log3(R_BC.transpose()@R_BC_est), \
                                  _Bpdot_MC - _Bpdot_MC_est, \
                                  _Bw_WC - _Bw_WC_est))
        self.est_err_trj.append(est_err)
        self.cov_trj.append(np.copy(self.ekf.kf_cov))

    else: 
      self.x_est = np.copy(self.x)

    self.update_kinematics_est()

  def get_mrv_tip_pos(self):
    return copy.deepcopy(self.pin_data.oMf[self.peg_fid].translation)
  
  def get_mrv_tip_rmat(self): 
    return copy.deepcopy(self.pin_data.oMf[self.peg_fid].rotation)
  
  def get_mrv_wrist_pos(self): 
    return copy.deepcopy(self.pin_data.oMf[self.wrist_fid].translation)
  
  def get_mrv_wrist_rmat(self):
    return copy.deepcopy(self.pin_data.oMf[self.wrist_fid].rotation)
  
  def get_mrv_base_twist(self): 
    mrv_v_est = self.x_est[self.pin_model.nq + self.mrv_vidx:self.pin_model.nq + self.mrv_vidx + 3]
    mrv_w_est = self.x_est[self.pin_model.nq + self.mrv_vidx + 3:self.pin_model.nq + self.mrv_vidx + 6]

    mrv_vdot = np.zeros(3) # We do not apply any additional forces to the MRV. This does not account for contact forces
    mrv_wdot = self.unforced_eulers_eqns_for_mrv(self.do_noisy_state_estimation)

    return np.concatenate((mrv_v_est,mrv_w_est)), np.concatenate((mrv_vdot,mrv_wdot))
  
  def get_goal_pos(self):
    return np.copy(self.pin_data.oMf[self.goal_fid].translation)
  
  def get_dist_nozzle_opening_from_goal(self):
    nozzle_rmat = self.pin_data.oMf[self.nozzle_fid].rotation
    goal_pos = np.transpose(nozzle_rmat)@self.pin_data.oMf[self.goal_fid].translation
    nozzle_pos = np.transpose(nozzle_rmat)@self.pin_data.oMf[self.nozzle_fid].translation
    return np.abs(goal_pos[2] - nozzle_pos[2])
  
  def get_dist_throat_opening_to_goal(self):
    nozzle_rmat = self.pin_data.oMf[self.nozzle_fid].rotation
    throat_opening_pos = np.transpose(nozzle_rmat)@self.pin_data.oMf[self.hole_fid].translation
    goal_pos = np.transpose(nozzle_rmat)@self.pin_data.oMf[self.goal_fid].translation
    return np.abs(throat_opening_pos[2] - goal_pos[2])

  def is_peg_in_throat(self,depth_offset=0):
    ''' Returns True if peg is inside the throat of the nozzle, False otherwise 

    Inputs: 
    depth_offset - The function will continue to return False until the peg is inside the throat by a distance of depth_offset'''
    if self.do_noisy_state_estimation:
      sw_peg_pos, _, _, sw_nozzle_pos, sw_nozzle_rmat, _ = self.get_peg_and_nozzle_info()
    else:
      sw_peg_pos, _, _, sw_nozzle_pos, sw_nozzle_rmat, _ = self.get_peg_and_nozzle_info_est()

    sw_pos_goal = sw_nozzle_pos + sw_nozzle_rmat[:, 2]*self.get_dist_nozzle_opening_from_goal()
    sw_rel_pos = sw_nozzle_rmat.transpose()@(sw_peg_pos - sw_pos_goal)

    return np.abs(sw_rel_pos[2]) < (self.get_dist_throat_opening_to_goal() - depth_offset)
  
  def dist_to_throat_opening(self):
    ''' Returns the distance between the peg and the throat opening of the nozzle'''
    if self.do_noisy_state_estimation:
      sw_peg_pos, _, _, sw_nozzle_pos, sw_nozzle_rmat, _ = self.get_peg_and_nozzle_info()
    else:
      sw_peg_pos, _, _, sw_nozzle_pos, sw_nozzle_rmat, _ = self.get_peg_and_nozzle_info_est()

    sw_pos_goal = sw_nozzle_pos + sw_nozzle_rmat[:, 2]*self.get_dist_nozzle_opening_from_goal()
    sw_rel_pos = sw_nozzle_rmat.transpose()@(sw_peg_pos - sw_pos_goal)

    # print('Distance to throat opening', np.abs(sw_rel_pos[2]))
    return np.abs(sw_rel_pos[2])
  
  def is_peg_past_nozzle_opening(self,depth_offset = 0):
    ''' Returns True if peg is inside the nozzle, False otherwise 

    Inputs: 
    depth_offset - The function will continue to return False until the peg is inside the nozzle by a distance of depth_offset'''

    if self.do_noisy_state_estimation:
      sw_peg_pos, _, _, sw_nozzle_pos, sw_nozzle_rmat, _ = self.get_peg_and_nozzle_info()
    else:
      sw_peg_pos, _, _, sw_nozzle_pos, sw_nozzle_rmat, _ = self.get_peg_and_nozzle_info_est()

    sw_rel_pos = sw_nozzle_rmat.transpose()@(sw_peg_pos - sw_nozzle_pos)

    return sw_rel_pos[2] > depth_offset
  
  def get_nozzle_pos(self):
    return np.copy(self.pin_data.oMf[self.nozzle_fid].translation)

  def get_peg_and_nozzle_info(self):
    # Get peg and nozzle kinematics expressed in world frame
    vtip = pin.getFrameVelocity(self.pin_model, self.pin_data, self.peg_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).vector
    vnozzle = pin.getFrameVelocity(self.pin_model, self.pin_data, self.nozzle_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).vector

    return np.copy(self.pin_data.oMf[self.peg_fid].translation), \
           np.copy(self.pin_data.oMf[self.peg_fid].rotation), \
           np.copy(vtip), \
           self.get_nozzle_pos(), \
           np.copy(self.pin_data.oMf[self.nozzle_fid].rotation), \
           np.copy(vnozzle)
  
  def get_client_angular_vel(self):
    return np.copy(self.x[self.pin_model.nq + self.cv_vidx + 3:self.pin_model.nq + self.cv_vidx + 6])
  
  def get_client_linear_vel(self):
    return np.copy(self.x[self.pin_model.nq + self.cv_vidx:self.pin_model.nq + self.cv_vidx + 3])

  def get_peg_and_nozzle_info_est(self):
    vtip_est = pin.getFrameVelocity(self.pin_model, self.pin_data_est, self.peg_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).vector
    vnozzle_est = pin.getFrameVelocity(self.pin_model, self.pin_data_est, self.nozzle_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).vector

    return self.pin_data_est.oMf[self.peg_fid].translation, \
           self.pin_data_est.oMf[self.peg_fid].rotation, \
           vtip_est, \
           self.pin_data_est.oMf[self.nozzle_fid].translation, \
           self.pin_data_est.oMf[self.nozzle_fid].rotation, \
           vnozzle_est
    
  # step or update_kinematics should've been called first
  def check_if_client_is_out_of_reach(self):
    hole_pos = self.pin_data.oMf[self.hole_fid].translation
    mrv_q = self.x[self.mrv_qidx:self.mrv_qidx + self.mrv_pin_model.nq]
    com = pin.centerOfMass(self.mrv_pin_model, self.mrv_pin_data, mrv_q)
    dist_hole_com = np.linalg.norm(hole_pos - com)
    return dist_hole_com > self.reach

  def modify_x(self, peg_pos, peg_rmat):
    rotation_correction = R.from_matrix(peg_rmat@self.pin_data.oMf[self.peg_fid].rotation.transpose())
    self.x[self.mrv_qidx + 3:self.mrv_qidx + 7] = (rotation_correction*R.from_quat(self.x[self.mrv_qidx + 3:self.mrv_qidx + 7])).as_quat()
    pin.forwardKinematics(self.pin_model, self.pin_data, self.x[:self.pin_model.nq], self.x[self.pin_model.nq:])
    pin.updateFramePlacement(self.pin_model, self.pin_data, self.peg_fid)
    position_correction = peg_pos - self.pin_data.oMf[self.peg_fid].translation
    self.x[self.mrv_qidx:self.mrv_qidx + 3] += position_correction

  def vel_feasible(self, twist_world):
    q = self.x[self.mrv_qidx:self.mrv_qidx + 7 + self.num_rotary]
    v = self.x[self.pin_model.nq + self.mrv_vidx:self.pin_model.nq + self.mrv_vidx + 6 + self.num_rotary]
    pin.forwardKinematics(self.mrv_pin_model, self.mrv_pin_data, q, v)
    pin.computeJointJacobians(self.mrv_pin_model, self.mrv_pin_data)
    pin.updateFramePlacement(self.mrv_pin_model, self.mrv_pin_data, self.mrv_peg_fid)

    J_ee_tip = pin.getFrameJacobian(self.mrv_pin_model, self.mrv_pin_data, self.mrv_peg_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    Ag = pin.computeCentroidalMap(self.mrv_pin_model, self.mrv_pin_data, q)

    '''
    # Transform to client frame
    client_pos = self.x[self.cv_qidx:self.cv_qidx + 3]
    client_rmat = R.from_quat(self.x[self.cv_qidx + 3:self.cv_qidx + 7]).as_matrix()
    client_w = self.x[self.pin_model.nq + self.cv_vidx + 3:self.pin_model.nq + self.cv_vidx + 6]
    client_v = self.x[self.pin_model.nq + self.cv_vidx:self.pin_model.nq + self.cv_vidx + 3]
    J_ee_tip[:3] = client_rmat.transpose()@J_ee_tip[:3]
    J_ee_tip[3:] = client_rmat.transpose()@J_ee_tip[3:]
    rel_vel_bias = np.concatenate((-client_v - np.cross(client_w, client_rmat.transpose()@(self.pin_data.oMf[self.peg_fid].translation - client_pos)), -client_w))
    '''

    '''
    Jdot_ee_tip[:3] = -pin.skew(client_w)@J_ee_tip[:3] + client_rmat.transpose()@Jdot_ee_tip[:3]
    Jdot_ee_tip[3:] = -pin.skew(client_w)@J_ee_tip[3:] + client_rmat.transpose()@Jdot_ee_tip[3:]
    '''

    J_ee_tip_gen = J_ee_tip[:, 6:] - J_ee_tip[:, :6]@np.linalg.solve(Ag[:, :6], Ag[:, 6:])
    mrv_h_0 = Ag@v
    schur_bias = J_ee_tip[:, :6]@np.linalg.solve(Ag[:, :6], mrv_h_0)

    ee_vel_vertices = self.joint_vel_vertices@J_ee_tip_gen.transpose() + schur_bias #+ rel_vel_bias

    return in_hull(ee_vel_vertices, twist_world)

  def acc_feasible(self, acc_world):
    q = self.x[self.mrv_qidx:self.mrv_qidx + 7 + self.num_rotary]
    v = self.x[self.pin_model.nq + self.mrv_vidx:self.pin_model.nq + self.mrv_vidx + 6 + self.num_rotary]
    pin.forwardKinematics(self.mrv_pin_model, self.mrv_pin_data, q, v)
    pin.computeJointJacobians(self.mrv_pin_model, self.mrv_pin_data)
    pin.computeJointJacobiansTimeVariation(self.mrv_pin_model, self.mrv_pin_data, q, v)
    pin.updateFramePlacement(self.mrv_pin_model, self.mrv_pin_data, self.mrv_peg_fid)

    J_ee_tip = pin.getFrameJacobian(self.mrv_pin_model, self.mrv_pin_data, self.mrv_peg_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    Ag = pin.computeCentroidalMap(self.mrv_pin_model, self.mrv_pin_data, q)
    Jdot_ee_tip = pin.getFrameJacobianTimeVariation(self.mrv_pin_model, self.mrv_pin_data, self.mrv_peg_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    Agdot = pin.computeCentroidalMapTimeVariation(self.mrv_pin_model, self.mrv_pin_data, q, v)

    J_ee_tip_gen = J_ee_tip[:, 6:] - J_ee_tip[:, :6]@np.linalg.solve(Ag[:, :6], Ag[:, 6:])

    schur_bias_acc = J_ee_tip[:, :6]@np.linalg.solve(Ag[:, :6], -Agdot@v) + Jdot_ee_tip@v

    ee_acc_vertices = self.joint_acc_vertices@J_ee_tip_gen.transpose() + schur_bias_acc

    return in_hull(ee_acc_vertices, acc_world)

  def get_constraint_vertices(self):
    q = self.x[self.mrv_qidx:self.mrv_qidx + 7 + self.num_rotary]
    v = self.x[self.pin_model.nq + self.mrv_vidx:self.pin_model.nq + self.mrv_vidx + 6 + self.num_rotary]
    pin.forwardKinematics(self.mrv_pin_model, self.mrv_pin_data, q, v)
    pin.computeJointJacobians(self.mrv_pin_model, self.mrv_pin_data)
    pin.computeJointJacobiansTimeVariation(self.mrv_pin_model, self.mrv_pin_data, q, v)
    pin.updateFramePlacement(self.mrv_pin_model, self.mrv_pin_data, self.mrv_peg_fid)

    J_ee_tip = pin.getFrameJacobian(self.mrv_pin_model, self.mrv_pin_data, self.mrv_peg_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    Ag = pin.computeCentroidalMap(self.mrv_pin_model, self.mrv_pin_data, q)
    Jdot_ee_tip = pin.getFrameJacobianTimeVariation(self.mrv_pin_model, self.mrv_pin_data, self.mrv_peg_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    Agdot = pin.computeCentroidalMapTimeVariation(self.mrv_pin_model, self.mrv_pin_data, q, v)

    J_ee_tip_gen = J_ee_tip[:, 6:] - J_ee_tip[:, :6]@np.linalg.solve(Ag[:, :6], Ag[:, 6:])
    mrv_h_0 = Ag@v
    schur_bias = J_ee_tip[:, :6]@np.linalg.solve(Ag[:, :6], mrv_h_0)

    ee_vel_vertices = self.joint_vel_vertices@J_ee_tip_gen.transpose() + schur_bias

    schur_bias_acc = J_ee_tip[:, :6]@np.linalg.solve(Ag[:, :6], -Agdot@v) + Jdot_ee_tip@v

    ee_acc_vertices = self.joint_acc_vertices@J_ee_tip_gen.transpose() + schur_bias_acc

    return ee_vel_vertices, ee_acc_vertices

  def get_reqs(self, des_twist_world):
    q = self.x[self.mrv_qidx:self.mrv_qidx + 7 + self.num_rotary]
    v = self.x[self.pin_model.nq + self.mrv_vidx:self.pin_model.nq + self.mrv_vidx + 6 + self.num_rotary]
    pin.forwardKinematics(self.mrv_pin_model, self.mrv_pin_data, q, v)
    pin.computeJointJacobians(self.mrv_pin_model, self.mrv_pin_data)
    pin.computeJointJacobiansTimeVariation(self.mrv_pin_model, self.mrv_pin_data, q, v)
    pin.updateFramePlacement(self.mrv_pin_model, self.mrv_pin_data, self.mrv_peg_fid)

    J_ee_tip = pin.getFrameJacobian(self.mrv_pin_model, self.mrv_pin_data, self.mrv_peg_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    Ag = pin.computeCentroidalMap(self.mrv_pin_model, self.mrv_pin_data, q)

    J_ee_tip_gen = J_ee_tip[:, 6:] - J_ee_tip[:, :6]@np.linalg.solve(Ag[:, :6], Ag[:, 6:])
    mrv_h_0 = Ag@v
    schur_bias = J_ee_tip[:, :6]@np.linalg.solve(Ag[:, :6], mrv_h_0)

    req_joint_vels = np.linalg.lstsq(J_ee_tip_gen, des_twist_world - schur_bias)[0]
    req_joint_accs = (req_joint_vels - v[6:])/self.dt

    return req_joint_vels, req_joint_accs

  def save(self, save_path):
    np.save(save_path + '/sw_is_in_throat_trj.npy', self.sw_is_in_throat_trj)
    np.save(save_path + '/sw_is_in_nozzle_trj.npy', self.sw_is_in_nozzle_trj)
    np.save(save_path + '/sw_peg_pos_trj.npy', self.sw_peg_pos_trj)
    np.save(save_path + '/sw_peg_rmat_trj.npy', self.sw_peg_rmat_trj)
    np.save(save_path + '/sw_peg_twist_trj.npy', self.sw_peg_twist_trj)

    np.save(save_path + '/sw_nozzle_pos_trj.npy', self.sw_nozzle_pos_trj)
    np.save(save_path + '/sw_nozzle_rmat_trj.npy', self.sw_nozzle_rmat_trj)
    np.save(save_path + '/sw_nozzle_twist_trj.npy', self.sw_nozzle_twist_trj)

    np.save(save_path + '/mrv_peg_force_trj.npy', self.mrv_peg_force_trj)

    np.save(save_path + '/mrv_peg_torque_trj.npy', self.mrv_peg_torque_trj)

    np.save(save_path + '/sw_base_pos_trj.npy', self.sw_base_pos_trj)
    np.save(save_path + '/sw_base_rmat_trj.npy', self.sw_base_rmat_trj)
    np.save(save_path + '/sw_joint_angles_trj.npy', self.sw_joint_angles_trj)
    np.save(save_path + '/sw_base_v_trj.npy', self.sw_base_v_trj)
    np.save(save_path + '/sw_base_w_trj.npy', self.sw_base_w_trj)
    np.save(save_path + '/sw_joint_vels_trj.npy', self.sw_joint_vels_trj)
    np.save(save_path + '/sw_joint_cmd_trj.npy', self.sw_joint_cmd_trj)

    np.save(save_path + '/sw_client_pos_trj.npy', self.sw_client_pos_trj)
    np.save(save_path + '/sw_client_rmat_trj.npy', self.sw_client_rmat_trj)
    np.save(save_path + '/sw_client_v_trj.npy', self.sw_client_v_trj)
    np.save(save_path + '/sw_client_w_trj.npy', self.sw_client_w_trj)

    np.save(save_path + '/sw_x_trj.npy', self.sw_x_trj)

    np.save(save_path + '/solar_panel_angles_trj.npy', self.solar_panel_angles_trj)
    np.save(save_path + '/solar_panel_vels_trj.npy', self.solar_panel_vels_trj)

    np.save(save_path + '/kinetic_energy_trj.npy', self.kinetic_energy_trj)
    np.save(save_path + '/joint_work_trj.npy', self.joint_work_trj)
    np.save(save_path + '/sim_ts.npy', self.sim_ts)
