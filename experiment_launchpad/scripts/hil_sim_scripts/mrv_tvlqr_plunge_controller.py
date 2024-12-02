import numpy as np
import pinocchio as pin
from mrv_client_sim import MRVClientSim
from scipy.spatial.transform import Rotation as R
import bisect
from actuation_cw_contact import ActuationModelCWContact
import time
from ref_traj_point import RefTrajPoint
from test_trajectories import TestTrajectories
import copy
import gc

# Controllers
from planning_scripts.tvlqr_controller import TVLQRController
from within_nozzle_admittance import WithinNozzleAdmittance

class TVLQR_PLUNGE_controller(object):
  def __init__(self, mrv_cv_urdf_file, mrv_urdf_file, pybullet_mrv_urdf_file, pybullet_cv_urdf_file, mrv_joint_angle_lower_limits, mrv_joint_angle_upper_limits, mrv_joint_vel_limits, 
               mrv_joint_acc_limits, mrv_joint_torque_limits, dt, cone_slope, clip_joint_commands,
              time_steps_between_measurements, cw_a, cw_mu, cw_orbit_dir, do_noisy_state_estimation, nozzle_opening_rad, 
               peg_rad, velocity_noise_ang_amp, time_limit, debug_with_test_traj, test_traj_id, lock_client, lock_mrv, probe_z_axis_plunge_velocity, use_variable_plunge_speed, 
               use_scheduled_gains, use_cw=True , use_ekf = True):  
    self.mrv_cv_urdf_file = mrv_cv_urdf_file
    self.mrv_urdf_file = mrv_urdf_file
    self.pybullet_mrv_urdf_file = pybullet_mrv_urdf_file
    self.pybullet_cv_urdf_file = pybullet_cv_urdf_file
    self.mrv_joint_angle_lower_limits = mrv_joint_angle_lower_limits
    self.mrv_joint_angle_upper_limits = mrv_joint_angle_upper_limits
    self.mrv_joint_vel_limits = mrv_joint_vel_limits
    self.mrv_joint_acc_limits = mrv_joint_acc_limits
    self.mrv_joint_torque_limits = mrv_joint_torque_limits
    self.dt = dt
    self.cone_slope = cone_slope
    self.clip_joint_commands = clip_joint_commands
    self.time_steps_between_measurements = time_steps_between_measurements
    self.cw_a = cw_a
    self.cw_mu = cw_mu
    self.cw_orbit_dir = cw_orbit_dir
    self.do_noisy_state_estimation = do_noisy_state_estimation
    self.nozzle_opening_rad = nozzle_opening_rad
    self.peg_rad = peg_rad
    self.use_cw = use_cw
    self.velocity_noise_ang_amp = velocity_noise_ang_amp
    self.time_limit = time_limit
    self.check_joint_angle_limit_violation = False # whether to check joint limit violations
    self.lock_client = lock_client
    self.lock_mrv = lock_mrv
    self.plunging = False

    self.init_time = time.time()

    self.use_ekf = use_ekf

    '''Input URDFs:
    These are all simply passed direclty to MRVCLientSim
    mrv_cv_urdf_file - currently using robot_cv_detached.urdf. 
    mrv_urdf_file - currently using robot.urdf. 
    pybullet_mrv_urdf_file - currently using barebones_robot.urdf. 
    pybullet_cv_urdf_file - currently using cv.urdf.
    '''

    self.mrv_client_sim = MRVClientSim(self.mrv_cv_urdf_file, self.mrv_urdf_file, self.pybullet_mrv_urdf_file, self.pybullet_cv_urdf_file, self.mrv_joint_angle_lower_limits, self.mrv_joint_angle_upper_limits, 
                               self.mrv_joint_vel_limits, self.mrv_joint_acc_limits, self.mrv_joint_torque_limits, self.dt, self.cone_slope, self.time_steps_between_measurements, 
                               self.cw_a, self.cw_mu, self.cw_orbit_dir, self.lock_client, self.lock_mrv, self.use_cw , use_ekf= self.use_ekf)

    pin.forwardKinematics(self.mrv_client_sim.pin_model, self.mrv_client_sim.pin_data, pin.neutral(self.mrv_client_sim.pin_model))
    pin.updateFramePlacement(self.mrv_client_sim.pin_model, self.mrv_client_sim.pin_data, self.mrv_client_sim.nozzle_fid)
    pin.updateFramePlacement(self.mrv_client_sim.pin_model, self.mrv_client_sim.pin_data, self.mrv_client_sim.goal_fid)
    self.dist_nozzle_opening_from_goal = self.mrv_client_sim.get_dist_nozzle_opening_from_goal()

    self.sw_f_kp = 0.0
    self.sw_f_ki = 0.0
    self.sw_tau_kp = 0.0
    self.sw_tau_ki = 0.0
    self.sw_pos_kp = 1
    self.sw_pos_kd = 2*np.sqrt(self.sw_pos_kp)
    self.sw_rot_kp = 1
    self.sw_rot_kd = 2*np.sqrt(self.sw_rot_kp)

    self.pin_model = self.mrv_client_sim.pin_model
    self.pin_data = pin.Data(self.pin_model)
    self.num_rotary = self.mrv_client_sim.num_rotary
    self.pd_target_fid = self.pin_model.getFrameId('pd_target')
    self.mrv_pin_model = self.mrv_client_sim.mrv_pin_model
    self.mrv_pin_data = pin.Data(self.mrv_pin_model)
    self.mrv_wrist_fid = self.mrv_pin_model.getFrameId('wrist')
    self.mrv_peg_fid = self.mrv_pin_model.getFrameId('ee_tip')

    self.cv_jidx = self.mrv_client_sim.cv_jidx
    self.cv_qidx = self.mrv_client_sim.cv_qidx
    self.cv_vidx = self.mrv_client_sim.cv_vidx

    self.mrv_jidx = self.mrv_client_sim.mrv_jidx
    self.mrv_qidx = self.mrv_client_sim.mrv_qidx
    self.mrv_vidx = self.mrv_client_sim.mrv_vidx

    self.ref_ee_pos_trj_world = []

    self.flop_counts = []
    self.times = []
    self.force_err_integral = np.zeros(3)
    self.joint_torque_err_integral = np.zeros(self.num_rotary)

    # The last trajectory point used to generate a control
    self.last_traj_point = []

    self.last_joint_vels_cmd = np.zeros(self.mrv_client_sim.num_rotary)
    
    # Within nozzle admittance controller is used once we reach the alignment waypoint, before that we will be tracking using
    # a TVLQR controller wich is initialized in self.reset_wrt_capture_box() and called by the simulator node
    self.within_nozzle_admittance = WithinNozzleAdmittance(probe_z_axis_plunge_velocity, use_variable_plunge_speed,use_scheduled_gains)

    '''Overrides the commanded velocity using a hard-code end effector twist. Use this for debugging'''
    self.override_twist = False

    '''Apply a sinusoidal velocity to the client for the first part of the simulation
    This was used to as a quick attempt to simulate client "Brownian motion"'''
    self.apply_sinusoidal_velocity_to_client = False
    self.sinusoidal_velocity_amp = 0.1*np.pi/180.
    self.sinusoidal_velocity_freq = 0.05
    self.sinusoidal_velocity_time = 6 #after this time, we stop setting the sinusoidal velocity 

    '''Apply a random initial velocity to the client'''
    self.apply_random_client_velocity = False
    self.random_client_velocity_time = 0 # for storing the sim time and determining timeout
    self.random_client_velocity_timeout = 1 # amount of time to wait before setting a new angular velocity

    if self.apply_sinusoidal_velocity_to_client and self.apply_random_client_velocity:
      raise Exception("Cannot use both slient sinusoidal velocity and client velocity noise at the same time.")
    
    ''' In lieu of the trajectory library, use a trajectory from TestTrajectory, normally for debugging purposes'''
    self.debug_with_test_traj = debug_with_test_traj
    self.test_traj_id = test_traj_id

    ''' Whether or not to check failure critierion. It's useful to turn this off for debugging purposes.'''
    self.check_failure_criterion = False

    '''Stop the controller after reaching the goal'''
    self.stop_after_success = True

    '''Whether or not to print a message when either joint velocity or acceleration is saturated to the limits'''
    self.print_joint_limit_violations = True

    '''Do not change this manually. Instead, use disable_joint_control()'''
    self.joint_control_enabled = True

    # Start the EKF with several steps to converge
    # wrench_peg_peg = np.zeros(6)
    # for _ in range(50):
    #   self.mrv_client_sim.update_state_estimate(wrench_peg_peg)

    self.ref_ee_pos_world_real = []

  def get_state_in_pieces(self):
    mrv_client_sim = self.mrv_client_sim

    sw_base_pos = np.copy(mrv_client_sim.x[mrv_client_sim.mrv_qidx:mrv_client_sim.mrv_qidx + 3])
    sw_base_rmat = R.from_quat(mrv_client_sim.x[mrv_client_sim.mrv_qidx + 3:mrv_client_sim.mrv_qidx + 7]).as_matrix()
    sw_joint_angles = np.copy(mrv_client_sim.x[mrv_client_sim.mrv_qidx + 7:mrv_client_sim.mrv_qidx + 7 + mrv_client_sim.num_rotary])
    sw_base_v = np.copy(mrv_client_sim.x[mrv_client_sim.pin_model.nq + mrv_client_sim.mrv_vidx:mrv_client_sim.pin_model.nq + mrv_client_sim.mrv_vidx + 3])
    sw_base_w = np.copy(mrv_client_sim.x[mrv_client_sim.pin_model.nq + mrv_client_sim.mrv_vidx + 3:mrv_client_sim.pin_model.nq + mrv_client_sim.mrv_vidx + 6])
    sw_joint_vels = np.copy(mrv_client_sim.x[mrv_client_sim.pin_model.nq + mrv_client_sim.mrv_vidx + 6:mrv_client_sim.pin_model.nq + mrv_client_sim.mrv_vidx + 6 + mrv_client_sim.num_rotary])

    sw_client_pos = np.copy(mrv_client_sim.x[mrv_client_sim.cv_qidx:mrv_client_sim.cv_qidx + 3])
    sw_client_rmat = R.from_quat(mrv_client_sim.x[mrv_client_sim.cv_qidx + 3:mrv_client_sim.cv_qidx + 7]).as_matrix()
    sw_client_v = np.copy(mrv_client_sim.x[mrv_client_sim.pin_model.nq + mrv_client_sim.cv_vidx:mrv_client_sim.pin_model.nq + mrv_client_sim.cv_vidx + 3])
    sw_client_w = np.copy(mrv_client_sim.x[mrv_client_sim.pin_model.nq + mrv_client_sim.cv_vidx + 3:mrv_client_sim.pin_model.nq + mrv_client_sim.cv_vidx + 6])

    return sw_base_pos, sw_base_rmat, sw_joint_angles, sw_base_v, sw_base_w, sw_joint_vels, sw_client_pos, sw_client_rmat, sw_client_v, sw_client_w

  def get_traj(self):
    return self.last_traj_point.p, self.last_traj_point.rmat

  def interp_x(self, x1, x2, alpha):
    x_interp = np.zeros_like(x1)
    x_interp[:self.pin_model.nq] = pin.interpolate(self.pin_model, x1[:self.pin_model.nq], x2[:self.pin_model.nq], alpha)
    x_interp[self.pin_model.nq:] = (1 - alpha)*x1[self.pin_model.nq:] + alpha*x2[self.pin_model.nq:]
    return x_interp

  def get_x_u_plan_from_load_paths(self, load_paths , weights):
    x1 = np.load(load_paths[0] + '/xs.npy')
    u1 = np.load(load_paths[0] + '/us.npy')
    x2 = np.load(load_paths[1] + '/xs.npy')
    u2 = np.load(load_paths[1] + '/us.npy')

    x_trj = np.array([self.interp_x(x1[i], x2[i], weights[0]) for i in range(len(x1))])
    u_trj = np.array([weights[0]*u1[i] + weights[1]*u2[i] for i in range(len(u1))])

    return x_trj, u_trj
  
  def get_reference_from_load_path(self, load_path, impact_dyn):
    ref_xs = np.load(load_path + '/xs.npy')
    ref_us = np.load(load_path + '/us.npy')
    phase_starts = np.load(load_path + '/phase_starts.npy')
    dts = np.load(load_path + '/dts.npy')
    x0 = ref_xs[0]

    if impact_dyn:
      impact_nozzle_idx = 3
      impact_hole_idx = 4
      about_to_impact_nozzle_idx = impact_nozzle_idx - 1
      about_to_impact_hole_idx = impact_hole_idx - 1
      contact_nozzle_idx = impact_nozzle_idx + 1
      contact_hole_idx = impact_hole_idx + 1

      about_to_impact_nozzle_step = phase_starts[about_to_impact_nozzle_idx]
      about_to_impact_hole_step = phase_starts[about_to_impact_hole_idx]

      impact_nozzle_step = phase_starts[impact_nozzle_idx]
      impact_hole_step = phase_starts[impact_hole_idx]

      contact_nozzle_step = phase_starts[contact_nozzle_idx]
      contact_hole_step = phase_starts[contact_hole_idx]

      impact_nozzle_impulse = ref_us[impact_nozzle_step, -3:]
      impact_hole_impulse = ref_us[impact_hole_step, -3:]

      ref_xs = np.concatenate((ref_xs[:impact_nozzle_step], ref_xs[contact_nozzle_step:impact_hole_step], ref_xs[contact_hole_step:]), 0)
      ref_us = np.concatenate((ref_us[:impact_nozzle_step], ref_us[contact_nozzle_step:impact_hole_step], ref_us[contact_hole_step:]), 0)
      phase_starts = np.concatenate((phase_starts[:impact_nozzle_idx], phase_starts[contact_nozzle_idx:impact_hole_idx] - 1, phase_starts[contact_hole_idx:] - 2))
      dts = np.concatenate((dts[:impact_nozzle_idx], dts[contact_nozzle_idx:impact_hole_idx], dts[contact_hole_idx:]))
    else:
      impact_nozzle_impulse = np.zeros(3)
      impact_hole_impulse = np.zeros(3)
    t = 0
    ts = []
    for dt, phase_start, phase_end in zip(dts, phase_starts[:-1], phase_starts[1:]):
      for step in range(phase_start, phase_end):
        ts.append(t)
        t = t + dt

    steps = ts[-1]/self.dt
    interp_xs = []
    interp_us = []
    interp_ts = []

    did_impact_nozzle = False
    did_impact_hole = False
    for step in range(int(steps)):
      t = step*self.dt
      interp_ts.append(t)
      pre_step = bisect.bisect_right(ts, t) - 1
      if pre_step == len(ref_xs) - 1:
        break
      alpha = (t - ts[pre_step])/(ts[pre_step + 1] - ts[pre_step])
      pre_x = ref_xs[pre_step]
      post_x = ref_xs[pre_step + 1]
      x_interp = self.interp_x(pre_x, post_x, alpha)
      interp_xs.append(x_interp)

      pre_u = ref_us[pre_step]
      post_u = ref_us[pre_step + 1]
      interp_us.append(alpha*post_u + (1 - alpha)*pre_u)

      if impact_dyn and (pre_step == about_to_impact_nozzle_step or pre_step == about_to_impact_hole_step):
        interp_us[-1][-3:] = pre_u[-3:]

      if impact_dyn and not did_impact_nozzle and pre_step == impact_nozzle_step:
        interp_us[-1][-3:] += impact_nozzle_impulse/self.dt
        did_impact_nozzle = True

      if impact_dyn and not did_impact_hole and pre_step == impact_hole_step:
        interp_us[-1][-3:] += impact_hole_impulse/self.dt
        did_impact_hole = True

    ref_xs = interp_xs
    ref_us = interp_us
    ref_ts = interp_ts

    mrv_client_sim = self.mrv_client_sim

    self.pin_model = mrv_client_sim.pin_model
    self.pin_data = pin.Data(self.pin_model)

    actuation = ActuationModelCWContact(self.pin_model, self.mrv_client_sim.mrv_pin_model, self.cw_a, self.cw_mu, self.cw_orbit_dir, np.eye(3), self.use_cw)

    v0 = x0[self.pin_model.nq:]
    initial_client_w = v0[self.cv_vidx + 3:self.cv_vidx + 6]
    initial_client_w_xy = np.copy(initial_client_w)[:2]

    theta = np.arctan2(initial_client_w_xy[1], initial_client_w_xy[0]) - np.pi/2
    if theta < 0:
      theta += 2*np.pi
    plane_idx = int(theta/(np.pi/4))
    nozzle_geom_fid = self.pin_model.getFrameId('nozzle_geom' + str(plane_idx))

    ref_ee_pos_trj = []
    ref_ee_rmat_trj = []
    ref_ee_v_trj = []
    ref_ee_w_trj = []
    ref_ee_vdot_trj = []
    ref_ee_wdot_trj = []
    ref_joint_angles_trj = []
    ref_joint_vels_trj = []
    ref_joint_accs_trj = []
    ref_forces_trj = []

    local_force_params = False

    for trj_idx, (x, joint_cmd) in enumerate(zip(ref_xs, ref_us)):
      q = x[:self.pin_model.nq]
      v = x[self.pin_model.nq:]
      if local_force_params:
        pin.forwardKinematics(self.pin_model, self.pin_data, q)
        pin.updateFramePlacement(self.pin_model, self.pin_data, nozzle_geom_fid)
        force = -joint_cmd[-3]*self.pin_data.oMf[nozzle_geom_fid].rotation[:, 0] + self.pin_data.oMf[self.mrv_client_sim.hole_fid].rotation[:, :2]@joint_cmd[-2:]
        u_for_actuation = np.concatenate((joint_cmd[:-3], force))
        tau = actuation.calc(x, u_for_actuation)
      else:
        tau = actuation.calc(x, joint_cmd)
        force = joint_cmd[-3:]
      vdot = pin.aba(self.pin_model, self.pin_data, q, v, tau)
  
      pin.forwardKinematics(self.pin_model, self.pin_data, q, v, vdot)
      pin.updateFramePlacement(self.pin_model, self.pin_data, self.mrv_client_sim.peg_fid)
      pin.updateFramePlacement(self.pin_model, self.pin_data, nozzle_geom_fid)
      pin.updateFramePlacement(self.pin_model, self.pin_data, self.mrv_client_sim.hole_fid)

      tip_pos = self.pin_data.oMf[self.mrv_client_sim.peg_fid].translation
      tip_rmat = self.pin_data.oMf[self.mrv_client_sim.peg_fid].rotation
      tip_twist = pin.getFrameVelocity(self.pin_model, self.pin_data, self.mrv_client_sim.peg_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).vector

      tip_v = tip_twist[:3]
      tip_w = tip_twist[3:]
      tip_acc = pin.getFrameAcceleration(self.pin_model, self.pin_data, self.mrv_client_sim.peg_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).vector
      tip_vdot = tip_acc[:3]
      tip_wdot = tip_acc[3:]

      client_pos = q[self.cv_qidx:self.cv_qidx + 3]
      client_quat = q[self.cv_qidx + 3:self.cv_qidx + 7]
      client_rmat = R.from_quat(client_quat).as_matrix()
      client_v = v[self.cv_vidx:self.cv_vidx + 3]
      client_w = v[self.cv_vidx + 3:self.cv_vidx + 6]
      client_vdot = vdot[self.cv_vidx:self.cv_vidx + 3]
      client_wdot = vdot[self.cv_vidx + 3:self.cv_vidx + 6]

      ref_ee_pos_trj.append(client_rmat.transpose()@(tip_pos - client_pos))
      ref_ee_rmat_trj.append(client_rmat.transpose()@tip_rmat)
      ref_ee_v_trj.append(client_rmat.transpose()@tip_v - \
                          client_v - \
                          np.cross(client_w, ref_ee_pos_trj[-1]))
      ref_ee_w_trj.append(client_rmat.transpose()@tip_w - \
                          client_w)
      ref_ee_vdot_trj.append(client_rmat.transpose()@tip_vdot - \
                             np.cross(client_w, client_rmat.transpose()@tip_v) - \
                             client_vdot - \
                             np.cross(client_w, ref_ee_v_trj[-1]) - \
                             np.cross(client_wdot, ref_ee_pos_trj[-1]))
      ref_ee_wdot_trj.append(client_rmat.transpose()@tip_wdot - \
                             np.cross(client_w, client_rmat.transpose()@tip_w) - \
                             client_wdot)

      ref_forces_trj.append(client_rmat.transpose()@force)

      ref_joint_angles_trj.append(np.copy(q[mrv_client_sim.mrv_qidx + 7:mrv_client_sim.mrv_qidx + mrv_client_sim.mrv_nq]))
      ref_joint_vels_trj.append(np.copy(v[mrv_client_sim.mrv_vidx + 6:mrv_client_sim.mrv_vidx + mrv_client_sim.mrv_nv]))
      ref_joint_accs_trj.append(np.copy(vdot[mrv_client_sim.mrv_vidx + 6:mrv_client_sim.mrv_vidx + mrv_client_sim.mrv_nv]))

    return ref_ee_pos_trj, \
           ref_ee_rmat_trj, \
           ref_ee_v_trj, \
           ref_ee_w_trj, \
           ref_ee_vdot_trj, \
           ref_ee_wdot_trj, \
           ref_joint_angles_trj, \
           ref_joint_vels_trj, \
           ref_joint_accs_trj, \
           ref_forces_trj, \
           np.copy(ref_xs), \
           np.copy(ref_us), \
           np.copy(ref_ts)

  def reset_wrt_capture_box(self, load_paths, weights, delta_pos, delta_rot, delta_v, initial_client_w, initial_mrv_w, rng, dist_centering_waypoint_from_goal):
    '''delta_pos is change in position relative to front of capture box'''

    mrv_client_sim = self.mrv_client_sim
    self.initial_client_w = initial_client_w

    # Define nozzle and peg positions with respect to nozzle frame
    sw_nozzle_pos = np.zeros(3)
    sw_nozzle_rmat = np.eye(3)

    sw_peg_pos = sw_nozzle_pos + sw_nozzle_rmat@delta_pos - sw_nozzle_rmat[:, 2]*(dist_centering_waypoint_from_goal - self.dist_nozzle_opening_from_goal)
    sw_peg_rmat = sw_nozzle_rmat@pin.exp3(delta_rot)

    # Angles from April 2024 URDF update (NG's starting joint angles)
    #initial_joint_angles = np.array([1.588249619, 0.5925392811, -0.2136283004, -1.062207383, -0.8237954069, 0.3869394952, -3.284709652])
    #initial_joint_velocities = np.array([0., 0., 0., 0., 0., 0., 0.])

    # Joint angles determined by starting from NG's joint angles in April 2024 URDF update and moving along (-z) approximately 0.75 seconds
    initial_joint_angles = np.array([1.38580076, 1.17069419, -0.2494883, -2.23758596, -0.51633639, 0.94903096, -3.64965438])
    initial_joint_velocities = np.array([0., 0., 0., 0., 0., 0., 0.])

    # Joint angles determined by starting from NG's joint angles in April 2024 URDF update and moving along (-z) approximately 0.5 seconds
    #initial_joint_angles = np.array([1.4082254, 0.99829701, -0.20852452, -1.90511499, -0.51490648, 0.79190962, -3.58750692])
    #initial_joint_velocities = np.array([0., 0., 0., 0., 0., 0., 0.])

    # Joint angles determined by starting from NG's joint angles in April 2024 URDF update and moving along (-z) approximately 1 second
    #initial_joint_angles = np.array([1.37657866, 1.33621139, -0.34258679, -2.55257545, -0.5846146, 1.09879969, -3.70471875])
    #initial_joint_velocities = np.array([0., 0., 0., 0., 0., 0., 0.])
  
    q_sw = pin.neutral(mrv_client_sim.pin_model)
    q_sw[mrv_client_sim.mrv_qidx + 7:mrv_client_sim.mrv_qidx + 7 + mrv_client_sim.num_rotary] = initial_joint_angles

    pin.forwardKinematics(mrv_client_sim.pin_model, mrv_client_sim.pin_data, q_sw)
    pin.updateFramePlacement(mrv_client_sim.pin_model, mrv_client_sim.pin_data, mrv_client_sim.peg_fid)
    pin.updateFramePlacement(mrv_client_sim.pin_model, mrv_client_sim.pin_data, mrv_client_sim.nozzle_fid)
    g_client_nozzle = mrv_client_sim.pin_data.oMf[mrv_client_sim.nozzle_fid].inverse()
    g_base_peg = mrv_client_sim.pin_data.oMf[mrv_client_sim.peg_fid].inverse()

    initial_client_pos = sw_nozzle_rmat@g_client_nozzle.translation + sw_nozzle_pos
    initial_client_quat = R.from_matrix(sw_nozzle_rmat@g_client_nozzle.rotation).as_quat()
    initial_client_v = np.zeros(3)

    initial_mrv_pos = sw_peg_rmat@g_base_peg.translation + sw_peg_pos
    initial_mrv_quat = R.from_matrix(sw_peg_rmat@g_base_peg.rotation).as_quat()

    initial_mrv_v = R.from_quat(initial_mrv_quat).as_matrix().transpose()@R.from_quat(initial_client_quat).as_matrix()@delta_v

    if load_paths is not None: 
      self.reset_trajectory_library(load_paths,weights)
      initial_client_rmat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
      self.tv_lqr_controller = TVLQRController(self.mrv_cv_urdf_file, self.mrv_urdf_file, self.ref_x_trj, self.ref_u_trj,
                                               self.dt, self.dt, self.cw_a, self.cw_mu, self.cw_orbit_dir, initial_client_rmat)
      self.tv_lqr_controller.backward_solve_ricatti()
      self.k_cntrl = 0
      self.solve_times = []
      

    self.mrv_client_sim = MRVClientSim(self.mrv_cv_urdf_file, self.mrv_urdf_file, self.pybullet_mrv_urdf_file, self.pybullet_cv_urdf_file, self.mrv_joint_angle_lower_limits, self.mrv_joint_angle_upper_limits, 
                               self.mrv_joint_vel_limits, self.mrv_joint_acc_limits, self.mrv_joint_torque_limits, self.dt, self.cone_slope, self.time_steps_between_measurements, 
                               self.cw_a, self.cw_mu, self.cw_orbit_dir, self.do_noisy_state_estimation, self.lock_client, self.lock_mrv, self.use_cw , use_ekf= self.use_ekf)
    mrv_client_sim = self.mrv_client_sim

    self.pin_model = mrv_client_sim.pin_model
    self.pin_data = pin.Data(self.pin_model)

    mrv_client_sim.reset(initial_client_pos, initial_client_quat, initial_client_v, initial_client_w, 
                 initial_mrv_pos, initial_mrv_quat, initial_joint_angles, initial_mrv_v, initial_mrv_w, initial_joint_velocities, rng)
    mrv_client_sim.update_kinematics()

    self.init_mrv_tip_pos = mrv_client_sim.get_mrv_tip_pos()
    self.init_mrv_joint_angles = mrv_client_sim.get_mrv_joint_angles()

    self.mrv_client_sim.update_state_estimate(np.zeros(6))

    self.prev_joint_vels = np.copy(initial_joint_velocities)
    self.initial_client_quat = np.copy(initial_client_quat)

    self.ref_trj_idx = 0

    self.joint_torque_meas_trj = []
    self.joint_torque_meas_d_trj = []

  def reset_trajectory_library(self, load_paths, weights):

    impact_dyn = False

    self.load_paths = load_paths
    self.load_path_weights = weights
    
    ref_ee_pos_trj_list = []
    ref_ee_rmat_trj_list = []
    ref_ee_v_trj_list = []
    ref_ee_w_trj_list = []
    ref_ee_vdot_trj_list = []
    ref_ee_wdot_trj_list = []
    ref_joint_angles_trj_list = []
    ref_joint_vels_trj_list = []
    ref_joint_accs_trj_list = []
    ref_forces_trj_list = []
    ref_x_trj_list = []
    ref_u_trj_list = []
    ref_t_trj_list = []

    for load_path in load_paths:
      ref_ee_pos_trj, \
      ref_ee_rmat_trj, \
      ref_ee_v_trj, \
      ref_ee_w_trj, \
      ref_ee_vdot_trj, \
      ref_ee_wdot_trj, \
      ref_joint_angles_trj, \
      ref_joint_vels_trj, \
      ref_joint_accs_trj, \
      ref_forces_trj, \
      ref_x_trj, \
      ref_u_trj, \
      ref_t_trj, = self.get_reference_from_load_path(load_path, impact_dyn)

      ref_ee_pos_trj_list.append(ref_ee_pos_trj)
      ref_ee_rmat_trj_list.append(ref_ee_rmat_trj)
      ref_ee_v_trj_list.append(ref_ee_v_trj)
      ref_ee_w_trj_list.append(ref_ee_w_trj)
      ref_ee_vdot_trj_list.append(ref_ee_vdot_trj)
      ref_ee_wdot_trj_list.append(ref_ee_wdot_trj)
      ref_joint_angles_trj_list.append(ref_joint_angles_trj)
      ref_joint_vels_trj_list.append(ref_joint_vels_trj)
      ref_joint_accs_trj_list.append(ref_joint_accs_trj)
      ref_forces_trj_list.append(ref_forces_trj)
      ref_x_trj_list.append(ref_x_trj)
      ref_u_trj_list.append(ref_u_trj)
      ref_t_trj_list.append(ref_t_trj)

    # Interpolate between reference trajectories
    max_steps = np.amax([len(ref_ee_pos_trj) for ref_ee_pos_trj in ref_ee_pos_trj_list])
    self.ref_ee_pos_trj = []
    self.ref_ee_rmat_trj = []
    self.ref_ee_v_trj = []
    self.ref_ee_w_trj = []
    self.ref_ee_vdot_trj = []
    self.ref_ee_wdot_trj = []
    self.ref_joint_angles_trj = []
    self.ref_joint_vels_trj = []
    self.ref_joint_accs_trj = []
    self.ref_forces_trj = []
    self.ref_x_trj = []
    self.ref_u_trj = []
    self.ref_t_trj = []

    get_trj_elem = lambda trj_idx, trj: trj[trj_idx] if trj_idx < len(trj) else trj[-1]

    for step in range(max_steps):
      self.ref_ee_pos_trj.append(sum([weight*get_trj_elem(step, trj) for weight, trj in zip(weights, ref_ee_pos_trj_list)]))

      # Weighted average of reference ee rotation matrices needs special handling
      ref_ee_R = R.concatenate([R.from_matrix(get_trj_elem(step, trj)) for trj in ref_ee_rmat_trj_list])
      self.ref_ee_rmat_trj.append(R.mean(ref_ee_R, weights).as_matrix())

      self.ref_ee_v_trj.append(sum([weight*get_trj_elem(step, trj) for weight, trj in zip(weights, ref_ee_v_trj_list)]))
      self.ref_ee_w_trj.append(sum([weight*get_trj_elem(step, trj) for weight, trj in zip(weights, ref_ee_w_trj_list)]))
      self.ref_ee_vdot_trj.append(sum([weight*get_trj_elem(step, trj) for weight, trj in zip(weights, ref_ee_vdot_trj_list)]))
      self.ref_ee_wdot_trj.append(sum([weight*get_trj_elem(step, trj) for weight, trj in zip(weights, ref_ee_wdot_trj_list)]))

      self.ref_forces_trj.append(sum([weight*get_trj_elem(step, trj) for weight, trj in zip(weights, ref_forces_trj_list)]))

      self.ref_joint_angles_trj.append(sum([weight*get_trj_elem(step, trj) for weight, trj in zip(weights, ref_joint_angles_trj_list)]))
      self.ref_joint_vels_trj.append(sum([weight*get_trj_elem(step, trj) for weight, trj in zip(weights, ref_joint_vels_trj_list)]))
      self.ref_joint_accs_trj.append(sum([weight*get_trj_elem(step, trj) for weight, trj in zip(weights, ref_joint_accs_trj_list)]))

      # Weighted average of reference x needs special handling
      ref_base_R = R.concatenate([R.from_quat(get_trj_elem(step, trj)[self.mrv_qidx + 3:self.mrv_qidx + 7]) for trj in ref_x_trj_list])
      ref_base_quat = R.mean(ref_base_R, weights).as_quat()

      ref_client_R = R.concatenate([R.from_quat(get_trj_elem(step, trj)[self.cv_qidx + 3:self.cv_qidx + 7]) for trj in ref_x_trj_list])
      ref_client_quat = R.mean(ref_client_R, weights).as_quat()

      ref_x = np.zeros(self.pin_model.nq + self.pin_model.nv)
      ref_x[self.mrv_qidx:self.mrv_qidx + 3] = sum([weight*get_trj_elem(step, trj)[self.mrv_qidx:self.mrv_qidx + 3] for weight, trj in zip(weights, ref_x_trj_list)])
      ref_x[self.mrv_qidx + 3:self.mrv_qidx + 7] = ref_base_quat
      ref_x[self.mrv_qidx + 7:self.mrv_qidx + self.mrv_client_sim.mrv_pin_model.nq] = self.ref_joint_angles_trj[step]
      ref_x[self.cv_qidx:self.cv_qidx + 3] = sum([weight*get_trj_elem(step, trj)[self.cv_qidx:self.cv_qidx + 3] for weight, trj in zip(weights, ref_x_trj_list)])
      ref_x[self.cv_qidx + 3:self.cv_qidx + 7] = ref_client_quat
      ref_x[self.pin_model.nq:] = sum([weight*get_trj_elem(step, trj)[self.pin_model.nq:] for weight, trj in zip(weights, ref_x_trj_list)])

      self.ref_x_trj.append(ref_x)
      self.ref_u_trj.append(sum([weight*get_trj_elem(step, trj) for weight, trj in zip(weights, ref_u_trj_list)]))
      self.ref_t_trj.append(sum([weight*get_trj_elem(step, trj) for weight, trj in zip(weights, ref_t_trj_list)]))

  def get_peg_and_nozzle_info(self):
    return self.mrv_client_sim.get_peg_and_nozzle_info()
  
  def step(self, wrench_peg_peg=None, apply_wrench_only_when_close=True):
    # If wrench_peg_peg is None, this signifies we are not using the F/T sensor so we much simulate the contact mechanics.
    # If wrench_peg_peg is zeros, this signifies we are using the F/T sensor but there is no wrench.
    # If this function returns success, insertion succeeded. If it returns 'nothing', neither success
    # nor fail happened. If it returns anything else, we assume failure and the returned string
    # is the reason for failure

    mrv_client_sim = self.mrv_client_sim

    if self.apply_sinusoidal_velocity_to_client:
        if mrv_client_sim.sim_time < self.sinusoidal_velocity_time:
          print("Overriding client velocity with a sinusoidal profile.")
          amp = self.sinusoidal_velocity_amp
          freq = self.sinusoidal_velocity_freq
          angular_vel= amp*np.sin(freq*(2*np.pi)*self.mrv_client_sim.sim_time)
          mrv_client_sim.set_client_velocity([0,0,0],[-angular_vel,0,0])
        else:
          print("Stopped overriding client velocity.")

    if self.apply_random_client_velocity and ((self.mrv_client_sim.sim_time - self.random_client_velocity_time) > self.random_client_velocity_timeout):
      print("Overriding client velocity with velocity noise")
      x_ang_vel = np.random.uniform(-self.velocity_noise_ang_amp, self.velocity_noise_ang_amp)
      y_ang_vel = np.random.uniform(-self.velocity_noise_ang_amp, self.velocity_noise_ang_amp)
      z_ang_vel = np.random.uniform(-self.velocity_noise_ang_amp, self.velocity_noise_ang_amp)
      mrv_client_sim.set_client_angular_velocity_disturbance(self.initial_client_w + [x_ang_vel,y_ang_vel,z_ang_vel])
      self.random_client_velocity_time = self.mrv_client_sim.sim_time

    use_pybullet = True
    use_contact_sim = wrench_peg_peg is None

    if use_pybullet and not use_contact_sim:
      mrv_client_sim.disable_pybullet_collisions()

    sw_peg_pos, sw_peg_rmat, sw_peg_twist, sw_nozzle_pos, sw_nozzle_rmat, sw_nozzle_twist = mrv_client_sim.get_peg_and_nozzle_info()
    sw_peg_pos_est, sw_peg_rmat_est, sw_peg_twist_est, sw_nozzle_pos_est, sw_nozzle_rmat_est, sw_nozzle_twist_est = self.mrv_client_sim.get_peg_and_nozzle_info_est()

    sw_pos_goal_est = sw_nozzle_pos_est + sw_nozzle_rmat_est[:, 2]*self.dist_nozzle_opening_from_goal
    sw_rmat_goal_est = np.copy(sw_nozzle_rmat_est)
    #sw_v_goal_est = sw_nozzle_twist_est[:3] + np.cross(sw_nozzle_twist_est[3:], sw_nozzle_rmat_est[:, 2]*self.dist_nozzle_opening_from_goal)
    #sw_w_goal_est = np.copy(sw_nozzle_twist_est[3:])

    sw_pos_goal = sw_nozzle_pos + sw_nozzle_rmat[:, 2]*self.dist_nozzle_opening_from_goal
    sw_rmat_goal = np.copy(sw_nozzle_rmat)
    sw_rel_pos = sw_rmat_goal.transpose()@(sw_peg_pos - sw_pos_goal)

    # Don't apply a measured force to the simulation unless peg is close to nozzle
    if apply_wrench_only_when_close and not use_contact_sim and self.is_peg_close_to_nozzle(sw_rel_pos):
      wrench_peg_peg = np.zeros(6)

    if not use_contact_sim:
      self.mrv_client_sim.pb_peg_wrench = wrench_peg_peg
    else:
      wrench_peg_peg = np.copy(self.mrv_client_sim.pb_peg_wrench)

    wrench_peg_peg_meas = np.copy(wrench_peg_peg)
    wrench_peg_peg_est = np.copy(wrench_peg_peg)

    sw_rel_pos_est = sw_rmat_goal_est.transpose()@(sw_peg_pos_est - sw_pos_goal_est)
    #sw_rel_vel_est = sw_rmat_goal_est.transpose()@(sw_peg_twist_est[:3] - sw_v_goal_est) - sw_rmat_goal_est.transpose()@np.cross(sw_w_goal_est, sw_peg_pos_est - sw_pos_goal_est)

    if self.is_peg_close_to_nozzle(sw_rel_pos_est):
      wrench_peg_peg_est[:] = 0

    if self.stop_after_success:
      if sw_rel_pos[2] > 0.:
      # if np.linalg.norm(sw_rel_pos[:2]) < self.peg_rad and sw_rel_pos[2] > 0.:
        print('Success')

        # This is currently creating an additional time step at the end where the wrist wrench is not available.
        mrv_client_sim.update_gt_state_trj()
        return 'success'

    # Check if software-simulated manipulator is close to singularity
    q_mrv = mrv_client_sim.get_mrv_config()
    v_mrv = mrv_client_sim.x[mrv_client_sim.pin_model.nq + mrv_client_sim.mrv_vidx:mrv_client_sim.pin_model.nq + mrv_client_sim.mrv_vidx + mrv_client_sim.mrv_pin_model.nv]

    # Check if we're close to a singularity
    pin.forwardKinematics(mrv_client_sim.mrv_pin_model, mrv_client_sim.mrv_pin_data, q_mrv)
    pin.computeJointJacobians(mrv_client_sim.mrv_pin_model, mrv_client_sim.mrv_pin_data)
    pin.computeJointJacobiansTimeVariation(self.mrv_pin_model, mrv_client_sim.mrv_pin_data, q_mrv, v_mrv)
    pin.updateFramePlacement(mrv_client_sim.mrv_pin_model, mrv_client_sim.mrv_pin_data, mrv_client_sim.mrv_peg_fid)
    pin.updateFramePlacement(mrv_client_sim.mrv_pin_model, mrv_client_sim.mrv_pin_data, self.mrv_wrist_fid)

    self.wrench_peg_peg = np.copy(wrench_peg_peg)
    self.wrench_peg_peg_meas = np.copy(wrench_peg_peg_meas)
    self.wrench_peg_peg_est = np.copy(wrench_peg_peg_est)

    client_rmat_est = R.from_quat(mrv_client_sim.x_est[mrv_client_sim.cv_qidx + 3:mrv_client_sim.cv_qidx + 7]).as_matrix()
    client_rmat = R.from_quat(mrv_client_sim.x[mrv_client_sim.cv_qidx + 3:mrv_client_sim.cv_qidx + 7]).as_matrix()
    
    target_wrench = np.zeros(6)

    if self.ref_trj_idx >= len(self.ref_ee_pos_trj):
      ref_ee_pos = self.ref_ee_pos_trj[self.ref_trj_idx - 1]
      ref_ee_rmat = self.ref_ee_rmat_trj[self.ref_trj_idx - 1]
      ref_joint_angles = self.ref_joint_angles_trj[self.ref_trj_idx - 1]

      ref_ee_v = np.zeros(3)
      ref_ee_w = np.zeros(3)

      ref_ee_vdot = np.zeros(3)
      ref_ee_wdot = np.zeros(3)

      ref_joint_vels = np.zeros(self.num_rotary)
      ref_joint_accs = np.zeros(self.num_rotary)
    else:
      ref_ee_pos = self.ref_ee_pos_trj[self.ref_trj_idx]
      ref_ee_rmat = self.ref_ee_rmat_trj[self.ref_trj_idx]
      ref_joint_angles = self.ref_joint_angles_trj[self.ref_trj_idx]

      ref_ee_v = self.ref_ee_v_trj[self.ref_trj_idx]
      ref_ee_w = self.ref_ee_w_trj[self.ref_trj_idx]

      ref_ee_vdot = self.ref_ee_vdot_trj[self.ref_trj_idx]
      ref_ee_wdot = self.ref_ee_wdot_trj[self.ref_trj_idx]

      ref_joint_vels = self.ref_joint_vels_trj[self.ref_trj_idx]
      ref_joint_accs = self.ref_joint_accs_trj[self.ref_trj_idx]

      # Target wrench is expressed in the peg frame
      target_wrench[:3] = sw_peg_rmat.transpose()@client_rmat_est@self.ref_forces_trj[self.ref_trj_idx]

      self.ref_trj_idx += 1

    q_mrv = mrv_client_sim.get_mrv_config()
    v_mrv = mrv_client_sim.get_mrv_config_dot()

    M = pin.crba(self.mrv_pin_model, self.mrv_pin_data, q_mrv)
    F_bias = pin.nonLinearEffects(self.mrv_pin_model, self.mrv_pin_data, q_mrv, v_mrv)

    # Convert task-space reference trajectories originally expressed in client frame, into world frame, accounting for current state of client
    client_pos_est = mrv_client_sim.x_est[mrv_client_sim.cv_qidx:mrv_client_sim.cv_qidx + 3]
    client_v_est = mrv_client_sim.x_est[mrv_client_sim.pin_model.nq + mrv_client_sim.cv_vidx:mrv_client_sim.pin_model.nq + mrv_client_sim.cv_vidx + 3]
    client_w_est = mrv_client_sim.x_est[mrv_client_sim.pin_model.nq + mrv_client_sim.cv_vidx + 3:mrv_client_sim.pin_model.nq + mrv_client_sim.cv_vidx + 6]

    if self.use_cw:
      client_vdot = client_rmat.transpose()@mrv_client_sim.compute_cw_force(True)/mrv_client_sim.client_mass
    else:
      client_vdot = np.zeros(3)

    use_estimated_state = True
    client_wdot = mrv_client_sim.unforced_eulers_eqns_for_client(use_estimated_state)
    client_w_world = client_rmat_est@client_w_est #TODO: Is this correct? Isn't client_w_est already in world frame?
    #client_w_world = client_w_est

    # Get actual ee_pose in the world frame for plotting
    client_pos = mrv_client_sim.x[mrv_client_sim.cv_qidx:mrv_client_sim.cv_qidx + 3]
    ref_pos_d_real = client_rmat@ref_ee_pos + client_pos
    self.ref_ee_pos_world_real.append(ref_pos_d_real)

    ref_pos_d = client_rmat_est@ref_ee_pos + client_pos_est
    ref_rmat_d = client_rmat_est@ref_ee_rmat
    ref_v_d = client_rmat_est@(ref_ee_v + client_v_est + np.cross(client_w_est, ref_ee_pos))
    ref_w_d = client_rmat_est@(ref_ee_w + client_w_est)      

    ref_vdot_d = client_rmat_est@(ref_ee_vdot + \
                          client_vdot + \
                          np.cross(client_w_est, ref_ee_v) + \
                          np.cross(client_wdot, ref_ee_pos)) + \
              np.cross(client_w_world, ref_v_d)
    ref_wdot_d = client_rmat_est@(ref_ee_wdot + client_wdot) + \
              np.cross(client_w_world, ref_w_d)

    ref_traj_point = RefTrajPoint()
    ref_traj_point.theta = ref_joint_angles
    ref_traj_point.theta_dot = ref_joint_vels
    ref_traj_point.theta_ddot = ref_joint_accs
    
    ref_traj_point.p = ref_pos_d
    ref_traj_point.v = ref_v_d
    ref_traj_point.vdot = ref_vdot_d

    ref_traj_point.rmat = ref_rmat_d
    ref_traj_point.omega = ref_w_d
    ref_traj_point.omega_dot = ref_wdot_d

    ref_traj_point.wrench = target_wrench

    if self.debug_with_test_traj: 
      test_traj = TestTrajectories(self.test_traj_id,self.init_mrv_tip_pos,self.init_mrv_joint_angles)
      ref_traj_point = test_traj.get_traj_point(mrv_client_sim.sim_time)    

    self.last_traj_point = copy.deepcopy(ref_traj_point)

    mrv_config = mrv_client_sim.get_mrv_config()
    mrv_config_dot = mrv_client_sim.get_mrv_config_dot()

    # We do not flop count on this block because the Jacobians are computed within the controllers
    # This block may be unnecessary in the future, but take care of how Jwrist is computed.
    pin.forwardKinematics(self.mrv_pin_model, self.mrv_pin_data, q_mrv)
    pin.computeJointJacobians(self.mrv_pin_model, self.mrv_pin_data)
    pin.computeJointJacobiansTimeVariation(self.mrv_pin_model, self.mrv_pin_data, q_mrv, v_mrv)
    pin.updateFramePlacement(self.mrv_pin_model, self.mrv_pin_data, self.mrv_wrist_fid)
    pin.updateFramePlacement(self.mrv_pin_model, self.mrv_pin_data, mrv_client_sim.mrv_peg_fid)

    # Not flop counting for now, but we may need to add this back in the future
    # count_flop = False
    # if count_flop:
    #   gc.disable()
    #   output_file = 'perf_output.txt'
    #   perf_proc = start_perf_proc(output_file)

    bt = time.process_time()
    if self.do_noisy_state_estimation:
      x_k = mrv_client_sim.x_est
      self.k_cntrl += 1
    else:
      x_k = mrv_client_sim.x
      self.k_cntrl += 1
    
    if mrv_client_sim.dist_to_throat_opening() < 0.175 or self.plunging:
      if not self.within_nozzle_admittance.admittance_traj_reset:
        print("Resetting admittance trajectory")
        self.within_nozzle_admittance.reset_admittance_traj(mrv_client_sim)
        self.plunging = True
        
      joint_acc_cmd = self.within_nozzle_admittance.compute_control(ref_traj_point, mrv_client_sim, wrench_peg_peg, self.dt, mrv_config, mrv_config_dot)
    else:
      print("Distance to throat opening: ", mrv_client_sim.dist_to_throat_opening())
      time_before = time.time()
      joint_acc_cmd = self.tv_lqr_controller.compute_control(x_k , self.k_cntrl, torques=False)
      elapsed_time = time.time() - time_before
      self.solve_times.append(elapsed_time)
      
  
    
    at = time.process_time()
    self.times.append(at - bt)  

    # Not flop counting for now, but we may need to add this back in the future
    # if count_flop:
    #   gc.enable()
    #   self.flop_counts.append(stop_perf_proc(output_file, perf_proc) - 2) # -2 to account for time.perf_counter()
        
    self.ref_ee_pos_trj_world.append(ref_pos_d)

    if self.clip_joint_commands:
      acc_eps = 0.001
      acc_min = -self.mrv_joint_acc_limits + acc_eps
      acc_max = self.mrv_joint_acc_limits - acc_eps
      acc_min_from_vel_min = (-self.mrv_joint_vel_limits - v_mrv[6:])/self.dt + acc_eps
      acc_max_from_vel_max = (self.mrv_joint_vel_limits - v_mrv[6:])/self.dt - acc_eps
      if (joint_acc_cmd < acc_min).any() or (joint_acc_cmd > acc_max).any():
          
          if self.print_joint_limit_violations:
            print('Joint acceleration limits reached.')
          joint_acc_cmd = np.clip(joint_acc_cmd, acc_min, acc_max)

      if (joint_acc_cmd < acc_min_from_vel_min).any() or (joint_acc_cmd > acc_max_from_vel_max).any():
          if self.print_joint_limit_violations:
            print('Joint velocity limits reached.')
          joint_acc_cmd = np.clip(joint_acc_cmd, acc_min_from_vel_min, acc_max_from_vel_max)

    joint_vel_cmd = self.last_joint_vels_cmd  + joint_acc_cmd*self.dt

    if self.override_twist: 
      Jstar, _ = mrv_client_sim.get_mrv_generalized_jacobian()

      '''Move through different twists'''
      # time_step = 3.
      # speed = 0.05
      # if mrv_client_sim.sim_time < time_step:
      #   twist = [0.0, 0.0, 0.0, speed, 0.0, 0]
      # elif mrv_client_sim.sim_time < 2*time_step: 
      #   twist = [0.0, 0.0, 0.0, -speed, 0.0, 0]
      # elif mrv_client_sim.sim_time < 3*time_step: 
      #   twist = [0.0, 0.0, 0.0, -speed, 0.0, 0]
      # elif mrv_client_sim.sim_time < 4*time_step: 
      #   twist = [0.0, 0.0, 0.0, speed, 0.0, 0]
      # elif mrv_client_sim.sim_time < 5*time_step: 
      #   twist = [0.0, 0.0, 0.0, speed, 0.0, 0]

      '''One hard-coded twist'''
      if mrv_client_sim.sim_time < 50:
        twist = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
      else: 
        twist = [0,0,0,0,0,0]

      twist_base, _ = mrv_client_sim.get_mrv_base_twist()
      joint_vel_cmd= np.linalg.pinv(Jstar)@np.array(twist - twist_base)

    self.last_joint_vels_cmd = joint_vel_cmd
    joint_cmd = joint_vel_cmd
    
    '''Need to recompute Jacobian in case task-space controller is being used'''
    # Previous commit used Jwrist to get the joint torques. This is incorrect
    #Jwrist = pin.getFrameJacobian(self.mrv_pin_model, self.mrv_pin_data, self.mrv_wrist_fid, pin.ReferenceFrame.LOCAL)
    #self.joint_torque_meas_trj.append(Jwrist[:, 6:].transpose()@wrench_peg_peg)
    #self.joint_torque_meas_d_trj.append(Jwrist[:, 6:].transpose()@target_wrench)
    Jpeg = pin.getFrameJacobian(self.mrv_pin_model, self.mrv_pin_data, self.mrv_peg_fid, pin.ReferenceFrame.LOCAL)
    self.joint_torque_meas_trj.append(Jpeg[:, 6:].transpose()@wrench_peg_peg)
    self.joint_torque_meas_d_trj.append(Jpeg[:, 6:].transpose()@target_wrench)

    if self.check_failure_criterion:
      # Failure criteria 2: goal is out of reach
      mrv_com = pin.centerOfMass(mrv_client_sim.mrv_pin_model, mrv_client_sim.mrv_pin_data, q_mrv)
      if np.linalg.norm(sw_pos_goal - mrv_com) > mrv_client_sim.reach:
        print('Fail')
        fail_reason = 'goal is out of reach'
        print('Reason: ' + fail_reason)
        mrv_client_sim.update_gt_state_trj()
        return fail_reason

      # Failure criteria 3: joint angle/vel/acc limit violation
      if self.check_joint_angle_limit_violation:
        q_mrv = self.get_mrv_state_angles(mrv_client_sim)

        v_mrv = mrv_client_sim.x[mrv_client_sim.pin_model.nq + mrv_client_sim.mrv_vidx:mrv_client_sim.pin_model.nq + mrv_client_sim.mrv_vidx + mrv_client_sim.mrv_pin_model.nv]
        if np.any(q_mrv[7:] > self.mrv_joint_angle_upper_limits) or np.any(q_mrv[7:] < self.mrv_joint_angle_lower_limits):
          print('Fail')
          fail_reason = 'joint angle limit violation'
          print('Reason: ' + fail_reason)
          mrv_client_sim.update_gt_state_trj()
          return fail_reason
        if np.any(np.abs(v_mrv[6:]) > self.mrv_joint_vel_limits):
          print('Fail')
          fail_reason = 'joint vel limit violation'
          print('Reason: ' + fail_reason)
          mrv_client_sim.update_gt_state_trj()
          return fail_reason
        if np.any(np.abs((v_mrv[6:] - self.prev_joint_vels)/self.dt) > self.mrv_joint_acc_limits):
          print('Fail')
          fail_reason = 'joint acc limit violation'
          print('Reason: ' + fail_reason)
          mrv_client_sim.update_gt_state_trj()
          return fail_reason

      # Failure criteria 4: plunge outside nozzle opening
      sw_pos_peg_wrt_nozzle = sw_nozzle_rmat.transpose()@(sw_peg_pos - sw_nozzle_pos)
      if sw_pos_peg_wrt_nozzle[2] > 0 and \
          sw_pos_peg_wrt_nozzle[2] < 0.05 and \
          np.linalg.norm(sw_pos_peg_wrt_nozzle[:2]) > self.nozzle_opening_rad + self.peg_rad:
        print('Fail')
        fail_reason = 'plunge outside nozzle opening'
        print('Reason: ' + fail_reason)
        mrv_client_sim.update_gt_state_trj()
        return fail_reason

    self.prev_joint_vels = np.copy(v_mrv[6:])
    mrv_client_sim.update_state_estimate(wrench_peg_peg_est)

    if not self.joint_control_enabled: 
      joint_cmd = None


    if use_contact_sim:
      mrv_client_sim.step(joint_cmd,None)
    else:
      mrv_client_sim.step(joint_cmd,wrench_peg_peg)

    

    if mrv_client_sim.sim_time > self.time_limit:
      print('Fail')
      fail_reason = 'Exceeded time limit'
      print('Reason: ' + fail_reason)
      mrv_client_sim.update_gt_state_trj()
      return fail_reason

    return 'nothing'
  
  def is_peg_close_to_nozzle(self,sw_rel_pos):
    return np.linalg.norm(sw_rel_pos[:2]) > self.nozzle_opening_rad + self.peg_rad + 0.001 or sw_rel_pos[2] < -self.dist_nozzle_opening_from_goal - 0.001
  
  def disable_joint_control(self):
    '''This has the effect of disabling the controller'''
    self.mrv_client_sim.set_pybullet_joint_torque(np.zeros(7))  
    self.joint_control_enabled = False
  
  def get_particle_positions(self):
    particle_positions , highest_weight_particle = self.mrv_client_sim.get_particle_filter_positions()
    return particle_positions, highest_weight_particle
  
  def get_ekf_estimate(self):
    return [],self.mrv_client_sim.get_ekf_estimate()

  def save(self, save_path):
    self.end_time = time.time()
    self.total_time = self.end_time - self.init_time

    self.mrv_client_sim.save(save_path)

    np.save(save_path + '/run_time.npy', self.total_time)
    np.save(save_path + '/solve_times.npy', self.solve_times)
    np.save(save_path + '/load_paths.npy', self.load_paths)
    np.save(save_path + '/load_path_weights.npy', self.load_path_weights)

    np.save(save_path + '/ref_ee_pos_world_real', self.ref_ee_pos_world_real)

    np.save(save_path + '/use_ekf.npy', self.use_ekf)

    np.save(save_path + '/joint_torque_meas_trj.npy', self.joint_torque_meas_trj)
    np.save(save_path + '/joint_torque_meas_d_trj.npy', self.joint_torque_meas_d_trj)

    np.save(save_path + '/ref_ee_pos_trj_world.npy', self.ref_ee_pos_trj_world)
    np.save(save_path + '/ref_ee_pos_trj.npy', self.ref_ee_pos_trj)
    np.save(save_path + '/ref_ee_rmat_trj.npy', self.ref_ee_rmat_trj)
    np.save(save_path + '/ref_ee_v_trj.npy', self.ref_ee_v_trj)
    np.save(save_path + '/ref_ee_w_trj.npy', self.ref_ee_w_trj)
    np.save(save_path + '/ref_ee_vdot_trj.npy', self.ref_ee_vdot_trj)
    np.save(save_path + '/ref_ee_wdot_trj.npy', self.ref_ee_wdot_trj)
    np.save(save_path + '/ref_joint_angles_trj.npy', self.ref_joint_angles_trj)
    np.save(save_path + '/ref_joint_vels_trj.npy', self.ref_joint_vels_trj)
    np.save(save_path + '/ref_joint_accs_trj.npy', self.ref_joint_accs_trj)
    np.save(save_path + '/ref_forces_trj.npy', self.ref_forces_trj)
    np.save(save_path + '/ref_x_trj.npy', self.ref_x_trj)
    np.save(save_path + '/ref_u_trj.npy', self.ref_u_trj)
    np.save(save_path + '/ref_t_trj.npy', self.ref_t_trj)

    np.save(save_path + '/flop_counts.npy', self.flop_counts)
    np.save(save_path + '/times.npy', self.times)
