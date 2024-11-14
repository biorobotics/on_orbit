import numpy as np
import pinocchio as pin
from mrv_client_sim import MRVClientSim
from scipy.spatial.transform import Rotation as R
import bisect
from actuation_cw_contact import ActuationModelCWContact
import time
from ref_traj_point import RefTrajPoint
from test_trajectories import TestTrajectories
import rospkg
import copy
import gc

# Controllers
from mrv_controller import MrvController
from resolved_accel import ResolvedAccel
from joint_space_tracking import JointSpaceTracking
from planar_admittance import PlanarAdmittance
from within_nozzle_admittance import WithinNozzleAdmittance

# Planner
from planning_scripts.ipopt_MPC_planner import MPCNozzleAlignPlanner


class IpoptMPC(object):
  def __init__(self, mrv_cv_urdf_file, mrv_urdf_file, pybullet_mrv_urdf_file, pybullet_cv_urdf_file, mrv_joint_angle_lower_limits, mrv_joint_angle_upper_limits, mrv_joint_vel_limits, 
               mrv_joint_acc_limits, mrv_joint_torque_limits, dt, cone_slope, clip_joint_commands,
              time_steps_between_measurements, cw_a, cw_mu, cw_orbit_dir, do_noisy_state_estimation, nozzle_opening_rad, 
               peg_rad, velocity_noise_ang_amp, time_limit, debug_with_test_traj, test_traj_id, lock_client, lock_mrv, probe_z_axis_plunge_velocity, use_variable_plunge_speed, 
               use_scheduled_gains,delta_pos, delta_rot, delta_v, initial_client_w, initial_mrv_w, rng,
                                                 dist_centering_waypoint_from_goal, use_cw=True , use_ekf = True):  
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
    self.ipopt_dt = 0.08
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
    self.internal_idx = 0
    self.plunging = False
    self.first_solve = True
    self.run_times = []

    self.init_time = time.time()

    self.use_ekf = use_ekf

    self.one_run = True

    '''Input URDFs:
    These are all simply passed direclty to MRVCLientSim
    mrv_cv_urdf_file - currently using robot_cv_detached.urdf. 
    mrv_urdf_file - currently using robot.urdf. 
    pybullet_mrv_urdf_file - currently using barebones_robot.urdf. 
    pybullet_cv_urdf_file - currently using cv.urdf.
    '''

    self.mrv_client_sim = MRVClientSim(self.mrv_cv_urdf_file, 
                                       self.mrv_urdf_file, 
                                       self.pybullet_mrv_urdf_file, 
                                       self.pybullet_cv_urdf_file, 
                                       self.mrv_joint_angle_lower_limits, 
                                       self.mrv_joint_angle_upper_limits, 
                                       self.mrv_joint_vel_limits, 
                                       self.mrv_joint_acc_limits, 
                                       self.mrv_joint_torque_limits, 
                                       self.dt, 
                                       self.cone_slope, 
                                       self.time_steps_between_measurements, 
                                       self.cw_a, 
                                       self.cw_mu, 
                                       self.cw_orbit_dir, 
                                       self.lock_client, 
                                       self.lock_mrv, 
                                       self.use_cw , 
                                       use_ekf= self.use_ekf)
    # print(self.mrv_cv_urdf_file)
    # quit()
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

    # Online planner
    '''
    This is an ipopt planner that will be called at each time step to generate a trajectory from the current state to the goal.
    '''
    initial_client_rmat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    control_cost_weight = 0.0001
    phase_lengths_sec = np.array([4,11.5 - (2*self.ipopt_dt) ,2*self.ipopt_dt,self.ipopt_dt])
    
    self.path = '/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit'
    mrv_joint_vel_lim_ipopt = 0.75*self.mrv_joint_vel_limits
    mrv_joint_acc_lim_ipopt = 0.75*self.mrv_joint_acc_limits
    # Print all the parameters
    print("Initializing MPCNozzleAlignPlanner with the following parameters:")
    print("URDF Path (robot_cv_detached):", self.path + '/urdf/robot_cv_detached.urdf')
    print("URDF Path (robot):", self.path + '/urdf/robot.urdf')
    print("dt:", self.ipopt_dt)
    print("Joint Angle Lower Limits:", self.mrv_joint_angle_lower_limits)
    print("Joint Angle Upper Limits:", self.mrv_joint_angle_upper_limits)
    print("Joint Torque Limits:", self.mrv_joint_torque_limits)
    print("Joint Velocity Limits:", mrv_joint_vel_lim_ipopt)
    print("Joint Acceleration Limits:", mrv_joint_acc_lim_ipopt)
    print("Control Cost Weight:", control_cost_weight)
    print("Phase Lengths (sec):", phase_lengths_sec)
    print("CW a:", self.cw_a)
    print("CW mu:", self.cw_mu)
    print("CW Orbit Direction:", self.cw_orbit_dir)
    print("Initial Client Rotation Matrix:", initial_client_rmat)
    print("Cone Slope:", self.cone_slope)
    print("Use CW:", self.use_cw)
    print("Meshes Path:", self.path + '/meshes/')
    # quit()

    self.elapsed_ipopt_steps = 0
    self.ipopt_planner = MPCNozzleAlignPlanner(self.path + '/urdf/robot_cv_detached.urdf', \
                                    self.path + '/urdf/robot.urdf', \
                                    self.ipopt_dt, \
                                    self.mrv_joint_angle_lower_limits, \
                                    self.mrv_joint_angle_upper_limits, \
                                    self.mrv_joint_torque_limits, \
                                    mrv_joint_vel_lim_ipopt, \
                                    mrv_joint_acc_lim_ipopt, \
                                    control_cost_weight, phase_lengths_sec, \
                                    self.cw_a, self.cw_mu, self.cw_orbit_dir, initial_client_rmat, \
                                    self.cone_slope, self.use_cw, self.path + '/meshes/')
    
    self.ref_ee_pos_trj_cum = []
    self.ref_ee_rmat_trj_cum = []
    self.ref_ee_v_trj_cum = []
    self.ref_ee_w_trj_cum = []
    self.ref_ee_vdot_trj_cum = []
    self.ref_ee_wdot_trj_cum = []
    self.ref_joint_angles_trj_cum = []
    self.ref_joint_vels_trj_cum = []
    self.ref_joint_accs_trj_cum = []
    self.ref_forces_trj_cum = []
    self.ref_x_trj_cum = []
    self.ref_u_trj_cum = []
    self.ref_t_trj_cum = []




    self.all_q_from_sim = []
    self.all_v_from_sim = []
    self.all_q_from_ipopt = []
    self.all_v_from_ipopt = []
    
    # Trajectory storage for saving and replay
    self.ref_trj_idx = 0
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

    # Controllers
    self.resolved_accel = ResolvedAccel(use_scheduled_gains)
    self.joint_space_tracking = JointSpaceTracking(self.mrv_client_sim)
    self.planar_admittance = PlanarAdmittance(use_scheduled_gains)
    self.within_nozzle_admittance = WithinNozzleAdmittance(probe_z_axis_plunge_velocity, use_variable_plunge_speed,use_scheduled_gains)

    '''Controller type:
    0 - Resolved acceleration with standard task-space admittance control
    1 - joint-space control only
    2 - Planar admitttance controller that generates it's own local "trajectory"
    3 - 3D admittance controller to be used when you're already within the nozzle and do not have a reference trajectory
    4 - Use controller 0 outside the nozzle and controller 3 within after alignement
    '''
    self.controller_type = 4

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
  
  # def get_initial_sw_client_rmat(self,delta_pos, delta_rot, delta_v, initial_client_w, initial_mrv_w, rng,
  #                                                dist_centering_waypoint_from_goal):
  #   self.reset_wrt_capture_box(delta_pos, delta_rot, delta_v, initial_client_w, initial_mrv_w, rng,
  #                                                dist_centering_waypoint_from_goal)
  #   mrv_client_sim = self.mrv_client_sim
  #   sw_client_rmat = R.from_quat(mrv_client_sim.x[mrv_client_sim.cv_qidx + 3:mrv_client_sim.cv_qidx + 7]).as_matrix()
  #   return sw_client_rmat
  
  def get_traj(self):
    return self.last_traj_point.p, self.last_traj_point.rmat

  def interp_x(self, x1, x2, alpha):
    x_interp = np.zeros_like(x1)
    x_interp[:self.pin_model.nq] = pin.interpolate(self.pin_model, x1[:self.pin_model.nq], x2[:self.pin_model.nq], alpha)
    x_interp[self.pin_model.nq:] = (1 - alpha)*x1[self.pin_model.nq:] + alpha*x2[self.pin_model.nq:]
    return x_interp

  def get_reference_from_ipopt(self, x0, elapsed_steps, impact_dyn):
    # Solve the optimization problem using the IPOPT planner
    # if not hasattr(self, 'prev_xs') :
    time_before_solve = time.time()
    xs, us, dts, phase_starts, solved, obj_value = self.ipopt_planner.plan(x0, elapsed_steps, max_iter=1500)
    self.run_times.append(time.time() - time_before_solve)

    # if not hasattr(self, 'prev_xs'):
    #     first_solve = True
    # else:
    #     first_solve = False

    # if not first_solve:
    #   solved = False

    if hasattr(self, 'prev_obj_value') and self.prev_obj_value is not None:
        if obj_value > 2.5 * self.prev_obj_value or obj_value > 1:
            print(f"Current objective value ({obj_value}) is signficantly higher({self.prev_obj_value}). Rejecting solution.")
            solved = False  # Reject the solution if the objective value increased too much


    if solved:

      self.prev_xs = xs
      self.prev_us = us
      self.prev_dts = dts
      self.prev_idx = 0
      self.prev_obj_value = obj_value

      idx0 = self.prev_idx
      idx1 = self.prev_idx + 1
      x0 = xs[idx0]
      x1 = xs[idx1]
      u0 = us[idx0]
      u1 = us[idx1]
      ipopt_dt = dts[0]
      
    else:
      print("IPOPT failed to solve the optimization problem.")
      # Check if previous solution exists
      if hasattr(self, 'prev_xs') and hasattr(self, 'prev_us'):
          # Advance the index by one
          self.prev_idx += 1
          if self.prev_idx >= len(self.prev_xs):
            self.prev_idx = len(self.prev_xs) - 1

          idx0 = self.prev_idx
          idx1 = self.prev_idx + 1

          if idx1 > len(self.prev_xs):
            idx1 = idx0
          
          x0 = self.prev_xs[idx0]
          print("x0:", x0)
          x1 = self.prev_xs[idx1]
          print("x1:", x1)
          u0 = self.prev_us[idx0]
          u1 = self.prev_us[idx1]
          ipopt_dt = self.prev_dts[0]
      else:
          raise Exception("No solution was generated.")

    # Calculate elapsed time and reference time steps
    elapsed_time = elapsed_steps * ipopt_dt
    num_controller_steps = int(ipopt_dt / self.dt)
    ref_ts = [elapsed_time + i * self.dt for i in range(num_controller_steps + 1)]


    # Interpolate between x0 and x1 and u0 and u1
    ref_xs = []
    ref_us = []
    ref_ts = []
    for i in range(num_controller_steps):
        t = i * self.dt
        alpha = t / ipopt_dt if ipopt_dt != 0 else 0
        x_interp = self.interp_x(x0, x1, alpha)
        u_interp = (1 - alpha) * u0 + alpha * u1
        ref_xs.append(x_interp)
        ref_us.append(u_interp)
        ref_ts.append(t)
        

    # Initialize the actuation model
    mrv_client_sim = self.mrv_client_sim
    self.pin_model = mrv_client_sim.pin_model
    self.pin_data = pin.Data(self.pin_model)

    actuation = ActuationModelCWContact(
        self.pin_model,
        self.mrv_client_sim.mrv_pin_model,
        self.cw_a,
        self.cw_mu,
        self.cw_orbit_dir,
        np.eye(3),
        self.use_cw
    )

    # Prepare lists to store the reference trajectories
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

    # Process each interpolated state
    self.q_from_ipopt = []
    self.v_from_ipopt = []
    for trj_idx, (x, u) in enumerate(zip(ref_xs, ref_us)):
        q = x[:self.pin_model.nq]
        v = x[self.pin_model.nq:]

        # Calculate joint torques and forces using the actuation model
        tau = actuation.calc(x, u)
        vdot = pin.aba(self.pin_model, self.pin_data, q, v, tau)

        # Update kinematics and frame placements
        pin.forwardKinematics(self.pin_model, self.pin_data, q, v, vdot)
        pin.updateFramePlacement(self.pin_model, self.pin_data, self.mrv_client_sim.peg_fid)
        pin.updateFramePlacement(self.pin_model, self.pin_data, self.mrv_client_sim.client_fid)

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

        # End-effector positions and rotations relative to client frame
        ee_frame_id = self.mrv_client_sim.peg_fid
        client_frame_id = self.mrv_client_sim.client_fid
        R3SO3_world_ee = self.pin_data.oMf[ee_frame_id]
        R3SO3_world_client = self.pin_data.oMf[client_frame_id]
        client_rmat = R3SO3_world_client.rotation
        client_pos = R3SO3_world_client.translation
        ee_pos_world = R3SO3_world_ee.translation
        ee_rmat_world = R3SO3_world_ee.rotation

        # Relative end-effector position and rotation matrix
        ref_ee_pos = client_rmat.T @ (ee_pos_world - client_pos)
        ref_ee_rmat = client_rmat.T @ ee_rmat_world

        # Relative velocities and accelerations
        ee_twist = pin.getFrameVelocity(self.pin_model, self.pin_data, ee_frame_id, pin.LOCAL_WORLD_ALIGNED).vector
        ee_acc = pin.getFrameAcceleration(self.pin_model, self.pin_data, ee_frame_id, pin.LOCAL_WORLD_ALIGNED).vector
        client_v = v[self.cv_vidx:self.cv_vidx + 3]
        client_w = v[self.cv_vidx + 3:self.cv_vidx + 6]
        client_vdot = vdot[self.cv_vidx:self.cv_vidx + 3]
        client_wdot = vdot[self.cv_vidx + 3:self.cv_vidx + 6]
        ref_ee_v = client_rmat.T @ ee_twist[:3] - client_v - np.cross(client_w, ref_ee_pos)
        ref_ee_w = client_rmat.T @ ee_twist[3:] - client_w
        ref_ee_vdot = client_rmat.T @ ee_acc[:3] - client_vdot - np.cross(client_w, ref_ee_v) - np.cross(client_wdot, ref_ee_pos)
        ref_ee_wdot = client_rmat.T @ ee_acc[3:] - client_wdot - np.cross(client_w, ref_ee_w)

        # Joint data and reference force
        mrv_qidx = self.mrv_client_sim.mrv_qidx
        mrv_vidx = self.mrv_client_sim.mrv_vidx
        mrv_nq = self.mrv_client_sim.mrv_nq
        mrv_nv = self.mrv_client_sim.mrv_nv
        ref_joint_angles = q[mrv_qidx + 7:mrv_qidx + mrv_nq]
        ref_joint_vels = v[mrv_vidx + 6:mrv_vidx + mrv_nv]
        ref_joint_accs = vdot[mrv_vidx + 6:mrv_vidx + mrv_nv]
        ref_force = u[-3:]

        # Append to trajectories
        ref_ee_pos_trj.append(ref_ee_pos)
        ref_ee_rmat_trj.append(ref_ee_rmat)
        ref_ee_v_trj.append(ref_ee_v)
        ref_ee_w_trj.append(ref_ee_w)
        ref_ee_vdot_trj.append(ref_ee_vdot)
        ref_ee_wdot_trj.append(ref_ee_wdot)
        ref_joint_angles_trj.append(ref_joint_angles)
        ref_joint_vels_trj.append(ref_joint_vels)
        ref_joint_accs_trj.append(ref_joint_accs)
        ref_forces_trj.append(client_rmat.T @ ref_force)

        self.q_from_ipopt.append(np.copy(q))
        self.v_from_ipopt.append(np.copy(v))


    return (
        ref_ee_pos_trj,
        ref_ee_rmat_trj,
        ref_ee_v_trj,
        ref_ee_w_trj,
        ref_ee_vdot_trj,
        ref_ee_wdot_trj,
        ref_joint_angles_trj,
        ref_joint_vels_trj,
        ref_joint_accs_trj,
        ref_forces_trj,
        ref_xs,
        ref_us,
        ref_ts
    )



  def reset_wrt_capture_box(self, delta_pos, delta_rot, delta_v, initial_client_w, initial_mrv_w, rng, dist_centering_waypoint_from_goal):
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

    # if load_paths is not None: 
    #   self.reset_trajectory_library(load_paths,weights)

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


    self.resolved_accel.reset_admittance_traj(self.mrv_client_sim, np.zeros(3),np.zeros(3))
    

    self.joint_space_tracking.reset_admittance_traj(self.mrv_client_sim)
    self.planar_admittance.reset_admittance_traj(self.mrv_client_sim)

    if self.controller_type == 3: 
      self.within_nozzle_admittance.reset_admittance_traj(self.mrv_client_sim) #call this once your inside the nozzle if self.controller_type == 4
  
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

    
    if self.ref_trj_idx >= len(self.ref_ee_pos_trj) and not self.plunging:
      if self.do_noisy_state_estimation:
        x0 = np.copy(self.mrv_client_sim.x_est)
      else:
        x0 = np.copy(self.mrv_client_sim.x)
      # with open('/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/x0.txt', 'w') as f:
      #   np.savetxt(f, x0)
      #   # np.savetxt(f, self.elapsed_ipopt_steps)
      #   print("elapsed_ipopt_steps", self.elapsed_ipopt_steps)
      #   quit()
      
      (
        ref_ee_pos_trj,
        ref_ee_rmat_trj,
        ref_ee_v_trj,
        ref_ee_w_trj,
        ref_ee_vdot_trj,
        ref_ee_wdot_trj,
        ref_joint_angles_trj,
        ref_joint_vels_trj,
        ref_joint_accs_trj,
        ref_forces_trj,
        ref_x_trj,
        ref_u_trj,
        ref_ts_trj,
      ) = self.get_reference_from_ipopt(x0, self.elapsed_ipopt_steps, impact_dyn=False)

      self.ref_trj_idx = 0
      self.elapsed_ipopt_steps += 1

      # # Store the udpdated trajectories
      self.ref_ee_pos_trj = ref_ee_pos_trj
      self.ref_ee_rmat_trj = ref_ee_rmat_trj
      self.ref_ee_v_trj = ref_ee_v_trj
      self.ref_ee_w_trj = ref_ee_w_trj
      self.ref_ee_vdot_trj = ref_ee_vdot_trj
      self.ref_ee_wdot_trj = ref_ee_wdot_trj
      self.ref_joint_angles_trj = ref_joint_angles_trj
      self.ref_joint_vels_trj = ref_joint_vels_trj
      self.ref_joint_accs_trj = ref_joint_accs_trj
      self.ref_forces_trj = ref_forces_trj
      self.ref_x_trj = ref_x_trj
      self.ref_u_trj = ref_u_trj
      self.ref_t_trj = ref_ts_trj

      # Update the lists that store the reference trajectories
      # self.ref_ee_pos_trj.extend(ref_ee_pos_trj)
      # self.ref_ee_rmat_trj.extend(ref_ee_rmat_trj)
      # self.ref_ee_v_trj.extend(ref_ee_v_trj)
      # self.ref_ee_w_trj.extend(ref_ee_w_trj)
      # self.ref_ee_vdot_trj.extend(ref_ee_vdot_trj)
      # self.ref_ee_wdot_trj.extend(ref_ee_wdot_trj)
      # self.ref_joint_angles_trj.extend(ref_joint_angles_trj)
      # self.ref_joint_vels_trj.extend(ref_joint_vels_trj)
      # self.ref_joint_accs_trj.extend(ref_joint_accs_trj)
      # self.ref_forces_trj.extend(ref_forces_trj)
      # self.ref_x_trj.extend(ref_x_trj)
      # self.ref_u_trj.extend(ref_u_trj)
      # self.ref_t_trj.extend(ref_ts_trj)

      # ref_forces = self.ref_forces_trj[self.ref_trj_idx]
      # ref_x = self.ref_x_trj[self.ref_trj_idx]
      # ref_u = self.ref_u_trj[self.ref_trj_idx]
      # ref_t = self.ref_t_trj[self.ref_trj_idx]


      ref_ee_pos = self.ref_ee_pos_trj[self.ref_trj_idx]
      ref_ee_rmat = self.ref_ee_rmat_trj[self.ref_trj_idx]
      ref_joint_angles = self.ref_joint_angles_trj[self.ref_trj_idx]

      ref_ee_v = self.ref_ee_v_trj[self.ref_trj_idx]
      ref_ee_w = self.ref_ee_w_trj[self.ref_trj_idx]

      ref_ee_vdot = self.ref_ee_vdot_trj[self.ref_trj_idx]
      ref_ee_wdot = self.ref_ee_wdot_trj[self.ref_trj_idx]

      ref_joint_vels = self.ref_joint_vels_trj[self.ref_trj_idx]
      ref_joint_accs = self.ref_joint_accs_trj[self.ref_trj_idx]

      ref_forces = self.ref_forces_trj[self.ref_trj_idx]
      ref_x = self.ref_x_trj[self.ref_trj_idx]
      ref_u = self.ref_u_trj[self.ref_trj_idx]
      ref_t = self.ref_t_trj[self.ref_trj_idx]

      target_wrench[:3] = sw_peg_rmat.T @ client_rmat_est @ self.ref_forces_trj[self.ref_trj_idx]
      self.ref_trj_idx += 1


    
    elif self.ref_trj_idx < len(self.ref_ee_pos_trj) and not self.plunging:
      print("Using reference trajectory")
      print("ref_ee_pos" , self.ref_ee_pos_trj[self.ref_trj_idx])
      ref_ee_pos = self.ref_ee_pos_trj[self.ref_trj_idx]
      ref_ee_rmat = self.ref_ee_rmat_trj[self.ref_trj_idx]
      ref_joint_angles = self.ref_joint_angles_trj[self.ref_trj_idx]

      ref_ee_v = self.ref_ee_v_trj[self.ref_trj_idx]
      ref_ee_w = self.ref_ee_w_trj[self.ref_trj_idx]

      ref_ee_vdot = self.ref_ee_vdot_trj[self.ref_trj_idx]
      ref_ee_wdot = self.ref_ee_wdot_trj[self.ref_trj_idx]

      ref_joint_vels = self.ref_joint_vels_trj[self.ref_trj_idx]
      ref_joint_accs = self.ref_joint_accs_trj[self.ref_trj_idx]

      ref_forces = self.ref_forces_trj[self.ref_trj_idx]
      ref_x = self.ref_x_trj[self.ref_trj_idx]
      ref_u = self.ref_u_trj[self.ref_trj_idx]
      ref_t = self.ref_t_trj[self.ref_trj_idx]

      # Target wrench is expressed in the peg frame
      target_wrench[:3] = sw_peg_rmat.T @ client_rmat_est @ self.ref_forces_trj[self.ref_trj_idx]
      self.ref_trj_idx += 1

      if self.one_run:
        self.one_run = False
      else:
        self.save('/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/11_11_24/charecterizing_time')
    
    
    else:
      # This is where we are after we start plunging 
      ref_ee_pos = self.ref_ee_pos_trj[-1]
      ref_ee_rmat = self.ref_ee_rmat_trj[-1]
      ref_joint_angles = self.ref_joint_angles_trj[-1]

      ref_ee_v = np.zeros(3)
      ref_ee_w = np.zeros(3)

      ref_ee_vdot = np.zeros(3)
      ref_ee_wdot = np.zeros(3)

      ref_joint_vels = np.zeros(self.num_rotary)
      ref_joint_accs = np.zeros(self.num_rotary)

      # Target wrench remains the same as the last
      target_wrench[:3] = sw_peg_rmat.T @ client_rmat_est @ self.ref_forces_trj[-1]

      ref_forces = self.ref_forces_trj[-1]
      ref_x = self.ref_x_trj[-1]
      ref_t = self.ref_t_trj[-1] + self.dt


    # Update the lists that store the reference trajectories
    self.ref_ee_pos_trj_cum.append(ref_ee_pos)
    self.ref_ee_rmat_trj_cum.append(ref_ee_rmat)
    self.ref_ee_v_trj_cum.append(ref_ee_v)
    self.ref_ee_w_trj_cum.append(ref_ee_w)
    self.ref_ee_vdot_trj_cum.append(ref_ee_vdot)
    self.ref_ee_wdot_trj_cum.append(ref_ee_wdot)
    self.ref_joint_angles_trj_cum.append(ref_joint_angles)
    self.ref_joint_vels_trj_cum.append(ref_joint_vels)
    self.ref_joint_accs_trj_cum.append(ref_joint_accs)
    self.ref_forces_trj_cum.append(ref_forces)
    self.ref_x_trj_cum.append(ref_x)
    self.ref_u_trj_cum.append(ref_u)
    self.ref_t_trj_cum.append(ref_t)
    
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
    if self.controller_type == 0:
      #print(f'ref_traj_point: {ref_traj_point}')
      joint_acc_cmd = self.resolved_accel.compute_control(ref_traj_point, mrv_client_sim, wrench_peg_peg, self.dt, mrv_config, mrv_config_dot)

    # TODO: This is where the control step is actually implemented, we need to make sure that this is using the upate from MPC and not
    elif self.controller_type == 1:
      # We don't plan to use joint-space tracking again, so I'm not fixing this issue at this time.
      raise Exception("Joint space tracking is using the incorrect Jacobian and wrench transformations. Fix before using")
      Jwrist = pin.getFrameJacobian(self.mrv_pin_model, self.mrv_pin_data, self.mrv_wrist_fid, pin.ReferenceFrame.LOCAL)
      joint_acc_cmd = self.joint_space_tracking.compute_control(ref_traj_point, wrench_peg_peg, self.dt, mrv_config, mrv_config_dot, Jwrist)
    elif self.controller_type == 2:
      joint_acc_cmd = self.planar_admittance.compute_control(ref_traj_point, mrv_client_sim, wrench_peg_peg, self.dt, mrv_config, mrv_config_dot)
    elif self.controller_type == 3: 
      joint_acc_cmd = self.within_nozzle_admittance.compute_control(ref_traj_point, mrv_client_sim, wrench_peg_peg, self.dt, mrv_config, mrv_config_dot)
    elif self.controller_type == 4: 
      # TODO: Lets turn this on only when we are some distance inside the nozzle
      # if mrv_client_sim.is_peg_past_nozzle_opening():         
      #    if not self.within_nozzle_admittance.admittance_traj_reset:
      #      print("Resetting admittance trajectory")
      #      self.within_nozzle_admittance.reset_admittance_traj(mrv_client_sim)

      #    joint_acc_cmd = self.within_nozzle_admittance.compute_control(ref_traj_point, mrv_client_sim, wrench_peg_peg, self.dt, mrv_config, mrv_config_dot)
      # else:
      #   joint_acc_cmd = self.resolved_accel.compute_control(ref_traj_point, mrv_client_sim, wrench_peg_peg, self.dt, mrv_config, mrv_config_dot)
      if mrv_client_sim.dist_to_throat_opening() < 0.16:
        if not self.within_nozzle_admittance.admittance_traj_reset:
          print("Resetting admittance trajectory")
          self.within_nozzle_admittance.reset_admittance_traj(mrv_client_sim)
        
        self.plunging = True
        print("We plunging babbbbby!")
        print("Distance to throat opening: ", mrv_client_sim.dist_to_throat_opening())
        joint_acc_cmd = self.within_nozzle_admittance.compute_control(ref_traj_point, mrv_client_sim, wrench_peg_peg, self.dt, mrv_config, mrv_config_dot)
      else:
        joint_acc_cmd = self.resolved_accel.compute_control(ref_traj_point, mrv_client_sim, wrench_peg_peg, self.dt, mrv_config, mrv_config_dot)
        self.internal_idx += 1
        print("We are at the following time step within the trajectory: ", self.internal_idx)
        print("Distance to throat opening: ", mrv_client_sim.dist_to_throat_opening()) 
    else:
      raise Exception('Invalid controller_type')
    
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

    if not self.joint_control_enabled: 
      joint_cmd = None

    if use_contact_sim:
      q_from_sim = self.mrv_client_sim.x[:self.pin_model.nq]
      v_from_sim = self.mrv_client_sim.x[self.pin_model.nq:]
      # Extract components from IPOPT output
      q_from_ipopt = self.q_from_ipopt
      v_from_ipopt = self.v_from_ipopt

      # Save all the states from simulation and from IPOPT for comparison
      # self.all_q_from_sim.append(np.copy(q_from_sim))
      # self.all_v_from_sim.append(np.copy(v_from_sim))
      # self.all_q_from_ipopt.append(q_from_ipopt[self.ref_trj_idx-1])
      # self.all_v_from_ipopt.append(v_from_ipopt[self.ref_trj_idx-1])
      print('Called step')
      mrv_client_sim.step(joint_cmd,None)
    else:
      mrv_client_sim.step(joint_cmd,wrench_peg_peg)

    mrv_client_sim.update_state_estimate(wrench_peg_peg_est)

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

    np.save(save_path + '/ipopt_qs', self.all_q_from_ipopt)
    np.save(save_path + '/ipopt_vs', self.all_v_from_ipopt)
    np.save(save_path + '/sim_qs', self.all_q_from_sim)
    np.save(save_path + '/sim_vs', self.all_v_from_sim)

    np.save(save_path + '/use_ekf.npy', self.use_ekf)

    np.save(save_path + '/joint_torque_meas_trj.npy', self.joint_torque_meas_trj)
    np.save(save_path + '/joint_torque_meas_d_trj.npy', self.joint_torque_meas_d_trj)

    np.save(save_path + '/ref_ee_pos_trj_world.npy', self.ref_ee_pos_trj_world)
    np.save(save_path + '/ref_ee_pos_trj.npy', self.ref_ee_pos_trj_cum)
    np.save(save_path + '/ref_ee_rmat_trj.npy', self.ref_ee_rmat_trj_cum)
    np.save(save_path + '/ref_ee_v_trj.npy', self.ref_ee_v_trj_cum)
    np.save(save_path + '/ref_ee_w_trj.npy', self.ref_ee_w_trj_cum)
    np.save(save_path + '/ref_ee_vdot_trj.npy', self.ref_ee_vdot_trj_cum)
    np.save(save_path + '/ref_ee_wdot_trj.npy', self.ref_ee_wdot_trj_cum)
    np.save(save_path + '/ref_joint_angles_trj.npy', self.ref_joint_angles_trj_cum)
    np.save(save_path + '/ref_joint_vels_trj.npy', self.ref_joint_vels_trj_cum)
    np.save(save_path + '/ref_joint_accs_trj.npy', self.ref_joint_accs_trj_cum)
    np.save(save_path + '/ref_forces_trj.npy', self.ref_forces_trj_cum)
    np.save(save_path + '/ref_x_trj.npy', self.ref_x_trj_cum)
    np.save(save_path + '/ref_u_trj.npy', self.ref_u_trj_cum)
    np.save(save_path + '/ref_t_trj.npy', self.ref_t_trj_cum)

    np.save(save_path + '/flop_counts.npy', self.flop_counts)
    np.save(save_path + '/times.npy', self.times)

    np.save(save_path + '/MPC_run_time.npy', self.run_times)
