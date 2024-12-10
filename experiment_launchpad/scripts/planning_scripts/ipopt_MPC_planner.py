#!/usr/bin/env python3
from implicit_midpoint_model import export_implicit_midpoint_model
# from implicit_midpoint_mk_model import export_implicit_midpoint_model
import casadi as ca
import pinocchio as pin
from pinocchio import casadi as cpin
from rp_conversion import *
from pinocchio.robot_wrapper import RobotWrapper

import rospkg
import numpy as np
from scipy.spatial.transform import Rotation as R
import time


from inf_def import inf

from stay_below_nozzle_constraint import export_stay_below_nozzle_constraint
from relative_frame_velocity_constraint import export_relative_velocity_constraint
from align_frames_constraint import export_align_frames_constraint
from relative_frame_acceleration_constraint import export_relative_acceleration_constraint
from stay_in_cone_approx_planes_constraint import export_stay_in_cone_approx_planes_constraint
from peg_direction_constraint import export_peg_direction_constraint
from contact_force_direction_constraint import export_contact_force_direction_constraint
from stay_in_hole_constraint import export_stay_in_hole_constraint
from contact_force_direction_in_hole_constraint import export_contact_force_direction_in_hole_constraint
from insertion_constraint import export_insertion_constraint
from joint_acc_constraint import export_joint_acc_constraint
from zero_joint_vel_constraint import zero_joint_vel_constraint

import os
import time

from ipopt_cpp.build.code_gen_caller_bindings import IPOPTContactCodeGenCaller, IPOPTContactSolver
import cyipopt

from scipy.sparse import coo_array

from perf_utils import start_perf_proc, stop_perf_proc
import subprocess

from floating_base_ik_problem import FloatingBaseIKProblem

class MPCNozzleAlignPlanner(object):
  def __init__(self, urdf_file, 
               mrv_urdf_file, dt, 
               joint_angle_lower_limits, joint_angle_upper_limits, 
               joint_torque_limits, joint_vel_limits, joint_acc_limits, 
               control_cost_weight, phase_lengths_sec, cw_a, cw_mu, cw_orbit_dir,
                initial_client_rmat, cone_slope, use_cw, meshdir):
    
    '''
    This is the constuctor for the MPC planner using ipopt. It initializes the planner with the necessary parameters
    and loads the URDF files for the MRV and Client Vehicle. Plan can then be called to return the solution to the optimization problem. This
    planner takes the end-effector to an intermediary waypoint pose that is aligned with the goal frame. From there, 
    a seperate plunge controller is used to insert the end-effector into the throat of the nozzle.


    Inputs:
      urdf_file: string 
      path to the URDF file representing the MRV and Client Vechicle

      mrv_urdf_file: string 
      path to the URDF file representing the MRV

      dt:int 
      time step

      joint_angle_lower_limits: array
      lower joint angle limits of FREND Arm on MRV

      joint_angle_upper_limits: array
      upper joint angle limits of FREND Arm on MRV

      joint_torque_limits: array
      joint torque limits of FREND Arm on MRV

      joint_vel_limits: array
      joint velocity limits of FREND Arm on MRV

      joint_acc_limits: array
      joint acceleration limits of FREND Arm on MRV

      control_cost_weight: float
      weight for the control cost matrix

      phase_lengths_sec: array
      lengths of each phase in secconds for the first run of the planner

      cw_a: int
      Clohessy-Wiltshire parameter a

      cw_mu: int
      Clohessy-Wiltshire parameter mu

      cw_orbit_dir: string, 'x' 'y' 'z'
      Clohessy-Wiltshire parameter orbit direction

      initial_client_rmat: matrix
      initial rotation matrix of the client vehicle

      cone_slope: int
      slope of the cone approximating the nozzle TODO: This is not a good estimation of the nozzle and should be improved
      this will involve editing both the URDF and the relevant constraints / planner code

      use_cw: boolean
      whether to use the Clohessy-Wiltshire approximation

      meshdir: string
      directory containing the mesh files for the URDF
    '''
    self.first_call = True
    self.dt = dt
    self.first_call = True
    self.phase_lengths_sec = phase_lengths_sec

    self.pin_model = RobotWrapper.BuildFromURDF(urdf_file).model
    self.pin_data = pin.Data(self.pin_model)
    self.pin_model.gravity.setZero()

    self.cpin_model = cpin.Model(self.pin_model)

    self.pin_geom = pin.buildGeomFromUrdf(self.pin_model, urdf_file, meshdir, pin.GeometryType.COLLISION)
    tip_geom_id = self.pin_geom.getGeometryId('ee_peg_0')
    for i in range(8):
      nozzle_geom_id = self.pin_geom.getGeometryId('nozzle_geom' + str(i) + '_0')
      self.pin_geom.addCollisionPair(pin.CollisionPair(tip_geom_id, nozzle_geom_id))
    self.geom_data = pin.GeometryData(self.pin_geom)

    self.cv_jidx = self.pin_model.getJointId('world_to_client')
    self.cv_qidx = self.pin_model.idx_qs[self.cv_jidx]
    self.cv_vidx = self.pin_model.idx_vs[self.cv_jidx]

    self.mrv_jidx = self.pin_model.getJointId('world_to_base')
    self.mrv_qidx = self.pin_model.idx_qs[self.mrv_jidx]
    self.mrv_vidx = self.pin_model.idx_vs[self.mrv_jidx]

    self.mrv_nq = self.pin_model.nq - 7
    self.mrv_nv = self.pin_model.nv - 6

    self.client_mass = self.pin_model.inertias[self.cv_jidx].mass
    self.client_inertia = self.pin_model.inertias[self.cv_jidx].matrix()[3:, 3:]
    self.client_inertia_inv = np.linalg.inv(self.client_inertia)

    self.ipopt_nx = 2*self.pin_model.nv # number of state variables

    self.tip_fid = self.pin_model.getFrameId('ee_tip')
    self.goal_fid = self.pin_model.getFrameId('goal')
    self.hole_fid = self.pin_model.getFrameId('hole')
    self.cone_vertex_fid = self.pin_model.getFrameId('cone_vertex')
    self.nozzle_fid = self.pin_model.getFrameId('nozzle')

    pin.forwardKinematics(self.pin_model, self.pin_data, pin.neutral(self.pin_model))
    pin.updateFramePlacements(self.pin_model, self.pin_data)
    self.hole_to_goal = self.pin_data.oMf[self.hole_fid].actInv(self.pin_data.oMf[self.goal_fid].translation)[2]
    self.client_to_nozzle = self.pin_data.oMf[self.nozzle_fid].translation[2]
    
    self.cone_slope = cone_slope

    self.num_phases = 4
    self.num_phases_without_dummy = 4 # Dummy phases are the nozzle impact, hole impact, and insertion phases

    self.joint_angle_lower_limits = np.array(joint_angle_lower_limits)
    self.joint_angle_upper_limits = np.array(joint_angle_upper_limits)
    self.joint_vel_limits = np.array(joint_vel_limits)
    self.joint_acc_limits = np.array(joint_acc_limits)
    self.joint_torque_limits = np.array(joint_torque_limits)

    self.num_rotary = self.pin_model.nv - 12
    self.nu = self.num_rotary + 3

    self.mrv_pin_model = RobotWrapper.BuildFromURDF(mrv_urdf_file, root_joint=pin.JointModelFreeFlyer()).model
    self.mrv_pin_model.gravity.setZero()
    self.mrv_pin_data = pin.Data(self.mrv_pin_model)
    self.mrv_cpin_model = cpin.Model(self.mrv_pin_model)

    joint_torque_weights = np.ones(7)
    contact_forces_weights = [1,1,1]
    self.R_mat = np.diag(control_cost_weight*np.concatenate((joint_torque_weights,contact_forces_weights)))

    self.xlb = -np.inf*np.ones(self.ipopt_nx)
    self.xub = np.inf*np.ones(self.ipopt_nx)

    revolute_idx = 0
    for j in range(1, self.pin_model.njoints):
      # We're assuming only revolute joints and floating joints
      if self.pin_model.joints[j].nv == 1:
        self.xlb[self.pin_model.idx_vs[j]] = self.joint_angle_lower_limits[revolute_idx] + 1e-3
        self.xub[self.pin_model.idx_vs[j]] = self.joint_angle_upper_limits[revolute_idx] - 1e-3

        self.xlb[self.pin_model.nv + self.pin_model.idx_vs[j]] = -self.joint_vel_limits[revolute_idx] + 1e-3
        self.xub[self.pin_model.nv + self.pin_model.idx_vs[j]] = self.joint_vel_limits[revolute_idx] - 1e-3

        revolute_idx += 1

    self.nu = self.num_rotary + 3
    self.u_lb = np.concatenate((-self.joint_torque_limits + 1e-3, [-10., -10., -10.]))
    self.u_ub = np.concatenate((self.joint_torque_limits - 1e-3, [10., 10., 10.]))

    self.optimize_codegen = False
    self.add_debug_flags = False

    self.cw_a = cw_a
    self.cw_mu = cw_mu
    self.cw_n = np.sqrt(cw_mu/cw_a**3)
    self.cw_orbit_dir = cw_orbit_dir
    self.initial_client_rmat = initial_client_rmat
    self.use_cw = use_cw

    self.cw_xproj = initial_client_rmat[:, 2]
    if self.cw_orbit_dir == 'x':
      self.cw_yproj = initial_client_rmat[:, 0]
    else:
      self.cw_yproj = initial_client_rmat[:, 1]
    self.cw_zproj = np.cross(self.cw_xproj, self.cw_yproj)

    self.apply_joint_acc_constraint = True
    # New parts for MPC integration
    self.current_phase = 0
    self.elapsed_steps = 0

  def interp_x(self, x1, x2, alpha):
    x_interp = np.zeros_like(x1)
    x_interp[:self.pin_model.nq] = pin.interpolate(self.pin_model, x1[:self.pin_model.nq], x2[:self.pin_model.nq], alpha)
    x_interp[self.pin_model.nq:] = (1 - alpha)*x1[self.pin_model.nq:] + alpha*x2[self.pin_model.nq:]
    return x_interp

  def generate_constraint_code_and_append_bounds(self, h_expr, h_lb, h_ub, q_sym, phase_lengths, phase_idx, ipopt_cl, ipopt_cu, folder_name, regen=True):
    '''
    Inputs: 
      h_expr: constraints?
      h_lb: constraint lower bounds?
      h_ub: constraint upper bounds?
      q_sim: symbolic CasADi variables representing input?
      phase_lengths: time steps in each phase 
      phase_idx: phase index
      ipopt_cl: constraint lower bounds
      ipopt_cu: constraint upper bounds
      folder_name: 
      regen:
    '''
    if regen:
      function_name = 'constraint_phase_' + str(phase_idx)
      constraint_fn = ca.Function(function_name, \
                                  [q_sym], \
                                  [ca.densify(h_expr)])
      filename = function_name + '.c'
      constraint_fn.generate(filename)
      if self.optimize_codegen:
        if self.add_debug_flags:
          subprocess.run(['gcc', '-fPIC', '-g', '-O2', '-shared', filename, '-o', filename[:-2] + '.so'])
        else:
          subprocess.run(['gcc', '-fPIC', '-O2', '-shared', filename, '-o', filename[:-2] + '.so'])
      else:
        if self.add_debug_flags:
          subprocess.run(['gcc', '-fPIC', '-g', '-shared', filename, '-o', filename[:-2] + '.so'])
        else:
          subprocess.run(['gcc', '-fPIC', '-shared', filename, '-o', filename[:-2] + '.so'])

      # Move generated files to the folder corresponding to this plane index
      os.replace(filename, folder_name + '/' + filename)
      os.replace(filename[:-2] + '.so', folder_name + '/' + filename[:-2] + '.so')

      function_name = 'constraint_jacobian_phase_' + str(phase_idx)
      constraint_jacobian_fn = ca.Function(function_name, [q_sym], [ca.jacobian(constraint_fn(q_sym), q_sym)])
      filename = function_name + '.c'
      constraint_jacobian_fn.generate(filename)
      if self.optimize_codegen:
        if self.add_debug_flags:
          subprocess.run(['gcc', '-fPIC', '-g', '-O2', '-shared', filename, '-o', filename[:-2] + '.so'])
        else:
          subprocess.run(['gcc', '-fPIC', '-O2', '-shared', filename, '-o', filename[:-2] + '.so'])
      else:
        if self.add_debug_flags:
          subprocess.run(['gcc', '-fPIC', '-g', '-shared', filename, '-o', filename[:-2] + '.so'])
        else:
          subprocess.run(['gcc', '-fPIC', '-shared', filename, '-o', filename[:-2] + '.so'])

      # Move generated file to the folder corresponding to this plane index
      os.replace(filename, folder_name + '/' + filename)
      os.replace(filename[:-2] + '.so', folder_name + '/' + filename[:-2] + '.so')

    ipopt_cl.append(np.tile(h_lb, phase_lengths[phase_idx]))
    ipopt_cu.append(np.tile(h_ub, phase_lengths[phase_idx]))

  def generate_cost_code(self, cost_input, phase_idx, folder_name, regen=True):
    if regen:
      u = cost_input[self.ipopt_nx:]

      cost_expr = 0.5*u.T @ self.R_mat @ u

      function_name = 'cost_phase_' + str(phase_idx)
      cost_fn = ca.Function(function_name, \
                            [cost_input], \
                            [ca.densify(cost_expr)])
      filename = function_name + '.c'
      cost_fn.generate(filename)
      if self.optimize_codegen:
        if self.add_debug_flags:
          subprocess.run(['gcc', '-fPIC', '-g', '-O2', '-shared', filename, '-o', filename[:-2] + '.so'])
        else:
          subprocess.run(['gcc', '-fPIC', '-O2', '-shared', filename, '-o', filename[:-2] + '.so'])
      else:
        if self.add_debug_flags:
          subprocess.run(['gcc', '-fPIC', '-g', '-shared', filename, '-o', filename[:-2] + '.so'])
        else:
          subprocess.run(['gcc', '-fPIC', '-shared', filename, '-o', filename[:-2] + '.so'])

      # Move generated files to the folder corresponding to this plane index
      os.replace(filename, folder_name + '/' + filename)
      os.replace(filename[:-2] + '.so', folder_name + '/' + filename[:-2] + '.so')

      function_name = 'cost_grad_phase_' + str(phase_idx)
      cost_grad_fn = ca.Function(function_name, [cost_input], [ca.densify(ca.jacobian(cost_fn(cost_input), cost_input).reshape((-1, 1)))])
      filename = function_name + '.c'
      cost_grad_fn.generate(function_name)
      if self.optimize_codegen:
        if self.add_debug_flags:
          subprocess.run(['gcc', '-fPIC', '-g', '-O2', '-shared', filename, '-o', filename[:-2] + '.so'])
        else:
          subprocess.run(['gcc', '-fPIC', '-O2', '-shared', filename, '-o', filename[:-2] + '.so'])
      else:
        if self.add_debug_flags:
          subprocess.run(['gcc', '-fPIC', '-g', '-shared', filename, '-o', filename[:-2] + '.so'])
        else:
          subprocess.run(['gcc', '-fPIC', '-shared', filename, '-o', filename[:-2] + '.so'])

      # Move generated files to the folder corresponding to this plane index
      os.replace(filename, folder_name + '/' + filename)
      os.replace(filename[:-2] + '.so', folder_name + '/' + filename[:-2] + '.so')

      cost_hess_tril = ca.tril(ca.jacobian(cost_grad_fn(cost_input), cost_input) + ca.diag(ca.vertcat(ca.SX.ones(self.ipopt_nx), ca.SX.zeros(self.nu))))
      # cost_hess_tril = ca.tril(ca.jacobian(cost_grad_fn(cost_input), cost_input))
      function_name = 'cost_hess_phase_' + str(phase_idx)
      cost_hess_fn = ca.Function(function_name, [cost_input], [cost_hess_tril])
      filename = function_name + '.c'
      cost_hess_fn.generate(filename)
      if self.optimize_codegen:
        if self.add_debug_flags:
          subprocess.run(['gcc', '-fPIC', '-g', '-O2', '-shared', filename, '-o', filename[:-2] + '.so'])
        else:
          subprocess.run(['gcc', '-fPIC', '-O2', '-shared', filename, '-o', filename[:-2] + '.so'])
      else:
        if self.add_debug_flags:
          subprocess.run(['gcc', '-fPIC', '-g', '-shared', filename, '-o', filename[:-2] + '.so'])
        else:
          subprocess.run(['gcc', '-fPIC', '-shared', filename, '-o', filename[:-2] + '.so'])

      # Move generated files to the folder corresponding to this plane index
      os.replace(filename, folder_name + '/' + filename)
      os.replace(filename[:-2] + '.so', folder_name + '/' + filename[:-2] + '.so')


  def plan(self, x0, elapsed_steps, save_path=None, max_iter=250, count_total_flop=False, stop_after_iter=-1):
    '''
    This is the function call for the MPC planner. It generates the constraints and costs for the optimization problem and solves it using IPOPT.
    It then returns the solution to the optimization problem and the lengths of the remaining phases.

    Inputs:
      x0: array 
      current state of the system

      elapsed_steps: int
      number of steps elapsed

      save_path: string , optional
      path to save the results

      max_iter: int, optional
      maximum number of iterations for the MPC solver

      count_total_flop: boolean, optional
      weather to count the total number of flops

      stop_after_iter: int, optional
      stop after a certain number of iterations
    
    Outputs:
      xs: Ipopt state solution
      us: Ipopt control solution
      phase_lengths: lengths of each phase
      phase_starts: start of each phase
      success: boolean indicating if the optimization was successful


    '''
    regen=True

    ## Adjust Phase Lengths Based on Elapsed Steps ##
    self.elapsed_steps = elapsed_steps
    phase_lengths_sec = np.array([self.phase_lengths_sec[0], \
                                  self.phase_lengths_sec[1],\
                                  self.phase_lengths_sec[2],\
                                  self.phase_lengths_sec[3]\
                                  ])
    phase_lengths = np.divide(phase_lengths_sec, self.dt).astype(int)
    phase_starts = np.concatenate([[0], np.cumsum(phase_lengths)])

    # Determine the current phase
    current_phase = None
    steps_into_phase = 0 
    for i in range(len(phase_lengths)):
      if phase_starts[i] <= elapsed_steps < phase_starts[i + 1]:
        current_phase = i
        steps_into_phase = elapsed_steps - phase_starts[i]
        break
    if current_phase is None:
        print("Elapsed steps exceed total phase lengths")
        return x0, np.zeros((0, self.nu)), self.dt*np.ones(self.num_phases), phase_starts, False
        
      
    # Determine the remaining phases and their lengths
    remaining_steps_in_phase = phase_lengths[current_phase] - steps_into_phase
    remaining_phase_lengths = phase_lengths[current_phase:].copy()
    remaining_phase_lengths[0] = remaining_steps_in_phase
    remaining_phase_starts = np.concatenate([[0], np.cumsum(remaining_phase_lengths)]).flatten()
    total_remaining_steps = np.sum(remaining_phase_lengths)

    # Determine direction where we should make contact. For now, ignore client z angular velocity
    v0 = x0[self.pin_model.nq:]
    initial_client_w = v0[self.cv_vidx + 3:self.cv_vidx + 6]
    initial_client_w_xy = np.copy(initial_client_w)[:2]

    theta = np.arctan2(initial_client_w_xy[1], initial_client_w_xy[0]) - np.pi/2
    if theta < 0:
      theta += 2*np.pi
    plane_idx = int(theta/(np.pi/4))

    folder_name = 'codegen_for_ipopt/plane_idx' + str(plane_idx) + '/'
    if regen:
      # Generate folder corresponding to this plane index
      if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Initialize from linear interpolation
    pin.forwardKinematics(self.pin_model, self.pin_data, x0[:self.pin_model.nq])
    pin.updateFramePlacement(self.pin_model, self.pin_data, self.nozzle_fid)
    mrv_q0 = x0[self.mrv_qidx:self.mrv_qidx + self.mrv_nq]
    ik_problem = FloatingBaseIKProblem(self.mrv_pin_model, \
                                       mrv_q0, \
                                       self.joint_angle_lower_limits, \
                                       self.joint_angle_upper_limits, \
                                       self.pin_data.oMf[self.nozzle_fid].translation, \
                                       self.pin_data.oMf[self.nozzle_fid].rotation)
    mrv_qf, solved = ik_problem.solve()

    print("MRV QF:", mrv_qf)

    if not solved:
      # No solution for initial guess
      return np.zeros((0, self.pin_model.nq + self.pin_model.nv)), np.zeros((0, self.nu)), self.dt*np.ones(self.num_phases), np.zeros(0, dtype=np.int64), False
    xf = np.copy(x0)
    xf[self.mrv_qidx:self.mrv_qidx + self.mrv_nq] = mrv_qf
    print("Initial guess solved")

    
    steps = total_remaining_steps - 1
    
    init_xs = np.array([from_pin(self.interp_x(x0, xf, step/steps), self.pin_model) for step in range(steps + 1)])
    init_us = np.zeros((steps + 1, self.nu))

    # Variable and constraint bounds
    ipopt_lb = []
    ipopt_ub = []
    ipopt_cl = []
    ipopt_cu = []
    constraint_sizes = []

    # Symbolic casadi variables
    x_sym = ca.SX.sym('x', self.ipopt_nx)
    u_sym = ca.SX.sym('u', self.nu)
    xnext_sym = ca.SX.sym('xnext', self.ipopt_nx)
    unext_sym = ca.SX.sym('unext', self.nu)
    dyn_input = ca.vertcat(x_sym, u_sym, xnext_sym, unext_sym)
    cost_input = ca.vertcat(x_sym, u_sym)
    constraint_input = dyn_input

    dynamics_expr = export_implicit_midpoint_model(self.cpin_model, 
                                                   plane_idx, 
                                                   dyn_input, 
                                                   self.dt, 
                                                   self.use_cw, 
                                                   self.mrv_cpin_model, 
                                                   self.initial_client_rmat, 
                                                   self.cw_mu, self.cw_a, 
                                                   self.cw_orbit_dir)
    dynamics_lb = np.zeros(self.ipopt_nx)
    dynamics_ub = np.zeros(self.ipopt_nx)


    # Loop over remaining phases"
    for phase_offset in range(len(remaining_phase_lengths)):
        
        state_fixed = False
        phase_idx = current_phase + phase_offset
        relative_phase_idx = phase_offset
        current_phase_steps = remaining_phase_lengths[phase_offset]

        x_lb = np.copy(self.xlb)
        x_ub = np.copy(self.xub)
        u_lb = np.copy(self.u_lb)
        u_ub = np.copy(self.u_ub)

        #### PHASE 0: STAY-BELOW-NOZZLE-OPENING-PLANE ####
        if phase_idx == 0:
            # No contact forces in this phase
            u_lb = np.copy(self.u_lb)
            u_ub = np.copy(self.u_ub)
            u_lb[self.num_rotary:self.num_rotary + 3] = 0
            u_ub[self.num_rotary:self.num_rotary + 3] = 0

            # Nonlinear constraints:
            # (1) stay below nozzle plane
            h1_expr, h1_lb, h1_ub = export_stay_below_nozzle_constraint(self.cpin_model, constraint_input)
            h_expr = ca.vertcat(dynamics_expr, h1_expr)
            h_lb = np.concatenate((dynamics_lb, h1_lb))
            h_ub = np.concatenate((dynamics_ub, h1_ub))

            if self.apply_joint_acc_constraint: 
                h_expr, h_lb, h_ub = MPCNozzleAlignPlanner.append_joint_acc_constraints(h_expr, h_lb, h_ub, self.cpin_model, 
                                                                                        constraint_input, self.joint_acc_limits, 
                                                                                        self.dt)

            constraint_sizes.append(h_expr.shape[0])
            self.generate_constraint_code_and_append_bounds(h_expr, h_lb, h_ub, constraint_input, 
                                                            remaining_phase_lengths, relative_phase_idx, ipopt_cl, 
                                                            ipopt_cu, folder_name, regen)
            


            # Quadratic control cost
            self.generate_cost_code(cost_input, relative_phase_idx, folder_name, regen)

            # State bounds
            single_step_lb = np.concatenate((x_lb, u_lb))
            single_step_ub = np.concatenate((x_ub, u_ub))
            # ipopt_lb.append(np.tile(single_step_lb, (current_phase_steps, 1)))
            # ipopt_ub.append(np.tile(single_step_ub, (current_phase_steps, 1)))
            ipopt_lb.append(np.tile(single_step_lb, remaining_phase_lengths[phase_offset]))
            ipopt_ub.append(np.tile(single_step_ub, remaining_phase_lengths[phase_offset]))
            
            if not state_fixed:
              # Append variable bounds for this phase
              ipopt_lb[0][:self.ipopt_nx] = init_xs[0]
              ipopt_ub[0][:self.ipopt_nx] = init_xs[0]
              state_fixed = True

            # Quadratic control cost
            self.generate_cost_code(cost_input, relative_phase_idx, folder_name, regen)

        
        elif phase_idx == 1:

            #### PHASE 1: STAY-IN-EXTENDED-NOZZLE ####
            # At the end of this phase, we should be aligned with the nozzle frame
            # Nonlinear constraints:
            # (1) stay in extended nozzle cone
            # (2) peg direction avoids side contact
            h1_expr, h1_lb, h1_ub = export_stay_in_cone_approx_planes_constraint(self.cpin_model, x_sym, self.cone_slope)
            h2_expr, h2_lb, h2_ub = export_peg_direction_constraint(self.cpin_model, constraint_input, self.cone_slope)


            h_expr = ca.vertcat(dynamics_expr, h1_expr, h2_expr)
            h_lb = np.concatenate((dynamics_lb, h1_lb, h2_lb))
            h_ub = np.concatenate((dynamics_ub, h1_ub, h2_ub))

            if self.apply_joint_acc_constraint: 
                h_expr, h_lb, h_ub = MPCNozzleAlignPlanner.append_joint_acc_constraints(h_expr, h_lb, h_ub, self.cpin_model, 
                                                                                        constraint_input, self.joint_acc_limits, 
                                                                                        self.dt)

            constraint_sizes.append(h_expr.shape[0])
            self.generate_constraint_code_and_append_bounds(h_expr, h_lb, h_ub, constraint_input, 
                                                            remaining_phase_lengths, relative_phase_idx, 
                                                            ipopt_cl, ipopt_cu, folder_name, regen)

            # Control limits (no contact forces in this phase)
            u_lb = np.copy(self.u_lb)
            u_ub = np.copy(self.u_ub)
            u_lb[self.num_rotary:self.num_rotary + 3] = 0
            u_ub[self.num_rotary:self.num_rotary + 3] = 0

            # Append variable bounds for this phase
            single_step_lb = np.concatenate((x_lb, u_lb))
            single_step_ub = np.concatenate((x_ub, u_ub))
            ipopt_lb.append(np.tile(single_step_lb, remaining_phase_lengths[phase_offset]))
            ipopt_ub.append(np.tile(single_step_ub, remaining_phase_lengths[phase_offset]))

            if not state_fixed:
              # Append variable bounds for this phase
              ipopt_lb[0][:self.ipopt_nx] = init_xs[0]
              ipopt_ub[0][:self.ipopt_nx] = init_xs[0]
              state_fixed = True
              
            # Quadratic control cost
            self.generate_cost_code(cost_input, relative_phase_idx, folder_name, regen)

        
        elif phase_idx == 2:        
            #### PHASE 2: PERIOD OF ALIGNMENT WITH NOZZLE ####
            # At the end of this phase, we should be aligned with the nozzle frame

            
            # Nonlinear constraints:
            # (1) stay in extended nozzle cone
            # (2) peg direction avoids side contact
            h1_expr, h1_lb, h1_ub = export_stay_in_cone_approx_planes_constraint(self.cpin_model, x_sym, self.cone_slope)
            h2_expr, h2_lb, h2_ub = export_peg_direction_constraint(self.cpin_model, constraint_input, self.cone_slope)


            # Start with loose tolerances to guide the optimization towards the correct solution
            position_upper_tol = np.array([0.1, 0.1, 0.3])
            position_lower_tol = -position_upper_tol

            # The angles should be within 2 degrees in the z axis and 5 degrees in the x and y axes
            rotation_upper_tol = np.array([0.2, 0.2, 0.2])
            rotation_lower_tol = -rotation_upper_tol

            upper_tol = np.concatenate((position_upper_tol, rotation_upper_tol))
            lower_tol = np.concatenate((position_lower_tol, rotation_lower_tol))

            pos_diff = np.array([0,0,0.3])
            
            
            h3_expr, h3_lb, h3_ub = export_align_frames_constraint(self.cpin_model, x_sym, 'ee_tip', 'nozzle', upper_tol, lower_tol, pos_diff = pos_diff)


            h_expr = ca.vertcat(dynamics_expr, h1_expr, h2_expr, h3_expr)
            h_lb = np.concatenate((dynamics_lb, h1_lb, h2_lb, h3_lb))
            h_ub = np.concatenate((dynamics_ub, h1_ub, h2_ub, h3_ub))

            if self.apply_joint_acc_constraint: 
                h_expr, h_lb, h_ub = MPCNozzleAlignPlanner.append_joint_acc_constraints(h_expr, h_lb, h_ub, self.cpin_model, constraint_input, self.joint_acc_limits, self.dt)

            constraint_sizes.append(h_expr.shape[0])
            self.generate_constraint_code_and_append_bounds(h_expr, h_lb, h_ub, constraint_input, remaining_phase_lengths, relative_phase_idx, ipopt_cl, ipopt_cu, folder_name, regen)

            # Control limits (no contact forces in this phase)
            u_lb = np.copy(self.u_lb)
            u_ub = np.copy(self.u_ub)
            u_lb[self.num_rotary:self.num_rotary + 3] = 0
            u_ub[self.num_rotary:self.num_rotary + 3] = 0

            # Append variable bounds for this phase
            single_step_lb = np.concatenate((x_lb, u_lb))
            single_step_ub = np.concatenate((x_ub, u_ub))
            ipopt_lb.append(np.tile(single_step_lb, remaining_phase_lengths[phase_offset]))
            ipopt_ub.append(np.tile(single_step_ub, remaining_phase_lengths[phase_offset]))

            if not state_fixed:
              # Append variable bounds for this phase
              ipopt_lb[0][:self.ipopt_nx] = init_xs[0]
              ipopt_ub[0][:self.ipopt_nx] = init_xs[0]
              state_fixed = True

            # Quadratic control cost
            self.generate_cost_code(cost_input, relative_phase_idx, folder_name, regen)

        elif phase_idx == 3:
            ### PHASE 3: Enforce Velocity Constraints ####

            # Start with loose tolerances to guide the optimization towards the correct solution
            position_upper_tol = np.array([0.05, 0.05, 0.01])
            position_lower_tol = -position_upper_tol

            # The angles should be within 2 degrees in the z axis and 5 degrees in the x and y axes
            rotation_upper_tol = np.array([0.1, 0.1, 0.1])
            rotation_lower_tol = -rotation_upper_tol

            upper_tol = np.concatenate((position_upper_tol, rotation_upper_tol))
            lower_tol = np.concatenate((position_lower_tol, rotation_lower_tol))

            pos_diff = np.array([0,0,0.3])
            
            h1_expr, h1_lb, h1_ub = export_align_frames_constraint(self.cpin_model, x_sym, 'ee_tip', 'nozzle', upper_tol, lower_tol, pos_diff = pos_diff)

            
            # Enforce that the ee tip velocity is 0mm/s in the z direction relative to the client frame
            # h2_expr, h2_lb, h2_ub = export_relative_velocity_constraint(self.cpin_model, constraint_input, 'ee_tip', 'client', vel_diff= vel_diff)

            # Enforce that the ee tip has no angular velocity
            vel_diff = np.array([0, 0, 0.005, 0, 0, 0])
            enforce = np.array([False, False, False, True, True, True])
            h2_expr, h2_lb, h2_ub = export_relative_velocity_constraint(self.cpin_model, constraint_input, 'ee_tip', 'client', vel_diff= vel_diff, enforce = enforce)

            h_expr = ca.vertcat(h1_expr, h2_expr)
            h_lb = np.concatenate((h1_lb, h2_lb))
            h_ub = np.concatenate((h1_ub, h2_ub))

            constraint_sizes.append(h_expr.shape[0])
            self.generate_constraint_code_and_append_bounds(h_expr, h_lb, h_ub, constraint_input, remaining_phase_lengths, relative_phase_idx, ipopt_cl, ipopt_cu, folder_name, regen)

            # Control limits (no contact forces in this phase)
            u_lb = np.copy(self.u_lb)
            u_ub = np.copy(self.u_ub)
            u_lb[self.num_rotary:self.num_rotary + 3] = 0
            u_ub[self.num_rotary:self.num_rotary + 3] = 0

            # Append variable bounds for this phase
            single_step_lb = np.concatenate((x_lb, u_lb))
            single_step_ub = np.concatenate((x_ub, u_ub))
            ipopt_lb.append(np.tile(single_step_lb, remaining_phase_lengths[phase_offset]))
            ipopt_ub.append(np.tile(single_step_ub, remaining_phase_lengths[phase_offset]))

            if not state_fixed:
              # Append variable bounds for this phase
              ipopt_lb[0][:self.ipopt_nx] = init_xs[0]
              ipopt_ub[0][:self.ipopt_nx] = init_xs[0]
              state_fixed = True

            # Quadratic control cost
            self.generate_cost_code(cost_input, relative_phase_idx, folder_name, regen)


    #END OF PHASES
    ipopt_lb = np.concatenate(ipopt_lb)
    ipopt_ub = np.concatenate(ipopt_ub)
    ipopt_cl = np.concatenate(ipopt_cl)
    ipopt_cu = np.concatenate(ipopt_cu)

    # Warm start
    vars_per_step = self.ipopt_nx + self.nu
    num_decision_vars = int(vars_per_step * total_remaining_steps)

    if not hasattr(self, 'previous_soln') or self.previous_soln is None:
        # First call warm start with initial guess
        warm_start = np.zeros(num_decision_vars)
        for step in range(total_remaining_steps):
            xstart = vars_per_step * step
            ustart = xstart + self.ipopt_nx
            warm_start[xstart:ustart] = init_xs[step]
            warm_start[ustart:ustart + self.nu] = init_us[step]
    else:
        # Subsequentlly warm start with previous solution
        previous_soln = self.previous_soln
        shifted_soln = previous_soln[vars_per_step:]
        steps_in_previous_soln = len(shifted_soln) // vars_per_step

        steps_to_copy = min(steps_in_previous_soln, total_remaining_steps)
        warm_start = np.zeros(num_decision_vars)
        warm_start[:vars_per_step * steps_to_copy] = shifted_soln[:vars_per_step * steps_to_copy]

        if steps_to_copy < total_remaining_steps:
            # Append last known state and control
            last_state = shifted_soln[-vars_per_step:-vars_per_step + self.ipopt_nx]
            last_control = shifted_soln[-self.nu:]
            for step in range(steps_to_copy, total_remaining_steps):
                xstart = vars_per_step * step
                ustart = xstart + self.ipopt_nx
                warm_start[xstart:ustart] = last_state
                warm_start[ustart:ustart + self.nu] = last_control
    

    # Save the generated code and variables if indicated
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        np.savetxt(os.path.join(save_path, 'nx.txt'), np.array([self.ipopt_nx]))
        np.savetxt(os.path.join(save_path, 'nu.txt'), np.array([self.nu]))
        np.savetxt(os.path.join(save_path, 'plane_idx.txt'), np.array([plane_idx]))
        np.savetxt(os.path.join(save_path, 'phase_starts.txt'), remaining_phase_starts)
        np.savetxt(os.path.join(save_path, 'constraint_sizes.txt'), constraint_sizes)
        np.savetxt(os.path.join(save_path, 'ipopt_lb.txt'), ipopt_lb)
        np.savetxt(os.path.join(save_path, 'ipopt_ub.txt'), ipopt_ub)
        np.savetxt(os.path.join(save_path, 'ipopt_cl.txt'), ipopt_cl)
        np.savetxt(os.path.join(save_path, 'ipopt_cu.txt'), ipopt_cu)
        np.savetxt(os.path.join(save_path, 'warm_start.txt'), warm_start)
        np.save(os.path.join(folder_name, 'constraint_sizes.npy'), constraint_sizes)

    ## Solve using the IPOPT solver ##
    print("Creating IPOPTContactSolver")
    solver = IPOPTContactSolver(self.ipopt_nx, self.nu,
                                remaining_phase_starts, constraint_sizes,
                                ipopt_lb, ipopt_ub,
                                ipopt_cl, ipopt_cu,
                                warm_start, stop_after_iter,
                                plane_idx, max_iter)

    soln = np.zeros_like(warm_start)
    obj_value = 0.0

    # Performance measurement
    perf_output_file = os.path.expanduser('~') + '/perf.txt'
    if count_total_flop:
        perf_proc = start_perf_proc(perf_output_file)
    else:
        perf_proc = None
    bt = time.perf_counter()

    # Solve the optimization problem
    solved = solver.solve(soln)
    obj_value = solver.get_obj_value()
    at = time.perf_counter()

    # Handle performance metrics
    if count_total_flop:
        total_flop = stop_perf_proc(perf_output_file, perf_proc) - 2
        if save_path is not None:
            np.save(os.path.join(save_path, 'total_flop.npy'), total_flop)
        print('perf count:', total_flop)
    if save_path is not None:
        np.save(os.path.join(save_path, 'total_time.npy'), at - bt)
    print('total time:', at - bt)

    if save_path is not None:
        np.save(os.path.join(save_path, 'num_iter.npy'), solver.get_num_iter())
        np.save(os.path.join(save_path, 'iteration_times.npy'), solver.get_iter_durations())

    #Process the solution
    self.xs = []
    self.us = []
    for step in range(int(total_remaining_steps)):
        xstart = vars_per_step * step
        ustart = xstart + self.ipopt_nx
        self.xs.append(soln[xstart:ustart])
        self.us.append(soln[ustart:ustart + self.nu])

    # Store previous solution for warm start
    self.previous_soln = soln.copy()

    if save_path is not None:
        np.save(os.path.join(save_path, 'success.npy'), solved)
        np.save(os.path.join(save_path, 'init_xs_ipopt.npy'), init_xs)
        np.save(os.path.join(save_path, 'xs_ipopt.npy'), self.xs)
        np.save(os.path.join(save_path, 'init_xs.npy'), [to_pin(x, self.pin_model) for x in init_xs])
        np.save(os.path.join(save_path, 'init_us.npy'), init_us)

    xs = [to_pin(x, self.pin_model) for x in self.xs]

    # Convert lists to numpy arrays
    xs = np.array(xs)
    us = np.array(self.us)
    dts = self.dt * np.ones(len(remaining_phase_lengths))
    self.xs = np.copy(xs)
    self.us = np.copy(us)
    self.dts = np.copy(dts)
    return xs, us, dts, remaining_phase_starts, solved, obj_value

  @staticmethod 
  def append_joint_acc_constraints(h_expr, h_lb, h_ub, cpin_model, constraint_input, joint_acc_limits, dt):
    h2_expr, h2_lb, h2_ub = export_joint_acc_constraint(cpin_model, constraint_input, joint_acc_limits, dt)
    h_expr = ca.vertcat(h_expr,h2_expr)
    h_lb = np.concatenate((h_lb,h2_lb))
    h_ub = np.concatenate((h_ub,h2_ub))
    return h_expr, h_lb, h_ub
