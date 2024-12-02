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
import os

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

class IpoptNozzleAlignPlanner(object):
  def __init__(self, urdf_file, 
               mrv_urdf_file, dt, 
               joint_angle_lower_limits, joint_angle_upper_limits, 
               joint_torque_limits, joint_vel_limits, joint_acc_limits, 
               control_cost_weight, phase_lengths_sec, cw_a, cw_mu, cw_orbit_dir, 
               initial_client_rmat,
               cone_slope, use_cw, meshdir):
    
    self.dt = dt

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

    self.num_phases = 3
    self.num_phases_without_dummy = 2 # Dummy phases are the nozzle impact, hole impact, and insertion phases

    self.joint_angle_lower_limits = np.array(joint_angle_lower_limits)
    self.joint_angle_upper_limits = np.array(joint_angle_upper_limits)
    self.joint_vel_limits = np.array(joint_vel_limits)
    self.joint_acc_limits = np.array(joint_acc_limits)
    self.joint_torque_limits = np.array(joint_torque_limits)

    self.num_rotary = self.pin_model.nv - 12

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
    print(initial_client_rmat)


    self.cw_xproj = initial_client_rmat[:, 2]
    if self.cw_orbit_dir == 'x':
      self.cw_yproj = initial_client_rmat[:, 0]
    else:
      self.cw_yproj = initial_client_rmat[:, 1]
    self.cw_zproj = np.cross(self.cw_xproj, self.cw_yproj)

    self.apply_joint_acc_constraint = True

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

      # Cost from orientation error and control effort
      desired_rot = ca.DM([0,0,0])
      actual_rot = cost_input[3:6]
      # actual_rot = ca.DM([0.04, -0.03 ,-3.13])
      rot_err = ca.norm_2(desired_rot - actual_rot)
      cost_expr = 0.5*u.T @ self.R_mat @ u + 0.5*rot_err**2

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

  def plan(self, x0, ecm_sim, ecm_sim_reset_args, save_path=None, max_iter=250, count_flop_per_iter=False, count_total_flop=False, stop_after_iter=-1, prop_time=0.):
    # regen = True
    regen=True

    if prop_time != 0.:
      raise Exception("prop_time is not supported in this planner.")

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
    print("Initial guess:", x0)
    print("Final guess:", xf)
    
    ## PHASE LENGTHS ##
    phase_lengths_sec = np.array([self.phase_lengths_sec[0], \
                                  self.phase_lengths_sec[1],\
                                  self.phase_lengths_sec[2],\
                                  self.phase_lengths_sec[3]\
                                  ])

    phase_lengths = np.divide(phase_lengths_sec,self.dt).astype(int) # Number of time instants per phase
    print("Phase lengths:", phase_lengths)
    phase_starts = np.concatenate([[0], np.cumsum(phase_lengths)])
    print("Phase starts:", phase_starts)
    # exit()
    steps = phase_starts[-1] - 1
    init_xs = np.array([from_pin(self.interp_x(x0, xf, step/steps), self.pin_model) for step in range(steps + 1)])
    init_us = np.zeros((steps + 1, self.nu))

    # Variable bounds
    ipopt_lb = []
    ipopt_ub = []

    # Constraint bounds
    ipopt_cl = []
    ipopt_cu = []

    x_sym = ca.SX.sym('x', self.ipopt_nx)
    u_sym = ca.SX.sym('u', self.nu)
    xnext_sym = ca.SX.sym('xnext', self.ipopt_nx)
    unext_sym = ca.SX.sym('unext', self.nu)
    dyn_input = ca.vertcat(x_sym, u_sym, xnext_sym, unext_sym)
    cost_input = ca.vertcat(x_sym, u_sym)
    # constraint_input = x_sym[:self.pin_model.nv]
    # constraint_input = cost_input
    constraint_input = dyn_input

    dynamics_expr = export_implicit_midpoint_model(self.cpin_model, plane_idx, dyn_input, self.dt, self.use_cw, self.mrv_cpin_model, self.initial_client_rmat, self.cw_mu, self.cw_a, self.cw_orbit_dir)
    dynamics_lb = np.zeros(self.ipopt_nx)
    dynamics_ub = np.zeros(self.ipopt_nx)

    constraint_sizes = []

    #### PHASE 0: STAY-BELOW-NOZZLE-OPENING-PLANE ####
    phase_idx = 0

    # Nonlinear constraints:
    # (1) stay below nozzle plane
    h1_expr, h1_lb, h1_ub = export_stay_below_nozzle_constraint(self.cpin_model, constraint_input)
    h_expr = ca.vertcat(dynamics_expr, h1_expr)
    h_lb = np.concatenate((dynamics_lb, h1_lb))
    h_ub = np.concatenate((dynamics_ub, h1_ub))

    if self.apply_joint_acc_constraint: 
      h_expr, h_lb, h_ub = IpoptNozzleAlignPlanner.append_joint_acc_constraints(h_expr, h_lb, h_ub, self.cpin_model, constraint_input, self.joint_acc_limits, self.dt)

    constraint_sizes.append(h_expr.shape[0])
    self.generate_constraint_code_and_append_bounds(h_expr, h_lb, h_ub, constraint_input, phase_lengths, phase_idx, ipopt_cl, ipopt_cu, folder_name, regen)

    # Control limits (no contact forces in this phase)
    u_lb = np.copy(self.u_lb)
    u_ub = np.copy(self.u_ub)
    u_lb[self.num_rotary:self.num_rotary + 3] = 0
    u_ub[self.num_rotary:self.num_rotary + 3] = 0

    # State bounds
    x_lb = self.xlb
    x_ub = self.xub

    ipopt_lb.append(np.tile(np.concatenate((x_lb, u_lb)), phase_lengths[phase_idx]))
    ipopt_ub.append(np.tile(np.concatenate((x_ub, u_ub)), phase_lengths[phase_idx]))

    # Fix initial state
    ipopt_lb[0][:self.ipopt_nx] = init_xs[0]
    ipopt_ub[0][:self.ipopt_nx] = init_xs[0]

    # Quadratic control cost
    self.generate_cost_code(cost_input, phase_idx, folder_name, regen)

    #### PHASE 1: STAY-IN-EXTENDED-NOZZLE ####
    # At the end of this phase, we should be aligned with the nozzle frame
    phase_idx += 1

    # Nonlinear constraints:
    # (1) stay in extended nozzle cone
    # (2) peg direction avoids side contact
    h1_expr, h1_lb, h1_ub = export_stay_in_cone_approx_planes_constraint(self.cpin_model, x_sym, self.cone_slope)
    h2_expr, h2_lb, h2_ub = export_peg_direction_constraint(self.cpin_model, constraint_input, self.cone_slope)


    h_expr = ca.vertcat(dynamics_expr, h1_expr, h2_expr)
    h_lb = np.concatenate((dynamics_lb, h1_lb, h2_lb))
    h_ub = np.concatenate((dynamics_ub, h1_ub, h2_ub))

    if self.apply_joint_acc_constraint: 
      h_expr, h_lb, h_ub = IpoptNozzleAlignPlanner.append_joint_acc_constraints(h_expr, h_lb, h_ub, self.cpin_model, constraint_input, self.joint_acc_limits, self.dt)

    constraint_sizes.append(h_expr.shape[0])
    self.generate_constraint_code_and_append_bounds(h_expr, h_lb, h_ub, constraint_input, phase_lengths, phase_idx, ipopt_cl, ipopt_cu, folder_name, regen)

    # Control limits (no contact forces in this phase)
    u_lb = np.copy(self.u_lb)
    u_ub = np.copy(self.u_ub)
    u_lb[self.num_rotary:self.num_rotary + 3] = 0
    u_ub[self.num_rotary:self.num_rotary + 3] = 0

    # State bounds
    x_lb = self.xlb
    x_ub = self.xub

    ipopt_lb.append(np.tile(np.concatenate((x_lb, u_lb)), phase_lengths[phase_idx]))
    ipopt_ub.append(np.tile(np.concatenate((x_ub, u_ub)), phase_lengths[phase_idx]))

    # Quadratic control cost
    self.generate_cost_code(cost_input, phase_idx, folder_name, regen)
    
    #### PHASE 2: PERIOD OF ALIGNMENT WITH NOZZLE ####
    # At the end of this phase, we should be aligned with the nozzle frame
    phase_idx += 1
    
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
      h_expr, h_lb, h_ub = IpoptNozzleAlignPlanner.append_joint_acc_constraints(h_expr, h_lb, h_ub, self.cpin_model, constraint_input, self.joint_acc_limits, self.dt)

    constraint_sizes.append(h_expr.shape[0])
    self.generate_constraint_code_and_append_bounds(h_expr, h_lb, h_ub, constraint_input, phase_lengths, phase_idx, ipopt_cl, ipopt_cu, folder_name, regen)

    # Control limits (no contact forces in this phase)
    u_lb = np.copy(self.u_lb)
    u_ub = np.copy(self.u_ub)
    u_lb[self.num_rotary:self.num_rotary + 3] = 0
    u_ub[self.num_rotary:self.num_rotary + 3] = 0

    # State bounds
    x_lb = self.xlb
    x_ub = self.xub

    ipopt_lb.append(np.tile(np.concatenate((x_lb, u_lb)), phase_lengths[phase_idx]))
    ipopt_ub.append(np.tile(np.concatenate((x_ub, u_ub)), phase_lengths[phase_idx]))

    # Quadratic control cost
    self.generate_cost_code(cost_input, phase_idx, folder_name, regen)
   
    ### PHASE 3: Enforce Velocity Constraints #### 
    phase_idx += 1

    # Start with loose tolerances to guide the optimization towards the correct solution
    position_upper_tol = np.array([0.01, 0.01, 0.001])
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
    self.generate_constraint_code_and_append_bounds(h_expr, h_lb, h_ub, constraint_input, phase_lengths, phase_idx, ipopt_cl, ipopt_cu, folder_name, regen)

    # Control limits (no contact forces in this phase)
    u_lb = np.copy(self.u_lb)
    u_ub = np.copy(self.u_ub)
    u_lb[self.num_rotary:self.num_rotary + 3] = 0
    u_ub[self.num_rotary:self.num_rotary + 3] = 0

    # State bounds
    x_lb = self.xlb
    x_ub = self.xub

    ipopt_lb.append(np.tile(np.concatenate((x_lb, u_lb)), phase_lengths[phase_idx]))
    ipopt_ub.append(np.tile(np.concatenate((x_ub, u_ub)), phase_lengths[phase_idx]))

    # Quadratic control cost
    self.generate_cost_code(cost_input, phase_idx, folder_name, regen)

    #END OF PHASES
    ipopt_lb = np.concatenate(ipopt_lb)
    ipopt_ub = np.concatenate(ipopt_ub)
    ipopt_cl = np.concatenate(ipopt_cl)
    ipopt_cu = np.concatenate(ipopt_cu)

    vars_per_step = self.ipopt_nx + self.nu
    num_decision_vars = int(vars_per_step*phase_starts[-1])
    warm_start = np.zeros(num_decision_vars)
    for step in range(phase_starts[-1]):
      xstart = vars_per_step*step
      ustart = xstart + self.ipopt_nx
      uend = ustart + self.nu
      warm_start[xstart:ustart] = init_xs[step]
      warm_start[ustart:uend] = init_us[step]

    np.savetxt('nx.txt', np.array([self.ipopt_nx]))
    np.savetxt('nu.txt', np.array([self.nu]))
    np.savetxt('plane_idx.txt', np.array([plane_idx]))
    np.savetxt('phase_starts.txt', phase_starts)
    np.savetxt('constraint_sizes.txt', constraint_sizes)
    np.savetxt('ipopt_lb.txt', ipopt_lb)
    np.savetxt('ipopt_ub.txt', ipopt_ub)
    np.savetxt('ipopt_cl.txt', ipopt_cl)
    np.savetxt('ipopt_cu.txt', ipopt_cu)
    np.savetxt('warm_start.txt', warm_start)
    np.save(folder_name + '/constraint_sizes.npy', constraint_sizes)
    print("Creating IPOPTContactSolver")
    solver = IPOPTContactSolver(self.ipopt_nx, self.nu, 
                                phase_starts, constraint_sizes, 
                                ipopt_lb, ipopt_ub, 
                                ipopt_cl, ipopt_cu, 
                                warm_start, stop_after_iter, 
                                plane_idx, max_iter)
    soln = np.zeros_like(warm_start)

    perf_output_file = os.path.expanduser('~') + '/perf.txt'
    if count_total_flop:
      perf_proc = start_perf_proc(perf_output_file)
    else:
      perf_proc = None
    bt = time.perf_counter()
    solved = solver.solve(soln)
    at = time.perf_counter()
    if count_total_flop:
      total_flop = stop_perf_proc(perf_output_file, perf_proc) - 2
      if save_path is not None:
        np.save(save_path + '/total_flop.npy', total_flop)
      print('perf count: ', total_flop)
    if save_path is not None:
      np.save(save_path + '/total_time.npy', at - bt)
    print('total time: ', at - bt)

    if save_path is not None:
      np.save(save_path + '/num_iter.npy', solver.get_num_iter())
      np.save(save_path + '/iteration_times.npy', solver.get_iter_durations())

    xs = []
    us = []
    for step in range(phase_starts[-1]):
      xs.append(soln[vars_per_step*step:vars_per_step*step + self.ipopt_nx])
      # print("X:", xs[-1])
      
      us.append(soln[vars_per_step*step + self.ipopt_nx:vars_per_step*(step + 1)])

    if save_path is not None:
      np.save(save_path  + '/success.npy', solved)

    if save_path is not None:
      np.save(save_path +'/init_xs_ipopt.npy', init_xs)
      np.save(save_path +'/xs_ipopt.npy', xs)
      np.save(save_path +'/init_xs.npy', [to_pin(x, self.pin_model) for x in init_xs])
      np.save(save_path +'/init_us.npy', init_us)

    xs = [to_pin(x, self.pin_model) for x in xs]



    xs = np.array(xs)
    us = np.array(us)
    dts = self.dt*np.ones(self.num_phases)
    self.xs = np.copy(xs)
    self.us = np.copy(us)
    self.dts = np.copy(dts)
    self.phase_starts = np.copy(phase_starts)
    return xs, us, dts, phase_starts, solved

  @staticmethod 
  def append_joint_acc_constraints(h_expr, h_lb, h_ub, cpin_model, constraint_input, joint_acc_limits, dt):
    h2_expr, h2_lb, h2_ub = export_joint_acc_constraint(cpin_model, constraint_input, joint_acc_limits, dt)
    h_expr = ca.vertcat(h_expr,h2_expr)
    h_lb = np.concatenate((h_lb,h2_lb))
    h_ub = np.concatenate((h_ub,h2_ub))
    return h_expr, h_lb, h_ub 
