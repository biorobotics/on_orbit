#include "ik_ee_trajectory_tracker.h"
#include <pinocchio/container/boost-container-limits.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include "pinocchio/algorithm/centroidal.hpp"
#include <chrono>

IKEETrajectoryTracker::IKEETrajectoryTracker(string urdf_file, VectorXRef_const joint_angle_lower_limits, VectorXRef_const joint_angle_upper_limits, VectorXRef_const joint_vel_limits, VectorXRef_const joint_acc_limits, double dt, double cone_slope) : joint_angle_lower_limits(joint_angle_lower_limits), joint_angle_upper_limits(joint_angle_upper_limits), joint_vel_limits(joint_vel_limits), joint_acc_limits(joint_acc_limits), dt(dt), cone_slope(cone_slope) {
  pin::urdf::buildModel(urdf_file, model);
  model.gravity.setZero();
  data = make_unique<pin::Data>(model);

  wrist_id = model.getFrameId("wrist");
  nozzle_id = model.getFrameId("nozzle");

  did_solve = false;

  tip_id = model.getFrameId("ee_tip");
  cone_vertex_id = model.getFrameId("cone_vertex");

  mrv_jidx = model.getJointId("world_to_base");
  mrv_qidx = model.idx_qs[mrv_jidx];
  mrv_vidx = model.idx_vs[mrv_jidx];
  cv_jidx = model.getJointId("world_to_client");
  cv_qidx = model.idx_qs[cv_jidx];
  cv_vidx = model.idx_vs[cv_jidx];

  num_rotary = model.nv - 12;

  for (int i = 0; i < num_rotary; ++i) {
    this->joint_angle_upper_limits(i) -= 1e-3;
    this->joint_angle_lower_limits(i) += 1e-3;
    this->joint_vel_limits(i) -= 1e-3;
    this->joint_acc_limits(i) -= 1e-3;
  }

  force_err_integral.setZero();

  client_inertia = Matrix3d::Zero();
  client_inertia(0, 0) = 8001.1405;
  client_inertia(1, 1) = 8430.025;
  client_inertia(2, 2) = 5268.8845;
  client_inertia_inverse = client_inertia.inverse();

  integration_LHS = MatrixXd::Zero(model.nv, model.nv);
  integration_LHS.block(6, cv_vidx, 6, 6).setIdentity();
  integration_LHS.block(12, mrv_vidx + 6, num_rotary, num_rotary).setIdentity();
  integration_RHS = VectorXd::Zero(model.nv);

  qp_time = 0;
  qp_setup_time = 0;
  dynamics_time = 0;

  prev_acc = VectorXd::Zero(num_rotary);

  force_kp = 0.008;
  force_ki = 0.004;
  made_contact = false;
  guard_speed = 1.0;
  use_cone_cbf = true;
}

VectorXd IKEETrajectoryTracker::get_control(VectorXRef_const x, Vector3Ref_const des_pos_wrt_client, Matrix3Ref_const des_rmat_wrt_client, Vector3Ref_const des_vel_wrt_client, Vector3Ref_const des_w_wrt_client, VectorXRef_const target_wrench, VectorXRef_const wrist_wrench, bool in_hole) {
  auto start_time = std::chrono::high_resolution_clock::now();
  VectorXd q = x.head(model.nq);
  q.segment<4>(mrv_qidx + 3).normalize();
  q.segment<4>(cv_qidx + 3).normalize();
  VectorXRef_const v = x.tail(model.nv);
  pin::forwardKinematics(model, *data, q, v);
  pin::computeJointJacobians(model, *data);
  pin::updateFramePlacements(model, *data);

  // The decision variables are system velocities (model.nv) and slacks
  int num_slack = 2*num_rotary + 1;
  int problem_size = model.nv + num_slack;
  solver.reset(problem_size);

  // Momentum conservation constraint
  MatrixXd momentum_LHS = MatrixXd::Zero(6, problem_size);
  momentum_LHS.leftCols(model.nv) = pin::computeCentroidalMap(model, *data, q);
  VectorXd momentum_RHS = momentum_LHS.leftCols(model.nv)*v;
  solver.add_constraint(momentum_LHS, momentum_RHS, momentum_RHS);

  // Client velocity constraint
  Vector3Ref_const client_pos = q.segment<3>(cv_qidx);
  Matrix3d client_rmat = Quaterniond(q.segment<4>(cv_qidx + 3)).toRotationMatrix();
  Vector3Ref_const client_v_local = v.segment<3>(cv_vidx);
  Vector3Ref_const client_w_local = v.segment<3>(cv_vidx + 3);

  MatrixXd client_LHS = MatrixXd::Zero(6, problem_size);
  client_LHS.block(0, cv_vidx, 6, 6).setIdentity();
  VectorXd client_RHS = VectorXd::Zero(6);
  Vector3d f_client_world = -data->oMf[wrist_id].rotation()*wrist_wrench.head<3>(); 
  Vector3d f_client_local = client_rmat.transpose()*f_client_world; 
  Vector3d client_vdot_local = f_client_local/client_mass - client_w_local.cross(client_v_local);
  client_RHS.head<3>() = client_v_local + client_vdot_local*dt;
  Vector3d tip_wrt_client_local = client_rmat.transpose()*(data->oMf[tip_id].translation() - client_pos);
  Vector3d tau_client_local = tip_wrt_client_local.cross(f_client_local);
  Vector3d client_wdot_local = client_inertia_inverse*(tau_client_local - client_w_local.cross(client_inertia*client_w_local));
  client_RHS.tail<3>() = client_w_local + client_wdot_local*dt;
  solver.add_constraint(client_LHS, client_RHS, client_RHS);

  Vector3d des_pos = client_rmat*des_pos_wrt_client + client_pos;
  Matrix3d des_rmat = client_rmat*des_rmat_wrt_client;
  Vector3d des_vel = client_rmat*(des_vel_wrt_client + client_v_local + client_w_local.cross(des_pos_wrt_client));
  Vector3d des_w = client_rmat*(des_w_wrt_client + client_w_local);

  // Joint acceleration constraint
  MatrixXd acc_constraint_LHS = MatrixXd::Zero(num_rotary, problem_size);
  acc_constraint_LHS.block(0, mrv_vidx + 6, num_rotary, num_rotary).setIdentity();
  acc_constraint_LHS.block(0, model.nv, num_rotary, num_rotary) = MatrixXd::Identity(num_rotary, num_rotary); // Slack
  solver.add_constraint(acc_constraint_LHS, -joint_acc_limits*dt + v.segment(mrv_vidx + 6, num_rotary), joint_acc_limits*dt + v.segment(mrv_vidx + 6, num_rotary));

  // LLSQ costs
  MatrixXd Jworld = MatrixXd::Zero(6, model.nv);

  pin::getFrameJacobian(model, *data, tip_id, pin::ReferenceFrame::LOCAL_WORLD_ALIGNED, Jworld);

  Vector3d pos_kp(1, 1, 1);
  double rot_kp = 0.1;
  double force_kp = this->force_kp;
  double force_ki = this->force_ki;
  double torque_kp = 0.0;
  if (in_hole) {
    force_ki = 0;
    torque_kp = 0.01;
  }

  MatrixXd position_LHS = MatrixXd::Zero(3, problem_size);
  position_LHS.leftCols(model.nv) = Jworld.topRows<3>();
  Vector3d force_err = data->oMf[wrist_id].rotation()*(wrist_wrench.head<3>() - target_wrench.head<3>());
  force_err_integral += force_err*dt;
  Vector3Ref_const tip_pos = data->oMf[tip_id].translation();
  Vector3d vel_without_force_control = des_vel - pos_kp.cwiseProduct(tip_pos - des_pos);
  if (!in_hole and use_cone_cbf) { 
    // Don't impact the cone wall at too high a speed
    double dist_from_client_com_to_cone_vertex = std::abs(client_rmat.col(2).dot(data->oMf[cone_vertex_id].translation() - client_pos));
    Vector3d tip_wrt_cone = client_rmat.transpose()*(tip_pos - (client_pos - client_rmat.col(2)*dist_from_client_com_to_cone_vertex));
    double xynorm = tip_wrt_cone.head<2>().norm();
    Vector3d cone_normal = client_rmat*Vector3d(-tip_wrt_cone(0), -tip_wrt_cone(1), -1/cone_slope*xynorm).normalized();
    double cone_dist = (-tip_wrt_cone(2)/cone_slope - xynorm)*sin(atan(cone_slope));
    Vector3d closest_point = tip_pos - cone_dist*cone_normal;
    Vector3d client_w_world = client_rmat*client_w_local;
    double vel_toward_cone = -cone_normal.dot(vel_without_force_control - client_rmat*v.segment<3>(cv_vidx) - client_w_world.cross(closest_point - client_pos));
    double excess = vel_toward_cone - 0.5*cone_dist;
    if (excess > 0) {
      vel_without_force_control -= -cone_normal*excess;
    }
  }

  if (wrist_wrench.norm() > 1) {
    made_contact = true;
  }

  Vector3d force_control_term = force_kp*force_err + force_ki*force_err_integral;
  if (!made_contact) {
    force_control_term = force_control_term.cwiseMin(guard_speed*Vector3d::Ones()).cwiseMax(-guard_speed*Vector3d::Ones());
  }

  Vector3d position_RHS = vel_without_force_control + force_control_term;
  solver.add_llsq_objective(position_LHS, position_RHS, 1);

  // Orientation controller: PD on SO(3)
  MatrixXd orientation_LHS = MatrixXd::Zero(3, problem_size);
  orientation_LHS.leftCols(model.nv) = Jworld.bottomRows<3>();
  Vector3d orientation_RHS = des_w - rot_kp*data->oMf[tip_id].rotation()*pin::log3(des_rmat.transpose()*data->oMf[tip_id].rotation()) + data->oMf[wrist_id].rotation()*(torque_kp*(wrist_wrench.tail<3>() - target_wrench.tail<3>()));
  solver.add_llsq_objective(orientation_LHS, orientation_RHS, 1);

  // Joint angle limits
  MatrixXd joint_limit_LHS = MatrixXd::Zero(num_rotary, problem_size);
  joint_limit_LHS.block(0, mrv_vidx + 6, num_rotary, num_rotary).setIdentity();
  // joint_limit_LHS.block(0, problem_size - num_slack, num_rotary, num_rotary).setIdentity(); // Slack
  solver.add_constraint(joint_limit_LHS, (joint_angle_lower_limits - q.segment(mrv_qidx + 7, num_rotary))/dt, (joint_angle_upper_limits - q.segment(mrv_qidx + 7, num_rotary))/dt);

  // Joint velocity limits
  joint_limit_LHS = MatrixXd::Zero(num_rotary, problem_size);
  joint_limit_LHS.block(0, mrv_vidx + 6, num_rotary, num_rotary).setIdentity();
  // joint_limit_LHS.block(0, problem_size - num_slack + num_rotary, num_rotary, num_rotary).setIdentity(); // Slack
  solver.add_constraint(joint_limit_LHS, -joint_vel_limits, joint_vel_limits);

  // Rim collision CBF
  double nozzle_opening_radius = 0.143;

  Vector3d t_tip_nozzle = client_rmat.transpose()*(tip_pos - data->oMf[nozzle_id].translation());
  MatrixXd Jnozzle = MatrixXd::Zero(6, model.nv);
  pin::getFrameJacobian(model, *data, nozzle_id, pin::ReferenceFrame::LOCAL_WORLD_ALIGNED, Jnozzle);

  double xynorm = t_tip_nozzle.head<2>().norm();
  double d = sqrt(pow(nozzle_opening_radius, 2) - 2*nozzle_opening_radius*xynorm + t_tip_nozzle.squaredNorm());
  double h = d - 0.015;
  Vector3d grad_h = 1/d*Vector3d((1 - nozzle_opening_radius/xynorm)*t_tip_nozzle(0), (1 - nozzle_opening_radius/xynorm)*t_tip_nozzle(1), t_tip_nozzle(2));
  MatrixXd cbf_LHS = MatrixXd::Zero(2, problem_size);
  cbf_LHS.block(0, 0, 1, model.nv) = grad_h.transpose()*client_rmat.transpose()*(Jworld.topRows<3>() - Jnozzle.topRows<3>());
  cbf_LHS(0, problem_size - num_slack + 2*num_rotary) = 1; // Slack
  cbf_LHS(1, problem_size - num_slack + 2*num_rotary) = 1; // Slack nonnegative
  cbf_LHS.block(0, cv_vidx + 3, 1, 3) = grad_h.transpose()*pin::skew(t_tip_nozzle);
  VectorXd cbf_lb = -h/dt*VectorXd::Ones(2);
  cbf_lb(1) = 0;
  // solver.add_constraint(cbf_LHS, cbf_lb, std::numeric_limits<double>::infinity()*VectorXd::Ones(1));

  MatrixXd quadratic_cost_R = MatrixXd::Zero(problem_size, problem_size);
  quadratic_cost_R.block(problem_size - num_slack, problem_size - num_slack, num_slack, num_slack) = 10000*MatrixXd::Identity(num_slack, num_slack);
  solver.add_quadratic_cost(quadratic_cost_R, 1);

  if (!in_hole) {
    MatrixXd acc_cost_LHS = MatrixXd::Zero(num_rotary, problem_size);
    acc_cost_LHS.block(0, mrv_vidx + 6, num_rotary, num_rotary).setIdentity();
    solver.add_llsq_objective(acc_cost_LHS, v.segment(mrv_vidx + 6, num_rotary) + prev_acc*dt, 10);
  }

  auto stop_time = std::chrono::high_resolution_clock::now();
  qp_setup_time += std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time).count();

  if (primal.size() != problem_size) {
    solver.solve(primal, dual, primal, dual, false, false);
  } else {
    solver.solve(primal, dual, primal, dual, did_solve, did_solve);
  }

  did_solve = true;

  cached_x = x;
  integration_RHS.segment<6>(6) = client_RHS;

  qp_setup_time += solver.get_prev_setup_time();
  qp_time += solver.get_prev_solve_time();

  for (int i = 0; i < num_rotary; ++i) {
    if (primal(mrv_vidx + 6 + i) > joint_acc_limits(i)) {
      primal(mrv_vidx + 6 + i) = joint_acc_limits(i);
    } else if (primal(mrv_vidx + 6 + i) < -joint_acc_limits(i)) {
      primal(mrv_vidx + 6 + i) = -joint_acc_limits(i);
    }
  }

  prev_acc = (primal.segment(mrv_vidx + 6, num_rotary) - v.segment(mrv_vidx + 6, num_rotary))/dt;

  return primal.segment(mrv_vidx + 6, num_rotary);
}

VectorXd IKEETrajectoryTracker::integrate_from_cache(VectorXRef_const h_0) {
  auto start_time = std::chrono::high_resolution_clock::now();
  integration_LHS.topRows<6>() = data->Ag;
  integration_RHS.head<6>() = h_0;
  integration_RHS.tail(num_rotary) = primal.segment(mrv_vidx + 6, num_rotary);
  cached_x.tail(model.nv) = integration_LHS.colPivHouseholderQr().solve(integration_RHS);
  cached_x.head(model.nq) = pin::integrate(model, cached_x.head(model.nq), cached_x.tail(model.nv)*dt);

  auto stop_time = std::chrono::high_resolution_clock::now();
  dynamics_time += std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time).count();

  return cached_x;
}

void IKEETrajectoryTracker::reset() {
  did_solve = false;
  force_err_integral.setZero();
  made_contact = false;
  prev_acc.setZero();
}

double IKEETrajectoryTracker::get_qp_setup_time() {
  return qp_setup_time;
}

double IKEETrajectoryTracker::get_qp_time() {
  return qp_time;
}

double IKEETrajectoryTracker::get_dynamics_time() {
  return dynamics_time;
}

void IKEETrajectoryTracker::set_gains(double force_kp, double force_ki) {
  this->force_kp = force_kp;
  this->force_ki = force_ki;
}

void IKEETrajectoryTracker::set_guard_speed(double guard_speed) {
  this->guard_speed = guard_speed;
}

void IKEETrajectoryTracker::set_use_cone_cbf(bool use_cone_cbf) {
  this->use_cone_cbf = use_cone_cbf;
}
