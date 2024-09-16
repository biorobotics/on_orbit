#include "ik_waypoint_follower.h"
#include <pinocchio/container/boost-container-limits.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include "pinocchio/algorithm/centroidal.hpp"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include <chrono>

IKWaypointFollower::IKWaypointFollower(string urdf_file, VectorXRef_const joint_angle_lower_limits, VectorXRef_const joint_angle_upper_limits, VectorXRef_const joint_vel_limits, VectorXRef_const joint_acc_limits, double dt) : joint_angle_lower_limits(joint_angle_lower_limits), joint_angle_upper_limits(joint_angle_upper_limits), joint_vel_limits(joint_vel_limits), joint_acc_limits(joint_acc_limits), dt(dt) {
  pin::urdf::buildModel(urdf_file, pin::JointModelFreeFlyer(), model);
  model.gravity.setZero();
  data = make_unique<pin::Data>(model);

  wrist_id = model.getFrameId("wrist");

  did_solve = false;

  tip_id = model.getFrameId("ee_tip");

  num_rotary = model.nv - 6;

  force_err_integral.setZero();

  integration_LHS = MatrixXd::Zero(model.nv, model.nv);
  integration_LHS.block(6, 6, num_rotary, num_rotary).setIdentity();
  integration_RHS = VectorXd::Zero(model.nv);

  qp_time = 0;
  qp_setup_time = 0;
  dynamics_time = 0;

  prev_acc = VectorXd::Zero(num_rotary);

  force_kp = 0.008;
  force_ki = 0.004;

  torque_kp = 0.01;

  pos_kp = Vector3d(1, 1, 1);
  rot_kp = 0.1;
}

VectorXd IKWaypointFollower::get_control(VectorXRef_const x, Vector3Ref_const pos_waypoint, Matrix3Ref_const rmat_waypoint, Vector3Ref_const v_waypoint, Vector3Ref_const w_waypoint, VectorXRef_const wrist_wrench, bool use_pos_waypoint, bool use_rmat_waypoint) {
  auto start_time = std::chrono::high_resolution_clock::now();
  VectorXd q = x.head(model.nq);
  q.segment<4>(3).normalize();
  VectorXRef_const v = x.tail(model.nv);
  pin::forwardKinematics(model, *data, q, v);
  pin::computeJointJacobians(model, *data);
  pin::updateFramePlacements(model, *data);
  pin::centerOfMass(model, *data);

  // The decision variables are system velocities (model.nv) and slacks
  int num_slack = 2*num_rotary;
  int problem_size = model.nv + num_slack;
  solver.reset(problem_size);

  // Momentum conservation constraint
  MatrixXd momentum_LHS = MatrixXd::Zero(6, problem_size);
  momentum_LHS.leftCols(model.nv) = pin::computeCentroidalMap(model, *data, q);
  VectorXd momentum_RHS = momentum_LHS.leftCols(model.nv)*v;
  Vector3d f_com_world = data->oMf[wrist_id].rotation()*wrist_wrench.head<3>();
  Vector3d tau_com_world = (data->oMf[tip_id].translation() - data->com[0]).cross(f_com_world);
  momentum_RHS.head<3>() += f_com_world*dt; // Linear impulse = change in linear momentum
  momentum_RHS.tail<3>() += tau_com_world*dt; // Angular impulse = change in angular momentum
  solver.add_constraint(momentum_LHS, momentum_RHS, momentum_RHS);

  // Joint acceleration constraint
  MatrixXd acc_constraint_LHS = MatrixXd::Zero(num_rotary, problem_size);
  acc_constraint_LHS.block(0, 6, num_rotary, num_rotary).setIdentity();
  solver.add_constraint(acc_constraint_LHS, -joint_acc_limits*dt + v.segment(6, num_rotary), joint_acc_limits*dt + v.segment(6, num_rotary));

  // LLSQ costs
  MatrixXd Jworld = MatrixXd::Zero(6, model.nv);

  pin::getFrameJacobian(model, *data, tip_id, pin::ReferenceFrame::LOCAL_WORLD_ALIGNED, Jworld);

  Vector3d pos_kp = this->pos_kp;
  double rot_kp = this->rot_kp;
  double force_kp = this->force_kp;
  double force_ki = this->force_ki;
  double torque_kp = this->torque_kp;

  MatrixXd position_LHS = MatrixXd::Zero(3, problem_size);
  position_LHS.leftCols(model.nv) = Jworld.topRows<3>();
  Vector3d force_err = data->oMf[wrist_id].rotation()*(wrist_wrench.head<3>());
  force_err_integral += force_err*dt;
  Vector3Ref_const tip_pos = data->oMf[tip_id].translation();
  Vector3d vel_without_force_control = v_waypoint + (w_waypoint.cross(tip_pos - pos_waypoint) - pos_kp.cwiseProduct(tip_pos - pos_waypoint))*(double)use_pos_waypoint;
  Vector3d force_control_term = force_kp*force_err + force_ki*force_err_integral;
  Vector3d position_RHS = vel_without_force_control + force_control_term;
  solver.add_llsq_objective(position_LHS, position_RHS, 1);

  // Orientation controller: PD on SO(3)
  MatrixXd orientation_LHS = MatrixXd::Zero(3, problem_size);
  orientation_LHS.leftCols(model.nv) = Jworld.bottomRows<3>();
  Vector3d orientation_RHS = w_waypoint - rot_kp*data->oMf[tip_id].rotation()*pin::log3(rmat_waypoint.transpose()*data->oMf[tip_id].rotation())*(double)use_rmat_waypoint + data->oMf[wrist_id].rotation()*(torque_kp*wrist_wrench.tail<3>());
  solver.add_llsq_objective(orientation_LHS, orientation_RHS, 1);

  // Joint angle limits
  MatrixXd joint_limit_LHS = MatrixXd::Zero(num_rotary, problem_size);
  joint_limit_LHS.block(0, 6, num_rotary, num_rotary).setIdentity();
  joint_limit_LHS.block(0, problem_size - num_slack, num_rotary, num_rotary).setIdentity(); // Slack
  solver.add_constraint(joint_limit_LHS, (joint_angle_lower_limits - q.segment(7, num_rotary))/dt, (joint_angle_upper_limits - q.segment(7, num_rotary))/dt);

  // Joint velocity limits
  joint_limit_LHS = MatrixXd::Zero(num_rotary, problem_size);
  joint_limit_LHS.block(0, 6, num_rotary, num_rotary).setIdentity();
  joint_limit_LHS.block(0, problem_size - num_slack + num_rotary, num_rotary, num_rotary).setIdentity(); // Slack
  solver.add_constraint(joint_limit_LHS, -joint_vel_limits, joint_vel_limits);

  MatrixXd quadratic_cost_R = MatrixXd::Zero(problem_size, problem_size);
  quadratic_cost_R.block(problem_size - num_slack, problem_size - num_slack, num_slack, num_slack) = 10000*MatrixXd::Identity(num_slack, num_slack);
  solver.add_quadratic_cost(quadratic_cost_R, 1);

  auto stop_time = std::chrono::high_resolution_clock::now();
  qp_setup_time += std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time).count();

  if (primal.size() != problem_size) {
    solver.solve(primal, dual, primal, dual, false, false);
  } else {
    solver.solve(primal, dual, primal, dual, did_solve, did_solve);
  }

  did_solve = true;

  cached_x = x;

  qp_setup_time += solver.get_prev_setup_time();
  qp_time += solver.get_prev_solve_time();

  prev_acc = (primal.segment(6, num_rotary) - v.segment(6, num_rotary))/dt;

  return primal.segment(6, num_rotary);
}

VectorXd IKWaypointFollower::integrate_from_cache(VectorXRef_const h_0) {
  auto start_time = std::chrono::high_resolution_clock::now();
  integration_LHS.topRows<6>() = data->Ag;
  integration_RHS.head<6>() = h_0;
  integration_RHS.tail(num_rotary) = primal.segment(6, num_rotary);
  cached_x.tail(model.nv) = integration_LHS.colPivHouseholderQr().solve(integration_RHS);
  cached_x.head(model.nq) = pin::integrate(model, cached_x.head(model.nq), cached_x.tail(model.nv)*dt);

  auto stop_time = std::chrono::high_resolution_clock::now();
  dynamics_time += std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time).count();

  return cached_x;
}

void IKWaypointFollower::reset() {
  did_solve = false;
  force_err_integral.setZero();
  prev_acc.setZero();
}

double IKWaypointFollower::get_qp_setup_time() {
  return qp_setup_time;
}

double IKWaypointFollower::get_qp_time() {
  return qp_time;
}

double IKWaypointFollower::get_dynamics_time() {
  return dynamics_time;
}

void IKWaypointFollower::set_gains(double force_kp, double force_ki, double torque_kp, Vector3Ref_const pos_kp, double rot_kp) {
  this->force_kp = force_kp;
  this->force_ki = force_ki;
  this->torque_kp = torque_kp;

  this->pos_kp = pos_kp;
  this->rot_kp = rot_kp;
}
