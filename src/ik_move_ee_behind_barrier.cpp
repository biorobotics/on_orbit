#include "ik_move_ee_behind_barrier.h"
#include <pinocchio/container/boost-container-limits.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <chrono>

IKMoveEEBehindBarrier::IKMoveEEBehindBarrier(string urdf_file, VectorXRef_const joint_angle_lower_limits, VectorXRef_const joint_angle_upper_limits, VectorXRef_const joint_vel_limits, VectorXRef_const joint_acc_limits, double dt, string ee_name) : joint_angle_lower_limits(joint_angle_lower_limits), joint_angle_upper_limits(joint_angle_upper_limits), joint_vel_limits(joint_vel_limits), joint_acc_limits(joint_acc_limits), dt(dt) {
  pin::urdf::buildModel(urdf_file, model);
  data = make_unique<pin::Data>(model);

  did_solve = false;

  ee_fid = model.getFrameId(ee_name);

  pos_kp = 2;
  rot_kp = 1;

  prev_pos_err_norm = std::numeric_limits<double>::infinity();
  prev_rot_err_norm = std::numeric_limits<double>::infinity();
}

VectorXd IKMoveEEBehindBarrier::get_vel_cmd(VectorXRef_const x, Vector3Ref_const des_ee_pos, Matrix3Ref_const des_ee_rmat, Vector3Ref_const barrier_pos, Vector3Ref_const barrier_vec) {
  VectorXRef_const q = x.head(model.nq);
  VectorXRef_const v = x.tail(model.nv);
  pin::forwardKinematics(model, *data, q, v);
  pin::computeJointJacobians(model, *data);
  pin::updateFramePlacement(model, *data, ee_fid);

  // The decision variables are joint velocity commands and slacks
  int num_slack = 3*model.nv + 1;
  int problem_size = model.nv + num_slack;
  solver.reset(problem_size);

  // LLSQ costs
  MatrixXd Jworld = MatrixXd::Zero(6, model.nv);
  pin::getFrameJacobian(model, *data, ee_fid, pin::ReferenceFrame::LOCAL_WORLD_ALIGNED, Jworld);

  MatrixXd position_LHS = MatrixXd::Zero(3, problem_size);
  position_LHS.leftCols(model.nv) = Jworld.topRows<3>();
  Vector3d pos_err = data->oMf[ee_fid].translation() - des_ee_pos;
  prev_pos_err_norm = pos_err.norm();
  Vector3d position_RHS = -pos_kp*pos_err;
  solver.add_llsq_objective(position_LHS, position_RHS, 1);

  // Orientation controller: PD on SO(3)
  MatrixXd orientation_LHS = MatrixXd::Zero(3, problem_size);
  orientation_LHS.leftCols(model.nv) = Jworld.bottomRows<3>();
  Vector3d rot_err = data->oMf[ee_fid].rotation()*pin::log3(des_ee_rmat.transpose()*data->oMf[ee_fid].rotation());
  prev_rot_err_norm = rot_err.norm();
  Vector3d orientation_RHS = -rot_kp*rot_err;
  solver.add_llsq_objective(orientation_LHS, orientation_RHS, 1);

  MatrixXd joint_limit_LHS = MatrixXd::Zero(model.nv, problem_size);

  // Joint angle limits
  joint_limit_LHS = MatrixXd::Zero(model.nv, problem_size);
  joint_limit_LHS.block(0, 0, model.nv, model.nv) = MatrixXd::Identity(model.nv, model.nv);
  joint_limit_LHS.block(0, model.nv, model.nv, model.nv) = MatrixXd::Identity(model.nv, model.nv); // Slack
  solver.add_constraint(joint_limit_LHS, 
                        (joint_angle_lower_limits - q.segment(0, model.nv))/(2*dt),
                        (joint_angle_upper_limits - q.segment(0, model.nv))/(2*dt));

  // Joint velocity limits
  joint_limit_LHS = MatrixXd::Zero(model.nv, problem_size);
  joint_limit_LHS.block(0, 0, model.nv, model.nv) = MatrixXd::Identity(model.nv, model.nv);
  joint_limit_LHS.block(0, model.nv + model.nv, model.nv, model.nv) = MatrixXd::Identity(model.nv, model.nv); // Slack
  solver.add_constraint(joint_limit_LHS, -joint_vel_limits, joint_vel_limits);

  // Joint acceleration limits
  joint_limit_LHS = MatrixXd::Zero(model.nv, problem_size);
  joint_limit_LHS.block(0, 0, model.nv, model.nv) = MatrixXd::Identity(model.nv, model.nv);
  joint_limit_LHS.block(0, model.nv + 2*model.nv, model.nv, model.nv) = MatrixXd::Identity(model.nv, model.nv); // Slack
  solver.add_constraint(joint_limit_LHS, v - joint_acc_limits*dt, v + joint_acc_limits*dt);

  MatrixXd cbf_LHS = MatrixXd::Zero(1, problem_size);
  cbf_LHS.block(0, 0, 1, model.nv) = barrier_vec.transpose()*Jworld.topRows<3>();
  cbf_LHS(0, model.nv + 3*model.nv) = 1; // Slack
  VectorXd cbf_ub = (barrier_vec.dot(barrier_pos - data->oMf[ee_fid].translation())/(10*dt))*VectorXd::Ones(1);
  solver.add_constraint(cbf_LHS, -1*std::numeric_limits<double>::infinity()*VectorXd::Ones(1), cbf_ub);

  MatrixXd quadratic_cost_R = MatrixXd::Zero(problem_size, problem_size);
  quadratic_cost_R.block(problem_size - num_slack, problem_size - num_slack, num_slack, num_slack) = 10000*MatrixXd::Identity(num_slack, num_slack);
  solver.add_quadratic_cost(quadratic_cost_R, 1);

  if (primal.size() != problem_size || dual.size() != problem_size) {
    solver.solve(primal, dual, primal, dual, false, false);
  } else {
    solver.solve(primal, dual, primal, dual, did_solve, did_solve);
  }

  did_solve = true;

  return primal.segment(0, model.nv);
}

void IKMoveEEBehindBarrier::reset() {
  did_solve = false;
}

void IKMoveEEBehindBarrier::set_gains(double pos_kp, double rot_kp) {
  this->pos_kp = pos_kp;
  this->rot_kp = rot_kp;
}

double IKMoveEEBehindBarrier::get_prev_pos_err_norm() {
  return prev_pos_err_norm;
}

double IKMoveEEBehindBarrier::get_prev_rot_err_norm() {
  return prev_rot_err_norm;
}
