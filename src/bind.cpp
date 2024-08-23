#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl_bind.h>

#include "ik_ee_trajectory_tracker.h"
#include "ik_waypoint_follower.h"
#include "ik_move_ee_behind_barrier.h"
#include "osqp_wrapper.h"


namespace py = pybind11;

PYBIND11_MODULE(on_orbit_bindings, m) {
  py::class_<IKEETrajectoryTracker>(m, "IKEETrajectoryTracker")
      .def(py::init<string, VectorXRef_const, VectorXRef_const, VectorXRef_const, VectorXRef_const, double, double>())
      .def("get_control", &IKEETrajectoryTracker::get_control)
      .def("integrate_from_cache", &IKEETrajectoryTracker::integrate_from_cache)
      .def("reset", &IKEETrajectoryTracker::reset)
      .def("get_qp_setup_time", &IKEETrajectoryTracker::get_qp_setup_time)
      .def("get_qp_time", &IKEETrajectoryTracker::get_qp_time)
      .def("get_dynamics_time", &IKEETrajectoryTracker::get_dynamics_time)
      .def("set_gains", &IKEETrajectoryTracker::set_gains)
      .def("set_guard_speed", &IKEETrajectoryTracker::set_guard_speed)
      .def("set_use_cone_cbf", &IKEETrajectoryTracker::set_use_cone_cbf)
      ;
  py::class_<IKWaypointFollower>(m, "IKWaypointFollower")
      .def(py::init<string, VectorXRef_const, VectorXRef_const, VectorXRef_const, VectorXRef_const, double>())
      .def("get_control", &IKWaypointFollower::get_control)
      .def("integrate_from_cache", &IKWaypointFollower::integrate_from_cache)
      .def("reset", &IKWaypointFollower::reset)
      .def("get_qp_setup_time", &IKWaypointFollower::get_qp_setup_time)
      .def("get_qp_time", &IKWaypointFollower::get_qp_time)
      .def("get_dynamics_time", &IKWaypointFollower::get_dynamics_time)
      .def("set_gains", &IKWaypointFollower::set_gains)
      ;
 
  py::class_<IKMoveEEBehindBarrier>(m, "IKMoveEEBehindBarrier")
      .def(py::init<string, VectorXRef_const, VectorXRef_const, VectorXRef_const, VectorXRef_const, double, string>())
      .def("get_vel_cmd", &IKMoveEEBehindBarrier::get_vel_cmd)
      .def("get_prev_pos_err_norm", &IKMoveEEBehindBarrier::get_prev_pos_err_norm)
      .def("get_prev_rot_err_norm", &IKMoveEEBehindBarrier::get_prev_rot_err_norm)
      .def("reset", &IKMoveEEBehindBarrier::reset)
      .def("set_gains", &IKMoveEEBehindBarrier::set_gains)
      ;
  py::class_<OsqpWrapper>(m, "OsqpWrapper")
      .def(py::init<int, int>())
      .def("reset_num_decision_vars", &OsqpWrapper::reset_num_decision_vars)
      .def("reset_num_constraints", &OsqpWrapper::reset_num_constraints)
      .def("update_hessian", &OsqpWrapper::update_hessian)
      .def("update_constraint_matrix", &OsqpWrapper::update_constraint_matrix)
      .def("update_constraint_lb", &OsqpWrapper::update_constraint_lb)
      .def("update_constraint_ub", &OsqpWrapper::update_constraint_ub)
      .def("update_gradient", &OsqpWrapper::update_gradient)
      .def("solve", &OsqpWrapper::solve)
      ;

}
