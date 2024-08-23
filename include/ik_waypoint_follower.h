#pragma once

#include <pinocchio/fwd.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include "constrained_llsq.h"
#include <memory>

using namespace std;
using namespace Eigen;

namespace pin = pinocchio;

typedef const Ref<const VectorXd>& VectorXRef_const;
typedef const Ref<const Vector3d>& Vector3Ref_const;
typedef const Ref<const Vector4d>& Vector4Ref_const;

typedef const Ref<const MatrixXd>& MatrixXRef_const;
typedef const Ref<const Matrix3d>& Matrix3Ref_const;

class IKWaypointFollower {
  public:
    IKWaypointFollower(string urdf_file, VectorXRef_const joint_angle_lower_limits, VectorXRef_const joint_angle_upper_limits, VectorXRef_const joint_vel_limits, VectorXRef_const joint_acc_limits, double dt);

    // Goal vel and w should be in world frame
    VectorXd get_control(VectorXRef_const x, Vector3Ref_const pos_waypoint, Matrix3Ref_const rmat_waypoint, Vector3Ref_const v_waypoint, Vector3Ref_const w_waypoint, VectorXRef_const wrist_wrench, bool use_pos_waypoint, bool use_rmat_waypoint);

    VectorXd integrate_from_cache(VectorXRef_const h_0);

    void reset();

    double get_qp_time();

    double get_qp_setup_time();

    double get_dynamics_time();

    void set_gains(double force_kp, double force_ki, double torque_kp, Vector3Ref_const pos_kp, double rot_kp);

  private:
    int wrist_id;
    ConstrainedLLSQ solver;
    pin::Model model;
    unique_ptr<pin::Data> data;
    VectorXd joint_angle_lower_limits;
    VectorXd joint_angle_upper_limits;
    VectorXd joint_vel_limits;
    VectorXd joint_acc_limits;
    VectorXd primal;
    VectorXd dual;
    bool did_solve; // Needs to be reset in the reset() function
    pin::FrameIndex tip_id;
    double dt;
    int num_rotary;
    Vector3d force_err_integral; // Needs to be reset in the reset() function
    VectorXd prev_acc; // Needs to be reset in the reset() function

    double force_kp;
    double force_ki;
    double torque_kp;

    Vector3d pos_kp;
    double rot_kp;

    VectorXd cached_x;
    MatrixXd integration_LHS;
    VectorXd integration_RHS;

    double qp_time;
    double qp_setup_time;
    double dynamics_time;
};
