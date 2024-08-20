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

class IKEETrajectoryTracker {
  public:
    IKEETrajectoryTracker(string urdf_file, VectorXRef_const joint_angle_lower_limits, VectorXRef_const joint_angle_upper_limits, VectorXRef_const joint_vel_limits, VectorXRef_const joint_acc_limits, double dt, double cone_slope);

    VectorXd get_control(VectorXRef_const x, Vector3Ref_const des_pos_wrt_client, Matrix3Ref_const des_rmat_wrt_client, Vector3Ref_const des_vel_wrt_client, Vector3Ref_const des_w_wrt_client, VectorXRef_const target_wrench, VectorXRef_const wrist_wrench, bool in_hole);

    VectorXd integrate_from_cache(VectorXRef_const h_0);

    void reset();

    double get_qp_time();

    double get_qp_setup_time();

    double get_dynamics_time();

    void set_gains(double force_kp, double force_ki);

    void set_guard_speed(double guard_speed);
    
    void set_use_cone_cbf(bool use_cone_cbf);

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
    pin::FrameIndex cone_vertex_id;
    pin::FrameIndex nozzle_id;
    double dt;
    int mrv_jidx;
    int mrv_qidx;
    int mrv_vidx;
    int cv_jidx;
    int cv_qidx;
    int cv_vidx;
    int num_rotary;
    Vector3d force_err_integral; // Needs to be reset in the reset() function
    VectorXd prev_acc; // Needs to be reset in the reset() function

    double force_kp;
    double force_ki;
    bool made_contact; // Needs to be reset in the reset() function
    double guard_speed;
    bool use_cone_cbf;

    double cone_slope;

    const double client_mass = 6000; // kg
    Matrix3d client_inertia;
    Matrix3d client_inertia_inverse;

    VectorXd cached_x;
    MatrixXd integration_LHS;
    VectorXd integration_RHS;

    double qp_time;
    double qp_setup_time;
    double dynamics_time;
};
