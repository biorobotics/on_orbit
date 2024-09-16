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

class IKMoveEEBehindBarrier {
  public:
    IKMoveEEBehindBarrier(string urdf_file, VectorXRef_const joint_angle_lower_limits, VectorXRef_const joint_angle_upper_limits, VectorXRef_const joint_vel_limits, VectorXRef_const joint_acc_limits, double dt, string ee_name);

    VectorXd get_vel_cmd(VectorXRef_const x, Vector3Ref_const des_ee_pos, Matrix3Ref_const des_ee_rmat, Vector3Ref_const barrier_pos, Vector3Ref_const barrier_vec);

    void reset();

    void set_gains(double pos_kp, double rot_kp);

    double get_prev_pos_err_norm();
    double get_prev_rot_err_norm();

  private:
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
    pin::FrameIndex ee_fid;
    double dt;

    double prev_pos_err_norm;
    double prev_rot_err_norm;

    double pos_kp;
    double rot_kp;
};
