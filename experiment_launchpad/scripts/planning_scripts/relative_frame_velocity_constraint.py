import casadi as ca
import pinocchio as pin
from pinocchio import casadi as cpin
from casadi_rp_conversion import *
import numpy as np
from inf_def import inf

def export_relative_velocity_constraint(cpin_model, input_vector, frame1, frame2, upper_tol= np.zeros(6), lower_tol=np.zeros(6) ,vel_diff=np.zeros(6), enforce = np.array([True, True, True, True, True, True])):
    '''
    This function enforces that the relative velocity between two frames is equal to a given value.

    Parameters:
    cpin_model: pinocchio model
    input_vector: casadi vector
    frame1: string containing the name of the first frame which will be used as the reference frame
    frame2: string containing the name of the second frame
    upper_tol: numpy array containing the upper bound for the constraints
    lower_tol: numpy array containing the lower bound for the constraints
    vel_diff: numpy array containing the desired relative linear and angular velocity between the two frames
              Format: [x_lin, y_lin, z_lin, x_rot, y_rot, z_rot]
    enforce: numpy array containing boolean values to enforce the constraints
   
    '''

    frame1_fid = cpin_model.getFrameId(frame1)
    frame2_fid = cpin_model.getFrameId(frame2)

    cv_jidx = cpin_model.getJointId('world_to_client')
    cv_qidx = cpin_model.idx_qs[cv_jidx]
    cv_vidx = cpin_model.idx_vs[cv_jidx]

    mrv_jidx = cpin_model.getJointId('world_to_base')
    mrv_qidx = cpin_model.idx_qs[mrv_jidx]
    mrv_vidx = cpin_model.idx_vs[mrv_jidx]
    num_rotary = cpin_model.nv - 12
    mrv_nq = cpin_model.nq - 7
    mrv_nv = cpin_model.nv - 6

    cpin_data = cpin_model.createData()

    nx_ipopt = 2 * cpin_model.nv
    nu = num_rotary + 3

    x_ipopt = input_vector[:nx_ipopt]
    q_ipopt = x_ipopt[:cpin_model.nv]
    v_ipopt = x_ipopt[cpin_model.nv:]

    q = q_to_pin(q_ipopt, cpin_model)
    v = v_ipopt

    cpin.forwardKinematics(cpin_model, cpin_data, q, v)
    cpin.updateFramePlacement(cpin_model, cpin_data, frame1_fid)
    cpin.updateFramePlacement(cpin_model, cpin_data, frame2_fid)

    frame1_pos = cpin_data.oMf[frame1_fid].translation
    frame1_pos_wrt_frame2 = cpin_data.oMf[frame2_fid].actInv(frame1_pos)

    frame1_v = cpin.getFrameVelocity(cpin_model, cpin_data, frame1_fid, cpin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    frame2_twist = cpin.getFrameVelocity(cpin_model, cpin_data, frame2_fid, cpin.ReferenceFrame.LOCAL)
    
    # Relative linear velocity expression
    rel_linear_v_expr = cpin_data.oMf[frame2_fid].rotation.T @ frame1_v.linear - frame2_twist.linear - ca.cross(frame2_twist.angular, frame1_pos_wrt_frame2)
    
    # Relative angular velocity expression
    rel_angular_v_expr = frame1_v.angular - frame2_twist.angular

    # Combine linear and angular velocity constraints
    rel_v_expr = ca.vertcat(rel_linear_v_expr, rel_angular_v_expr)
    vel_diff_combined = rel_v_expr - vel_diff

    # Define the bounds for the constraints
    lb = np.full(6, -inf)
    ub = np.full(6, inf)

    lb[enforce] = lower_tol[enforce]
    ub[enforce] = upper_tol[enforce]

    return vel_diff_combined, lb, ub
