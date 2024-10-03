import casadi as ca
import pinocchio as pin
from pinocchio import casadi as cpin
from casadi_rp_conversion import *
import numpy as np
from inf_def import inf

def export_relative_acceleration_constraint(cpin_model, input_vector, frame1, frame2, dt, acc_limits=np.zeros(6), enforce=np.array([True, True, True, True, True, True])):
    '''
    This function enforces that the relative acceleration between two frames is within given limits.

    Parameters:
    cpin_model: pinocchio model
    input_vector: casadi vector
    frame1: string containing the name of the first frame which will be used as the reference frame
    frame2: string containing the name of the second frame
    acc_limits: numpy array containing the limits for the relative linear and angular acceleration between the two frames
                Format: [x_lin, y_lin, z_lin, x_rot, y_rot, z_rot]
    dt: time step used for finite difference
    enforce: numpy array containing boolean values to enforce the constraints
    '''

    frame1_fid = cpin_model.getFrameId(frame1)
    frame2_fid = cpin_model.getFrameId(frame2)

    cpin_data = cpin_model.createData()

    num_rotary = cpin_model.nv - 12

    nv = cpin_model.nv
    nx_ipopt = 2 * nv
    nu = num_rotary + 3

    x_ipopt = input_vector[:nx_ipopt]
    u_ipopt = input_vector[nx_ipopt:nx_ipopt + nu]
    xnext_ipopt = input_vector[nx_ipopt + nu:2 * nx_ipopt + nu]
    unext_ipopt = input_vector[2 * nx_ipopt + nu:]

    q_ipopt = x_ipopt[:nv]
    v_ipopt = x_ipopt[nv:]

    qnext_ipopt = xnext_ipopt[:nv]
    vnext_ipopt = xnext_ipopt[nv:]

    q = q_to_pin(q_ipopt, cpin_model)
    v = v_ipopt

    qnext = q_to_pin(qnext_ipopt, cpin_model)
    vnext = vnext_ipopt

    # Forward kinematics for the current state
    cpin.forwardKinematics(cpin_model, cpin_data, q, v)
    cpin.updateFramePlacement(cpin_model, cpin_data, frame1_fid)
    cpin.updateFramePlacement(cpin_model, cpin_data, frame2_fid)

    frame1_pos = cpin_data.oMf[frame1_fid].translation
    frame1_pos_wrt_frame2 = cpin_data.oMf[frame2_fid].actInv(frame1_pos)

    frame1_v = cpin.getFrameVelocity(cpin_model, cpin_data, frame1_fid, cpin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    frame2_twist = cpin.getFrameVelocity(cpin_model, cpin_data, frame2_fid, cpin.ReferenceFrame.LOCAL)

    # Forward kinematics for the next state
    cpin.forwardKinematics(cpin_model, cpin_data, qnext, vnext)
    cpin.updateFramePlacement(cpin_model, cpin_data, frame1_fid)
    cpin.updateFramePlacement(cpin_model, cpin_data, frame2_fid)

    frame1_pos_next = cpin_data.oMf[frame1_fid].translation
    frame1_pos_wrt_frame2_next = cpin_data.oMf[frame2_fid].actInv(frame1_pos_next)

    frame1_v_next = cpin.getFrameVelocity(cpin_model, cpin_data, frame1_fid, cpin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    frame2_twist_next = cpin.getFrameVelocity(cpin_model, cpin_data, frame2_fid, cpin.ReferenceFrame.LOCAL)

    # Relative velocity current
    rel_linear_v_expr = cpin_data.oMf[frame2_fid].rotation.T @ frame1_v.linear - frame2_twist.linear - ca.cross(frame2_twist.angular, frame1_pos_wrt_frame2)
    rel_angular_v_expr = frame1_v.angular - frame2_twist.angular

    # Relative velocity next
    rel_linear_v_expr_next = cpin_data.oMf[frame2_fid].rotation.T @ frame1_v_next.linear - frame2_twist_next.linear - ca.cross(frame2_twist_next.angular, frame1_pos_wrt_frame2_next)
    rel_angular_v_expr_next = frame1_v_next.angular - frame2_twist_next.angular

    # Relative acceleration
    rel_linear_a_expr = (rel_linear_v_expr_next - rel_linear_v_expr) / dt
    rel_angular_a_expr = (rel_angular_v_expr_next - rel_angular_v_expr) / dt

    # Combine linear and angular acceleration constraints
    rel_a_expr = ca.vertcat(rel_linear_a_expr, rel_angular_a_expr)

    # Define the bounds for the constraints
    lb = np.full(6, -inf)
    ub = np.full(6, inf)

    lb[enforce] = -acc_limits[enforce]
    ub[enforce] = acc_limits[enforce]

    return rel_a_expr, lb, ub
