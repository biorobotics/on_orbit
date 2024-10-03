import casadi as ca
import pinocchio as pin
from pinocchio import casadi as cpin
from casadi_rp_conversion import *
import numpy as np
from inf_def import inf

def export_algin_frames_constraint(cpin_model, x_acados, frame1, frame2, upper_tol=np.zeros(6), lower_tol=np.zeros(6), pos_diff=np.array([0, 0, 0]), rot_diff=np.array([0, 0, 0]), enforce = np.array([True, True, True, True, True, True])):
  '''
  This function enforces that the relative position and orientation between two frames is equal to a given value.

  Parameters:
  cpin_model: pinocchio model
  input_vector: casadi vector
  frame1: string containing the name of the first frame which will be used as the reference frame
  frame2: string containing the name of the second frame
  pos_diff: numpy array containing the desired relative position between the two frames
  rot_diff: numpy array containing the desired relative orientation between the two frames
  '''
   
  cpin_data = cpin_model.createData()

  frame1_fid = cpin_model.getFrameId(frame1)
  frame2_fid = cpin_model.getFrameId(frame2)

  q = q_to_pin(x_acados[:cpin_model.nv], cpin_model)

  cpin.forwardKinematics(cpin_model, cpin_data, q)
  cpin.updateFramePlacement(cpin_model, cpin_data, frame1_fid)
  cpin.updateFramePlacement(cpin_model, cpin_data, frame2_fid)

  frame1_pos_wrt_frame2 = cpin_data.oMf[frame2_fid].rotation.T@(cpin_data.oMf[frame1_fid].translation - cpin_data.oMf[frame2_fid].translation)
  frame1_rmat_wrt_frame2 = cpin_data.oMf[frame2_fid].rotation.T@cpin_data.oMf[frame1_fid].rotation
  rot_err = cpin.unSkew(0.5*(frame1_rmat_wrt_frame2 - frame1_rmat_wrt_frame2.T))
  
  # Define the constraints
  position_err = frame1_pos_wrt_frame2 - cpin_data.oMf[frame2_fid].rotation.T@pos_diff
  position_con = position_err
  rotation_con = rot_err - rot_diff

  constraints = ca.vertcat(position_con, rotation_con)


  # Define the bounds for the constraints
  lb = np.full(6, -inf)
  ub = np.full(6, inf)

  lb[enforce] = lower_tol[enforce]
  ub[enforce] = upper_tol[enforce]

  return constraints, lb, ub


