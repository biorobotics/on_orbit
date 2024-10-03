import casadi as ca
import pinocchio as pin
from pinocchio import casadi as cpin
from casadi_rp_conversion import *
import numpy as np
from inf_def import inf

def zero_joint_vel_constraint(cpin_model, input_vector):

  mrv_jidx = cpin_model.getJointId('world_to_base')
  mrv_qidx = cpin_model.idx_qs[mrv_jidx]
  mrv_vidx = cpin_model.idx_vs[mrv_jidx]
  num_rotary = cpin_model.nv - 12
  mrv_nv = cpin_model.nv - 6

  nq_acados = cpin_model.nv
  nx_acados = 2*cpin_model.nv
  v_acados = input_vector[nq_acados:nx_acados]
  
  lb = np.zeros(num_rotary)
  ub = np.zeros(num_rotary)
  return ca.vertcat(v_acados[mrv_vidx + 6:mrv_vidx + mrv_nv]), lb, ub
