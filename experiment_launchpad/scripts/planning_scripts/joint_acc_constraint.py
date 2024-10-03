import casadi as ca
import pinocchio as pin
from pinocchio import casadi as cpin
from casadi_rp_conversion import *
import numpy as np
from inf_def import inf

def export_joint_acc_constraint(cpin_model, input_vector, joint_acc_limits, dt):
  cpin_data = cpin_model.createData()

  mrv_jidx = cpin_model.getJointId('world_to_base')
  mrv_qidx = cpin_model.idx_qs[mrv_jidx]
  mrv_vidx = cpin_model.idx_vs[mrv_jidx]
  num_rotary = cpin_model.nv - 12
  mrv_nv = cpin_model.nv - 6

  nq_acados = cpin_model.nv
  nx_acados = 2*cpin_model.nv
  nu = num_rotary + 3
  v_acados = input_vector[nq_acados:nx_acados]
  vnext_acados = input_vector[nx_acados + nu + nq_acados:nx_acados + nu + nx_acados]
  vdiff = vnext_acados[mrv_vidx + 6:mrv_vidx + mrv_nv] - v_acados[mrv_vidx + 6:mrv_vidx + mrv_nv]
  joint_acc_limits = ca.DM(joint_acc_limits)

  size = 2*joint_acc_limits.shape[0]
  lb = -inf*np.ones(size)
  ub = np.zeros(size)
  return ca.vertcat(vdiff - dt*joint_acc_limits, -vdiff - dt*joint_acc_limits), lb, ub
