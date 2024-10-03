import casadi as ca
import pinocchio as pin
from pinocchio import casadi as cpin

# RP to quat
def cayley_map(phi):
  return 1/ca.sqrt(1 + ca.dot(phi, phi))*ca.vertcat(phi, 1)

def Jrp_to_SO3(phi):
  return 2/(1 + ca.dot(phi, phi))*(ca.SX.eye(3) - cpin.skew(phi[:3]))

def JSO3_to_rp(q):
  return 0.5*(ca.SX.eye(3) + cpin.skew(q[:3])/q[3] + q[:3]@q[:3].T/q[3]**2)

# Currently, we have mrv, then cv in the q and v vectors in the urdf.
# I couldn't find a casadi 'contacatenate-by-index' type of function in the docs,
# but maybe it exists
def q_to_pin(q_acados, cpin_model):
  cv_jidx = cpin_model.getJointId('world_to_client')
  cv_qidx = cpin_model.idx_qs[cv_jidx]
  cv_vidx = cpin_model.idx_vs[cv_jidx]
  
  mrv_jidx = cpin_model.getJointId('world_to_base')
  mrv_qidx = cpin_model.idx_qs[mrv_jidx]
  mrv_vidx = cpin_model.idx_vs[mrv_jidx]
  num_rotary = cpin_model.nv - 12
  mrv_nv = cpin_model.nv - 6

  return  ca.vertcat(q_acados[mrv_vidx:mrv_vidx + 3], \
                     cayley_map(q_acados[mrv_vidx + 3:mrv_vidx + 6]), \
                     q_acados[mrv_vidx + 6:mrv_vidx + mrv_nv], \
                     q_acados[cv_vidx:cv_vidx + 3], \
                     cayley_map(q_acados[cv_vidx + 3:cv_vidx + 6]))
