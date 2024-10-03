import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from scipy.spatial.transform import Rotation as R

# RP to quat
def cayley_map(phi):
  return 1/np.sqrt(1 + phi@phi)*np.concatenate((phi, [1]))

# Quat to RP
def inv_cayley_map(q):
  return q[:3]/q[3]

def JSO3_to_rp(q):
  return 0.5*(np.eye(3) + pin.skew(q[:3])/q[3] + np.outer(q[:3], q[:3])/q[3]**2)

def Jrp_to_SO3(phi):
  return 2/(1 + phi@phi)*(np.eye(3) - pin.skew(phi[:3]))

def rp_to_rmat(phi):
  phi_skew = pin.skew(phi)
  return 1/(1 + phi@phi)*(np.eye(3) + 2*phi_skew + phi_skew@phi_skew + np.outer(phi, phi))

# All functions here assume that the pinocchio model only contains revolute and 
# free-floating joints

# Converts from pinocchio coordinates to IPOPT coordinates
def from_pin(x, pin_model, q_only=False):
  if q_only:
    x_ipopt = np.zeros(pin_model.nv, x.dtype)
  else:
    x_ipopt = np.zeros(2*pin_model.nv, x.dtype)
    x_ipopt[pin_model.nv:] = x[pin_model.nq:] # Velocity (no conversion)

  # Configuration
  for j in range(1, pin_model.njoints):
    if pin_model.joints[j].nv == 1:
      # Revolute
      x_ipopt[pin_model.idx_vs[j]] = x[pin_model.idx_qs[j]]
    else:
      # Free-floating
      x_ipopt[pin_model.idx_vs[j]:pin_model.idx_vs[j] + 3] = x[pin_model.idx_qs[j]:pin_model.idx_qs[j] + 3] # Position
      x_ipopt[pin_model.idx_vs[j] + 3:pin_model.idx_vs[j] + 6] = inv_cayley_map(x[pin_model.idx_qs[j] + 3:pin_model.idx_qs[j] + 7]) # Orientation

  return x_ipopt

# Converts to pinocchio coordinates from IPOPT coordinates
def to_pin(x_ipopt, pin_model, q_only=False):
  if q_only:
    x = np.zeros(pin_model.nq, x_ipopt.dtype)
  else:
    x = np.zeros(pin_model.nq + pin_model.nv, x_ipopt.dtype)
    x[pin_model.nq:] = x_ipopt[pin_model.nv:] # Velocity (no conversion)

  # Configuration
  for j in range(1, pin_model.njoints):
    if pin_model.joints[j].nv == 1:
      # Revolute
      x[pin_model.idx_qs[j]] = x_ipopt[pin_model.idx_vs[j]]
    else:
      # Free-floating
      x[pin_model.idx_qs[j]:pin_model.idx_qs[j] + 3] = x_ipopt[pin_model.idx_vs[j]:pin_model.idx_vs[j] + 3] # Position
      x[pin_model.idx_qs[j] + 3:pin_model.idx_qs[j] + 7] = cayley_map(x_ipopt[pin_model.idx_vs[j] + 3:pin_model.idx_vs[j] + 6])

  return x

# Jacobian of the conversion from IPOPT coordinates to pinocchio coordinates
def J_to_pin(x_ipopt, pin_model, q_only=False):
  if q_only:
    J = np.eye(pin_model.nv)
  else:
    J = np.eye(2*pin_model.nv)
  # Configuration
  for j in range(1, pin_model.njoints):
    if pin_model.joints[j].nv == 6:
      # Free-floating
      pos_start = pin_model.idx_vs[j]
      pos_end = pos_start + 3
      rp_start = pos_end
      rp_end = rp_start + 3
      J[pos_start:pos_end, pos_start:pos_end] = rp_to_rmat(x_ipopt[rp_start:rp_end]).transpose() # Position
      J[rp_start:rp_end, rp_start:rp_end] = Jrp_to_SO3(x_ipopt[rp_start:rp_end]) # Orientation
  return J

# Jacobian of the conversion from pinocchio coordinates to IPOPT coordinates
def J_from_pin(x, pin_model, q_only=False):
  if q_only:
    J = np.eye(pin_model.nv)
  else:
    J = np.eye(2*pin_model.nv)
  # Configuration
  for j in range(1, pin_model.njoints):
    if pin_model.joints[j].nv == 6:
      # Free-floating
      pos_start = pin_model.idx_vs[j]
      pos_end = pos_start + 3
      rp_start = pos_end
      rp_end = rp_start + 3
      quat_start = pin_model.idx_qs[j] + 3
      quat_end = quat_start + 4
      quat = x[quat_start:quat_end]
      rmat = R.from_quat(quat).as_matrix()
      J[pos_start:pos_end, pos_start:pos_end] = rmat # Position
      J[rp_start:rp_end, rp_start:rp_end] = JSO3_to_rp(quat) # orientation
  return J

if __name__ == '__main__':
  urdf_file = '../urdf/robot.urdf'
  pin_model = RobotWrapper.BuildFromURDF(urdf_file, root_joint=pin.JointModelFreeFlyer()).model
  pin_model.gravity.setZero()
  pin_data = pin.Data(pin_model)

  eps = 1e-7

  q = pin.integrate(pin_model, pin.neutral(pin_model), np.ones(pin_model.nv))
  q_ipopt = from_pin(q, pin_model, True)

  Jfrom = J_from_pin(q, pin_model, True)
  Jfrom_fd = np.zeros_like(Jfrom)
  dq = np.zeros(pin_model.nv)
  for i in range(pin_model.nv):
    dq[i] = eps
    q_plus = pin.integrate(pin_model, q, dq)
    q_ipopt_plus = from_pin(q_plus, pin_model, True)
    Jfrom_fd[:, i] = (q_ipopt_plus - q_ipopt)/eps
    dq[i] = 0

  print(np.amax(np.abs(Jfrom)))
  print(Jfrom)
  print(np.amax(np.abs(Jfrom - Jfrom_fd)))

  q_back = to_pin(q_ipopt, pin_model, True)
  Jto = J_to_pin(q_ipopt, pin_model, True)
  Jto_fd = np.zeros_like(Jto)
  q_ipopt_plus = np.copy(q_ipopt)
  for i in range(pin_model.nv):
    q_ipopt_plus[i] += eps
    q_back_plus = to_pin(q_ipopt_plus, pin_model, True)
    Jto_fd[:, i] = pin.difference(pin_model, q_back, q_back_plus)/eps
    q_ipopt_plus[i] = q_ipopt[i]

  print(np.amax(np.abs(Jto)))
  print(Jto)
  print(np.amax(np.abs(Jto - Jto_fd)))
