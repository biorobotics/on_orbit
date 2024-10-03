from scipy.spatial.transform import Rotation as R
import numpy as np
import pinocchio as pin

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
      x_ipopt[pin_model.idx_vs[j] + 3:pin_model.idx_vs[j] + 6] = R.from_quat(x[pin_model.idx_qs[j] + 3:pin_model.idx_qs[j] + 7]).as_rotvec() # Orientation

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
      x[pin_model.idx_qs[j]:pin_model.idx_qs[j] + 3] = x_ipopt[pin_model.idx_vs[j]:pin_model.idx_vs[j] + 3]
      x[pin_model.idx_qs[j] + 3:pin_model.idx_qs[j] + 7] = R.from_rotvec(x_ipopt[pin_model.idx_vs[j] + 3:pin_model.idx_vs[j] + 6]).as_quat()

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
      rot_start = pos_end
      rot_end = rot_start + 3
      J[pos_start:pos_end, pos_start:pos_end] = R.from_rotvec(x_ipopt[rot_start:rot_end]).as_matrix().transpose() # Position
      J[rot_start:rot_end, rot_start:rot_end] = pin.Jexp3(x_ipopt[rot_start:rot_end]) # Orientation
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
      rot_start = pos_end
      rot_end = rot_start + 3
      quat_start = pin_model.idx_qs[j] + 3
      quat_end = quat_start + 4
      rmat = R.from_quat(x[quat_start:quat_end]).as_matrix()
      J[pos_start:pos_end, pos_start:pos_end] = rmat # Position
      J[rot_start:rot_end, rot_start:rot_end] = pin.Jlog3(rmat) # orientation
  return J
