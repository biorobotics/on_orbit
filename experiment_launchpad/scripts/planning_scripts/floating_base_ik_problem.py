import numpy as np
import cyipopt
import scipy.sparse
from scipy.spatial.transform import Rotation as R
from rp_conversion import *

class FloatingBaseIKProblem(object):
  def __init__(self, pin_model, q0, joint_angle_lower_limits, joint_angle_upper_limits, goal_ee_pos, goal_ee_rmat, fix_base_pose=False):
    self.pin_model = pin_model
    self.pin_data = pin.Data(self.pin_model)
    self.num_rotary = pin_model.nv - 6

    self.tip_fid = pin_model.getFrameId('ee_tip')

    self.com = np.copy(pin.centerOfMass(self.pin_model, self.pin_data, q0))

    self.q0 = np.copy(q0)
    self.q0_ipopt = from_pin(q0, self.pin_model, True)

    self.fix_base_pose = fix_base_pose
    if self.fix_base_pose:
      self.num_constraints = 12 # Nonlinear equality constraints on base pose, EE pose
    else:
      self.num_constraints = 9 # Nonlinear equality constraints on CoM position, EE pose
    self.cl = np.zeros(self.num_constraints)
    self.cu = np.zeros(self.num_constraints)

    self.num_decision_vars = pin_model.nv

    self.lb = -10000*np.ones(self.num_decision_vars)
    self.ub = 10000*np.ones(self.num_decision_vars)

    # Joint angles within limits
    self.lb[6:6 + self.num_rotary] = joint_angle_lower_limits
    self.ub[6:6 + self.num_rotary] = joint_angle_upper_limits

    for i in range(self.num_rotary):
      if np.isinf(joint_angle_lower_limits[i]):
        self.lb[6 + i] = -3.14
      if np.isinf(joint_angle_upper_limits[i]):
        self.ub[6 + i] = 3.14

    self.goal_ee_pos = np.copy(goal_ee_pos)
    self.goal_ee_rmat = np.copy(goal_ee_rmat)

  def solve(self, warm_start=None, verbose=True):
    if warm_start is None:
      warm_start = np.copy(self.q0_ipopt)

    nlp = cyipopt.Problem(
       n=self.num_decision_vars,
       m=self.num_constraints,
       problem_obj=self,
       lb=self.lb,
       ub=self.ub,
       cl=self.cl,
       cu=self.cu,
    )
    if not verbose:
      nlp.addOption(b'print_level', 0)
    nlp.addOption(b'tol', 1.0)
    nlp.addOption(b'compl_inf_tol', 1e-2)
    nlp.addOption(b'warm_start_init_point', 'yes')
    nlp.addOption(b'hessian_approximation', 'limited-memory')

    soln, info = nlp.solve(warm_start)

    return to_pin(soln, self.pin_model, True), info['status'] in [0, 1, 5]

  def get_num_decision_vars(self):
    return self.num_decision_vars
    
  def get_num_constraints(self):
    return self.num_constraints

  def objective(self, q_ipopt):
    return 0

  def gradient(self, q_ipopt):
    """Returns the gradient of the objective with respect to v."""
    obj = self.objective(q_ipopt)
    q_ipopt_plus = np.copy(q_ipopt)
    eps = 1e-5
    grad = np.zeros(self.num_decision_vars)
    for i in range(self.num_decision_vars):
      q_ipopt_plus[i] += eps
      obj_plus = self.objective(q_ipopt_plus)
      grad[i] = (obj_plus - obj)/eps
      q_ipopt_plus[i] = q_ipopt[i]
    return grad

  def constraints(self, q_ipopt):
    """Returns the constraints."""
    q = to_pin(q_ipopt, self.pin_model, True)
    com = pin.centerOfMass(self.pin_model, self.pin_data, q)
    pin.forwardKinematics(self.pin_model, self.pin_data, q)
    pin.updateFramePlacement(self.pin_model, self.pin_data, self.tip_fid)
    if self.fix_base_pose:
      return np.concatenate((q_ipopt[:6] - self.q0_ipopt[:6], \
                             self.pin_data.oMf[self.tip_fid].translation - self.goal_ee_pos, \
                             pin.log3(self.goal_ee_rmat@self.pin_data.oMf[self.tip_fid].rotation.transpose())))
    else:
      return np.concatenate((com - self.com, \
                             self.pin_data.oMf[self.tip_fid].translation - self.goal_ee_pos, \
                             pin.log3(self.goal_ee_rmat@self.pin_data.oMf[self.tip_fid].rotation.transpose())))

  def jacobian(self, q_ipopt):
    """Returns the Jacobian of the constraints with respect to v."""
    c = self.constraints(q_ipopt)
    q_ipopt_plus = np.copy(q_ipopt)
    eps = 1e-5
    J = np.zeros((self.num_constraints, self.num_decision_vars))
    for i in range(self.num_decision_vars):
      q_ipopt_plus[i] += eps
      c_plus = self.constraints(q_ipopt_plus)
      J[:, i] = (c_plus - c)/eps
      q_ipopt_plus[i] = q_ipopt[i]
    return J.flatten()

  def jacobianstructure(self):
    """Returns the row and column indices for non-zero vales of the
    Jacobian."""
    rows = [row*np.ones(self.num_decision_vars, dtype=np.int64) for row in range(self.num_constraints)]
    cols = [np.arange(self.num_decision_vars) for row in range(self.num_constraints)]
    return np.concatenate(rows), np.concatenate(cols)

  def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                   d_norm, regularization_size, alpha_du, alpha_pr,
                   ls_trials):
    """Prints information at every Ipopt iteration."""
    return True

    # msg = "Objective value at iteration #{:d} is - {:g}"

    #print(msg.format(iter_count, obj_value))
