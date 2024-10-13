import pinocchio as pin
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import cho_solve, cho_factor

class R3SO3EKF(object):
  def __init__(self):
    self.SO3_pin_model = pin.Model()
    self.SO3_pin_model.addJoint(0, pin.JointModelSpherical(), pin.SE3.Identity(), 'base_joint')
    self.nq = 7 #number of configuration variables
    self.nv = 6 #number of velocity variables
    self.nx = self.nq + self.nv
    self.ndx = 2*self.nv
    self.kf_x = np.zeros(self.nx)
    self.kf_cov0 = np.eye(self.ndx)
    self.kf_cov = np.copy(self.kf_cov0)
    self.kf_Q = 0.01*np.eye(self.ndx)
    self.nz = 7
    self.ndz = 6
    self.kf_R = 10*np.eye(self.ndz)
    self.initialized = False

    self.client_mass = 1050
    self.client_inertia = np.diag([1792.419, 2015.632, 630.731])
    self.client_inertia_inv = np.linalg.inv(self.client_inertia)

  # w should global, not local
  def initialize(self, pos, quat, posdot, w):
    self.kf_x[:3] = np.copy(pos)
    self.kf_x[3:7] = np.copy(quat)
    self.kf_x[self.nq:self.nq + 3] = np.copy(posdot)
    self.kf_x[self.nq + 3:self.nq + 6] = np.copy(w)

    self.kf_cov = np.copy(self.kf_cov0)

    self.initialized = True

  def get_w_next(self, rmat, w, tau_client, dt):
    wlocal = rmat.transpose()@w
    wlocal_next = wlocal + self.client_inertia_inv@(rmat.transpose()@tau_client - np.cross(wlocal, self.client_inertia@wlocal))*dt
    w_next = rmat@wlocal_next
    return w_next

  # f_client should be in the base frame
  def get_pred_cov(self, base_v, base_w, f_client, tau_client, dt): 
    if not self.initialized:
      print('Attempting to run prediction on uninitialized EKF')
      return

    # Kalman prediction
    pos = self.kf_x[:3]
    rmat = R.from_quat(self.kf_x[3:7]).as_matrix()
    posdot = self.kf_x[self.nq:self.nq + 3]
    posddot = f_client/self.client_mass
    w = self.kf_x[self.nq + 3:self.nq + 6]

    wlocal = rmat.transpose()@w
    wlocal_next = wlocal + self.client_inertia_inv@(rmat.transpose()@tau_client - np.cross(wlocal, self.client_inertia@wlocal))*dt
    w_next = rmat@wlocal_next

    dwlocal_drmat = pin.skew(rmat.transpose()@w)
    dwlocal_dw = rmat.transpose()
    dwlocal_next_dwlocal = np.eye(3) + dt*self.client_inertia_inv@(-pin.skew(wlocal)@self.client_inertia + pin.skew(self.client_inertia@wlocal))
    dwlocal_next_drmat = dt*self.client_inertia_inv@pin.skew(rmat.transpose()@tau_client)
    dw_next_dwlocal_next = rmat
    dw_next_drmat = -rmat@pin.skew(wlocal_next) + dw_next_dwlocal_next@dwlocal_next_drmat + dw_next_dwlocal_next@dwlocal_next_dwlocal@dwlocal_drmat
    dw_next_dw = dw_next_dwlocal_next@dwlocal_next_dwlocal@dwlocal_dw

    base_w_skew = pin.skew(base_w)
    dpos = (posdot + 0.5*posddot*dt - base_v - base_w_skew@pos)*dt
    dR = pin.exp3((w - base_w)*dt)
    kf_F = np.eye(self.ndx)
    kf_F[:3, :3] += -base_w_skew*dt
    kf_F[:3, 6:9] = dt*np.eye(3)
    kf_F[3:6, 9:12] = rmat.transpose()@pin.Jexp3((w - base_w)*dt)*dt
    kf_F[9:12, 3:6] = dw_next_drmat
    kf_F[9:12, 9:12] = dw_next_dw
    return kf_F@self.kf_cov@kf_F.transpose() + self.kf_Q

  # f_client should be in the base frame
  def predict(self, base_v, base_w, f_client, tau_client, dt): 
    if not self.initialized:
      print('Attempting to run prediction on uninitialized EKF')
      return

    # Kalman prediction with constant acceleration model
    pos = self.kf_x[:3]
    rmat = R.from_quat(self.kf_x[3:7]).as_matrix()
    posdot = self.kf_x[self.nq:self.nq + 3]
    posddot = f_client/self.client_mass
    w = self.kf_x[self.nq + 3:self.nq + 6]

    wlocal = rmat.transpose()@w
    wlocal_next = wlocal + self.client_inertia_inv@(rmat.transpose()@tau_client - np.cross(wlocal, self.client_inertia@wlocal))*dt
    w_next = rmat@wlocal_next

    dwlocal_drmat = pin.skew(rmat.transpose()@w)
    dwlocal_dw = rmat.transpose()
    dwlocal_next_dwlocal = np.eye(3) + dt*self.client_inertia_inv@(-pin.skew(wlocal)@self.client_inertia + pin.skew(self.client_inertia@wlocal))
    dwlocal_next_drmat = dt*self.client_inertia_inv@pin.skew(rmat.transpose()@tau_client)
    dw_next_dwlocal_next = rmat
    dw_next_drmat = -rmat@pin.skew(wlocal_next) + dw_next_dwlocal_next@dwlocal_next_drmat + dw_next_dwlocal_next@dwlocal_next_dwlocal@dwlocal_drmat
    dw_next_dw = dw_next_dwlocal_next@dwlocal_next_dwlocal@dwlocal_dw

    '''
    w_next = self.get_w_next(rmat, w, tau_client, dt)
    dw_next_drmat_fd = np.zeros((3, 3))
    dw_next_dw_fd = np.zeros((3, 3))
    eps = 1e-7
    delta = np.zeros(3)
    for i in range(3):
        delta[i] = eps
        rmat_plus = rmat@pin.exp3(delta)
        dw_next_drmat_fd[:, i] = (self.get_w_next(rmat_plus, w, tau_client, dt) - w_next)/eps
        w_plus = w + delta
        dw_next_dw_fd[:, i] = (self.get_w_next(rmat, w_plus, tau_client, dt) - w_next)/eps
        delta[i] = 0

    print(dw_next_dw_fd - dw_next_dw)
    print(dw_next_drmat_fd - dw_next_drmat)
    print()
    '''

    base_w_skew = pin.skew(base_w)
    dpos = (posdot + 0.5*posddot*dt - base_v - base_w_skew@pos)*dt
    dR = pin.exp3((w - base_w)*dt)
    kf_F = np.eye(self.ndx)
    kf_F[:3, :3] += -base_w_skew*dt
    kf_F[:3, 6:9] = dt*np.eye(3)
    kf_F[3:6, 9:12] = rmat.transpose()@pin.Jexp3((w - base_w)*dt)*dt
    kf_F[9:12, 3:6] = dw_next_drmat
    kf_F[9:12, 9:12] = dw_next_dw
    self.kf_x[:3] = pos + dpos
    self.kf_x[3:7] = R.from_matrix(dR@rmat).as_quat()
    self.kf_x[7:10] = posdot + posddot*dt
    self.kf_x[10:13] = w_next
    self.kf_cov = kf_F@self.kf_cov@kf_F.transpose() + self.kf_Q

  def correct(self, pos, quat):
    if not self.initialized:
      print('Attempting to run correction on uninitialized EKF')

    # Kalman correction
    zhat = np.copy(self.kf_x[:self.nq])
    z = np.concatenate((pos, quat))

    inn = np.zeros(self.ndz)
    inn[:3] = z[:3] - zhat[:3]
    inn[3:6] = pin.difference(self.SO3_pin_model, zhat[3:7], z[3:7])

    kf_H = np.zeros((self.ndz, self.ndx))
    kf_H[:3, :3] = np.eye(3)
    kf_H[3:6, 3:6] = -pin.dDifference(self.SO3_pin_model, zhat[3:7], z[3:7], pin.ArgumentPosition.ARG0)

    inn_cov_chol = cho_factor(kf_H@self.kf_cov@kf_H.transpose() + self.kf_R, lower=True)

    dx = self.kf_cov@kf_H.transpose()@cho_solve(inn_cov_chol, inn)
    self.kf_x[:3] += dx[:3]
    self.kf_x[3:7] = pin.integrate(self.SO3_pin_model, self.kf_x[3:7], dx[3:6])
    self.kf_cov -= self.kf_cov@kf_H.transpose()@cho_solve(inn_cov_chol, kf_H@self.kf_cov)
    self.kf_cov = (self.kf_cov + self.kf_cov.transpose())/2
