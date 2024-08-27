import pinocchio as pin
import numpy as np
from scipy.spatial.transform import Rotation as R

class ActuationModelCWContact():
  def __init__(self, pin_model, mrv_pin_model, cw_a, cw_mu, cw_orbit_dir, initial_client_rmat, use_cw=True):
    self.pin_model = pin_model
    self.pin_data = pin.Data(self.pin_model)
    self.num_rotary = self.pin_model.nv - 12

    self.mrv_pin_model = mrv_pin_model
    self.mrv_pin_data = pin.Data(self.mrv_pin_model)

    self.cv_jidx = self.pin_model.getJointId('world_to_client')
    self.cv_qidx = self.pin_model.idx_qs[self.cv_jidx]
    self.cv_vidx = self.pin_model.idx_vs[self.cv_jidx]

    self.mrv_jidx = self.pin_model.getJointId('world_to_base')
    self.mrv_qidx = self.pin_model.idx_qs[self.mrv_jidx]
    self.mrv_vidx = self.pin_model.idx_vs[self.mrv_jidx]

    self.mrv_nq = self.mrv_pin_model.nq
    self.mrv_nv = self.mrv_pin_model.nv

    self.mrv_joint7_jidx = self.mrv_pin_model.getJointId('joint7')
    self.mrv_ee_fid = self.mrv_pin_model.getFrameId('ee_tip')

    # Assume contact always occurs at the tip for now. Precompute its position wrt joint 7
    pin.forwardKinematics(self.mrv_pin_model, self.mrv_pin_data, pin.neutral(self.mrv_pin_model))
    pin.updateFramePlacement(self.mrv_pin_model, self.mrv_pin_data, self.mrv_ee_fid)
    contact_point = self.mrv_pin_data.oMf[self.mrv_ee_fid].translation
    self.contact_point_wrt_joint7 = self.mrv_pin_data.oMi[self.mrv_joint7_jidx].actInv(contact_point)

    self.client_mass = self.pin_model.inertias[self.cv_jidx].mass

    self.cw_n = np.sqrt(cw_mu/cw_a**3)
    self.cw_orbit_dir = cw_orbit_dir
    self.cw_xproj = initial_client_rmat[:, 2]
    if self.cw_orbit_dir == 'x':
      self.cw_yproj = initial_client_rmat[:, 0]
    else:
      self.cw_yproj = initial_client_rmat[:, 1]
    self.cw_zproj = np.cross(self.cw_xproj, self.cw_yproj)

    # Joint torques and contact force
    self.nu = self.num_rotary + 3

    self.use_cw = use_cw

  # Last 3 elements of u are the contact force in the world frame
  def calc(self, x, u): 
    jid = self.pin_model.getJointId('joint1')
    vidx = self.pin_model.idx_vs[jid]
    tau = np.zeros(self.pin_model.nv)
    tau[vidx:vidx + self.num_rotary] = u[:self.num_rotary]

    mrv_q = x[self.mrv_qidx:self.mrv_qidx + self.mrv_pin_model.nq]
    mrv_v = x[self.pin_model.nq + self.mrv_vidx:self.pin_model.nq + self.mrv_vidx + self.mrv_pin_model.nv]

    client_pos = x[self.cv_qidx:self.cv_qidx + 3]
    client_quat = x[self.cv_qidx + 3:self.cv_qidx + 7]
    client_rmat = R.from_quat(client_quat).as_matrix()
    client_v = x[self.pin_model.nq + self.cv_vidx:self.pin_model.nq + self.cv_vidx + 3]

    if self.use_cw:
      # Get CW states from sim state
      com = pin.centerOfMass(self.mrv_pin_model, self.mrv_pin_data, mrv_q, mrv_v)
      Jcom = pin.jacobianCenterOfMass(self.mrv_pin_model, self.mrv_pin_data, mrv_q)
      vcom = Jcom@mrv_v

      separation = com - client_pos
      cw_x = self.cw_xproj@separation 
      cw_y = self.cw_yproj@separation
      cw_z = self.cw_zproj@separation

      vdiff = vcom - client_rmat@client_v
      cw_xdot = self.cw_xproj@vdiff
      cw_ydot = self.cw_yproj@vdiff
      cw_zdot = self.cw_zproj@vdiff

      cw_xddot = 3*self.cw_n**2*cw_x + 2*self.cw_n*cw_ydot
      cw_yddot = -2*self.cw_n*cw_xdot
      cw_zddot = -self.cw_n**2*cw_z

      # Compute fictitious forces on the client vehicle arising from CW eqns
      cw_fx = -self.client_mass*cw_xddot
      cw_fy = -self.client_mass*cw_yddot
      cw_fz = -self.client_mass*cw_zddot

      cw_force = cw_fx*self.cw_xproj + cw_fy*self.cw_yproj + cw_fz*self.cw_zproj
      tau[self.cv_vidx:self.cv_vidx + 3] = client_rmat.transpose()@cw_force

    # Incorporate contact forces

    # Compute contact point coordinates in mrv joint7 frame and client frame
    pin.forwardKinematics(self.mrv_pin_model, self.mrv_pin_data, mrv_q) 
    pin.computeJointJacobians(self.mrv_pin_model, self.mrv_pin_data)
    pin.updateFramePlacement(self.mrv_pin_model, self.mrv_pin_data, self.mrv_ee_fid)
    contact_point = self.mrv_pin_data.oMf[self.mrv_ee_fid].translation
    contact_point_wrt_client = client_rmat.transpose()@(contact_point - client_pos)

    # Compute wrench in each frame
    f_ext = [pin.Force.Zero() for j in range(self.pin_model.njoints)]

    contact_force = u[self.num_rotary:]
    f_ext[self.mrv_joint7_jidx].linear[:] = self.mrv_pin_data.oMi[self.mrv_joint7_jidx].rotation.transpose()@contact_force
    f_ext[self.mrv_joint7_jidx].angular[:] = np.cross(self.contact_point_wrt_joint7, f_ext[self.mrv_joint7_jidx].linear[:])

    f_ext[self.cv_jidx].linear[:] = -client_rmat.transpose()@contact_force
    f_ext[self.cv_jidx].angular[:] = np.cross(contact_point_wrt_client, f_ext[self.cv_jidx].linear[:])

    q = x[:self.pin_model.nq]
    tau[:] -= pin.computeStaticTorque(self.pin_model, self.pin_data, q, f_ext)

    return tau

  def calcDiff(self, x, u): 
    dtau_du = np.zeros((self.pin_model.nv, self.nu))
    dtau_dx = np.zeros((self.pin_model.nv, 2*self.pin_model.nv))

    jid = self.pin_model.getJointId('joint1')
    vidx = self.pin_model.idx_vs[jid]
    tau = np.zeros(self.pin_model.nv)
    tau[vidx:vidx + self.num_rotary] = u[:self.num_rotary]

    dtau_du[vidx:vidx + self.num_rotary, :self.num_rotary] = np.eye(self.num_rotary)

    mrv_q = x[self.mrv_qidx:self.mrv_qidx + self.mrv_pin_model.nq]
    mrv_v = x[self.pin_model.nq + self.mrv_vidx:self.pin_model.nq + self.mrv_vidx + self.mrv_nv]

    client_pos = x[self.cv_qidx:self.cv_qidx + 3]
    client_quat = x[self.cv_qidx + 3:self.cv_qidx + 7]
    client_rmat = R.from_quat(client_quat).as_matrix()
    client_v = x[self.pin_model.nq + self.cv_vidx:self.pin_model.nq + self.cv_vidx + 3]

    if self.use_cw:
      com = pin.centerOfMass(self.mrv_pin_model, self.mrv_pin_data, mrv_q, mrv_v)
      Jcom = pin.jacobianCenterOfMass(self.mrv_pin_model, self.mrv_pin_data, mrv_q)
      vcom = Jcom@mrv_v

      pin.computeAllTerms(self.mrv_pin_model, self.mrv_pin_data, mrv_q, mrv_v)
      dvcom_dq = pin.getCenterOfMassVelocityDerivatives(self.mrv_pin_model, self.mrv_pin_data)

      separation = com - client_pos
      cw_x = self.cw_xproj@separation 
      cw_y = self.cw_yproj@separation
      cw_z = self.cw_zproj@separation

      dcw_pos_dq = np.zeros((3, self.pin_model.nv))
      dcw_pos_dq[:, self.mrv_vidx:self.mrv_vidx + self.mrv_nv] = Jcom
      dcw_pos_dq[:, self.cv_vidx:self.cv_vidx + 3] = -client_rmat

      vdiff = vcom - client_rmat.transpose()@client_v
      cw_xdot = self.cw_xproj@vdiff
      cw_ydot = self.cw_yproj@vdiff
      cw_zdot = self.cw_zproj@vdiff

      dcw_vel_dq = np.zeros((3, self.pin_model.nv))
      dcw_vel_dq[:, self.mrv_vidx:self.mrv_vidx + self.mrv_nv] = dvcom_dq
      dcw_vel_dq[:, self.cv_vidx + 3:self.cv_vidx + 6] = client_rmat@pin.skew(client_v)

      dcw_vel_dv = np.zeros((3, self.pin_model.nv))
      dcw_vel_dv[:, self.mrv_vidx:self.mrv_vidx + self.mrv_nv] = Jcom
      dcw_vel_dv[:, self.cv_vidx:self.cv_vidx + 3] = -client_rmat

      cw_proj_matrix = np.stack((self.cw_xproj, self.cw_yproj, self.cw_zproj), 0)
      dcw_pos_dq = cw_proj_matrix@dcw_pos_dq
      dcw_vel_dq = cw_proj_matrix@dcw_vel_dq # Something's wrong with this one
      dcw_vel_dv = cw_proj_matrix@dcw_vel_dv # Something's wrong with this one

      cw_xddot = 3*self.cw_n**2*cw_x + 2*self.cw_n*cw_ydot
      cw_yddot = -2*self.cw_n*cw_xdot
      cw_zddot = -self.cw_n**2*cw_z

      dcw_acc_dcw_pos = np.zeros((3, 3))
      dcw_acc_dcw_pos[0, 0] = 3*self.cw_n**2
      dcw_acc_dcw_pos[2, 2] = -self.cw_n**2

      dcw_acc_dcw_vel = np.zeros((3, 3))
      dcw_acc_dcw_vel[0, 1] = 2*self.cw_n
      dcw_acc_dcw_vel[1, 0] = -2*self.cw_n

      # Compute fictitious forces on the client vehicle arising from CW eqns
      cw_fx = -self.client_mass*cw_xddot
      cw_fy = -self.client_mass*cw_yddot
      cw_fz = -self.client_mass*cw_zddot

      cw_force = cw_fx*self.cw_xproj + cw_fy*self.cw_yproj + cw_fz*self.cw_zproj
      tau[self.cv_vidx:self.cv_vidx + 3] = client_rmat.transpose()@cw_force

      dcw_force_dcw_acc = -self.client_mass*cw_proj_matrix.transpose()

      dcw_force_dq = dcw_force_dcw_acc@(dcw_acc_dcw_pos@dcw_pos_dq + dcw_acc_dcw_vel@dcw_vel_dq)
      dcw_force_dv = dcw_force_dcw_acc@dcw_acc_dcw_vel@dcw_vel_dv
      dcw_force_dx = np.concatenate((dcw_force_dq, dcw_force_dv), 1)

      dtau_dx[self.cv_vidx:self.cv_vidx + 3] = client_rmat.transpose()@dcw_force_dx
      dtau_dx[self.cv_vidx:self.cv_vidx + 3, self.cv_vidx + 3:self.cv_vidx + 6] += pin.skew(tau[self.cv_vidx:self.cv_vidx + 3])

    # Incorporate contact forces

    # Compute contact point coordinates in mrv joint7 frame and client frame
    pin.forwardKinematics(self.mrv_pin_model, self.mrv_pin_data, mrv_q) 
    pin.computeJointJacobians(self.mrv_pin_model, self.mrv_pin_data)
    pin.updateFramePlacement(self.mrv_pin_model, self.mrv_pin_data, self.mrv_ee_fid)
    contact_point = self.mrv_pin_data.oMf[self.mrv_ee_fid].translation
    Jee = pin.getFrameJacobian(self.mrv_pin_model, self.mrv_pin_data, self.mrv_ee_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

    contact_point_wrt_client = client_rmat.transpose()@(contact_point - client_pos)
    dcontact_point_wrt_client_dq = np.zeros((3, self.pin_model.nv))
    dcontact_point_wrt_client_dq[:, self.mrv_vidx:self.mrv_vidx + self.mrv_nv] = client_rmat.transpose()@Jee[:3]
    dcontact_point_wrt_client_dq[:, self.cv_vidx:self.cv_vidx + 3] = -np.eye(3)
    dcontact_point_wrt_client_dq[:, self.cv_vidx + 3:self.cv_vidx + 6] = pin.skew(contact_point_wrt_client)

    Jjoint7 = pin.getJointJacobian(self.mrv_pin_model, self.mrv_pin_data, self.mrv_joint7_jidx, pin.ReferenceFrame.LOCAL)

    Jclient = np.zeros((3, 6))
    Jclient[:, :3] = client_rmat
    Jclient[:3, 3:] = -client_rmat@pin.skew(contact_point_wrt_client)

    dtau_du[self.mrv_vidx:self.mrv_vidx + self.mrv_nv, self.num_rotary:] += Jee[:3].transpose()
    dtau_du[self.cv_vidx:self.cv_vidx + 6, self.num_rotary:] -= Jclient.transpose()

    # Compute wrench in each frame
    f_ext = [pin.Force.Zero() for j in range(self.pin_model.njoints)]

    contact_force = u[self.num_rotary:]
    f_ext[self.mrv_joint7_jidx].linear[:] = self.mrv_pin_data.oMi[self.mrv_joint7_jidx].rotation.transpose()@contact_force
    f_ext[self.mrv_joint7_jidx].angular[:] = np.cross(self.contact_point_wrt_joint7, f_ext[self.mrv_joint7_jidx].linear[:])

    f_ext[self.cv_jidx].linear[:] = -client_rmat.transpose()@contact_force
    f_ext[self.cv_jidx].angular[:] = np.cross(contact_point_wrt_client, f_ext[self.cv_jidx].linear[:])

    q = x[:self.pin_model.nq]
    dtau_dx[:, :self.pin_model.nv] -= pin.computeStaticTorqueDerivatives(self.pin_model, self.pin_data, q, f_ext)

    djoint7_wrench_dmrv_q = np.zeros((6, self.mrv_pin_model.nv))
    djoint7_wrench_dmrv_q[:3] = pin.skew(f_ext[self.mrv_joint7_jidx].linear)@Jjoint7[3:]
    djoint7_wrench_dmrv_q[3:] = pin.skew(self.contact_point_wrt_joint7)@djoint7_wrench_dmrv_q[:3]

    dtau_dx[self.mrv_vidx:\
            self.mrv_vidx + self.mrv_nv, \
            self.mrv_vidx:\
            self.mrv_vidx + self.mrv_nv] += Jjoint7.transpose()@djoint7_wrench_dmrv_q

    dclient_wrench_dq = np.zeros((6, self.pin_model.nv))
    dclient_wrench_dq[:3, self.cv_vidx + 3:self.cv_vidx + 6] = pin.skew(f_ext[self.cv_jidx].linear)
    dclient_wrench_dq[3:] = pin.skew(contact_point_wrt_client)@dclient_wrench_dq[:3] + \
                            -pin.skew(f_ext[self.cv_jidx].linear)@dcontact_point_wrt_client_dq

    dtau_dx[self.cv_vidx:\
            self.cv_vidx + 6, \
            :self.pin_model.nv] += dclient_wrench_dq

    return dtau_dx, dtau_du
