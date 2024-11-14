import casadi as ca
import pinocchio as pin
from pinocchio import casadi as cpin
from casadi_rp_conversion import *
import numpy as np

def export_implicit_midpoint_model(cpin_model, plane_idx, dyn_input, dt, use_cw, mrv_cpin_model, initial_client_rmat, cw_mu, cw_a, cw_orbit_dir, interp_joint_torque=True, interp_contact_force=True):
    # cpin_model:
    # 
    tip_fid = cpin_model.getFrameId('ee_tip')
    nozzle_geom_fid = cpin_model.getFrameId('nozzle_geom' + str(plane_idx))
    hole_fid = cpin_model.getFrameId('hole')

    cv_jidx = cpin_model.getJointId('world_to_client')
    cv_qidx = cpin_model.idx_qs[cv_jidx]
    cv_vidx = cpin_model.idx_vs[cv_jidx]

    mrv_jidx = cpin_model.getJointId('world_to_base')
    mrv_qidx = cpin_model.idx_qs[mrv_jidx]
    mrv_vidx = cpin_model.idx_vs[mrv_jidx]
    num_rotary = cpin_model.nv - 12
    mrv_nq = cpin_model.nq - 7
    mrv_nv = cpin_model.nv - 6

    cpin_data = cpin_model.createData()

    nx_ipopt = 2*cpin_model.nv
    nu = num_rotary + 3
    
    x0_ipopt = dyn_input[:nx_ipopt]
    q0_ipopt = x0_ipopt[:cpin_model.nv]
    v0_ipopt = x0_ipopt[cpin_model.nv:]
    u0 = dyn_input[nx_ipopt:nx_ipopt + nu]

    x1_ipopt = dyn_input[nx_ipopt + nu:nx_ipopt + nu + nx_ipopt]
    q1_ipopt = x1_ipopt[:cpin_model.nv]
    v1_ipopt = x1_ipopt[cpin_model.nv:]
    u1 = dyn_input[nx_ipopt + nu + nx_ipopt:nx_ipopt + nu + nx_ipopt + nu]

    x_ipopt = 0.5*(x0_ipopt + x1_ipopt)
    q_ipopt = x_ipopt[:cpin_model.nv]
    v_ipopt = x_ipopt[cpin_model.nv:]

    if interp_joint_torque:
      joint_torque = 0.5*u0[:num_rotary] + 0.5*u1[:num_rotary]
    else:
      joint_torque = u0[:num_rotary]

    if interp_contact_force:
      contact_force = 0.5*u0[num_rotary:] + 0.5*u1[num_rotary:]
    else:
      contact_force = u0[num_rotary:]

    u = ca.vertcat(joint_torque, contact_force)

    q = q_to_pin(q_ipopt, cpin_model)
    v = v_ipopt

    mrv_SE3 = cpin.XYZQUATToSE3(q[mrv_qidx:mrv_qidx + 7])
    client_SE3 = cpin.XYZQUATToSE3(q[cv_qidx:cv_qidx + 7])

    cpin.forwardKinematics(cpin_model, cpin_data, q)
    cpin.computeJointJacobians(cpin_model, cpin_data)
    cpin.updateFramePlacement(cpin_model, cpin_data, tip_fid)
    cpin.updateFramePlacement(cpin_model, cpin_data, nozzle_geom_fid)
    cpin.updateFramePlacement(cpin_model, cpin_data, hole_fid)
    contact_point = cpin_data.oMf[tip_fid].translation
    contact_point_wrt_client = client_SE3.actInv(contact_point)

    Jtip = cpin.getFrameJacobian(cpin_model, cpin_data, tip_fid, cpin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
    Jclient = ca.horzcat(ca.SX.zeros(3, mrv_nv), client_SE3.rotation, -client_SE3.rotation@cpin.skew(contact_point_wrt_client))
    # contact_frame = ca.horzcat(-cpin_data.oMf[nozzle_geom_fid].rotation[:, :1], cpin_data.oMf[hole_fid].rotation[:, :2])
    # Jcon = contact_frame.T@(Jtip - Jclient)
    Jcon = Jtip - Jclient

    qdot_ipopt = ca.vertcat(mrv_SE3.rotation@v_ipopt[mrv_vidx:mrv_vidx + 3], \
                            JSO3_to_rp(q[mrv_qidx + 3:mrv_qidx + 7])@v[mrv_vidx + 3:mrv_vidx + 6], \
                            v_ipopt[mrv_vidx + 6:mrv_vidx + mrv_nv], \
                            client_SE3.rotation@v_ipopt[cv_vidx:cv_vidx + 3], \
                            JSO3_to_rp(q[cv_qidx + 3:cv_qidx + 7])@v[cv_vidx + 3:cv_vidx + 6])
 
    q_dynamics_expr = qdot_ipopt - (q1_ipopt - q0_ipopt)/dt

    if use_cw:
      client_pos = q[cv_qidx:cv_qidx + 3]
      client_quat = q[cv_qidx + 3:cv_qidx + 7]
      client_rmat = client_SE3.rotation
      client_v = v[cv_vidx:cv_vidx + 3]

      mrv_cpin_data = cpin.Data(mrv_cpin_model)
      client_mass = cpin_model.inertias[cv_jidx].mass

      cw_n = np.sqrt(cw_mu/cw_a**3)
      cw_xproj = ca.DM(initial_client_rmat[:, 2])
      if cw_orbit_dir == 'x':
        cw_yproj = ca.DM(initial_client_rmat[:, 0])
      else:
        cw_yproj = ca.DM(initial_client_rmat[:, 1])
      cw_zproj = ca.cross(cw_xproj, cw_yproj)

      mrv_q = q[mrv_qidx:mrv_qidx + mrv_nq]
      mrv_v = v[mrv_vidx:mrv_vidx + mrv_nv]

      # Get CW states from sim state
      com = cpin.centerOfMass(mrv_cpin_model, mrv_cpin_data, mrv_q, mrv_v)
      vcom = mrv_cpin_data.vcom[0]

      separation = com - client_pos
      cw_x = ca.dot(cw_xproj, separation)
      cw_y = ca.dot(cw_yproj, separation)
      cw_z = ca.dot(cw_zproj, separation)

      vdiff = vcom - client_rmat@client_v
      cw_xdot = ca.dot(cw_xproj, vdiff)
      cw_ydot = ca.dot(cw_yproj, vdiff)
      cw_zdot = ca.dot(cw_zproj, vdiff)

      cw_xddot = 3*cw_n**2*cw_x + 2*cw_n*cw_ydot
      cw_yddot = -2*cw_n*cw_xdot
      cw_zddot = -cw_n**2*cw_z

      # Compute fictitious forces on the client vehicle arising from CW eqns
      cw_fx = -client_mass*cw_xddot
      cw_fy = -client_mass*cw_yddot
      cw_fz = -client_mass*cw_zddot

      cw_force = cw_fx*cw_xproj + cw_fy*cw_yproj + cw_fz*cw_zproj
      client_force_local = client_rmat.T@cw_force
    else:
      client_force_local = ca.SX.zeros(3)

    tau = ca.vertcat(ca.SX.zeros(6), u[:num_rotary], client_force_local, ca.SX.zeros(3)) + Jcon.T@u[num_rotary:num_rotary + 3]
    v_dynamics_expr = (tau - cpin.rnea(cpin_model, cpin_data, q, v, (v1_ipopt - v0_ipopt)/dt))*dt
    # print('v_dyn:', v_dynamics_expr.size())
    # quit()

    print(u.shape)
    quit()

    return ca.vertcat(q_dynamics_expr, v_dynamics_expr)
