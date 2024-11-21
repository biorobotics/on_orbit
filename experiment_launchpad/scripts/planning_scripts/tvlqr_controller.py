import casadi as ca
import time
import pinocchio as pin
from pinocchio import casadi as cpin
from pinocchio.robot_wrapper import RobotWrapper
from rp_conversion import *
from casadi_rp_conversion import *
import numpy as np
import cProfile

class TVLQRController(object):
    '''
    This class implements a TVLQR controller for the client vehicle and the MRV system. The controller is used to linearize about and 
    track a reference trajectory
    '''
    def __init__(self, mrv_cv_urdf_file , mrv_urdf_file, x_ref_traj, u_ref_traj, dt_traj, dt_cntrl , cw_a , cw_mu , cw_orbit_dir, initial_client_rmat):
        
        # Model creation for the client bechicle / mrv system
        self.pin_model = RobotWrapper.BuildFromURDF(mrv_cv_urdf_file).model
        self.pin_data = pin.Data(self.pin_model)
        self.pin_model.gravity.setZero()
        self.cpin_model = cpin.Model(self.pin_model)

        # Model creation for the MRV
        self.mrv_pin_model = RobotWrapper.BuildFromURDF(mrv_urdf_file, root_joint=pin.JointModelFreeFlyer()).model
        self.mrv_pin_model.gravity.setZero()
        self.mrv_pin_data = pin.Data(self.mrv_pin_model)
        mrv_cpin_model = cpin.Model(self.mrv_pin_model)
        
        # Get the number of rotary joints and the number of generalized coordinates
        num_rotary = self.cpin_model.nv -12
        self.mrv_nq = self.cpin_model.nq - 7
        self.mrv_nv = self.cpin_model.nv - 6
        self.cpin_data = self.cpin_model.createData()
        self.dt_traj = dt_traj
        self.dt_cntrl = dt_cntrl
        nx_rp = 2*self.cpin_model.nv
        nu = num_rotary + 3

        # Get the joint ids for the client cv and the robot mrv base
        cv_jidx = self.cpin_model.getJointId('world_to_client')
        cv_qidx = self.pin_model.idx_qs[cv_jidx]
        cv_vidx = self.cpin_model.idx_vs[cv_jidx]
        mrv_jidx = self.cpin_model.getJointId('world_to_base')
        self.mrv_qidx = self.cpin_model.idx_qs[mrv_jidx]
        self.mrv_vidx = self.cpin_model.idx_vs[mrv_jidx]

        # Define the cost vectors and matricies for the TVLQR controller
        self.Q_f = np.eye(nx_rp) # Final state cost matrix
        self.Q = np.eye(nx_rp) # State cost matrix
        self.R = np.eye(nu) * 10 # Control cost matrix

        # Process the reference trajectory and store the pinocchio and rodrigues parameterized states
        self.pin_xs = []
        self.rp_xs = []
        self.us = []
        self.pin_xs = np.zeros((len(x_ref_traj), self.cpin_model.nq + self.cpin_model.nv))
        self.rp_xs = np.zeros((len(x_ref_traj), nx_rp))
        self.us = np.zeros((len(u_ref_traj), nu))
        for i in range(len(x_ref_traj)):
            x_i = x_ref_traj[i]
            x_rp = from_pin(x_i, self.cpin_model)
            self.pin_xs[i, :] = x_i
            self.rp_xs[i, :] = x_rp
            self.us[i, :] = u_ref_traj[i]

        # Create the symbolic variables for the dynamics
        x_sym = ca.SX.sym('x', nx_rp)
        u_sym = ca.SX.sym('u', nu)
        dyn_input = ca.vertcat(x_sym, u_sym)
        x_rp = dyn_input[:nx_rp]
        q_rp = x_rp[:self.cpin_model.nv]
        v_rp = x_rp[self.cpin_model.nv:]
        u = dyn_input[nx_rp:nx_rp + nu]

        # Convert the rodrigues paramters to pinocchio states
        q_pin = q_to_pin(q_rp, self.cpin_model)
        v_pin = v_rp

        mrv_SE3 = cpin.XYZQUATToSE3(q_pin[self.mrv_qidx:self.mrv_qidx + 7])
        client_SE3 = cpin.XYZQUATToSE3(q_pin[cv_qidx:cv_qidx + 7])

        cpin.forwardKinematics(self.cpin_model, self.cpin_data, q_pin)
        cpin.computeJointJacobians(self.cpin_model, self.cpin_data)

        qdot_rp = ca.vertcat(mrv_SE3.rotation@v_rp[self.mrv_vidx:self.mrv_vidx + 3], \
                            JSO3_to_rp(q_pin[self.mrv_qidx + 3:self.mrv_qidx + 7])@v_pin[self.mrv_vidx + 3:self.mrv_vidx + 6], \
                            v_rp[self.mrv_vidx + 6:self.mrv_vidx + self.mrv_nv], \
                            client_SE3.rotation@v_rp[cv_vidx:cv_vidx + 3], \
                            JSO3_to_rp(q_pin[cv_qidx + 3:cv_qidx + 7])@v_pin[cv_vidx + 3:cv_vidx + 6])
        

        client_pos = q_pin[cv_qidx:cv_qidx + 3]
        client_rmat = client_SE3.rotation
        client_v = v_pin[cv_vidx:cv_vidx + 3]

        mrv_cpin_data = cpin.Data(mrv_cpin_model)
        client_mass = self.cpin_model.inertias[cv_jidx].mass

        cw_n = np.sqrt(cw_mu/cw_a**3)
        cw_xproj = ca.DM(initial_client_rmat[:, 2])
        if cw_orbit_dir == 'x':
            cw_yproj = ca.DM(initial_client_rmat[:, 0])
        else:
            cw_yproj = ca.DM(initial_client_rmat[:, 1])
        cw_zproj = ca.cross(cw_xproj, cw_yproj)

        mrv_q = q_pin[self.mrv_qidx:self.mrv_qidx + self.mrv_nq]
        mrv_v = v_pin[self.mrv_vidx:self.mrv_vidx + self.mrv_nv]

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

        # This tau is assuming that there is no contact force between the client and the mrv, reasonable only for the free space case
        # before the final plunge sequence, which will be handled by the admittance controller
        tau = ca.vertcat(ca.SX.zeros(6), u[:num_rotary], client_force_local, ca.SX.zeros(3))
        acc = cpin.aba(self.cpin_model, self.cpin_data, q_pin, v_pin, tau)

        # Get the next state
        q_k_p_1 = q_rp + qdot_rp*self.dt_traj
        v_k_p_1 = v_rp + acc*self.dt_traj
        x_k_p_1 = ca.vertcat(q_k_p_1, v_k_p_1)

        # Get the Jacobians of the dynamics
        A = ca.jacobian(x_k_p_1, x_sym)
        B = ca.jacobian(x_k_p_1, u_sym)

        # Create the casadi functions for the dynamics
        self.A_func = ca.Function('A_func', [x_sym, u_sym], [A])
        self.B_func = ca.Function('B_func', [x_sym, u_sym], [B])
    
    def compute_A_B(self, x_k, u_k):
        Ak = self.A_func(x_k, u_k)
        Bk = self.B_func(x_k, u_k)
        return Ak, Bk
    
    def backward_solve_ricatti(self):

        N = len(self.rp_xs)

        # Cost matricies to CasADi DM types
        Q = self.Q
        R = self.R
        Q_f = self.Q_f

        # Initialize P with the terminal state cost
        P_k_p_1 = Q_f

        K_k_t = []
        for k in range(N-1, -1, -1):
            x_k = self.rp_xs[k]
            u_k = self.us[k]

            A_k , B_k = self.compute_A_B(x_k, u_k)
            A_k = np.array(A_k)
            B_k = np.array(B_k)

            # Solve the Ricatti equation
            S_k = R + B_k.T@P_k_p_1@B_k
            K_k = np.linalg.solve(S_k, B_k.T@P_k_p_1@A_k)
            P_k = Q + A_k.T @ P_k_p_1 @ (A_k - B_k @ K_k)
            P_k += K_k.T @ R @ K_k
            
            # Store the gains
            K_k_t.insert(0, K_k)

            # Update Pk1 and pk1 for the next solve
            P_k_p_1 = P_k
    
        self.K_k_t_list = K_k_t
    
    def compute_control(self, x_k, k_cntrl, torques=False):

        # Seperate the dt_cntrl and dt_traj
        dt_cntrl = self.dt_cntrl
        dt_traj = self.dt_traj
        N = len(self.rp_xs)

        # Get the fractional trajectory index
        dt_ratio = dt_cntrl/dt_traj
        k_traj_float = k_cntrl * dt_ratio

        # Edge case handling
        if k_traj_float >= N-1:
            k0 = N-2
            k1 = N-1
            alpha = 1
        elif k_traj_float <= 0:
            k0 = 0
            k1 = 1
            alpha = 0.0
        else:
            k0 = int(k_traj_float)
            k1 = k0 + 1
            alpha = k_traj_float - k0
        
        # Interpolate between the gains
        K_k0 = self.K_k_t_list[k0]
        K_k1 = self.K_k_t_list[k1]
        K_k = (1-alpha)*K_k0 + alpha*K_k1

        # Interpolate the nominal control input u_nom
        u_nom_k0 = self.us[k0]
        u_nom_k1 = self.us[k1]
        u_nom_interp = (1 - alpha) * u_nom_k0 + alpha * u_nom_k1

        # Interpolate the nominal state x_nom
        x_nom_k0 = self.rp_xs[k0]
        x_nom_k1 = self.rp_xs[k1]
        x_nom_interp = (1 - alpha) * x_nom_k0 + alpha * x_nom_k1
        x_nom_interp_pin = to_pin(x_nom_interp, self.pin_model)


        # Get rotation differences from pinnochio
        d_x_pin = pin.difference(self.pin_model, x_nom_interp_pin[:self.pin_model.nq], x_k[:self.pin_model.nq])
        rot_vec_mrv = d_x_pin[3:6]
        rot_vec_client = d_x_pin[16:19]
        length_vec_mrv = np.linalg.norm(rot_vec_mrv)
        length_vec_client = np.linalg.norm(rot_vec_client)
        rod_diff_mrv = rot_vec_mrv/length_vec_mrv * np.tan(length_vec_mrv/2)
        rod_diff_client = rot_vec_client/length_vec_client * np.tan(length_vec_client/2)

        d_x_rp = np.hstack([d_x_pin[:3], rod_diff_mrv, d_x_pin[6:16], rod_diff_client,self.pin_xs[k_cntrl][self.pin_model.nq:]-x_k[self.pin_model.nq:]]) 

        # Compute the control input
        u_k = u_nom_interp + K_k @ d_x_rp
        u_k = np.clip(u_k, -150, 150)

        #tau = np.hstack([np.zeros(3), u_k])
        tau = np.hstack([np.zeros(6), u_k[:7]])
        q_pin = x_k[:self.cpin_model.nq]
        v_pin = x_k[self.cpin_model.nq:]
        mrv_q = q_pin[self.mrv_qidx:self.mrv_qidx + self.mrv_nq]
        mrv_v = v_pin[self.mrv_vidx:self.mrv_vidx + self.mrv_nv]
        acc = pin.aba(self.mrv_pin_model, self.mrv_pin_data, mrv_q, mrv_v, tau)
        joint_acc = acc[6:]
        # time.sleep(0.1)

        if torques:
            return u_k[:7]
        else:
            return joint_acc

if __name__ == "__main__":
    import time
    urdf_file_mrv_cv = '/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/urdf/robot_cv_detached.urdf'
    urdf_file_mrv = '/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/urdf/robot.urdf'
    x_ref_traj = np.load('/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/10_03_24/insertion_traj/pos_0.1_0.1_-0.1_rot_0.0_0.0_0.0_delta_v_0.0_0.0_0.0_client_w_0.0_0.0_0.0/control/20241003-133845/xs.npy')
    u_ref_traj = np.load('/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/10_03_24/insertion_traj/pos_0.1_0.1_-0.1_rot_0.0_0.0_0.0_delta_v_0.0_0.0_0.0_client_w_0.0_0.0_0.0/control/20241003-133845/us.npy')
    dt_traj = 0.01
    dt_cntrl = 0.01
    cw_a = 6793137
    cw_mu = 3.986e+14
    cw_orbit_dir = 'x'
    initial_client_rmat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    
    controller = TVLQRController(urdf_file_mrv_cv, urdf_file_mrv, x_ref_traj, u_ref_traj, dt_traj, dt_cntrl, cw_a, cw_mu, cw_orbit_dir, initial_client_rmat)
    # cProfile.run('controller.backward_solve_ricatti()', filename='/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/profiles/tvlqr_controller.prof')
    controller.backward_solve_ricatti()
    x_k = controller.pin_xs[9]
    x_k[8] += 0.00
    u_k = controller.us[9]
    k = 9
    start_time = time.time()
    u_k_controller = controller.compute_control(x_k, k, torques=True)
    print('Time taken:', time.time() - start_time)
    print('Torque Difference:', u_k_controller - u_k[:7])
    print('Commanded Torque:', u_k_controller)


