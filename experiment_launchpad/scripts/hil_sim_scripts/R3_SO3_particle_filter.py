import numpy as np
from scipy.spatial.transform import Rotation as R
import pinocchio as pin
from sksurgerycore.algorithms.averagequaternions import weighted_average_quaternions
class R3SO3ParticleFilter(object):
    def __init__(self,  pose_pos_std , pose_rot_std,num_particles=800):

        self.SO3_pin_model = pin.Model()
        self.SO3_pin_model.addJoint(0, pin.JointModelSpherical(), pin.SE3.Identity(), 'base_joint')
        self.nq = 7  # number of configuration variables
        self.nv = 6  # number of velocity variables
        self.nx = self.nq + self.nv  # Total state dimension
        self.ndx = 2 * self.nv  # Dimenstion of the state error vector
        self.num_particles = num_particles

        self.particles = np.zeros((self.num_particles, self.nx))
        self.weights = np.ones(self.num_particles) / self.num_particles

        self.pose_pos_std = pose_pos_std / 3
        self.pose_rot_std = pose_rot_std / 3


        self.process_noise_std = np.array([
            2.5e-4, 2.5e-4, 2.5e-4,  # Position
            1.0607e-4, 1.0607e-4, 1.0607e-4,  # Orientation
            1.0607e-4, 1.0607e-4, 1.0607e-4,  # Linear velocity
            1.0607e-4, 1.0607e-4, 1.0607e-4  # Angular velocity
        ])

        # Measurement noise standard deviations
        self.measurement_noise_std = np.hstack([self.pose_pos_std, self.pose_rot_std])

        
        # Physical Client Parameters
        self.client_mass = 1050
        self.client_inertia = np.diag([1792.419, 2015.632, 630.731])
        self.client_inertia_inv = np.linalg.inv(self.client_inertia)


        self.initialized = False

    def initialize(self, pos, quat, posdot, w):
        """Initialize particles around the initial state."""
        for i in range(self.num_particles):
            # Position initialization with noise
            self.particles[i, :3] = pos + np.random.randn(3) * self.process_noise_std[:3]
            
            # Orientation initialization with noise
            delta_rotvec = np.random.randn(3) * self.process_noise_std[3:6]
            delta_quat = R.from_rotvec(delta_rotvec).as_quat()
            self.particles[i, 3:7] = (R.from_quat(quat) * R.from_quat(delta_quat)).as_quat()
            
            # Linear velocity initialization with noise
            self.particles[i, 7:10] = posdot + np.random.randn(3) * self.process_noise_std[6:9]
            
            # Angular velocity initialization with noise
            self.particles[i, 10:13] = w + np.random.randn(3) * self.process_noise_std[9:12]

        self.initialized = True

    def predict(self, base_v, base_w, f_client, tau_client, dt):
        """Propagate each particle forward using the system dynamics."""
        if not self.initialized:
            print('Attempting to run prediction on uninitialized Particle Filter')
            return

        for i in range(self.num_particles):
            # Extract particle state
            pos = self.particles[i, :3]
            quat = self.particles[i, 3:self.nq]
            posdot = self.particles[i, self.nq:self.nq + 3]
            posddot = f_client / self.client_mass
            w = self.particles[i, self.nq + 3:self.nq + 6]
            rmat = R.from_quat(quat).as_matrix()

            # Constant acceleration model
            wlocal = rmat.transpose() @ w
            wlocal_next = wlocal + self.client_inertia_inv@(rmat.transpose()@tau_client - np.cross(wlocal, self.client_inertia@wlocal))*dt
            w_next = rmat@wlocal_next

            # Update position and velocity
            base_w_skew = pin.skew(base_w)
            dpos = (posdot + 0.5*posddot*dt - base_v - base_w_skew@pos)*dt
            pos_new = pos + dpos
            posdot_new = posdot + posddot * dt

            # Update orientation
            delta_theta = (w - base_w) * dt
            delta_quat = R.from_rotvec(delta_theta).as_quat()
            quat_new = (R.from_quat(quat) * R.from_quat(delta_quat)).as_quat()

            # Add process noise
            pos_new += np.random.randn(3) * self.process_noise_std[:3]
            delta_rotvec_noise = np.random.randn(3) * self.process_noise_std[3:6]
            delta_quat_noise = R.from_rotvec(delta_rotvec_noise).as_quat()
            quat_new = (R.from_quat(quat_new) * R.from_quat(delta_quat_noise)).as_quat()
            posdot_new += np.random.randn(3) * self.process_noise_std[6:9]
            w_next += np.random.randn(3) * self.process_noise_std[9:12]


            # Store updated state back into the particle
            self.particles[i, :3] = pos_new
            self.particles[i, 3:7] = quat_new
            self.particles[i, 7:10] = posdot_new
            self.particles[i, 10:13] = w_next

    def update(self, pos_meas, quat_meas):
        """Update particle weights based on measurement likelihood."""
        if not self.initialized:
            print('Attempting to run update on uninitialized Particle Filter')
            return

        # Precompute measurement noise covariance inverse
        pos_cov_inv = np.diag(1.0 / (self.measurement_noise_std[:3] ** 2))
        ori_cov_inv = np.diag(1.0 / (self.measurement_noise_std[3:6] ** 2))

        for i in range(self.num_particles):
            # Extract particle state
            pos = self.particles[i, :3]
            quat = self.particles[i, 3:7]

            # Position error and likelihood
            pos_error = pos_meas - pos
            pos_likelihood = np.exp(-0.5 * pos_error @ pos_cov_inv @ pos_error)

            # Orientation error and likelihood
            quat_error = R.from_quat(quat).inv() * R.from_quat(quat_meas)
            angle_error = quat_error.as_rotvec()
            ori_likelihood = np.exp(-0.5 * angle_error @ ori_cov_inv @ angle_error)

            # Update particle weight
            self.weights[i] *= pos_likelihood * ori_likelihood

            # TODO: Lets incorpoarte the F/T measurements that we have


        # Normalize weights
        self.weights += 1.e-300  # Avoid zeros
        self.weights /= np.sum(self.weights)

        # Resample particles when the effective number of particles is low
        self.resample()

    def resample(self):
        """Resample particles based on their weights to avoid degeneracy."""
        N_eff = 1.0 / np.sum(np.square(self.weights))
        if N_eff < self.num_particles / 2:
            indices = np.random.choice(self.num_particles, size=self.num_particles, replace=True, p=self.weights)
            self.particles = self.particles[indices]
            self.weights = np.ones(self.num_particles) / self.num_particles
    
    def get_standard_deviation(self):
        all_positions = self.particles[:, :3]
        all_quats = self.particles[:, 3:7]
        all_euler_angles = np.array([R.from_quat(q).as_euler('xyz', degrees=True) for q in all_quats])

        pos_std = np.std(all_positions, axis=0)
        ori_std = np.std(all_euler_angles, axis=0)

        return pos_std, ori_std
        

    def get_estimate(self):
        """Compute the weighted average of the particles to get the state estimate."""
        # Position estimate
        pos_est = np.average(self.particles[:, :3], weights=self.weights, axis=0)

        # Orientation estimate using quaternion averaging
        # weighted_average_quaternions expects quaternions in WXYZ format wheres we have XYZW
        # quaternions = np.roll(self.particles[:, 3:7], shift=1, axis=1)
        quaternions = self.particles[:, 3:7]
        quat_est = weighted_average_quaternions(quaternions, self.weights)

        # Linear velocity estimate
        posdot_est = np.average(self.particles[:, 7:10], weights=self.weights, axis=0)

        # Angular velocity estimate
        w_est = np.average(self.particles[:, 10:13], weights=self.weights, axis=0)

        # Indicies of the particle with the highest weight
        max_weight_idx = np.argmax(self.weights)
        pos_max_weight = self.particles[max_weight_idx, :3]

        # Indicies of all particles
        all_positions = self.particles[:, :3]


        return pos_est, quat_est, posdot_est, w_est, pos_max_weight, all_positions


