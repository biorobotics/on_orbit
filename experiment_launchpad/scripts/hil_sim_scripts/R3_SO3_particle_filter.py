import numpy as np
from scipy.spatial.transform import Rotation as R
import pinocchio as pin
from shapely.geometry import Point , Polygon
from sksurgerycore.algorithms.averagequaternions import weighted_average_quaternions


class R3SO3ParticleFilter(object):
    def __init__(self,  pose_pos_std , pose_rot_std,num_particles=100 , use_ft = True):

        self.SO3_pin_model = pin.Model()
        self.SO3_pin_model.addJoint(0, pin.JointModelSpherical(), pin.SE3.Identity(), 'base_joint')
        self.nq = 7  # number of configuration variables
        self.nv = 6  # number of velocity variables
        self.nx = self.nq + self.nv  # Total state dimension
        self.ndx = 2 * self.nv  # Dimenstion of the state error vector
        self.num_particles = num_particles
        self.use_ft = use_ft

        self.particles = np.zeros((self.num_particles, self.nx))
        self.weights = np.ones(self.num_particles) / self.num_particles

        self.pose_pos_std = pose_pos_std / 3
        self.pose_rot_std = pose_rot_std / 3


        self.process_noise_std = np.array([
            4.5e-4, 4.5e-4, 4.5e-4,  # Position
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
    
    
    def diff_nozzle_profile(self,z):
        '''This is the profile of the large nozzle that reresents the nozzle of the LAE'''
        h,k = 0.0, 0.0 # center of the cirlce that defines the nozzle profile
        R = .760744339 # radius of the circle that defines the nozzle profile
        arc_radius_offset = .1420 # offset from the edge of the circle to the centerline of the nozzle
        if z >= 0.0 and z <= .4290:
            x_arc = h + np.sqrt(R**2 - (z-k)**2) - R + arc_radius_offset
        else:
            x_arc = .0095

        z_transition_start = .429
        transition_width = .020
        t = (z - z_transition_start) / transition_width
        # Sigmoid based blend to make the nozzle profile differentiable
        blend_factor = 1 / (1 + np.exp(-10 * t))
        nozzle_profile_radius = blend_factor * .0095 + (1 - blend_factor) * x_arc
        return nozzle_profile_radius

    def dist_to_nozzle_wall(self,peg_pos):
        '''This function uses the nozzle profile to determine the distance of the peg to the nozzle wall'''

        z = peg_pos[2]
        if z > 0:
            nozzle_radius = self.diff_nozzle_profile(z)
            x , y = peg_pos[0] , peg_pos[1]
            dist_to_nozzle = np.sqrt(x**2 + y**2) - nozzle_radius
        else:
            dist_to_nozzle = -np.inf
        return dist_to_nozzle
    
    def weight_from_distance(self,dist):
        ''' This function returns the weight of the particle based on the distance to the nozzle wall'''
        # Currently this is calibrated for use with octagonal approximation of the nozzle wall
        # k = 200 
        # k_n = 70000 
        # if dist < 0:
        #     return np.exp(k*dist)
        # else:
        #     return 1 / (1 + k_n*dist**2)
        # Gaussian distribution centered at -0.008 with std of 0.001
        return np.exp(-0.5 * ((dist + 0.008) / 0.001) ** 2)
        
    def plane_nozzle_approximation(self,z):
        # Constants
        num_planes = 8
        nozzle_wall_width = 0.005  
        entry_outer_diameter = 0.284
        entry_radius = entry_outer_diameter * 0.5

        # Slope angle from XACRO
        slope_angle = 0.285672797  # In radians
        cone_slope = np.tan(slope_angle)

        if z > 0.429:
            r_outer = .016  # Fixed radius once in the throat
        else:
            r_outer = entry_radius - cone_slope * z  # Original calculation
        r_inner = r_outer - nozzle_wall_width

        # Offet angle to aligned the plane with the x-axis
        angle_offset = np.pi / num_planes  # Half of central angle
        theta = np.linspace(0, 2 * np.pi, num_planes, endpoint=False) + angle_offset
        radius = r_inner / np.cos(angle_offset)

        # Compute x and y coordinates of the inner octagon vertices
        x_vertices = radius * np.cos(theta)
        y_vertices = radius * np.sin(theta)
        x_vertices = np.append(x_vertices, x_vertices[0])
        y_vertices = np.append(y_vertices, y_vertices[0])

        return x_vertices, y_vertices
    
    def dist_to_octagon_wall(self,peg_pos):
        '''Computes the minimum distance from the peg to the closest octagonal plane'''
        z = peg_pos[2]
        x_values, y_values = self.plane_nozzle_approximation(z)
        min_dist = np.inf
        x_p, y_p = peg_pos[0], peg_pos[1]

        # Compute distance from point to each segment of the octagon
        for i in range(len(x_values) - 1):
            x1, y1 = x_values[i], y_values[i]
            x2, y2 = x_values[i + 1], y_values[i + 1]
            edge_vector = np.array([x2 - x1, y2 - y1])
            point_vector = np.array([x_p - x1, y_p - y1])
            t = np.dot(point_vector, edge_vector) / np.dot(edge_vector, edge_vector)
            t = np.clip(t, 0, 1)  # Clamp t to [0, 1] to stay within the edge's endpoints
            # Closest point on the edge to the point
            closest_point = np.array([x1, y1]) + t * edge_vector

            # Distance from the point to the closest point on the edge
            dist = np.linalg.norm(np.array([x_p, y_p]) - closest_point)

            # Track the minimum distance
            if dist < min_dist:
                min_dist = dist
        
        # Check if the point is inside the octagon
        polygon = Polygon(zip(x_values, y_values))
        if polygon.contains(Point(x_p, y_p)):
            min_dist = -min_dist
            return min_dist
        else:
            min_dist = min_dist
            return min_dist


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

    def update_with_vision(self, pos_meas, quat_meas):

        """Update particle weights based on vision measurement"""
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
        
        # Normalize weights
        self.weights += 1.e-300  # Avoid zeros
        self.weights /= np.sum(self.weights)
    
    def update_with_force(self, pos_peg , quat_peg, SE3_client_nozzle):
        
        "Get the transforms that will remian constant throughout the update"
        SE3_est_peg = pin.SE3(R.from_quat(quat_peg).as_matrix(), pos_peg)
        SE3_peg_est = SE3_est_peg.inverse()
        SE3_client_nozzle = SE3_client_nozzle
        print('Using force measurement')
        pos_std , ori_std = self.get_standard_deviation()
        print('Position standard deviation: ', pos_std)
        print('Orientation standard deviation: ', ori_std)
        for i in range(self.num_particles):
            pos_part = self.particles[i, :3]
            quat_part = self.particles[i, 3:7]
            SE3_est_client = pin.SE3(R.from_quat(quat_part).as_matrix(), pos_part)
            SE3_peg_nozzle = SE3_peg_est * SE3_est_client * SE3_client_nozzle
            SE3_nozzle_peg = SE3_peg_nozzle.inverse()
            peg_wrt_nozzle = SE3_nozzle_peg.translation
            dist_to_nozzle = self.dist_to_octagon_wall(peg_wrt_nozzle)
            weight = self.weight_from_distance(dist_to_nozzle)
            print('weight: ', weight)
            print('weight_before: ', self.weights[i])   
            self.weights[i] *= weight
            print('weight_after: ', self.weights[i])
        
        # Normalize weights
        pos_std , ori_std = self.get_standard_deviation()
        self.weights += 1.e-300  # Avoid zeros
        self.weights /= np.sum(self.weights)
        pos_std , ori_std = self.get_standard_deviation()
        print('Position standard deviation: ', pos_std)
        print('Orientation standard deviation: ', ori_std)
        # dist_of_max_weight = self.dist_to_octagon_wall(self.particles[np.argmax(self.weights), :3])
        # print('Distance of max weight particle to nozzle wall: ', dist_of_max_weight)
        # Print the weight of the particle with the highest weight
        # idx_max_weight = np.argmax(self.weights)
        # print('Max weight: ', self.weights[idx_max_weight])
        # idx_min_weight = np.argmin(self.weights)
        # print('Min weight: ', self.weights[idx_min_weight])
          
    def update(self, pos_meas, quat_meas, f_t_meas, pos_peg, quat_peg, SE3_client_nozzle):
        """Update particle weights based on measurement likelihood."""
        if not self.initialized:
            print('Attempting to run update on uninitialized Particle Filter')
            return

        self.update_with_vision(pos_meas, quat_meas)
        if np.abs(np.linalg.norm(f_t_meas))> 0:
            if self.use_ft:
                self.update_with_force(pos_peg , quat_peg, SE3_client_nozzle)
            else:
                pass

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
        all_rotvecs = R.from_quat(all_quats).as_rotvec()

        pos_std = np.std(all_positions, axis=0)
        ori_std = np.std(all_rotvecs, axis=0)

        return pos_std, ori_std
        

    def get_estimate(self , SE3_world_est , SE3_client_peg):
        """Compute the weighted average of the particles to get the state estimate."""
        # Position estimate
        pos_est = np.average(self.particles[:, :3], weights=self.weights, axis=0)

        # Orientation estimate using quaternion averaging
        # weighted_average_quaternions expects quaternions in WXYZ format wheres we have XYZW
        # quaternions = np.roll(self.particles[:, 3:7], shift=1, axis=1)
        quaternions = self.particles[:, 3:7]
        quat_est = weighted_average_quaternions(quaternions, self.weights)

        # Velocity estimate
        posdot_est = np.average(self.particles[:, 7:10], weights=self.weights, axis=0)
        w_est = np.average(self.particles[:, 10:13], weights=self.weights, axis=0)

        # For visualization , transfer the particles to the client frame
        all_positions = np.zeros((self.num_particles, 3))
        for i in range(self.num_particles):
            pos_part = self.particles[i, :3]
            quat_part = self.particles[i, 3:7]
            SE3_est_client = pin.SE3(R.from_quat(quat_part).as_matrix(), pos_part)
            SE3_world_peg = SE3_world_est * SE3_est_client * SE3_client_peg
            all_positions[i] = SE3_world_peg.translation
        
        pos_max_weight = self.particles[np.argmax(self.weights), :3]


        return pos_est, quat_est, posdot_est, w_est, pos_max_weight, all_positions


