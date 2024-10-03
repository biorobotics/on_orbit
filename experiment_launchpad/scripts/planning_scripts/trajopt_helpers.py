import numpy as np

class TrajectoryHelper:
    def __init__(self, x_range, y_range, z_range,vx_range, vy_range, vz_range, max_ang_vel_for_contact_avoidance, ipopt_traj_opt,
                 nozzle_align, break_on_failure=True):
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.vx_range = vx_range
        self.vy_range = vy_range
        self.vz_range = vz_range
        self.max_ang_vel_for_contact_avoidance = max_ang_vel_for_contact_avoidance
        self.ipopt_traj_opt = ipopt_traj_opt
        self.nozzle_align = nozzle_align
        self.break_on_failure = break_on_failure

    def run_planner(self, ic_grid):
        num_grid_pts = len(ic_grid)
        print("Number of grid points:")
        print(num_grid_pts)

        init_grid_idx = 0
        final_grid_idx = num_grid_pts
        for grid_idx in range(init_grid_idx, final_grid_idx):
            print('Starting grid index %d/%d' % (grid_idx, num_grid_pts))

            delta_pos = ic_grid[grid_idx, :3]
            delta_rot = np.zeros(3)

            delta_v = np.copy(ic_grid[grid_idx, 3:6])
            initial_client_w = ic_grid[grid_idx, 6:9]

            if np.linalg.norm(initial_client_w) < self.max_ang_vel_for_contact_avoidance * np.pi / 180.:
                use_contact = False
            else:
                raise Exception("Should not be using contact in current testing.")
                use_contact = True

            success = self.ipopt_traj_opt.plan(delta_pos, delta_rot, initial_client_w, use_contact, self.nozzle_align)

            if not success:
                print("Planner failed.")
                print("grid_idx")
                print(grid_idx)
                if self.break_on_failure:
                    break

    def generate_and_run_wx(self, min_wx, max_wx, step_size):
        if min_wx < max_wx:
            wx_range = 0.01*np.arange(min_wx, max_wx + step_size, step_size)
        else:
            wx_range = 0.01*np.arange(min_wx, max_wx - step_size, -step_size)

        wy_range = np.array([0])
        wz_range = np.array([0])

        # Convert to radians
        wx_range = wx_range * np.pi / 180.
        wy_range = wy_range * np.pi / 180.
        wz_range = wz_range * np.pi / 180.

        x_grid, y_grid, z_grid, vx_grid, vy_grid, vz_grid, client_wx_grid, client_wy_grid, client_wz_grid = np.meshgrid(
            self.x_range, self.y_range, self.z_range, self.vx_range, self.vy_range, self.vz_range, wx_range, wy_range, wz_range, indexing='ij')
        ic_grid = np.stack((x_grid.flatten(), y_grid.flatten(), z_grid.flatten(),
                            vx_grid.flatten(), vy_grid.flatten(), vz_grid.flatten(),
                            client_wx_grid.flatten(), client_wy_grid.flatten(), client_wz_grid.flatten()), 1)

        print("Initial conditions grid:", ic_grid)
        self.run_planner(ic_grid)

    def generate_and_run_wy(self, min_wy, max_wy, step_size):
        if min_wy < max_wy:
            wy_range = 0.01*np.arange(min_wy, max_wy + step_size, step_size)
        else:
            wy_range = 0.01*np.arange(min_wy, max_wy - step_size, -step_size)

        wx_range = np.array([0])
        wz_range = np.array([0])

        # Convert to radians
        wx_range = wx_range * np.pi / 180.
        wy_range = wy_range * np.pi / 180.
        wz_range = wz_range * np.pi / 180.

        x_grid, y_grid, z_grid, vx_grid, vy_grid, vz_grid, client_wx_grid, client_wy_grid, client_wz_grid = np.meshgrid(
            self.x_range, self.y_range, self.z_range, self.vx_range, self.vy_range, self.vz_range, wx_range, wy_range, wz_range, indexing='ij')
        ic_grid = np.stack((x_grid.flatten(), y_grid.flatten(), z_grid.flatten(),
                            vx_grid.flatten(), vy_grid.flatten(), vz_grid.flatten(),
                            client_wx_grid.flatten(), client_wy_grid.flatten(), client_wz_grid.flatten()), 1)

        print("Initial conditions grid:", ic_grid)
        self.run_planner(ic_grid)

    def generate_and_run_cross(self, cross_velocities, step_size):
        wz_range = np.array([0])
        for cross_vel in cross_velocities:
            wx_max = cross_vel["wx_max"]
            wy_max = cross_vel["wy_max"]

            # Define ranges based on the signs of wx_max and wy_max
            if wx_max > 0 and wy_max > 0:
                wx_range = 0.01*np.arange(0, wx_max + step_size, step_size)
                wy_range = 0.01*np.arange(0, wy_max + step_size, step_size)
            elif wx_max < 0 and wy_max < 0:
                wx_range = 0.01*np.arange(wx_max, 0 - step_size, step_size)
                wy_range = 0.01*np.arange(wy_max, 0 - step_size, step_size)
            elif wx_max > 0 and wy_max < 0:
                wx_range = 0.01*np.arange(0, wx_max + step_size, step_size)
                wy_range = 0.01*np.arange(wy_max, 0 - step_size, step_size)
            elif wx_max < 0 and wy_max > 0:
                wx_range = 0.01*np.arange(wx_max, 0 - step_size, step_size)
                wy_range = 0.01*np.arange(0, wy_max + step_size, step_size)

            # Convert to radians
            wx_range = wx_range * np.pi / 180.
            wy_range = wy_range * np.pi / 180.

            # Filter for valid combinations
            valid_combinations = [(wx, wy) for wx in wx_range for wy in wy_range if (wx == wy) or (wx == -wy)]

            for wx, wy in valid_combinations:
                ic_grid = np.array([[self.x_range[0], self.y_range[0], self.z_range[0], self.vx_range[0], self.vy_range[0], self.vz_range[0], wx, wy, wz_range[0]]])
                self.run_planner(ic_grid)
