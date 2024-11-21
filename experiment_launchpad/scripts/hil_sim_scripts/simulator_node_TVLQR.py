#!/usr/bin/env python3
import casadi
import numpy as np
import cProfile
import time
from scipy.spatial.transform import Rotation as R
from get_grid import get_grid
from trajlib_util_tvlqr import get_trajlib_load_paths, interp_trajectories_on_initial_client_w, interp_trajectories_on_delta_pos, interp_trajectories_on_init_client_state

import os
import gc
import atexit
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hil_sim_scripts.mrv_tvlqr_plunge_controller import TVLQR_PLUNGE_controller

# Global variables for on_shutdown
mrv_controller = None
success = False
fail_reason = 'None'
# save_path = '/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/11_15_24/'
do_save = True

def on_shutdown():
    global mrv_controller, success, fail_reason, save_path, do_save
    gc.enable()
    if not do_save:
        return

    print('Saving data')
    print('success: ', success)

    mrv_controller.save(save_path)

    np.save(save_path + 'success.npy', success)
    np.save(save_path + 'fail_reason.npy', fail_reason)

    print('Saved data')

def main():
    global mrv_controller, success, fail_reason, save_path, do_save

    np.set_printoptions(linewidth=np.inf)
    np.set_printoptions(suppress=True)

    use_ekf = True

    grid_type = 17
    ic_grid = get_grid(grid_type)
    print("Grid of initial conditions:")
    print(ic_grid)

    initial_grid_idx = 0  # Use zero-indexing
    use_grid = True

    dist_centering_waypoint_from_goal = 0.4865  # meters

    do_noisy_state_estimation = True
    do_vision_delay = True
    num_trial = 1

    dt = 0.01
    path = '/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit'

    cone_slope = 3.1510649771783403

    mrv_joint_angle_lower_limits = np.array([-2.792526803, -1.745329252, -2.967059728, -2.967059728,
                                             -4.36332313, -1.919862177, -4.310963252])
    mrv_joint_angle_upper_limits = np.array([2.792526803, 1.745329252, 2.565634, 2.617993878,
                                             1.221730476, 1.919862177, 1.169370599])
    mrv_joint_vel_limits = np.array([0.069813, 0.069813, 0.10472, 0.10472, 0.12217, 0.12217, 0.12217])
    mrv_joint_acc_limits = np.array([0.12217, 0.12217, 0.22689, 0.22689, 0.3316, 0.33161, 0.33161])
    mrv_joint_torque_limits = np.array([150, 150, 150, 150, 150, 150, 150])

    client_velocity_noise_ang_amp = 0.0 * np.pi / 180.

    cw_a = 6793137
    cw_mu = 3.986e+14
    cw_orbit_dir = 'x'
    peg_rad = 0.008
    nozzle_opening_rad = 0.142

    traj_library_prefix = '/traj_lib/11_18_24/align_w_nozzle/'
    use_cw = True



    atexit.register(on_shutdown)

    num_grid_idx = len(ic_grid)
    final_grid_idx = num_grid_idx

    print("Starting with grid_idx:")
    print(initial_grid_idx)

    base_seed = 12345
    rng_sequences = np.random.SeedSequence(base_seed).spawn(num_trial)

    for x in range(0, num_trial):
        for grid_idx in range(initial_grid_idx, final_grid_idx):
            print('Starting grid index %d/%d' % (grid_idx, num_grid_idx))
            save_path = None

            if use_grid:
                delta_pos = ic_grid[grid_idx, :3]
            else:
                delta_pos = np.array([0.00, 0.0, -0.05])

            delta_rot = np.array([0., 0., 0.]) * np.pi / 180

            print('Desired position difference (m) in nozzle frame: ', delta_pos)
            print('Desired rotation difference (deg) in nozzle frame: ', delta_rot)

            if use_grid:
                delta_v = np.copy(ic_grid[grid_idx, 3:6])
                initial_mrv_w = ic_grid[grid_idx, 6:9] * np.pi / 180
            else:
                delta_v = np.array([0.0, 0.0, 0.0])
                initial_mrv_w = np.array([0.0, 0.0, 0.0]) * np.pi / 180

            if use_grid:
                initial_client_w = ic_grid[grid_idx, 9:12] * np.pi / 180
            else:
                initial_client_w = np.array([0.0, 0.0, 0.0]) * np.pi / 180

            if do_vision_delay:
                vision_measurement_rate = 10.0  # Hz
            else:
                vision_measurement_rate = dt

            time_steps_between_measurements = dt / (1 / vision_measurement_rate)

            # Set other parameters
            clip_joint_commands = False
            time_limit = 500  
            debug_with_test_traj = False
            test_traj_id = 0
            lock_client = False
            lock_mrv = False
            probe_z_axis_plunge_velocity = 0.005
            use_variable_plunge_speed = True
            use_scheduled_gains = True
            use_ekf = True
            load_paths, ws, dps = get_trajlib_load_paths(path + traj_library_prefix)
            load_paths_for_interpolation, weights = interp_trajectories_on_init_client_state(
                        load_paths, initial_client_w, ws, delta_pos, dps)
                       
            rng = np.random.default_rng(rng_sequences[x])

            # Initialize the controller
            mrv_controller = TVLQR_PLUNGE_controller(
                path + '/urdf/robot_cv_detached.urdf',
                path + '/urdf/robot.urdf',
                path + '/urdf/robot.urdf',
                path + '/urdf/cv.urdf',
                mrv_joint_angle_lower_limits, mrv_joint_angle_upper_limits,
                mrv_joint_vel_limits, mrv_joint_acc_limits, mrv_joint_torque_limits, dt,
                cone_slope, clip_joint_commands,
                time_steps_between_measurements, cw_a, cw_mu, cw_orbit_dir, do_noisy_state_estimation,
                nozzle_opening_rad, peg_rad, client_velocity_noise_ang_amp, time_limit,
                debug_with_test_traj, test_traj_id, lock_client, lock_mrv, probe_z_axis_plunge_velocity,
                use_variable_plunge_speed,
                use_scheduled_gains,
                use_cw=use_cw, use_ekf=use_ekf)
            
            mrv_controller.reset_wrt_capture_box( load_paths_for_interpolation, weights, delta_pos, delta_rot, delta_v, initial_client_w, initial_mrv_w, rng, dist_centering_waypoint_from_goal)
            

            # Get initial state information
            sw_peg_pos, sw_peg_rmat, sw_peg_twist, sw_nozzle_pos, sw_nozzle_rmat, sw_nozzle_twist = mrv_controller.get_peg_and_nozzle_info()
            sw_base_pos, sw_base_rmat, sw_joint_angles, sw_base_v, sw_base_w, sw_joint_vels, sw_client_pos, sw_client_rmat, sw_client_v, sw_client_w = mrv_controller.get_state_in_pieces()

            timestr = time.strftime("%Y%m%d-%H%M%S")

            # Save folder
            save_folder = 'experiment_logs/11_18_24'
            save_path_str = save_folder + '/' + '_'.join([
                'pos_', str(delta_pos[0]), str(delta_pos[1]), str(delta_pos[2]), 'rot', str(delta_rot[0]),
                str(delta_rot[1]), str(delta_rot[2]), 'delta_v', str(delta_v[0]), str(delta_v[1]),
                str(delta_v[2]), 'mrv_w', str(initial_mrv_w[0]), str(initial_mrv_w[1]),
                str(initial_mrv_w[2]), 'client_w', str(initial_client_w[0]), str(initial_client_w[1]),
                str(initial_client_w[2])])
            save_path = path + '/' + save_path_str + timestr + '/'

            if do_save:
                os.makedirs(save_path, exist_ok=True)

            success = False
            fail_reason = 'None'

            gc.disable()

            # Simulation step settings
            apply_wrench_only_when_close = False

            start_time = time.time()
            while True:
                do_step = True

                if do_step:
                    status = mrv_controller.step(None, apply_wrench_only_when_close)

                    if status == 'success':
                        success = True
                        print("Success.")
                        break
                    elif status != 'nothing':
                        fail_reason = status
                        print('Fail at step %d' % (len(mrv_controller.mrv_client_sim.sim_ts) - 1))
                        break

            gc.enable()
            print('we got here')
            print(do_save)
            if do_save:
                print('Saving data')
                print('success: ', success)

                mrv_controller.save(save_path)

                np.save(save_path + 'success.npy', success)
                np.save(save_path + 'fail_reason.npy', fail_reason)

                print('Saved data')

            print()

            if not use_grid:
                break

if __name__ == '__main__':
    # Run the main function under cProfile
    main()
    # cProfile.run('main()', filename='/home/medusar/bspin/on_orbit/profiler_info/profiling_particle_filter.prof')
