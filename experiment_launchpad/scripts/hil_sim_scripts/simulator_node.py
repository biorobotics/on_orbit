#!/usr/bin/env python3
import numpy as np
import cProfile
import time
from scipy.spatial.transform import Rotation as R
from get_grid import get_grid
from trajlib_util import get_trajlib_load_paths, interp_trajectories_on_initial_client_w, interp_trajectories_on_delta_pos, interp_trajectories_on_init_client_state
from sim_ros_vis_publisher import SimROSVisPublisher

import rospy
import rospkg

import os
import gc

from mrv_controller import  MrvController
# Global variables for on_shutdown
mrv_controller = None
success = False
fail_reason = 'None'
save_path = ''
do_save = False

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

    visualize = rospy.get_param('visualize')

    grid_type = rospy.get_param('grid_type')
    ic_grid = get_grid(grid_type)
    print("Grid of initial conditions:")
    print(ic_grid)

    initial_grid_idx = rospy.get_param('initial_grid_idx')
    use_grid = rospy.get_param('use_grid')

    dist_centering_waypoint_from_goal = rospy.get_param('dist_centering_waypoint_from_goal')

    do_noisy_state_estimation = rospy.get_param('do_noisy_state_estimation')
    do_vision_delay = rospy.get_param('do_vision_delay')
    num_trial = rospy.get_param('num_software_trials')
    dt = 0.01


    rospy.init_node('control_node')

    rospack = rospkg.RosPack()
    rospath = rospack.get_path('on_orbit')

    rate = rospy.Rate(1 / dt)

    sim_vis_publisher = SimROSVisPublisher(rospath)

    cone_slope = rospy.get_param('cone_slope')

    mrv_joint_angle_lower_limits = np.array(rospy.get_param('joint_angle_lower_limits'))
    mrv_joint_angle_upper_limits = np.array(rospy.get_param('joint_angle_upper_limits'))
    mrv_joint_vel_limits = np.array(rospy.get_param('joint_vel_limits'))
    mrv_joint_acc_limits = np.array(rospy.get_param('joint_acc_limits'))
    mrv_joint_torque_limits = np.array(rospy.get_param('joint_torque_limits'))

    client_velocity_noise_ang_amp = rospy.get_param('client_velocity_noise_ang_amp') * np.pi / 180.

    cw_a = rospy.get_param('cw_a')
    cw_mu = rospy.get_param('cw_mu')
    cw_orbit_dir = rospy.get_param('cw_orbit_dir')

    peg_rad = 0.01505
    nozzle_opening_rad = 0.142

    use_cw = rospy.get_param('use_cw')

    do_save = rospy.get_param('do_save')

    load_paths, ws, dps = get_trajlib_load_paths(rospath)

    # Register on_shutdown after it's defined
    rospy.on_shutdown(on_shutdown)

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
                delta_pos = np.array(rospy.get_param('delta_pos'))

            delta_rot = np.array(rospy.get_param('delta_rot')) * np.pi / 180

            print('Desired position difference (m) in nozzle frame: ', delta_pos)
            print('Desired rotation difference (deg) in nozzle frame: ', delta_rot)

            if use_grid:
                delta_v = np.copy(ic_grid[grid_idx, 3:6])
                initial_mrv_w = ic_grid[grid_idx, 6:9] * np.pi / 180
            else:
                delta_v = np.array(rospy.get_param('delta_v'))
                initial_mrv_w = np.array(rospy.get_param('initial_mrv_w')) * np.pi / 180

            if use_grid:
                initial_client_w = ic_grid[grid_idx, 9:12] * np.pi / 180
            else:
                initial_client_w = np.array(rospy.get_param('initial_client_w')) * np.pi / 180

            if do_vision_delay:
                vision_measurement_rate = rospy.get_param('vision_measurement_rate')
            else:
                vision_measurement_rate = dt

            time_steps_between_measurements = dt / (1 / vision_measurement_rate)

            clip_joint_commands = rospy.get_param('clip_joint_commands')
            time_limit = rospy.get_param('time_limit')
            debug_with_test_traj = rospy.get_param('debug_with_test_traj')
            test_traj_id = rospy.get_param('test_traj_id')
            lock_client = rospy.get_param('lock_client')
            lock_mrv = rospy.get_param('lock_mrv')
            probe_z_axis_plunge_velocity = rospy.get_param('probe_z_axis_plunge_velocity')
            use_variable_plunge_speed = rospy.get_param('use_variable_plunge_speed')
            use_scheduled_gains = rospy.get_param('use_scheduled_gains')
            use_ekf = rospy.get_param('use_ekf')

            mrv_controller = MrvController(
                rospath + '/urdf/robot_cv_detached.urdf',
                rospath + '/urdf/robot.urdf',
                rospath + '/urdf/robot.urdf',
                rospath + '/urdf/cv.urdf',
                mrv_joint_angle_lower_limits, mrv_joint_angle_upper_limits,
                mrv_joint_vel_limits, mrv_joint_acc_limits, mrv_joint_torque_limits, dt,
                cone_slope, clip_joint_commands,
                time_steps_between_measurements, cw_a, cw_mu, cw_orbit_dir, do_noisy_state_estimation,
                nozzle_opening_rad, peg_rad, client_velocity_noise_ang_amp, time_limit,
                debug_with_test_traj, test_traj_id, lock_client, lock_mrv, probe_z_axis_plunge_velocity,
                use_variable_plunge_speed,
                use_scheduled_gains, use_cw=use_cw, use_ekf=use_ekf)

            rng = np.random.default_rng(rng_sequences[x])

            load_paths_for_interpolation, weights = interp_trajectories_on_init_client_state(
                load_paths, initial_client_w, ws, delta_pos, dps)

            mrv_controller.reset_wrt_capture_box(
                load_paths_for_interpolation,
                weights, delta_pos, delta_rot, delta_v, initial_client_w, initial_mrv_w, rng,
                dist_centering_waypoint_from_goal)

            sw_peg_pos, sw_peg_rmat, sw_peg_twist, sw_nozzle_pos, sw_nozzle_rmat, sw_nozzle_twist = mrv_controller.get_peg_and_nozzle_info()
            sw_base_pos, sw_base_rmat, sw_joint_angles, sw_base_v, sw_base_w, sw_joint_vels, sw_client_pos, sw_client_rmat, sw_client_v, sw_client_w = mrv_controller.get_state_in_pieces()

            timestr = time.strftime("%Y%m%d-%H%M%S")

            save_path_str = rospy.get_param('save_folder') + '/' + '_'.join([
                'pos_', str(delta_pos[0]), str(delta_pos[1]), str(delta_pos[2]), 'rot', str(delta_rot[0]),
                str(delta_rot[1]), str(delta_rot[2]), 'delta_v', str(delta_v[0]), str(delta_v[1]),
                str(delta_v[2]), 'mrv_w', str(initial_mrv_w[0]), str(initial_mrv_w[1]),
                str(initial_mrv_w[2]), 'client_w', str(initial_client_w[0]), str(initial_client_w[1]),
                str(initial_client_w[2])])
            save_path = rospath + '/' + save_path_str + timestr + '/'

            if do_save:
                os.makedirs(save_path, exist_ok=True)

            success = False
            fail_reason = 'None'

            gc.disable()

            rate = rospy.Rate(1 / dt)

            # Sim step setting
            apply_wrench_only_when_close = False

            start_time = time.time()
            while not rospy.is_shutdown():
                # do_step = time.time() > start_time + 2
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
                sim_vis_publisher.delete_filter_marker()
                if visualize:
                    sw_base_pos, sw_base_rmat, sw_joint_angles, sw_base_v, sw_base_w, sw_joint_vels, sw_client_pos, sw_client_rmat, sw_client_v, sw_client_w = mrv_controller.get_state_in_pieces()
                    traj_pos, traj_rmat = mrv_controller.get_traj()
                    if use_ekf:
                        particle_positions, high_weight_particle_position = mrv_controller.get_ekf_estimate()
                    else:
                        particle_positions, high_weight_particle_position = mrv_controller.get_particle_positions()
                    high_weight_particle_position = high_weight_particle_position.reshape(1, 3)
                    sim_vis_publisher.publish(sw_joint_angles, sw_base_pos, sw_base_rmat, sw_client_pos, sw_client_rmat, traj_pos, traj_rmat)
                    sim_vis_publisher.publish_filter_marker(particle_positions, is_red=True)
                    sim_vis_publisher.publish_filter_marker(high_weight_particle_position, is_red=False)

                    rate.sleep()

            gc.enable()
            if do_save:
                print('Saving data')
                print('success: ', success)

                mrv_controller.save(save_path)

                np.save(save_path + 'success.npy', success)
                np.save(save_path + 'fail_reason.npy', fail_reason)

                print('Saved data')

            if rospy.is_shutdown():
                break

            print()

            if not use_grid:
                break

    if visualize:
        while not rospy.is_shutdown():
            sim_vis_publisher.publish(sw_joint_angles, sw_base_pos, sw_base_rmat, sw_client_pos, sw_client_rmat, traj_pos, traj_rmat)
            rate.sleep()

if __name__ == '__main__':
    # Run the main function under cProfile
    cProfile.run('main()', filename='/home/medusar/bspin/on_orbit/profiler_info/profiling_particle_filter.prof')
