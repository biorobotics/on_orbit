#!/usr/bin/env python3
import rospy
import numpy as np
import rospkg
from hil_runner import HILRunner
from sim_ros_vis_publisher import SimROSVisPublisher
from holodeck_interface import HolodeckInterface

calibrated = False

def main():
    global calibrated

    rospack = rospkg.RosPack()
    rospath = rospack.get_path('on_orbit')
    rospy.init_node('debug_holo', anonymous=True)

    mrv_hil_home_angles = np.array(rospy.get_param('/mrv_hil_home_angles'))
    client_hil_home_angles = np.array(rospy.get_param('/client_hil_home_angles'))


    hil_runner = HILRunner(rospath, mrv_hil_home_angles, client_hil_home_angles, dt=0.01)
    sim_vis_publisher = SimROSVisPublisher(rospath)

    cone_slope = rospy.get_param('cone_slope')

    mrv_joint_angle_lower_limits = np.array(rospy.get_param('joint_angle_lower_limits'))
    mrv_joint_angle_upper_limits = np.array(rospy.get_param('joint_angle_upper_limits'))
    mrv_joint_vel_limits = np.array(rospy.get_param('joint_vel_limits'))
    mrv_joint_acc_limits = np.array(rospy.get_param('joint_acc_limits'))
    mrv_joint_torque_limits = np.array(rospy.get_param('joint_torque_limits'))
    apply_wrench_only_when_close = rospy.get_param('apply_wrench_only_when_close')

    client_velocity_noise_ang_amp = rospy.get_param('client_velocity_noise_ang_amp')*np.pi/180.

    cw_a = rospy.get_param('cw_a')
    cw_mu = rospy.get_param('cw_mu')
    cw_orbit_dir = rospy.get_param('cw_orbit_dir')

    peg_rad = 0.01505
    nozzle_opening_rad = 0.142

    use_cw = rospy.get_param('use_cw')

    do_save = rospy.get_param('do_save')

    holo_control = HolodeckInterface(dt=0.01)

    holo_control.ur_idle_mode('mrv')
    holo_control.ur_idle_mode('client')

    hil_runner.reset_to_home_angles(check_for_continue=True)
    
    holo_control.ur_idle_mode('mrv')
    holo_control.ur_idle_mode('client')

    hil_runner.move_peg_out_of_hole()
    # if not calibrated:
    #     hil_runner.calibrate_ft_bias()
    #     calibrated = True
    
    # rate = rospy.Rate(1/0.01)
    # while not rospy.is_shutdown():
    #     if calibrated:
    #         hil_runner.print_ft_compensated()
    #     rate.sleep()

        
    rospy.spin()

if __name__ == '__main__':
    main()