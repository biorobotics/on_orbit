import os
import numpy as np
import matplotlib.pyplot as plt

def load_npy_data(file_path):
    """Load .npy file from specified file path."""
    if os.path.exists(file_path):
        return np.load(file_path, allow_pickle=True)
    print(f"File not found: {file_path}")
    return None

def finite_difference(data, time_step):
    """Calculate finite difference to approximate acceleration."""
    return np.diff(data, axis=0) / time_step

def overall_velocity(joint_velocities):
    """Calculate the overall joint velocity (e.g., average across joints)."""
    return np.mean(np.abs(joint_velocities), axis=1)

def gather_velocity_data(directory):
    """Gather joint velocity data from real and planned trajectories."""
    real_velocities = load_npy_data(os.path.join(directory, 'sw_joint_vels_trj.npy'))
    planned_velocities = load_npy_data(os.path.join(directory, 'ref_joint_vels_trj.npy'))

    if real_velocities is None or planned_velocities is None:
        print("Error loading velocity data.")
        return None, None

    return real_velocities, planned_velocities

def plot_velocity_and_acceleration(real_velocities, planned_velocities, time_step=0.01):
    """Plot overall joint velocity and acceleration for both real and planned trajectories."""
    # Find the shorter length between the two datasets
    min_length = min(len(real_velocities), len(planned_velocities))
    real_velocities = real_velocities[:min_length]
    planned_velocities = planned_velocities[:min_length]

    # Calculate overall velocity (mean absolute velocity across joints)
    real_overall_velocity = overall_velocity(real_velocities)
    planned_overall_velocity = overall_velocity(planned_velocities)

    # Create time arrays based on the shortened length
    velocity_time = np.arange(0, min_length * time_step, time_step)
    acceleration_time = np.arange(0, (min_length - 1) * time_step, time_step)

    # Calculate overall acceleration using finite difference
    real_overall_acceleration = finite_difference(real_overall_velocity, time_step)
    planned_overall_acceleration = finite_difference(planned_overall_velocity, time_step)

    # Plot overall velocity
    plt.figure(figsize=(10, 6))
    plt.plot(velocity_time, real_overall_velocity, label='Real Overall Velocity', color='blue')
    plt.plot(velocity_time, planned_overall_velocity, label='Planned Overall Velocity', color='red', linestyle='--')
    plt.xlabel('Simulation Time (s)')
    plt.ylabel('Overall Velocity')
    plt.title('Overall Joint Velocity')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot overall acceleration
    plt.figure(figsize=(10, 6))
    plt.plot(acceleration_time, real_overall_acceleration, label='Simulated Acceleration', color='blue', linewidth= 0.3)
    plt.plot(acceleration_time, planned_overall_acceleration, label='Planned Acceleration', color='orange', linestyle='--', linewidth=3)
    plt.xlabel('Simulation Time (s)')
    plt.ylabel('Overall Acceleration')
    plt.title('Overall Joint Acceleration')
    plt.legend()
    plt.grid(False)
    plt.show()

def main(directory):
    real_velocities, planned_velocities = gather_velocity_data(directory)
    if real_velocities is not None and planned_velocities is not None:
        plot_velocity_and_acceleration(real_velocities, planned_velocities)

# Example usage
main('/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/10_25_24/for_video_noise/pos__0.0_0.0_-0.05_rot_0.0_0.0_0.0_delta_v_0.0_0.0_0.0_mrv_w_0.0_0.0_0.0_client_w_0.0_0.0_0.020241026-154023')
