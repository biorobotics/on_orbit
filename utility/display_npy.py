import numpy as np

def print_npy_file(file_path):
    # Load the .npy file
    data = np.load(file_path, allow_pickle=True)
    x = data[0]
    quat = x[17:21]
    
    # Print the contents of the .npy file
    print("Contents of the .npy file:\n")
    print(quat)
    print("Shape of the data:", data.shape)

def main():
    # Specify the path to the .npy file you want to print
    root = '/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/traj_lib/11_18_24/align_w_nozzle/pos_0.1_0.1_0.0_rot_0.0_0.0_0.0_delta_v_0.0_0.0_0.0_client_w_0.0_0.0_0.0/control/20241118-093348'
    file_path = root + '/init_xs.npy'
    # Call the function to print the contents of the .npy file
    print_npy_file(file_path)

if __name__ == "__main__":
    main()
