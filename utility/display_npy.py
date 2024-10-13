import numpy as np

def print_npy_file(file_path):
    # Load the .npy file
    data = np.load(file_path, allow_pickle=True)
    
    # Print the contents of the .npy file
    print("Contents of the .npy file:\n")
    print(data)
    print("Shape of the data:", data.shape)

def main():
    # Specify the path to the .npy file you want to print
    file_path = '/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/10_09_24/2cm_noise_bad_orient/pos__0.0_0.0_-0.05_rot_0.0_0.0_0.0_delta_v_0.0_0.0_0.0_mrv_w_0.0_0.0_0.0_client_w_0.0_0.0_0.0_20241009-104126/ee_mrv_pos_trj.npy'
    
    # Call the function to print the contents of the .npy file
    print_npy_file(file_path)

if __name__ == "__main__":
    main()
