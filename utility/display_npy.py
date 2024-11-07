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
    root = '/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/11_1_24/little_pop_at_beginning'
    file_path = root + '/ref_x_trj.npy'
    
    # Call the function to print the contents of the .npy file
    print_npy_file(file_path)

if __name__ == "__main__":
    main()
