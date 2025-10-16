import numpy as np
import sys
import os

import scipy.io
def convert_mat_to_npy(mat_file_path, npy_file_path=None):
    """
    Convert a .mat file to a .npy file
    
    Args:
        mat_file_path: Path to the input .mat file
        npy_file_path: Path for the output .npy file (optional)
    """
    # Load the .mat file
    mat_data = scipy.io.loadmat(mat_file_path)
    
    # Remove metadata keys that scipy adds
    mat_data = {k: v for k, v in mat_data.items() if not k.startswith('__')}
    
    # If there's only one variable, save it directly
    if len(mat_data) == 1:
        data = list(mat_data.values())[0]
    else:
        # If multiple variables, save as a dictionary
        data = mat_data
    
    # Generate output filename if not provided
    if npy_file_path is None:
        npy_file_path = os.path.splitext(mat_file_path)[0] + '.npy'
    
    # Save as numpy file
    np.save(npy_file_path, data)
    print(f"Converted {mat_file_path} to {npy_file_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_mat_to_npy.py <input.mat> [output.npy]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} does not exist")
        sys.exit(1)
    
    convert_mat_to_npy(input_file, output_file)