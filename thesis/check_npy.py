import os
import numpy as np

# Step  1: Define the path to the directory containing the .npy files
folder_path = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/train/npy_seg'

# Create an empty list to store the filenames of .npy files that contain only zeros
zero_files = []

# Iterate over all files in the directory
for filename in os.listdir(folder_path):
    # Check if the file is a .npy file
    if filename.endswith('.npy'):
        # Construct the full file path
        filepath = os.path.join(folder_path, filename)
        
        # Load the .npy file
        try:
            data = np.load(filepath)
        except IOError:
            print(f"Could not open file {filepath}")
            continue  # Skip this file and move to the next one
        
        # Check if the loaded array contains only zeros
        if np.all(data ==   0):
            zero_files.append(filename)

# Save the list of filenames to a text file
with open('zero_files_data.txt', 'w') as f:
    for file in zero_files:
        f.write(file + '\n')

print("List of files containing only zeros saved to 'zero_files.txt'")

