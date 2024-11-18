import os
import shutil
import re

def rename_and_move_files(source_folder, destination_folder, alpha_value):
    # Ensure destination folder exists
    os.makedirs(destination_folder, exist_ok=True)
    folder_path = os.path.join(source_folder)
    #print(folder_name)
    #print(folder_path)
    for file_name in os.listdir(folder_path):
                print(file_name)
                # Check if the file name matches the scheme result_alpha_<alpha_value>_image_**_data.csv
                if file_name.startswith(f'result_alpha_{alpha_value}_'):
                    file_path = os.path.join(folder_path, file_name)
                    # Construct new file name
                    new_file_name = file_name.replace(f'result_alpha_{alpha_value}_', '').replace('.csv', '_results_output.csv')
                    # Move and rename the file to the destination folder
                    shutil.move(file_path, os.path.join(destination_folder, new_file_name))
                    print(f'Moved and renamed: {file_path} to {os.path.join(destination_folder, new_file_name)}')

# Define source folder, destination folder, and alpha value
source_folder = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/train/coordinates_and_hop/new_alpha_tuning/0_output'
destination_folder = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/train/coordinates_and_hop/new_alpha_tuning/0_output'
alpha_value = '0.0'

# Call the function
rename_and_move_files(source_folder, destination_folder, alpha_value)

# Directory containing the files
directory = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/train/coordinates_and_hop/new_alpha_tuning/0_output'

# Pattern to match the files you want to rename
pattern = re.compile(r'image_(.*?)_data_results_output\.csv')

# Iterate over the files in the directory
for filename in os.listdir(directory):
    # Check if the file matches the pattern
    match = pattern.match(filename)
    if match:
        # Extract the part of the filename to retain
        new_filename = f"{match.group(1)}_results_output.csv"
        
        # Construct the full old and new file paths
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_filename)
        
        # Rename the file
        os.rename(old_file, new_file)
        print(f"Renamed: {filename} to {new_filename}")

