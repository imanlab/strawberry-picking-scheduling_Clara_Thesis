import os
import shutil
import re

# Define the source directory containing the folders
source_directory = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/train/coordinates_and_hop/new_alpha_tuning/output'

# Define the destination base directory where new folders will be created
destination_base_directory = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/train/coordinates_and_hop/new_alpha_tuning/0_output'

# Pattern to match the files you want to move
file_pattern = re.compile(r'result_alpha_0\.0_image_(.*?)_data\.csv')

# Iterate over the folders in the source directory
for foldername in os.listdir(source_directory):
    print(foldername)
    folder_path = os.path.join(source_directory, foldername)
    if os.path.isdir(folder_path):
        # Iterate over the files in the subdirectory
        for filename in os.listdir(folder_path):
            print(f"Processing filename: {filename} in folder: {foldername}")
            # Check if the file matches the pattern
            match = file_pattern.match(filename)
            print(f"Match: {match}")
            if match:
                # Extract the part of the filename to use for the new folder name
                folder_name_part = foldername  # Use the folder name directly
                
                # Construct the full paths for the source file and the new destination folder
                source_file = os.path.join(folder_path, filename)
                new_folder = os.path.join(destination_base_directory, folder_name_part)
                
                # Create the new folder if it doesn't exist
                os.makedirs(new_folder, exist_ok=True)
                
                # Construct the destination file path
                destination_file = os.path.join(new_folder, filename)
                
                # Move the file
                shutil.move(source_file, destination_file)
                print(f"Moved: {filename} to {new_folder}")
            else:
                print(f"No match for filename: {filename} in folder: {foldername}")
