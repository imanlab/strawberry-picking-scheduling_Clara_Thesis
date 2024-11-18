import os
import re

# Define the input folder

input_folder = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/val/final_scheduling_annotations_allrows_ga'  # Replace with your input folder path

# Regular expression to match the old filename pattern and capture the number
pattern = re.compile(r'(\d+)_shortest_distances_paths_output\.csv')

# Iterate over the files in the folder
for filename in os.listdir(input_folder):
    match = pattern.match(filename)
    if match:
        # Extract the number from the filename
        number = match.group(1)
        # Create the new filename
        new_filename = f'{number}_results_output.csv'
        # Construct full file paths
        old_file_path = os.path.join(input_folder, filename)
        new_file_path = os.path.join(input_folder, new_filename)
        # Rename the file
        os.rename(old_file_path, new_file_path)
        print(f'Renamed: {filename} -> {new_filename}')
    else:
        print(f'Skipping file with unexpected name format: {filename}')

print("Renaming complete.")

