import os
import re

# Define the input folder
input_folder = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/val/final_scheduling_annotations_allrows_astar'  # Replace with your input folder path

# Regular expression to identify correct and incorrect file names
correct_pattern = re.compile(r'^\d+_results_output\.csv$')
incorrect_pattern = re.compile(r'(\d+)_results.*_output\.csv$')

# Iterate over the files in the folder
for file_name in os.listdir(input_folder):
    if not correct_pattern.match(file_name):
        match = incorrect_pattern.match(file_name)
        if match:
            file_number = match.group(1)
            correct_file_name = f'{file_number}_results_output.csv'
            old_file_path = os.path.join(input_folder, file_name)
            new_file_path = os.path.join(input_folder, correct_file_name)
            
            # Rename the file
            os.rename(old_file_path, new_file_path)
            print(f'Renamed: {file_name} -> {correct_file_name}')
        else:
            print(f'Skipping unrecognized file: {file_name}')
    else:
        print(f'Correct name: {file_name}')

print("Processing complete.")

