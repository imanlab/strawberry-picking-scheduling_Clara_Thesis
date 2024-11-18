import os
import re

def rename_files(directory):
    # Define the pattern to match the filenames
    pattern = re.compile(r'result_alpha_[\d.]+_image_(\d+)_data\.csv')

    # Loop through all files in the directory
    for filename in os.listdir(directory):
        # Match the pattern
        match = pattern.match(filename)
        if match:
            # Extract the number from the filename
            number = match.group(1)
            # Construct the new filename
            new_filename = f"{number}_results_output.csv"
            # Get the full file paths
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_filename)
            # Rename the file
            os.rename(old_file, new_file)
            print(f"Renamed: {old_file} to {new_file}")

# Define the directory containing the files
directory = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling_riseholme/train/final_scheduling_som'

# Run the rename function
rename_files(directory)

