import os
import re

# Directory where files need to be renamed
directory = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/train/final_scheduling_annotations_allrows_astar'

# Function to rename files
def rename_files(directory):
    for filename in os.listdir(directory):
        # Match the pattern 'results_<number>_output.csv'
        match = re.match(r'results_(\d+)_output.csv', filename)
        if match:
            number = match.group(1)
            # Construct the new filename '<number>_results_output.csv'
            new_filename = f'{number}_results_output.csv'
            # Get the full paths
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_filename)
            # Rename the file
            os.rename(old_file, new_file)
            print(f'Renamed: {filename} to {new_filename}')

# Run the renaming function
rename_files(directory)

