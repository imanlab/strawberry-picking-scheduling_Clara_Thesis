import os
import glob
import pandas as pd
import re

# Define input and output folders
input_folder = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/test/final_scheduling_annotations_allrows_astar'  # Replace with your input folder path
output_folder = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/test/final_scheduling_annotations_allrows_astar'  # Replace with your output folder path

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Regular expression to extract the number from the file name
file_number_pattern = re.compile(r'results_(\d+)_output\.csv')

# Get list of input CSV files
input_files = glob.glob(os.path.join(input_folder, '*.csv'))

# Process each input file
for file_path in input_files:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Sort the DataFrame by the 'Scheduling Order' column in ascending order
    df_sorted = df.sort_values(by='Scheduling Order', ascending=True)
    
    # Extract the number from the file name
    file_name = os.path.basename(file_path)
    match = file_number_pattern.match(file_name)
    if match:
        file_number = match.group(1)
        print(file_number)
        # Generate the output file name
        output_file_name = f'{file_number}_results_output.csv'
        output_file_path = os.path.join(output_folder, output_file_name)
        
        # Write the sorted DataFrame to a new CSV file
        df_sorted.to_csv(output_file_path, index=False)
    else:
        print(f"Skipping file with unexpected name format: {file_name}")

print("Processing complete.")

