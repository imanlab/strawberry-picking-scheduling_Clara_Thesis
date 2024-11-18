import os
import pandas as pd
import re

def clean_brackets(path_str):
    # Use regular expressions to remove square brackets
    return re.sub(r'[\[\]]', '', path_str)

def process_csv_files(directory):
    if not os.path.isdir(directory):
        print(f"The specified path is not a directory: {directory}")
        return
    
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in the directory: {directory}")
        return
    
    output_dir = os.path.join(directory, "cleaned_output")
    os.makedirs(output_dir, exist_ok=True)
    
    for csv_file in csv_files:
        file_path = os.path.join(directory, csv_file)
        try:
            df = pd.read_csv(file_path)
            df['Path'] = df['Path'].apply(clean_brackets)
            output_file_path = os.path.join(output_dir, csv_file)
            df.to_csv(output_file_path, index=False)
            print(f"Processed and saved cleaned data to {output_file_path}")
        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")

# Usage
directory = '/home/clara/Thesis/strawberry-pp-w-r-dataset-master/scheduling/scheduling_data_collection/data-collection/test/optimal_path_astar'  # Change this to the path of your folder containing the CSV files
process_csv_files(directory)

